import base64
import binascii
import hmac
import hashlib
import io
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageStat, UnidentifiedImageError
from rembg import remove


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("passpic-ai-bg-remover")
logger.setLevel(logging.INFO)

rate_limit_lock = Lock()
rate_limit_hits = defaultdict(deque)


def env_int(name: str, default: int, minimum: int = 1):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default=%s.", name, raw_value, default)
        return default
    return max(minimum, value)


MAX_REQUEST_BYTES = env_int("BG_REMOVER_MAX_REQUEST_BYTES", 12 * 1024 * 1024)
MAX_IMAGE_BYTES = env_int("BG_REMOVER_MAX_IMAGE_BYTES", 8 * 1024 * 1024)
MAX_IMAGE_PIXELS = env_int("BG_REMOVER_MAX_IMAGE_PIXELS", 20_000_000)
RATE_LIMIT_WINDOW_SECONDS = env_int("BG_REMOVER_RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_MAX_REQUESTS = env_int("BG_REMOVER_RATE_LIMIT_MAX_REQUESTS", 20)
RATE_LIMIT_ENABLED = os.getenv("BG_REMOVER_RATE_LIMIT_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
PHOTO_ENHANCEMENT_ENABLED = os.getenv("PHOTO_ENHANCEMENT_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
PHOTO_ENHANCEMENT_DEFAULT = os.getenv("PHOTO_ENHANCEMENT_DEFAULT", "auto").strip().lower()
PHOTO_ENHANCEMENT_MAX_BRIGHTNESS = env_int("PHOTO_ENHANCEMENT_MAX_BRIGHTNESS", 22, 0)
PHOTO_ENHANCEMENT_MAX_CONTRAST = env_int("PHOTO_ENHANCEMENT_MAX_CONTRAST", 14, 0)
PHOTO_ENHANCEMENT_MAX_SHARPNESS = env_int("PHOTO_ENHANCEMENT_MAX_SHARPNESS", 12, 0)

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def get_allowed_origins():
    raw_origins = os.getenv("BG_REMOVER_ALLOWED_ORIGINS") or os.getenv("CORS_ALLOWED_ORIGINS")
    if raw_origins:
        origins = [origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip()]
        if "*" in origins:
            logger.warning("CORS is configured with wildcard origin. Use explicit origins in production.")
        return origins

    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
    ]


allowed_origins = get_allowed_origins()
app = FastAPI(title="PassPic AI Background Remover")
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_headers=["*"],
    allow_methods=["*"],
    allow_origins=allowed_origins,
)


@app.middleware("http")
async def enforce_request_limits(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    if request.url.path in {"/remove-background", "/cloudinary/delete"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_bytes = int(content_length)
            except ValueError:
                logger.warning("[%s] Invalid Content-Length header: %s", request_id, content_length)
                return limit_error(400, "Invalid Content-Length header.", request_id)

            if request_bytes > MAX_REQUEST_BYTES:
                logger.warning(
                    "[%s] Request rejected: content_length=%s exceeds max_request_bytes=%s path=%s",
                    request_id,
                    request_bytes,
                    MAX_REQUEST_BYTES,
                    request.url.path,
                )
                return limit_error(413, f"Request is too large. Maximum allowed request size is {MAX_REQUEST_BYTES} bytes.", request_id)

        if RATE_LIMIT_ENABLED:
            client_ip = get_client_ip(request)
            allowed, retry_after = rate_limit_allow(client_ip, time.time())
            if not allowed:
                logger.warning(
                    "[%s] Rate limit exceeded: client_ip=%s path=%s retry_after=%s",
                    request_id,
                    client_ip,
                    request.url.path,
                    retry_after,
                )
                response = limit_error(429, "Too many requests. Please wait and try again.", request_id)
                response.headers["Retry-After"] = str(retry_after)
                return response

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def limit_error(status_code: int, detail: str, request_id: str):
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "requestId": request_id},
        headers={"X-Request-ID": request_id},
    )


def get_client_ip(request: Request):
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_allow(client_ip: str, now: float):
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    with rate_limit_lock:
        hits = rate_limit_hits[client_ip]
        while hits and hits[0] <= window_start:
            hits.popleft()

        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - hits[0])))
            return False, retry_after

        hits.append(now)

        if len(rate_limit_hits) > 10000:
            stale_clients = [ip for ip, timestamps in rate_limit_hits.items() if not timestamps or timestamps[-1] <= window_start]
            for ip in stale_clients:
                rate_limit_hits.pop(ip, None)

        return True, 0


class RemoveBackgroundRequest(BaseModel):
    imageBase64: str
    background: str = "white"
    autoEnhance: bool | None = None


class DeleteCloudinaryAssetRequest(BaseModel):
    publicId: str
    deletionToken: str


@app.on_event("startup")
def log_startup_config():
    upload_enabled = cloudinary_upload_enabled()
    logger.info(
        "Background remover startup: cloudinary_upload_enabled=%s, cloudinary_url_configured=%s, cloudinary_cloud_name_configured=%s, cloudinary_api_key_configured=%s, cloudinary_api_secret_configured=%s, cloudinary_folder=%s",
        upload_enabled,
        env_is_configured("CLOUDINARY_URL"),
        env_is_configured("CLOUDINARY_CLOUD_NAME"),
        env_is_configured("CLOUDINARY_API_KEY"),
        env_is_configured("CLOUDINARY_API_SECRET"),
        os.getenv("CLOUDINARY_FOLDER", "passpic-ai/processed"),
    )
    logger.info("CORS allowed origins: %s", allowed_origins)
    logger.info(
        "Request protection: max_request_bytes=%s, max_image_bytes=%s, max_image_pixels=%s, rate_limit_enabled=%s, rate_limit_max_requests=%s, rate_limit_window_seconds=%s",
        MAX_REQUEST_BYTES,
        MAX_IMAGE_BYTES,
        MAX_IMAGE_PIXELS,
        RATE_LIMIT_ENABLED,
        RATE_LIMIT_MAX_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS,
    )
    logger.info(
        "Photo enhancement: enabled=%s, default=%s, max_brightness=%s, max_contrast=%s, max_sharpness=%s",
        PHOTO_ENHANCEMENT_ENABLED,
        PHOTO_ENHANCEMENT_DEFAULT,
        PHOTO_ENHANCEMENT_MAX_BRIGHTNESS,
        PHOTO_ENHANCEMENT_MAX_CONTRAST,
        PHOTO_ENHANCEMENT_MAX_SHARPNESS,
    )
    logger.info(
        "Photo retention policy: uploaded source photos are processed in memory only; Cloudinary processed image deletion after client display confirmation is enabled=%s.",
        cloudinary_delete_enabled(),
    )
    if not upload_enabled:
        logger.info("Cloudinary upload mode is disabled. Service will return imageBase64 responses.")


@app.post("/remove-background")
async def remove_background(fastapi_request: Request):
    request_id = getattr(fastapi_request.state, "request_id", uuid.uuid4().hex[:12])
    started_at = time.perf_counter()
    request = await parse_remove_background_request(fastapi_request, request_id)
    logger.info(
        "[%s] /remove-background started: content_type=%s image_base64_chars=%s background=%s auto_enhance=%s",
        request_id,
        fastapi_request.headers.get("content-type", "unknown"),
        len(request.imageBase64),
        request.background,
        request.autoEnhance,
    )

    source_bytes = decode_image_base64(request.imageBase64, request_id)
    if len(source_bytes) > MAX_IMAGE_BYTES:
        logger.warning("[%s] Image rejected: bytes=%s exceeds max_image_bytes=%s", request_id, len(source_bytes), MAX_IMAGE_BYTES)
        raise HTTPException(status_code=413, detail=f"Image is too large. Maximum decoded image size is {MAX_IMAGE_BYTES} bytes.")

    try:
        source = Image.open(io.BytesIO(source_bytes)).convert("RGBA")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        logger.warning("[%s] Invalid or unsafe image upload: %s", request_id, error)
        raise HTTPException(status_code=400, detail="Uploaded image is invalid, unsupported, or too large.") from error
    validate_image_dimensions(source, request_id)
    logger.info(
        "[%s] Uploaded image received: dimensions=%sx%s, bytes=%s, background=%s",
        request_id,
        source.width,
        source.height,
        len(source_bytes),
        request.background,
    )

    if not contains_human_face(source):
        logger.warning("[%s] Human face validation failed", request_id)
        raise HTTPException(
            status_code=422,
            detail="No human face was detected. Please upload or capture a clear front-facing photo of a person.",
        )

    transparent = remove(source)
    logger.info(
        "[%s] Background removed: dimensions=%sx%s, alpha_bbox=%s",
        request_id,
        transparent.width,
        transparent.height,
        transparent.getchannel("A").getbbox(),
    )
    if not contains_person_sized_foreground(transparent):
        logger.warning("[%s] Foreground validation failed after background removal", request_id)
        raise HTTPException(
            status_code=422,
            detail="The person could not be separated clearly from the background. Please try a clearer photo with the face and shoulders visible.",
        )

    transparent = trim_to_subject(transparent)
    logger.info("[%s] Image trimmed to subject: dimensions=%sx%s", request_id, transparent.width, transparent.height)
    white_canvas = Image.new("RGBA", transparent.size, (255, 255, 255, 255))
    white_canvas.alpha_composite(transparent)
    final_image = white_canvas.convert("RGB")
    if should_auto_enhance(request):
        final_image = enhance_photo(final_image, transparent.getchannel("A"), request_id)
    else:
        logger.info("[%s] Photo enhancement skipped: enabled=%s requested=%s", request_id, PHOTO_ENHANCEMENT_ENABLED, request.autoEnhance)

    output = io.BytesIO()
    final_image.save(output, format="JPEG", quality=98)
    logger.info(
        "[%s] Prepared image encoded: dimensions=%sx%s, bytes=%s",
        request_id,
        final_image.width,
        final_image.height,
        output.tell(),
    )

    output_bytes = output.getvalue()
    if cloudinary_upload_enabled():
        logger.info("[%s] Cloudinary upload mode enabled. Uploading processed image.", request_id)
        cloudinary_asset = upload_to_cloudinary(output_bytes, request_id)
        logger.info("[%s] Returning imageUrl response after %.2fs", request_id, time.perf_counter() - started_at)
        return cloudinary_asset

    logger.info("[%s] Cloudinary upload mode disabled. Returning imageBase64 response after %.2fs", request_id, time.perf_counter() - started_at)
    return {"imageBase64": base64.b64encode(output_bytes).decode("utf-8")}


async def parse_remove_background_request(request: Request, request_id: str):
    content_type = request.headers.get("content-type", "").lower()
    image_base64 = None
    background = "white"
    auto_enhance = None

    if "application/json" in content_type or not content_type:
        try:
            payload = await request.json()
        except Exception as error:
            logger.warning("[%s] Invalid JSON request body for /remove-background: %s", request_id, error)
            raise HTTPException(status_code=400, detail="Request body must be valid JSON with imageBase64.") from error

        if not isinstance(payload, dict):
            logger.warning("[%s] Invalid JSON request body type for /remove-background: %s", request_id, type(payload).__name__)
            raise HTTPException(status_code=400, detail="Request body must be a JSON object with imageBase64.")

        image_base64, image_field = get_first_string(payload, "imageBase64", "image_base64", "base64Image", "image", "photo")
        background = str(payload.get("background") or "white")
        auto_enhance = parse_optional_bool(payload.get("autoEnhance", payload.get("auto_enhance")))
        logger.info("[%s] Parsed JSON /remove-background payload using field=%s", request_id, image_field or "missing")
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
        except Exception as error:
            logger.warning("[%s] Invalid form request body for /remove-background: %s", request_id, error)
            raise HTTPException(status_code=400, detail="Request body must include imageBase64 or an uploaded image file.") from error

        image_base64, image_field = get_first_string(form, "imageBase64", "image_base64", "base64Image")
        background = str(form.get("background") or "white")
        auto_enhance = parse_optional_bool(form.get("autoEnhance", form.get("auto_enhance")))

        if not image_base64:
            for file_field in ("file", "image", "photo"):
                uploaded_file = form.get(file_field)
                if uploaded_file is not None and hasattr(uploaded_file, "read"):
                    file_bytes = await uploaded_file.read()
                    image_base64 = base64.b64encode(file_bytes).decode("utf-8")
                    image_field = file_field
                    logger.info("[%s] Parsed multipart /remove-background upload using file_field=%s bytes=%s", request_id, file_field, len(file_bytes))
                    break

        logger.info("[%s] Parsed form /remove-background payload using field=%s", request_id, image_field or "missing")
    else:
        logger.warning("[%s] Unsupported /remove-background content_type=%s", request_id, content_type or "missing")
        raise HTTPException(status_code=415, detail="Unsupported request content type. Send JSON with imageBase64.")

    if not image_base64:
        logger.warning("[%s] /remove-background missing imageBase64. content_type=%s", request_id, content_type or "missing")
        raise HTTPException(status_code=400, detail="imageBase64 is required. Send JSON as {\"imageBase64\":\"...\"}.")

    return RemoveBackgroundRequest(imageBase64=image_base64, background=background, autoEnhance=auto_enhance)


def get_first_string(mapping, *keys: str):
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value, key
    return None, None


def parse_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def should_auto_enhance(request: RemoveBackgroundRequest):
    if not PHOTO_ENHANCEMENT_ENABLED:
        return False
    if request.autoEnhance is not None:
        return request.autoEnhance
    return PHOTO_ENHANCEMENT_DEFAULT in {"1", "true", "yes", "on", "auto"}


def enhance_photo(image: Image.Image, alpha_mask: Image.Image, request_id: str):
    rgb = image.convert("RGB")
    subject_mask = alpha_mask.point(lambda value: 255 if value > 16 else 0)
    if not subject_mask.getbbox():
        logger.info("[%s] Photo enhancement skipped: no foreground mask available", request_id)
        return rgb

    stat = ImageStat.Stat(rgb, subject_mask)
    mean_luma = sum(stat.mean) / 3
    brightness_delta = max(0, min(PHOTO_ENHANCEMENT_MAX_BRIGHTNESS, int((178 - mean_luma) / 3.8)))
    contrast_delta = max(0, min(PHOTO_ENHANCEMENT_MAX_CONTRAST, int((152 - min(stat.stddev)) / 9)))
    sharpness_delta = PHOTO_ENHANCEMENT_MAX_SHARPNESS if min(image.size) >= 480 else max(0, PHOTO_ENHANCEMENT_MAX_SHARPNESS // 2)

    brightness_factor = 1 + (brightness_delta / 100)
    contrast_factor = 1 + (contrast_delta / 100)
    sharpness_factor = 1 + (sharpness_delta / 100)

    enhanced = rgb
    if brightness_delta:
        enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness_factor)
    if contrast_delta:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)
    if sharpness_delta:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness_factor)
    final_image = Image.composite(enhanced, rgb, subject_mask)

    logger.info(
        "[%s] Photo enhancement applied to foreground: mean_luma=%.2f brightness_factor=%.3f contrast_factor=%.3f sharpness_factor=%.3f",
        request_id,
        mean_luma,
        brightness_factor,
        contrast_factor,
        sharpness_factor,
    )
    return final_image


def decode_image_base64(image_base64: str, request_id: str):
    if not image_base64:
        raise HTTPException(status_code=400, detail="imageBase64 is required.")

    if image_base64.startswith("data:"):
        _, _, image_base64 = image_base64.partition(",")

    max_base64_chars = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 128
    if len(image_base64) > max_base64_chars:
        logger.warning("[%s] Base64 image rejected: chars=%s exceeds max_base64_chars=%s", request_id, len(image_base64), max_base64_chars)
        raise HTTPException(status_code=413, detail=f"Image is too large. Maximum decoded image size is {MAX_IMAGE_BYTES} bytes.")

    try:
        return base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        logger.warning("[%s] Invalid base64 image payload: %s", request_id, error)
        raise HTTPException(status_code=400, detail="imageBase64 must be a valid base64 encoded image.") from error


def validate_image_dimensions(image: Image.Image, request_id: str):
    pixels = image.width * image.height
    if pixels > MAX_IMAGE_PIXELS:
        logger.warning("[%s] Image rejected: dimensions=%sx%s pixels=%s exceeds max_image_pixels=%s", request_id, image.width, image.height, pixels, MAX_IMAGE_PIXELS)
        raise HTTPException(status_code=413, detail=f"Image dimensions are too large. Maximum allowed pixels is {MAX_IMAGE_PIXELS}.")


@app.post("/cloudinary/delete")
def delete_cloudinary_asset(request: DeleteCloudinaryAssetRequest, fastapi_request: Request):
    request_id = getattr(fastapi_request.state, "request_id", uuid.uuid4().hex[:12])
    public_id = request.publicId.strip()
    logger.info("[%s] /cloudinary/delete started: public_id=%s", request_id, public_id)

    if not cloudinary_upload_enabled():
        logger.info("[%s] Cloudinary upload mode disabled. Delete request skipped.", request_id)
        return {"deleted": False, "skipped": True}

    if not cloudinary_delete_enabled():
        logger.info("[%s] Cloudinary delete mode disabled. Delete request skipped: public_id=%s", request_id, public_id)
        return {"deleted": False, "skipped": True}

    if not public_id:
        raise HTTPException(status_code=400, detail="Cloudinary public ID is required.")

    if not valid_deletion_token(public_id, request.deletionToken):
        logger.warning("[%s] Invalid Cloudinary deletion token for public_id=%s", request_id, public_id)
        raise HTTPException(status_code=403, detail="Invalid Cloudinary deletion token.")

    try:
        import cloudinary.uploader
    except ImportError as error:
        logger.exception("[%s] Cloudinary package import failed during delete", request_id)
        raise HTTPException(
            status_code=500,
            detail="Cloudinary upload is enabled, but the cloudinary package is not installed.",
        ) from error

    configure_cloudinary(request_id)
    try:
        result = cloudinary.uploader.destroy(public_id, resource_type="image", invalidate=True)
    except Exception as error:
        logger.exception("[%s] Cloudinary delete failed: public_id=%s", request_id, public_id)
        raise HTTPException(status_code=502, detail=f"Cloudinary delete failed: {error}") from error

    cloudinary_result = result.get("result")
    deleted = cloudinary_result in {"ok", "not found"}
    logger.info(
        "[%s] Cloudinary delete completed: public_id=%s, result=%s, deleted=%s",
        request_id,
        public_id,
        cloudinary_result,
        deleted,
    )
    return {"deleted": deleted, "result": cloudinary_result}


def cloudinary_upload_enabled():
    return os.getenv("CLOUDINARY_UPLOAD_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def cloudinary_delete_enabled():
    return os.getenv("CLOUDINARY_DELETE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def configure_cloudinary(request_id: str):
    try:
        import cloudinary
    except ImportError as error:
        logger.exception("[%s] Cloudinary package import failed", request_id)
        raise HTTPException(
            status_code=500,
            detail="Cloudinary upload is enabled, but the cloudinary package is not installed.",
        ) from error

    if os.getenv("CLOUDINARY_URL"):
        logger.info("[%s] Configuring Cloudinary from CLOUDINARY_URL", request_id)
        cloudinary.config(secure=True)
    else:
        cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        api_key = os.getenv("CLOUDINARY_API_KEY")
        api_secret = os.getenv("CLOUDINARY_API_SECRET")
        if not cloud_name or not api_key or not api_secret:
            logger.error(
                "[%s] Cloudinary credentials missing: cloud_name_configured=%s, api_key_configured=%s, api_secret_configured=%s",
                request_id,
                bool(cloud_name),
                bool(api_key),
                bool(api_secret),
            )
            raise HTTPException(
                status_code=500,
                detail="Cloudinary upload is enabled, but Cloudinary credentials are not configured.",
            )
        logger.info("[%s] Configuring Cloudinary from separate credentials: cloud_name=%s", request_id, cloud_name)
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )


def upload_to_cloudinary(image_bytes: bytes, request_id: str):
    try:
        import cloudinary.uploader
    except ImportError as error:
        logger.exception("[%s] Cloudinary package import failed", request_id)
        raise HTTPException(
            status_code=500,
            detail="Cloudinary upload is enabled, but the cloudinary package is not installed.",
        ) from error

    configure_cloudinary(request_id)

    folder = os.getenv("CLOUDINARY_FOLDER", "passpic-ai/processed")
    public_id = f"passpic-{uuid.uuid4().hex}"
    upload_options = {
        "folder": folder,
        "format": "jpg",
        "overwrite": False,
        "public_id": public_id,
        "resource_type": "image",
    }
    logger.info(
        "[%s] Cloudinary upload starting: folder=%s, public_id=%s, bytes=%s",
        request_id,
        folder,
        public_id,
        len(image_bytes),
    )

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    try:
        result = cloudinary.uploader.upload(f"data:image/jpeg;base64,{encoded}", **upload_options)
    except Exception as error:
        logger.exception("[%s] Cloudinary upload failed", request_id)
        raise HTTPException(status_code=502, detail=f"Cloudinary upload failed: {error}") from error

    secure_url = result.get("secure_url")
    if not secure_url:
        logger.error("[%s] Cloudinary upload response missing secure_url: keys=%s", request_id, sorted(result.keys()))
        raise HTTPException(status_code=500, detail="Cloudinary did not return an image URL.")

    uploaded_public_id = result.get("public_id")
    logger.info(
        "[%s] Cloudinary upload succeeded: secure_url=%s, public_id=%s, format=%s, width=%s, height=%s, bytes=%s",
        request_id,
        secure_url,
        uploaded_public_id,
        result.get("format"),
        result.get("width"),
        result.get("height"),
        result.get("bytes"),
    )
    return {
        "imageUrl": secure_url,
        "cloudinaryPublicId": uploaded_public_id,
        "cloudinaryDeleteToken": create_deletion_token(uploaded_public_id),
    }


def get_deletion_secret():
    secret = (
        os.getenv("CLOUDINARY_DELETE_SIGNING_SECRET")
        or os.getenv("CLOUDINARY_API_SECRET")
        or os.getenv("CLOUDINARY_URL")
    )
    if not secret:
        logger.error("Cloudinary delete signing secret is not configured.")
        raise HTTPException(status_code=500, detail="Cloudinary deletion signing secret is not configured.")
    return secret.encode("utf-8")


def create_deletion_token(public_id: str):
    if not public_id:
        return ""
    return hmac.new(get_deletion_secret(), public_id.encode("utf-8"), hashlib.sha256).hexdigest()


def valid_deletion_token(public_id: str, token: str):
    expected = create_deletion_token(public_id)
    return bool(expected and token and hmac.compare_digest(expected, token))


def env_is_configured(name: str):
    return bool(os.getenv(name))


def contains_human_face(image: Image.Image):
    rgb = image.convert("RGB")
    frame = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    frontal = load_cascade("haarcascade_frontalface_default.xml")
    profile = load_cascade("haarcascade_profileface.xml")
    min_side = max(48, min(gray.shape[:2]) // 12)

    frontal_faces = frontal.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_side, min_side),
    )
    if len(frontal_faces) > 0:
        return True

    profile_faces = profile.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_side, min_side),
    )
    return len(profile_faces) > 0


def load_cascade(file_name: str):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + file_name)
    if cascade.empty():
        raise RuntimeError(f"Unable to load OpenCV cascade: {file_name}")
    return cascade


def contains_person_sized_foreground(image: Image.Image):
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return False

    left, top, right, bottom = bbox
    width, height = image.size
    subject_width = right - left
    subject_height = bottom - top
    subject_area_ratio = (subject_width * subject_height) / (width * height)

    return subject_width >= width * 0.12 and subject_height >= height * 0.18 and subject_area_ratio >= 0.03


def trim_to_subject(image: Image.Image):
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return image

    left, top, right, bottom = bbox
    width, height = image.size
    subject_width = right - left
    subject_height = bottom - top

    pad_x = int(subject_width * 0.12)
    pad_top = int(subject_height * 0.14)
    pad_bottom = int(subject_height * 0.03)

    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_top),
        min(width, right + pad_x),
        min(height, bottom + pad_bottom),
    )

    return image.crop(crop_box)

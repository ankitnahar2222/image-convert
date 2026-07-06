import base64
import io

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from rembg import remove


app = FastAPI(title="FitForVisa Local Background Remover")


class RemoveBackgroundRequest(BaseModel):
    imageBase64: str
    background: str = "white"


@app.post("/remove-background")
def remove_background(request: RemoveBackgroundRequest):
    source_bytes = base64.b64decode(request.imageBase64)
    source = Image.open(io.BytesIO(source_bytes)).convert("RGBA")

    if not contains_human_face(source):
        raise HTTPException(
            status_code=422,
            detail="No human face was detected. Please upload or capture a clear front-facing photo of a person.",
        )

    transparent = remove(source)
    if not contains_person_sized_foreground(transparent):
        raise HTTPException(
            status_code=422,
            detail="The person could not be separated clearly from the background. Please try a clearer photo with the face and shoulders visible.",
        )

    transparent = trim_to_subject(transparent)
    white_canvas = Image.new("RGBA", transparent.size, (255, 255, 255, 255))
    white_canvas.alpha_composite(transparent)

    output = io.BytesIO()
    white_canvas.convert("RGB").save(output, format="JPEG", quality=98)

    return {"imageBase64": base64.b64encode(output.getvalue()).decode("utf-8")}


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

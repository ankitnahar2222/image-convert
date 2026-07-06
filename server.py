import importlib.util
from pathlib import Path


service_path = Path(__file__).resolve().parents[1] / "local-bg-remover" / "server.py"
spec = importlib.util.spec_from_file_location("fitforvisa_bg_remover", service_path)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load local-bg-remover/server.py")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app

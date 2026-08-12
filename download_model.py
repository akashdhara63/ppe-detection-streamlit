
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
target = MODEL_DIR / "best.pt"

source = hf_hub_download(
    repo_id="ayushgupta7777/safetyvision-yolov8",
    filename="v2/best.pt",
)

shutil.copy2(source, target)
print(f"Saved PPE model to: {target}")

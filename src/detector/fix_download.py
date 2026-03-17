import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

snapshot_download(
    repo_id="ai4bharat/indic-bert",
    token="your_actual_token_here",
    resume_download=True,
    max_workers=1
)
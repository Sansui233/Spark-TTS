import os

from huggingface_hub import snapshot_download

model_dir = "pretrained_models/Spark-TTS-0.5B"

if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
    print("Model files already exist. Skipping download.")
else:
    print("Downloading model files...")
    snapshot_download(
        repo_id="SparkAudio/Spark-TTS-0.5B",
        local_dir=model_dir,
        revision="main",
        resume_download=True,
    )
    print("Download complete!")

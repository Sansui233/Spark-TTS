"""
有参考克隆声音
```
curl -X POST "http://localhost:7860/voice_clone" \
     -H "Content-Type: application/json" \
     -d '{"text":"Hello","prompt_text":"你好","prompt_audio_path":"audio/sample.wav"}'
```

无参考合成声音
```
curl -X POST "http://localhost:7860/voice_creation" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "要合成的文本",
           "gender": "female",
           "pitch": 3,
           "speed": 2
         }'
```

"""

import argparse
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import Body, FastAPI, HTTPException, Response

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI
from tools.time import timer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 全局变量
model: SparkTTS
save_dir = "example/results"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # start code
    parser = argparse.ArgumentParser(description="Spark TTS FastAPI server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0).",
    )
    args, _ = parser.parse_known_args()
    model = initialize_model(model_dir=args.model_dir, device=args.device)
    os.makedirs(save_dir, exist_ok=True)

    yield
    # stop code here


app = FastAPI(
    title="Spark-TTS API",
    description="API for Spark Text-to-Speech synthesis",
    lifespan=lifespan,
)


@timer
def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(Path(model_dir), device)
    return model


@timer
def run_tts(
    text: str,
    model: SparkTTS,
    prompt_text: str | None = None,
    prompt_speech_path: str | None = None,
    gender: str | None = None,
    pitch: str | None = None,
    speed: str | None = None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    logging.info(f"prompt: {prompt_text}")
    logging.info(f"text: {text}")

    # Perform inference and save the output audio
    audio_bytes = io.BytesIO()
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech_path,  # type: ignore
            prompt_text,
            gender,
            pitch,
            speed,
        )

        # sf.write(save_path, wav, samplerate=16000)
        # logging.info(f"Audio saved at: {save_path}")
        sf.write(audio_bytes, wav, samplerate=16000, format="WAV")

    audio_bytes.seek(0)
    return audio_bytes


@app.post("/voice_clone")
async def voice_clone(
    text: str = Body(...),
    prompt_audio_path: str = Body(...),
    prompt_text: Optional[str] = Body(None),
):
    """
    Clone a voice based on reference audio.

    Args:
        text: The text to synthesize
        prompt_text: Text of the prompt speech (optional)
        prompt_audio: Reference audio file for voice cloning

    Returns:
        Audio file of synthesized speech
    """

    try:
        # 处理提示文本
        prompt_text_clean = (
            None if prompt_text is None or len(prompt_text) < 2 else prompt_text
        )

        # 执行TTS
        audio_bytes = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech_path=prompt_audio_path,
        )

        # 返回生成的音频文件
        return Response(
            content=audio_bytes.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=generated_audio.wav"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice_creation")
async def voice_creation(
    text: str = Body(..., embed=True),
    gender: str = Body(...),
    pitch: int = Body(3),
    speed: int = Body(3),
):
    """
    Create a synthetic voice with adjustable parameters.

    Args:
        text: The text to synthesize
        gender: 'male' or 'female'
        pitch: Value from 1-5
        speed: Value from 1-5

    Returns:
        Audio file of synthesized speech
    """
    try:
        # 验证参数
        if gender not in ["male", "female"]:
            raise HTTPException(
                status_code=400, detail="Gender must be 'male' or 'female'"
            )

        if not 1 <= pitch <= 5 or not 1 <= speed <= 5:
            raise HTTPException(
                status_code=400, detail="Pitch and speed must be between 1 and 5"
            )

        # 映射参数
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]

        # 执行TTS
        audio_bytes = run_tts(
            text, model, gender=gender, pitch=pitch_val, speed=speed_val
        )

        # 返回生成的音频文件
        return Response(
            audio_bytes.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=generated_audio.wav"
            },
        )

    except Exception as e:
        logging.error(f"Error during voice creation")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

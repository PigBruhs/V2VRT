# tts_engine.py
# Dependencies:
#   pip install TTS soundfile numpy
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from TTS.api import TTS


@dataclass
class TTSConfig:
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    output_dir: str = "outputs"
    sample_rate: int = 22050  # XTTS v2 default
    use_gpu: bool = False  # CPU only on the given server


class TTSEngine:
    """
    Cloud TTS using Coqui TTS XTTS v2 on CPU.
    API:
      - __init__(config: Dict)
      - synthesize(text: str, language: str, voice: Optional[str] = None,
                   speed: float = 1.0, emotion: Optional[str] = None,
                   style: Optional[str] = None, streaming: bool = False) -> str
    Notes:
      - voice can be a reference speaker wav path for XTTS v2 zero shot.
      - speed emotion style are placeholders; XTTS v2 does not natively expose them on CPU API.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = TTSConfig(**config)
        os.makedirs(cfg.output_dir, exist_ok=True)
        self._sr = cfg.sample_rate
        self._out_dir = cfg.output_dir
        self._tts = TTS(model_name=cfg.model_name, gpu=cfg.use_gpu)

    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        emotion: Optional[str] = None,
        style: Optional[str] = None,
        streaming: bool = False,
    ) -> str:
        # Generate to memory then write wav for consistent output
        # XTTS v2: use speaker_wav if provided
        wav: np.ndarray = self._tts.tts(
            text=text,
            speaker_wav=voice,
            language=language,
        )
        # Simple speed control by resampling if speed != 1.0
        if speed and abs(speed - 1.0) > 1e-3:
            # naive time scale via numpy indexing
            idx = np.arange(0, len(wav), speed, dtype=np.float32)
            idx = np.clip(idx, 0, len(wav) - 1)
            wav = np.interp(idx, np.arange(len(wav), dtype=np.float32), wav).astype(
                np.float32
            )

        out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex}.wav")
        sf.write(out_path, wav, self._sr)
        return out_path


if __name__ == "__main__":
    # Simple test
    config = {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "output_dir": "outputs",
        "sample_rate": 22050,
        "use_gpu": False,
    }
    tts_engine = TTSEngine(config)
    text = "Hello, this is a test of the text to speech synthesis."
    out_wav = tts_engine.synthesize(
        text=text,
        language="en",
        voice=None,
        speed=1.0,
    )
    print(f"Synthesized speech saved to: {out_wav}")
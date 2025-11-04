from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:
    from scipy.signal import resample_poly
except Exception:  # pragma: no cover
    resample_poly = None  # type: ignore

import sherpa_onnx


@dataclass
class ASRConfig:
    # Transducer model files from sherpa-onnx export
    encoder: str
    decoder: str
    joiner: str
    tokens: str
    provider: str = "cpu"  # cpu
    num_threads: int = os.cpu_count() or 4
    sample_rate: int = 16000
    debug: bool = False


class ASREngine:
    """
    Cloud ASR using sherpa-onnx OfflineRecognizer on CPU.
    API:
      - __init__(config: Dict)
      - transcribe(audio: str | Tuple[np.ndarray, int], language: Optional[str] = None,
                   streaming: bool = False, diarization: bool = False) -> Dict
    Notes:
      - Input should be mono waveform at config.sample_rate.
      - If sampling rate mismatch, will resample via scipy.signal.resample_poly.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = ASRConfig(**config)
        # Threads for ORT and OpenMP
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.num_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cfg.num_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cfg.num_threads))

        model_config = sherpa_onnx.OfflineModelConfig(
            transducer=sherpa_onnx.OfflineTransducerModelConfig(
                encoder=cfg.encoder,
                decoder=cfg.decoder,
                joiner=cfg.joiner,
            ),
            tokens=cfg.tokens,
            provider=cfg.provider,
            num_threads=cfg.num_threads,
            debug=cfg.debug,
        )
        self._recognizer = sherpa_onnx.OfflineRecognizer(model_config)
        self._sr = cfg.sample_rate

    def _load_audio(
        self, audio: Union[str, Tuple[np.ndarray, int]]
    ) -> Tuple[np.ndarray, int, float]:
        if isinstance(audio, str):
            wav, sr = sf.read(audio, dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
        else:
            wav, sr = audio
            wav = np.asarray(wav, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=-1)

        if sr != self._sr:
            if resample_poly is None:
                raise RuntimeError(
                    "Sampling rate mismatch and scipy is not available for resampling."
                )
            g = np.gcd(sr, self._sr)
            up, down = self._sr // g, sr // g
            wav = resample_poly(wav, up, down).astype("float32")
            sr = self._sr

        duration = float(len(wav)) / float(sr)
        return wav, sr, duration

    def transcribe(
        self,
        audio: Union[str, Tuple[np.ndarray, int]],
        language: Optional[str] = None,
        streaming: bool = False,
        diarization: bool = False,
    ) -> Dict[str, Any]:
        wav, sr, duration = self._load_audio(audio)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sr, wav)
        stream.input_finished()

        # Offline single stream decode
        self._recognizer.decode_stream(stream)

        # sherpa-onnx exposes result on stream.result
        result = getattr(stream, "result", None)
        text = result.text if result is not None else ""

        segments: List[Dict[str, Any]] = [
            {"start": 0.0, "end": duration, "text": text}
        ]
        out = {"text": text, "segments": segments, "lang": language}
        return out


if __name__ == "__main__":
    import pprint

    config = {
        "encoder": "sherpa-onnx-transducer-2024-06-27-utc-16k-encoder.onnx",
        "decoder": "sherpa-onnx-transducer-2024-06-27-utc-16k-decoder.onnx",
        "joiner": "sherpa-onnx-transducer-2024-06-27-utc-16k-joiner.onnx",
        "tokens": "sherpa-onnx-transducer-2024-06-27-utc-16k-tokens.txt",
        "provider": "cpu",
        "num_threads": 4,
        "sample_rate": 16000,
        "debug": False,
    }

    engine = ASREngine(config)
    audio_file = "test_wavs/0.wav"  # Replace with your audio file path
    result = engine.transcribe(audio_file)
    pprint.pprint(result)
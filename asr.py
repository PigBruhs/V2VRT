import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


try:
    import torch
except Exception:
    torch = None

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_pipeline_logger() -> logging.Logger:
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        handler = logging.FileHandler(_LOG_DIR / "pipeline.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOGGER = _get_pipeline_logger()


@dataclass
class ASRConfig:
    model_name_or_path: str = "small"  # 模型名或本地路径（例如: small, tiny, medium 等）
    device: Optional[str] = None  # "cuda" 或 "cpu"，None 表示自动检测
    compute_type: Optional[str] = None  # 例如 "float16" (cuda) 或 "int8" (cpu)，可留空由代码选择
    sample_rate: int = 16000
    beam_size: int = 5
    task: str = "transcribe"  # "transcribe" 或 "translate"
    language: Optional[str] = None  # 例如 "zh", "en"，None 表示自动检测


class ASREngine:
    def __init__(self, config: Union[ASRConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            self._cfg = ASRConfig(**config)
        else:
            self._cfg = config

        # 选择设备
        if self._cfg.device:
            device = self._cfg.device
        else:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        # 选择 compute_type
        if self._cfg.compute_type:
            compute_type = self._cfg.compute_type
        else:
            compute_type = "float16" if device.startswith("cuda") else "int8"

        # 加载模型（若是模型名会自动从 Hugging Face 下载并缓存）
        try:
            self._model = WhisperModel(
                self._cfg.model_name_or_path, device=device, compute_type=compute_type
            )
        except Exception as e:
            LOGGER.exception("ASR 初始化模型失败")
            raise RuntimeError(f"加载 Whisper 模型失败: {e}")

        self._sr = self._cfg.sample_rate
        self._beam_size = self._cfg.beam_size
        self._task = self._cfg.task
        self._language = self._cfg.language

    def _prepare_audio(self, audio: Union[str, bytes, Tuple[np.ndarray, int]]) -> str:
        """
        返回一个本地 wav 文件路径（临时文件），采样率为 self._sr。
        如果输入是路径则直接返回输入路径（仍会验证采样率）。
        """
        if isinstance(audio, bytes):
            # 从内存字节中读取
            wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
        elif isinstance(audio, str):
            # 验证文件采样率并重采样（如需）
            info = sf.info(audio)
            if info.samplerate != self._sr:
                if resample_poly is None:
                    raise RuntimeError("需要 `scipy` 才能重采样：pip install scipy")
                wav, sr = sf.read(audio, dtype="float32")
            else:
                return audio
        else:
            wav, sr = audio

        wav = np.asarray(wav, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if sr != self._sr:
            if resample_poly is None:
                raise RuntimeError("需要 `scipy` 才能重采样：pip install scipy")
            g = np.gcd(sr, self._sr)
            wav = resample_poly(wav, self._sr // g, sr // g).astype("float32")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, wav, self._sr, subtype="PCM_16")
        return tmp.name

    def transcribe(
        self,
        audio: Union[str, bytes, Tuple[np.ndarray, int]],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        返回: { "text": full_text, "segments": [ {start, end, text}, ... ], "lang": detected_or_given }
        """
        started = time.perf_counter()
        tmp_path = None
        try:
            src_path = self._prepare_audio(audio)
            # 仅当创建���临时文件时才记录 tmp_path
            if src_path != audio:
                tmp_path = src_path

            lang = language or self._language

            segments, info = self._model.transcribe(
                src_path,
                beam_size=self._beam_size,
                language=lang,
                task=self._task,
            )

            texts: List[str] = []
            segs = []
            for segment in segments:
                segs.append(
                    {"start": float(segment.start), "end": float(segment.end), "text": segment.text}
                )
                texts.append(segment.text)
            full_text = " ".join(texts).strip()
            result = {"text": full_text, "segments": segs, "lang": lang or info.language}
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "ASR.transcribe success | lang=%s | chars=%d | duration_ms=%.2f",
                result["lang"],
                len(full_text),
                duration_ms,
            )
            return result
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception(
                "ASR.transcribe failed | lang=%s | duration_ms=%.2f",
                language or self._language or "auto",
                duration_ms,
            )
            raise

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

import sys
import json


def main() -> None:
    """简单命令行测试入口：
    - 使用官方模型名 "small"（faster-whisper 会自动下载并缓存 CTranslate2 模型）
    - 对一个示例 wav 文件进行转写
    """
    # 这里直接使用模型名 "small"，交给 faster-whisper 自己处理缓存和转换
    model_name_or_path = "small"

    default_audio = os.path.join(".", "test_wavs", "8k.wav")
    audio_path = default_audio if os.path.isfile(default_audio) else None

    if audio_path is None:
        print("[ASR] 未找到示例音频文件 test_wavs/0.wav，请自行提供音频路径。", file=sys.stderr)
        return

    cfg = ASRConfig(
        model_name_or_path=model_name_or_path,
        device=None,         # 自动检测 cuda/cpu
        compute_type=None,   # 由 ASREngine 按 device 自动选择
        sample_rate=16000,
        beam_size=5,
        task="transcribe",
        language=None,
    )

    try:
        engine = ASREngine(cfg)
    except Exception as e:
        print("加载模型失败:", e, file=sys.stderr)
        sys.exit(1)

    try:
        result = engine.transcribe(audio_path)
    except Exception as e:
        print("转写失败:", e, file=sys.stderr)
        sys.exit(2)

    print("模型路径:", model_name_or_path)
    print("识别全文:")
    print(result.get("text", ""))
    print("\n分段:")
    print(json.dumps(result.get("segments", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

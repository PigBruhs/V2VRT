import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf

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
class TTSConfig:
    output_dir: str = "outputs"
    model_dir: str = "./models/tts/piper"  # 改为模型目录


class TTSEngine:
    """
    基于 piper-tts 的本地 TTS 引擎。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = TTSConfig(**config)
        os.makedirs(cfg.output_dir, exist_ok=True)
        self._out_dir = cfg.output_dir

        # 扫描并加载所有可用的 piper 模型
        self._models: Dict[str, str] = {}
        model_dir = Path(cfg.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Piper 模型目录不存在: {model_dir}")

        for onnx_file in model_dir.glob("*.onnx"):
            # 从文件名中提取语言代码, e.g., "en_US.onnx" -> "en"
            lang_code = onnx_file.name.split("_")[0].lower()
            self._models[lang_code] = str(onnx_file.absolute())
            LOGGER.info("发现 TTS 模型: lang=%s, path=%s", lang_code, self._models[lang_code])

        if not self._models:
            raise RuntimeError(f"在目录 {model_dir} 中未找到任何 .onnx 模型文件")

        LOGGER.info("TTS 初始化完成 | output_dir=%s", self._out_dir)

    def synthesize(
            self,
            text: str,
            language: str = "zh",
            voice: Optional[str] = None,
            speed: float = 1.0,
            emotion: Optional[str] = None,
            style: Optional[str] = None,
            streaming: bool = False,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text 不能为空")

        lang_code = (language or "unknown").lower().split('-')[0] # "zh-cn" -> "zh"

        # 根据语言选择模型
        model_path = self._models.get(lang_code)
        if not model_path:
            # 尝试一个备用/默认模型，例如英语
            LOGGER.warning("未找到语言 '%s' 的 TTS 模型，尝试使用 'en' 作为备用。", lang_code)
            model_path = self._models.get("en")
            if not model_path:
                # 如果连英语模型都没有，就用字典���的第一个
                LOGGER.warning("也未找到 'en' 模型，将使用任意一个可用模型。")
                model_path = next(iter(self._models.values()), None)

        if not model_path:
            raise RuntimeError(f"没有找到适用于语言 '{lang_code}' 或任何备用语言的 TTS 模型。")


        short_id = uuid.uuid4().hex[:8]
        out_filename = f"tts_{lang_code}_{short_id}.wav"
        out_path = os.path.abspath(os.path.join(self._out_dir, out_filename))

        cmd = [
            r"E:\V2VRT\.venv\Scripts\piper.exe",
            "--model",
            model_path,  # 使用动态选择的模型路径
            "--output_file",
            out_path,
        ]

        started = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "调用 piper 失败 (code={code}).\nstderr: {stderr}".format(
                        code=result.returncode,
                        stderr=result.stderr.decode(errors="ignore"),
                    )
                )
            if not os.path.isfile(out_path):
                raise RuntimeError(f"piper 未生成输出 wav 文件: {out_path}")

            info = sf.info(out_path)
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "TTS.synthesize success | lang=%s | chars=%d | file=%s | duration_ms=%.2f",
                lang_code,
                len(text),
                out_path,
                duration_ms,
            )
            LOGGER.debug(
                "TTS.synthesize details | samplerate=%s | channels=%s | frames=%s",
                info.samplerate,
                info.channels,
                info.frames,
            )
            return out_path
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception(
                "TTS.synthesize failed | lang=%s | duration_ms=%.2f",
                lang_code,
                duration_ms,
            )
            raise




if __name__ == "__main__":
    # 注意：请根据你本机实际模型文件名调整这些路径
    base_model_dir = "./models/tts/piper"

    lang_tests = [
        {
            "lang": "en",
            "text": "Hello this is an English test",
            "model": os.path.join(base_model_dir, "en_US.onnx"),
        },
        {
            "lang": "fr",
            "text": "Bonjour ceci est un test en français",
            "model": os.path.join(base_model_dir, "fr_FR.onnx"),
        },
        {
            "lang": "de",
            "text": "Guten Tag dies ist ein deutscher Test",
            "model": os.path.join(base_model_dir, "de_DE.onnx"),
        },
        {
            "lang": "zh",
            # 这里用你已经验证过正常的中文句子
            "text": "你好啊我的朋友欢迎来到哈尔滨！",
            "model": os.path.join(base_model_dir, "zh_CN.onnx"),
            # 如果你实际文件名是 zh_CN-huayan-medium.onnx，就改成那个
            # "model": os.path.join(base_model_dir, "zh_CN-huayan-medium.onnx"),
        },
        {
            "lang": "cz",
            "text": "Ahoj toto je test v češtině",
            "model": os.path.join(base_model_dir, "cs_CZ.onnx"),
        },
        {
            "lang": "it",
            "text": "Ciao questo è un test in italiano",
            "model": os.path.join(base_model_dir, "it_IT.onnx"),
        },
        {
            "lang": "ru",
            "text": "Привет это тест на русском языке",
            "model": os.path.join(base_model_dir, "ru_RU.onnx"),
        },
    ]

    for item in lang_tests:
        lang = item["lang"]
        text = item["text"]
        model_path = item["model"]

        print(f"\n=== 测试语言: {lang}, 模型: {model_path} ===")

        # 为每种语言单独创建一个引擎实例，指定对应的模型路径
        config = {
            "output_dir": "outputs",
            "model_path": model_path,
        }

        try:
            tts_engine = TTSEngine(config)
        except FileNotFoundError as e:
            print(f"[{lang}] 模型不存在，跳过: {e}")
            continue

        try:
            out_wav = tts_engine.synthesize(
                text=text,
                language="zh" if lang == "zh" else lang,
            )
            print(f"[{lang}] 合成成功, wav: {out_wav}")
        except Exception as e:
            print(f"[{lang}] 合成失败: {e}")
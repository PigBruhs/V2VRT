from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import uuid
import subprocess
import soundfile as sf


@dataclass
class TTSConfig:
    output_dir: str = "outputs"
    model_path: str = "./models/tts/piper/zh_CN.onnx"


class TTSEngine:
    """
    基于 piper-tts 的本地 TTS 引擎。
    通过 stdin 传入 UTF-8 文本。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = TTSConfig(**config)
        os.makedirs(cfg.output_dir, exist_ok=True)

        self._out_dir = cfg.output_dir
        self._model_path = os.path.abspath(cfg.model_path)

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Piper 模型文件不存在: {self._model_path}"
            )

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

        short_id = uuid.uuid4().hex[:8]
        # 这里不能有反斜杠
        out_filename = f"tts_{lang}_{short_id}.wav"
        out_path = os.path.abspath(os.path.join(self._out_dir, out_filename))

        cmd = [
            r"E:\V2VRT\.venv\Scripts\piper.exe",  # 建议用绝对路径
            "--model",
            self._model_path,
            "--output_file",
            out_path,
        ]

        # 明确以 UTF-8 字节流喂给 stdin
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print("piper stdout:", result.stdout.decode(errors="ignore"))
        print("piper stderr:", result.stderr.decode(errors="ignore"))

        if result.returncode != 0:
            stderr_text = result.stderr.decode(errors="ignore")
            raise RuntimeError(
                "调用 piper 失败 (code={code}).\nstderr: {stderr}".format(
                    code=result.returncode,
                    stderr=stderr_text,
                )
            )

        if not os.path.isfile(out_path):
            raise RuntimeError(f"piper 未生成输出 wav 文件: {out_path}")

        info = sf.info(out_path)
        print("Piper 输出 wav 信息:", info)

        return out_path




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
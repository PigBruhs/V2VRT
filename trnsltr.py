import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ctranslate2
from transformers import M2M100Tokenizer

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
class MTConfig:
    ct2_model_path: str  # path to converted CT2 model of facebook/m2m100_418M
    tokenizer_name: str = "facebook/m2m100_418M"
    tokenizer_path: Optional[str] = None
    local_files_only: bool = True
    device: str = "cpu"
    compute_type: str = "int8"  # int8 float16 float32
    num_threads: int = os.cpu_count() or 4
    beam_size: int = 4
    max_decoding_length: int = 256
    length_penalty: float = 1.0


class Translator:
    """
    Cloud MT using CTranslate2 with M2M100-418M int8 on CPU.
    API:
      - __init__(config: Dict)
      - translate(text: str, src_lang: str, tgt_lang: str,
                  domain: Optional[str] = None, formality: Optional[str] = None) -> str
    Notes:
      - Requires a CT2 converted model directory for M2M100 418M.
      - tokenizer_name should match the original HF tokenizer.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = MTConfig(**config)
        # Limit threads
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.num_threads))
        self._translator = ctranslate2.Translator(
            cfg.ct2_model_path,
            device=cfg.device,
            compute_type=cfg.compute_type,
            inter_threads=1,
            intra_threads=cfg.num_threads,
        )
        # 只从本地目录加载 tokenizer，不再回退到远程名称
        self._tokenizer = self._load_tokenizer(cfg)
        self._beam = cfg.beam_size
        self._max_len = cfg.max_decoding_length
        self._len_penalty = cfg.length_penalty
        LOGGER.info("Translator 初始化完成 | model=%s", cfg.ct2_model_path)

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        domain: Optional[str] = None,
        formality: Optional[str] = None,
    ) -> str:
        started = time.perf_counter()
        try:
            # Prepare source tokens with source language
            self._tokenizer.src_lang = src_lang
            src_ids = self._tokenizer(text, return_tensors=None).input_ids  # type: ignore
            # input_ids is a list of ids; we need tokens for CT2
            src_tokens = self._tokenizer.convert_ids_to_tokens(src_ids)

            # Target language prefix token for M2M100
            tgt_prefix_token = self._tokenizer.get_lang_token(tgt_lang)
            tgt_prefix = [tgt_prefix_token]

            result = self._translator.translate_batch(
                [src_tokens],
                beam_size=self._beam,
                max_decoding_length=self._max_len,
                length_penalty=self._len_penalty,
                target_prefix=[tgt_prefix],
            )

            hyp_tokens = result[0].hypotheses[0]
            hyp_ids = self._tokenizer.convert_tokens_to_ids(hyp_tokens)
            out_text = self._tokenizer.decode(hyp_ids, skip_special_tokens=True)
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "Translator.translate success | src=%s | tgt=%s | chars=%d | duration_ms=%.2f",
                src_lang,
                tgt_lang,
                len(out_text),
                duration_ms,
            )
            return out_text
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception(
                "Translator.translate failed | src=%s | tgt=%s | duration_ms=%.2f",
                src_lang,
                tgt_lang,
                duration_ms,
            )
            raise

    def _load_tokenizer(self, cfg: MTConfig) -> M2M100Tokenizer:
        """
        只从本地目录加载 tokenizer：
        - 期望目录中至少包含 vocab.json 和 sentencepiece.bpe.model
        - 如缺失则抛出明确错误，提示用户补齐文件
        """
        local_dir_hint = cfg.tokenizer_path or cfg.ct2_model_path
        if not local_dir_hint:
            raise RuntimeError(
                "Tokenizer 加载失败：未配置本地 tokenizer 路径，也不再回退到远程名称。"
                "请在 MTConfig 中设置 ct2_model_path/tokenizer_path 指向包含 tokenizer 文件的目录。"
            )

        local_dir = Path(local_dir_hint).expanduser()
        if not local_dir.is_dir():
            raise RuntimeError(f"Tokenizer 加载失败：目录不存在：{local_dir}")

        vocab_file = local_dir / "vocab.json"
        spm_file = local_dir / "sentencepiece.bpe.model"

        if not vocab_file.exists() or not spm_file.exists():
            missing = []
            if not vocab_file.exists():
                missing.append("vocab.json")
            if not spm_file.exists():
                missing.append("sentencepiece.bpe.model")
            raise RuntimeError(
                f"Tokenizer 加载失败：目录 {local_dir} 缺少必要文件：{', '.join(missing)}。\n"
                "请从一台可联网的机器上运行：\n"
                "  from transformers import M2M100Tokenizer\n"
                "  tok = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')\n"
                "  tok.save_pretrained('<你的本地目录>')"
            )

        # 本地目录已包含标准 HF tokenizer 文件
        return M2M100Tokenizer.from_pretrained(
            str(local_dir),
            local_files_only=True,
        )


def test_translator() -> None:
    """
    简单测试函数：
    - 构造 Translator
    - 翻译一条固定英文句子到法文
    - 打印输入/输出
    """
    config = {
        "ct2_model_path": "./models/nlp/m2m100_418M_int8",
        # 如有需要，也可以显式给 tokenizer_path
        # "tokenizer_path": "./models/nlp/m2m100_418M_int8",
    }

    print("[test] 初始化 Translator ...")
    translator = Translator(config)

    src_text = "u r a white dog"
    print(f"[test] 原文: {src_text}")
    translated_text = translator.translate(src_text, src_lang="en", tgt_lang="fr")
    print(f"[test] 译文: {translated_text}")


if __name__ == "__main__":
    try:
        test_translator()
    except Exception as e:
        import traceback

        print("[test] 发生异常：", e)
        traceback.print_exc()
        raise

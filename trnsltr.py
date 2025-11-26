# translator.py
# Dependencies:
#   pip install ctranslate2 transformers sentencepiece
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import ctranslate2
from transformers import M2M100Tokenizer


@dataclass
class MTConfig:
    ct2_model_path: str  # path to converted CT2 model of facebook/m2m100_418M
    tokenizer_name: str = "facebook/m2m100_418M"
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
        self._tokenizer = M2M100Tokenizer.from_pretrained(cfg.tokenizer_name)
        self._beam = cfg.beam_size
        self._max_len = cfg.max_decoding_length
        self._len_penalty = cfg.length_penalty

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        domain: Optional[str] = None,
        formality: Optional[str] = None,
    ) -> str:
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
        return out_text



if __name__ == "__main__":
    # Example usage
    config = {
        "ct2_model_path": "./models/nlp/m2m100_418M_int8",
    }
    translator = Translator(config)
    src_text = "Hello, how are you?"
    translated_text = translator.translate(src_text, src_lang="en", tgt_lang="fr")
    print(translated_text)  # Should print the French translation of the input text

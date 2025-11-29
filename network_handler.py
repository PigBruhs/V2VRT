# python
import base64
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

LOGGER = logging.getLogger("pipeline")


@dataclass
class PipelineConfig:
    """
    管线配置数据类 (此部分保留但不再由新逻辑直接使用)
    """
    run_asr: bool = True
    run_trnsltr: bool = True
    run_tts: bool = True


class NetworkCoordinator:
    """
    NetworkCoordinator

    负责手机前端与后端语音/翻译模块之间的网络协调与流程编排。
    """

    def __init__(
        self,
        asr_engine: Any,
        translator: Any,
        tts_engine: Any,
    ) -> None:
        """
        初始化协调器.
        """
        self._asr = asr_engine
        self._translator = translator
        self._tts = tts_engine
        self._transactions: Dict[str, Dict[str, Any]] = {}

    def negotiate(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理第一阶段的协商请求。
        接收 mode，返回下一步操作和 transaction_id。
        """
        print("\n" + "="*50)
        print(f"收到协商请求 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

        mode = request_json.get("mode")
        if mode is None or mode not in range(7):
            print(f"[错误] 协商失败：无效的模式: {mode}")
            return self._error_response(None, "INVALID_MODE", "mode 不能为空且必须为 0-6")

        # B-B-B, B-B-F, B-F-F, F-B-B, F-B-F, F-F-B, B-F-B
        # ASR:     B,   B,   B,   F,   F,   F,   B
        # TRNSLTR: B,   B,   F,   B,   B,   F,   F
        # TTS:     B,   F,   F,   B,   F,   B,   B

        # 确定前端需要发送什么数据
        if mode in [0, 1, 2, 6]:
            next_action = "send_audio"
        elif mode in [3, 4, 5]:
            next_action = "send_text"
        else:
            # 理论上不会发生
            return self._error_response(None, "INTERNAL_ERROR", f"无法为模式 {mode} 确定下一步操作")

        transaction_id = str(uuid.uuid4())
        # 存储事务状态，可以增加一个时间戳用于后续的过期清理
        self._transactions[transaction_id] = {
            "mode": mode,
            "timestamp": time.time()
        }

        response = {
            "status": "ok",
            "transaction_id": transaction_id,
            "next_action": next_action,
            "execute_endpoint": "/api/v1/execute"
        }
        print(f"协商成功: transaction_id={transaction_id}, next_action={next_action}")
        print("="*50 + "\n")
        return response

    def execute(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理第二阶段的执行请求。
        """
        print("\n" + "="*50)
        print(f"收到执行请求 @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        started = time.perf_counter()

        transaction_id = request_json.get("transaction_id")
        if not transaction_id or transaction_id not in self._transactions:
            print(f"[错误] 执行失败：无效或过期的 transaction_id: {transaction_id}")
            return self._error_response(None, "INVALID_TRANSACTION", "transaction_id 无效或已过期")

        # 取出并消耗事务
        transaction = self._transactions.pop(transaction_id)
        mode = transaction["mode"]

        print(f"事务 ID: {transaction_id}, 模式: {mode}")

        run_asr = mode in [0, 1, 2, 6]
        run_translator = mode in [0, 1, 3, 4]
        run_tts = mode in [0, 3, 5, 6]

        payload = request_json.get("payload", {}) or {}
        source_lang: Optional[str] = payload.get("source_lang")
        target_lang: Optional[str] = payload.get("target_lang")

        print(f"流程计划: ASR={'后端' if run_asr else '前端'}, 翻译={'后端' if run_translator else '前端'}, TTS={'后端' if run_tts else '前端'}")
        print(f"语言参数: 源='{source_lang}', 目标='{target_lang}'")

        timing_ms = {"asr": 0.0, "translator": 0.0, "tts": 0.0}
        current_text = payload.get("text")
        current_audio_b64 = payload.get("audio_base64")

        if current_audio_b64:
            print("Payload: 收到音频数据 (Base64)")
        if current_text:
            print(f"Payload: 收到文本数据: '{current_text[:50]}...'")

        # --- 1. ASR ---
        if run_asr:
            if not current_audio_b64:
                return self._error_response(transaction_id, "MISSING_AUDIO", f"模式 {mode} 需要 audio_base64")
            try:
                audio_data = base64.b64decode(current_audio_b64)
                t0 = time.perf_counter()
                print("--> 调用后端 ASR...")
                asr_result = self._asr.transcribe(audio_data, language=source_lang)
                timing_ms["asr"] = (time.perf_counter() - t0) * 1000.0
                current_text = asr_result["text"]
                print(f"<-- ASR 结果: '{current_text}' (耗时: {timing_ms['asr']:.2f}ms)")
                if not source_lang:
                    source_lang = asr_result.get("lang")
            except Exception as e:
                LOGGER.exception("ASR 模块在模式 %d 中失败", mode)
                return self._error_response(transaction_id, "ASR_FAILED", f"ASR 模块调用失败: {e}")

        if not current_text:
            return self._error_response(transaction_id, "MISSING_TEXT", f"模式 {mode} 在处理流程中未能获得文本")

        # --- 2. 翻译 ---
        if run_translator:
            if not source_lang or not target_lang:
                return self._error_response(transaction_id, "MISSING_LANG", "翻译需要 source_lang 和 target_lang")
            try:
                t0 = time.perf_counter()
                print(f"--> 调用后端翻译 (从 '{source_lang}' 到 '{target_lang}')...")
                current_text = self._translator.translate(current_text, src_lang=source_lang, tgt_lang=target_lang)
                timing_ms["translator"] = (time.perf_counter() - t0) * 1000.0
                print(f"<-- 翻译结果: '{current_text}' (耗时: {timing_ms['translator']:.2f}ms)")
            except Exception as e:
                LOGGER.exception("翻译模块在模式 %d 中失败", mode)
                return self._error_response(transaction_id, "TRANSLATOR_FAILED", f"翻译模块调用失败: {e}")

        # --- 3. TTS ---
        final_audio_b64: Optional[str] = None
        if run_tts:
            try:
                tts_lang = target_lang or source_lang or "zh"
                t0 = time.perf_counter()
                print(f"--> 调用后端 TTS (语言: '{tts_lang}')...")
                output_wav_path = self._tts.synthesize(current_text, language=tts_lang)
                timing_ms["tts"] = (time.perf_counter() - t0) * 1000.0
                with open(output_wav_path, "rb") as f:
                    final_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                print(f"<-- TTS 完成, 生成音频数据 (耗时: {timing_ms['tts']:.2f}ms)")
            except Exception as e:
                LOGGER.exception("TTS 模块在模式 %d 中失败", mode)
                return self._error_response(transaction_id, "TTS_FAILED", f"TTS 模块调用失败: {e}")

        total_ms = (time.perf_counter() - started) * 1000.0
        print(f"总处理时间: {total_ms:.2f}ms")

        # --- 4. 构造响应 ---
        response = {
            "transaction_id": transaction_id,
            "status": "ok",
            "results": {
                "text": current_text,
                "audio_base64": final_audio_b64,
                "meta": {
                    "mode": mode,
                    "timing_ms": {**timing_ms, "total": total_ms},
                },
            },
        }
        print(f"构造响应: text='{response['results']['text']}', audio_present={response['results']['audio_base64'] is not None}")
        print("="*50 + "\n")
        return response

    # ===== 内部工具方法 =====
    @staticmethod
    def _error_response(
        req_id: Optional[str], # 在这里，它可能是 transaction_id
        error_code: str,
        error_message: str,
    ) -> Dict[str, Any]:
        """
        统一错误响应构造.
        """
        return {
            "request_id": req_id, # 保持字段名一致
            "status": "error",
            "error_code": error_code,
            "error_message": error_message,
            "results": None,
        }

import logging
import os
from flask import Flask, jsonify, request

from asr import ASREngine, ASRConfig
from trnsltr import Translator
from tts import TTSEngine
from network_handler import NetworkCoordinator

logging.basicConfig(level=logging.INFO)

DEFAULT_MODEL_ROOT = "./models"


def _init_modules():
    asr_engine = ASREngine(
        ASRConfig(
            model_name_or_path="small",
            sample_rate=16000,
            beam_size=5,
            task="transcribe",
            language=None,
        )
    )

    translator = Translator(
        {
            "ct2_model_path": os.path.join(DEFAULT_MODEL_ROOT, "nlp", "m2m100_418M_int8"),
            "tokenizer_path": os.path.join(DEFAULT_MODEL_ROOT, "nlp", "m2m100_418M_int8"),
            "device": "cpu",
            "compute_type": "int8",
        }
    )

    tts_engine = TTSEngine(
        {
            "output_dir": "outputs",
            "model_dir": os.path.join(DEFAULT_MODEL_ROOT, "tts", "piper"),
        }
    )

    return asr_engine, translator, tts_engine


def create_app() -> Flask:
    asr_engine, translator, tts_engine = _init_modules()
    coordinator = NetworkCoordinator(asr_engine, translator, tts_engine)

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/api/v1/negotiate")
    def negotiate():
        payload = request.get_json(silent=True) or {}
        response = coordinator.negotiate(payload)
        status = 200 if response.get("status") == "ok" else 400
        return jsonify(response), status

    @app.post("/api/v1/execute")
    def execute():
        payload = request.get_json(silent=True) or {}
        response = coordinator.execute(payload)
        status = 200 if response.get("status") == "ok" else 400
        return jsonify(response), status

    app.coordinator = coordinator  # 可用于测试或后续扩展
    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=8000, threaded=True)

# V2VRT — Voice to Voice Realtime Translator

简要说明
---------
V2VRT 是一个本地化的实时语音到语音翻译系统管线，包含 ASR（语音识别）、NLP/翻译 和 TTS（文本到语音）模块。系统支持前端或后端任意一端完成 ASR 或 NLP，并保证：在“最后一次通信”中，响应必定包含 asr_result 与 nlp_result（无论这些结果来自前端透传还是后端生成）。

主要产出（每次执行的最终结果）
--------------------------------
在 /api/v1/execute 的成功响应（status: "ok"）的 results 字段中至少包含：
- text: 最终用于合成或展示的文本（翻译后或透传）
- audio_base64: 若触发后端 TTS，则返回 base64 编码的音频（wav）
- asr_result: 一定存在的 ASR 结果对象
  - text, lang, segments（可选）, provider ("backend" 或 "frontend")
- nlp_result: 一定存在的 NLP/翻译 结果对象
  - text, source_lang, target_lang（可选）, provider ("backend"、"frontend" 或 "passthrough")
- meta: 包含 mode、耗时统计等元信息

接口保证
--------
- 无论 ASR/NLP 在前端或后端执行，NetworkCoordinator 会在最终响应中返回 asr_result 与 nlp_result，保证前端能在最后一次通信中获取完整信息。
- 错误响应会返回 transaction_id（如有），便于前端定位事务。

典型交互模式（mode）
--------------------
系统支持 7 种典型职责切分模式（0-6），例如：
- 后端全流程：后端负责 ASR → 翻译 → TTS
- 前端 ASR / 后端 翻译+TTS：前端发送 asr_text，后端完成翻译与合成
- 前端文本输入：前端直接发送文本，后端可执行翻译或仅执行 TTS

API 概览
--------
1. POST /api/v1/negotiate
   - 请求：{ "mode": <0-6> }
   - 响应：{ "status": "ok", "transaction_id": "...", "next_action": "send_audio"|"send_text", "execute_endpoint": "/api/v1/execute" }

2. POST /api/v1/execute
   - 请求：{ "transaction_id": "...", "payload": { ... } }
   - payload 中可包含：
     - audio_base64（当 next_action 为 send_audio）
     - text（当 next_action 为 send_text）
     - source_lang / target_lang
     - asr_text / asr_lang / asr_segments（前端已完成 ASR 时）
     - nlp_text / nlp_source_lang / nlp_target_lang（前端已完成翻译时）
   - 响应：参见“主要产出”部分，最终响应包含 asr_result 与 nlp_result。

技术栈与依赖
------------
- 语言与框架
  - Python 3.8+
  - Flask — 提供 HTTP API（app.py）
- ASR
  - faster-whisper（WhisperModel）
  - numpy, soundfile, scipy（重采样）
- 翻译（NLP）
  - CTranslate2（ctranslate2） + transformers 的 M2M100Tokenizer（本地 tokenizer）
- TTS
  - Piper（本地可执行或二进制），输出 wav
  - soundfile（读取/验证 wav）
- 日志与工具
  - logging，base64，uuid，subprocess 等标准库

关键文件
---------
- app.py — Flask 启动与路由
- network_handler.py — 协调器，保证最终响应包含 asr_result 与 nlp_result
- asr.py — ASR 引擎封装（faster-whisper）
- trnsltr.py — 翻译封装（CTranslate2 + 本地 tokenizer）
- tts.py — TTS 封装（piper）
- API_DOCUMENT.md — 对外接口说明（已同步保证语义）

运行与部署（快速指南）
--------------------
1. 模型准备
   - ASR: faster-whisper 使用模型名或本地路径（例如 "small"）
   - 翻译: 将 CTranslate2 转换后的 m2m100 模型放置于 models/nlp/，同时将 tokenizer 的 vocab.json 和 sentencepiece.bpe.model 放入同一目录
   - TTS: 将 piper 的 onnx 模型放在 models/tts/ 下（文件名建议包含语言前缀，如 en_US.onnx）

2. 安装依赖（示例）
   - pip install flask faster-whisper ctranslate2 transformers soundfile scipy numpy

3. 配置
   - 如需修改模型路径或 piper 可执行路径，请在 app.py / tts.py 中相应调整。

4. 启动服务
   - python app.py
   - 默认监听 0.0.0.0:8000

示例响应（简化）
----------------
```json
{
  "transaction_id": "uuid",
  "status": "ok",
  "results": {
    "text": "Bonjour, comment allez-vous?",
    "audio_base64": "...",
    "asr_result": {
      "text": "Hello, how are you?",
      "lang": "en",
      "segments": [],
      "provider": "frontend"
    },
    "nlp_result": {
      "text": "Bonjour, comment allez-vous?",
      "source_lang": "en",
      "target_lang": "fr",
      "provider": "backend"
    },
    "meta": { "mode": 1, "timing_ms": { "asr": 12.3, "translator": 45.6, "tts": 30.1, "total": 88.0 } }
  }
}
```

测试建议
--------
- 针对各模块（ASR、Translator、TTS）编写单元与集成测试并使用 mock。
- 端到端测试：使用已知语音样本验证转写、翻译与合成流程。
- 记录 timing_ms 用于性能调优。

贡献与扩展
----------
欢迎提交 PR：
- 替换或新增 ASR/TTS 实现（请保持 NetworkCoordinator 返回结构不变）
- 增加流式支持或消息队列（注意最终响应仍需包含 asr_result 与 nlp_result）

许可证与联系方式
----------------
请在仓库根目录添加 LICENSE 与作者信息。

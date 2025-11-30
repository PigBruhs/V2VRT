# 日志格式说明

本文件描述 `V2VRT` 项目的日志格式、常见条目模板及排查建议。日志主要写入 `logs/pipeline.log`，同时关键流程在控制台输出更详细的运行信息（含前端收到/返回的关键部分）。

## 全局行格式
每条日志行遵循如下格式：

```
%(asctime)s | %(levelname)s | %(name)s | %(message)s
```
示例：

```
2025-11-30 12:34:56,789 | INFO | pipeline | ASR.transcribe success | lang=zh | chars=24 | duration_ms=123.45
```

## 日志级别说明
- `DEBUG`：最细粒度，包含参数、内部变量（仅用于调试环境）。
- `INFO`：正常运行信息（启动/成功/主要耗时）。
- `WARNING`：可恢复的异常或退化行为（使用备用模型等）。
- `ERROR`：模块调用失败，后面通常跟随堆栈信息（由 `logger.exception` 输出）。

## 模块常见消息模板

- ASR
  - 成功：
    ```
    ASR.transcribe success | lang=%s | chars=%d | duration_ms=%.2f
    ```
  - 失败（含异常堆栈）：
    ```
    ASR.transcribe failed | lang=%s | duration_ms=%.2f
    <stack trace...>
    ```

- 翻译（Translator）
  - 成功：
    ```
    Translator.translate success | src=%s | tgt=%s | chars=%d | duration_ms=%.2f
    ```
  - 失败（含异常堆栈）：
    ```
    Translator.translate failed | src=%s | tgt=%s | duration_ms=%.2f
    <stack trace...>
    ```

- TTS
  - 成功：
    ```
    TTS.synthesize success | lang=%s | chars=%d | file=%s | duration_ms=%.2f
    ```
  - 失败（含异常堆栈）：
    ```
    TTS.synthesize failed | lang=%s | duration_ms=%.2f
    <stack trace...>
    ```

## NetworkCoordinator（控制台输出为主）
控制台打印用于展示事务生命周期与关键 I/O（注意：日志文件仍使用 `pipeline` logger）：

- 协商（negotiate）示例：
  ```
  ==================================================
  收到协商请求 @ 2025-11-30 12:00:00
  协商成功: transaction_id= a1b2c3d4..., next_action=send_audio
  ==================================================
  ```

- 执行（execute）示例（重要：只打印关键字段，敏感或大体积字段需省略或截断）：
  ```
  ==================================================
  收到执行请求 @ 2025-11-30 12:00:01
  事务 ID: a1b2c3d4..., 模式: 0
  流程计划: ASR=后端, 翻译=后端, TTS=后端
  语言参数: 源='en', 目标='fr'
  Payload: 收到音频数据 (Base64)，大小约 512KB
  --> 调用后端 ASR...
  <-- 翻译结果: 'Bonjour, ...' (耗时: 120.00ms)
  总处理时间: 345.67ms
  构造响应: text='Bonjour, ...', audio_present=True
  ==================================================
  ```

- 建议打印细节
  - 当前收到的文本（`text`／`asr_text`／`nlp_text`）只打印前 200 字并截断：`'Hello world...'(len=123)`
  - 对于 `audio_base64` 仅标注存在与近似大小，不写入完整 base64。
  - 返回响应时打印 `transaction_id`、`status`、`results.text`（截断）与是否包含音频。

## 错误与异常记录
- 使用 `logger.exception` 保留堆栈信息，日志条目会包含 `ERROR` 及后续堆栈。
- 错误响应约定（NetworkCoordinator 返回给前端）：
  ```json
  {
    "request_id": "<transaction_id>",
    "status": "error",
    "error_code": "ASR_FAILED",
    "error_message": "ASR 模块调用失败: ...",
    "results": null
  }
  ```
- 日志中同时记录相关上下文（mode、source/target language、timing_ms）。

## Transaction / meta 字段
在 pipeline 的最终 `INFO` 或控制台输出中应包含 `meta`：
- `mode`：交互模式编号（0-6）。
- `timing_ms`：各阶段耗时字典，推荐结构：
  ```
  "timing_ms": { "asr": 120.0, "translator": 80.0, "tts": 140.0, "total": 340.0 }
  ```

## 隐私与容量注意
- 严禁将完整 `audio_base64` 写入日志或控制台（仅标注存在/大小/sha256 摘要）。
- 对敏感文本（个人信息）应考虑脱敏或不记录完整内容。

## 快速排查示例
- 查找某事务：
  ```bash
  grep "a1b2c3d4" logs/pipeline.log
  ```
- 查找最近的 ASR 错误：
  ```bash
  grep "ASR.transcribe failed" logs/pipeline.log | tail -n 50
  ```

## 日志轮转与保留（建议）
- 建议对 `logs/pipeline.log` 使用按天或按大小轮转（`logging.handlers.RotatingFileHandler` 或 `TimedRotatingFileHandler`）。
- 保留周期建议：7-30 天（视磁盘与合规要求）。

## 总结
- 日志行为分两类：文件日志（结构化、持久化，logger 名称为 `pipeline`）与控制台简要生命周期输出（用于实时观察请求/响应关键字段）。
- 保持消息统一模板（便于 grep/聚合），对大/敏感数据进行省略或摘要化。


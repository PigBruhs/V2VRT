### **V2VRT API 接口文档 (v3.0)**

说明：在当前工作的“最后一次通信”中，后端响应必定包含 asr_result 与 nlp_result（无论 ASR/NLP 是由前端提供还是由后端生成）。该保证用于前端统一展示与存档最终识别与翻译结果。

本 API 采用两阶段交互模型：**协商** 和 **执行**。

---

### **阶段 1: 协商 (Negotiate)**

前端首先发起协商请求，告知后端希望使用的处理模式。

*   **端点**: `POST /api/v1/negotiate`
*   **Content-Type**: `application/json`

#### **请求体**

```json
{
    "mode": 0 
}
```
*   `mode`: `integer` (必填) - `0`到`6`的模式码。

#### **成功响应 (`status: "ok"`)**

后端返回一个唯一的事务ID和下一步的指示。

```json
{
    "status": "ok",
    "transaction_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "next_action": "send_audio",
    "execute_endpoint": "/api/v1/execute"
}
```
*   `transaction_id`: `string` - 本次交互的唯一ID，在执行阶段必须回传。
*   `next_action`: `string` - 指示前端下一步应发送的数据类型。
    *   `"send_audio"`: 前端需要发送音频数据。
    *   `"send_text"`: 前端需要发送文本数据。

---

### **阶段 2: 执行 (Execute)**

前端根据协商结果，将数据和事务ID发送到执行端点。

*   **端点**: `POST /api/v1/execute`
*   **Content-Type**: `application/json`

#### **请求体**

```json
{
    "transaction_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "payload": {
        "audio_base64": "UklGRiQ...AA==",
        "source_lang": "en",
        "target_lang": "fr",
        "asr_text": "Hello, how are you?",
        "asr_lang": "en",
        "asr_segments": [ ... ],
        "nlp_text": "Bonjour, comment allez-vous?",
        "nlp_source_lang": "en",
        "nlp_target_lang": "fr"
    }
}
```
*   `transaction_id`: `string` (必填) - 从协商阶段获取的ID。
*   `payload`: `object` (必填) - 包含实际数据的载荷。
    *   如果 `next_action` 是 `"send_audio"`，`payload` 中必须包含 `audio_base64`。
    *   如果 `next_action` 是 `"send_text"`，`payload` 中必须包含 `text`。
    *   `source_lang` 和 `target_lang` 根据具体模式按需提供。
    *   当 ASR 在前端执行时，`asr_text`（可选配 `asr_lang`、`asr_segments`）应提供，用于后端透传并标注 provider。
    *   当 NLP/翻译在前端执行时，`nlp_text` 及其语言字段应提供，以便后端在最后一次通信中返回完整结果。

#### **最终响应**

响应格式与旧版 `/pipeline` 端点一致，且在“最后一次通信”中必定包含 asr_result 与 nlp_result 两个对象，便于前端统一展示。

##### **成功响应 (`status: "ok"`)**
```json
{
    "transaction_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "ok",
    "results": {
        "text": "Bonjour, comment allez-vous?",
        "audio_base64": "UklGRiQ...AA==",
        "asr_result": {
            "text": "Hello, how are you?",
            "lang": "en",
            "segments": [ ... ],
            "provider": "backend"
        },
        "nlp_result": {
            "text": "Bonjour, comment allez-vous?",
            "source_lang": "en",
            "target_lang": "fr",
            "provider": "backend"
        },
        "meta": { "mode": 0, "timing_ms": { "asr": 12.3, "translator": 45.6, "tts": 30.1, "total": 88.0 } }
    }
}
```

##### **失败响应 (`status: "error"`)**
错误响应中字段名采用 `transaction_id` 以便前端关联事务。

```json
{
    "transaction_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    "status": "error",
    "error_code": "ASR_FAILED",
    "error_message": "ASR 模块调用失败: ...",
    "results": null
}
```

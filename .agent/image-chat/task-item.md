# 实施计划

- [ ] 1. 扩展后端 API 请求模型以支持图片数据
  - 修改 `main.py` 中的 `ChatRequest` 模型，新增可选的 `image` 字段（Base64 字符串）和 `image_mime_type` 字段
  - 在 `/api/chat` 路由中添加图片数据校验逻辑（格式合法性、大小限制）
  - 确保不带图片的请求保持向后兼容
  - _需求：3.1、3.3、3.4_

- [ ] 2. 扩展 LLM 客户端支持多模态消息格式
  - 修改 `llm_client.py` 中 `HunyuanClient.chat()` 方法的 messages 类型声明，支持 content 为数组格式（包含 text 和 image_url 类型）
  - 新增 `chat_with_vision()` 方法，自动切换到 Vision 模型并构建多模态消息
  - 新增 `HunyuanConfig` 中的 `vision_model` 字段（如 `hunyuan-vision`）
  - 实现 Vision 调用失败时的降级逻辑：提取纯文本部分，回退到标准模型调用
  - 在 Mock 模式下返回模拟的图片分析响应
  - 在 `get_status()` 中新增 `vision_supported` 字段
  - _需求：4.1、4.2、4.3、4.4_

- [ ] 3. 修改 Agent Loop 以传递多模态消息
  - 修改 `agent_loop.py` 中 `_act()` 方法，检测当前消息是否包含图片，若包含则调用 `chat_with_vision()` 而非普通 `chat()`
  - 修改 `main.py` 中 `/api/chat` 路由，将图片数据注入到 Agent Loop 的上下文中
  - _需求：3.1、3.2、4.1_

- [ ] 4. 扩展上下文引擎处理多模态消息
  - 修改 `context_engine.py` 中 `ConversationMessage` 数据类，新增可选的 `image_data` 字段和 `has_image` 标记
  - 修改 `ContextWindow.to_prompt_messages()` 方法，当消息包含图片时构建 OpenAI Vision 格式的 content 数组
  - 修改 `_compact()` 压缩逻辑，将图片消息压缩为文本摘要（如"[用户发送了一张图片]"）
  - 图片消息的 token 估算按固定值（如 765 tokens）计算
  - _需求：5.1、5.2、5.3、5.4_

- [ ] 5. 扩展记忆系统处理图片相关内容
  - 修改 `context_engine.py` 中 `add_message()` 方法，当存储图片消息到记忆时仅保存文本描述而非原始图片数据
  - 确保 token 预算不足时优先丢弃历史图片数据
  - _需求：5.2、5.4_

- [ ] 6. 新增 LLM Vision 能力检测 API
  - 在 `main.py` 中新增 `/api/llm/capabilities` 端点，返回当前 LLM 的能力信息（包括 `vision_supported` 布尔值）
  - 或扩展现有 `/api/llm/status` 端点，增加 `vision_supported` 字段
  - _需求：6.1、6.3_

- [ ] 7. 前端：实现图片上传组件与预览功能
  - 在 `static/index.html` 的输入区域添加图片上传按钮（📎 图标）和隐藏的 file input
  - 在 `static/main.js` 中实现图片选择、预览缩略图、移除图片的交互逻辑
  - 实现拖拽上传（dragover/drop 事件）和粘贴上传（paste 事件）
  - 添加文件大小校验（5MB 限制）和文件类型校验
  - 在 `static/style.css` 中添加图片预览区、拖拽高亮等样式
  - _需求：1.1、1.2、1.3、1.4、1.5、1.6_

- [ ] 8. 前端：实现图文混合消息发送与展示
  - 修改 `static/main.js` 中 `sendMessage()` 函数，将图片转为 Base64 并附加到请求体
  - 修改 `appendMessage()` 函数，支持在用户消息气泡中展示图片缩略图
  - 允许纯图片消息发送（无文本时发送按钮仍可用）
  - 发送后清除图片预览状态
  - _需求：2.1、2.2、2.3、2.4_

- [ ] 9. 前端：LLM Vision 能力状态展示
  - 在 `static/main.js` 中扩展 `refreshLLMStatus()` 函数，获取 Vision 能力信息
  - 根据 Vision 支持状态控制图片上传按钮的启用/禁用
  - 在头部状态栏和底部状态栏展示多模态能力标识
  - _需求：6.1、6.2、6.3_

- [ ] 10. Dockerfile 与依赖更新
  - 更新 `Dockerfile` 中的环境变量，新增 `HUNYUAN_VISION_MODEL` 配置
  - 确认 `requirements.txt` 无需额外依赖（Base64 处理使用 Python 标准库）
  - _需求：4.1_

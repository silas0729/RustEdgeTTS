# 第三方服务、音色与生成内容声明

## 服务性质

Edge TTS Studio 是独立开发的第三方学习研究项目，通过 `edge-tts-rust` 客户端访问 Microsoft Edge Read Aloud 相关在线语音服务。它不是 Microsoft 官方产品，也未获得 Microsoft 的赞助、认证、授权或背书。

Microsoft、Microsoft Edge、Edge Read Aloud 以及相关名称和音色属于其各自权利人。本项目名称中的 “Edge TTS” 仅用于说明兼容或调用的技术对象，不表示任何合作关系。

## 使用限制

内置在线 TTS 功能仅供个人学习、实验和非商业研究。不得使用本项目或内置服务制作、交付或运营商业配音、广告、付费内容、商业视频、游戏、课程、SaaS、API、客户项目或其他营利用途。

本项目许可证只处理项目作者原创代码的授权。它不授予、转让或暗示授予任何第三方服务、端点、音色、语音模型、商标或生成音频的权利。

## 输入与输出责任

使用在线 TTS 时，输入文本及合成设置会发送到 Microsoft 相关在线服务。用户应确保自己有权处理和提交输入内容，并自行判断生成音频能否被保存、公开、传播或以其他方式使用。

生成音频并不当然获得版权、表演权、声音权、商标权、商业使用权或其他权利。用户不得利用生成内容冒充、误导、侵权、违法或规避任何技术与服务限制。

## 服务条款与可用性

用户必须自行查阅并遵守 Microsoft 当前有效的服务协议、产品条款、隐私声明及适用法律。在线端点、音色、返回格式和访问策略可能随时改变、限制或停止，本项目不承诺服务可用性。

截至本声明版本日期，没有发现 Microsoft 发布的、专门且明确授予第三方非官方 Edge Read Aloud 客户端商业使用这些音色和生成音频权利的公开许可。因此，本项目不提供也不暗示提供此类商业授权。

参考资料：

- Microsoft Services Agreement: <https://www.microsoft.com/servicesagreement>
- Microsoft Edge Read Aloud: <https://support.microsoft.com/edge/use-immersive-reader-in-microsoft-edge>
- PolyForm Noncommercial License 1.0.0: <https://polyformproject.org/licenses/noncommercial/1.0.0>

## 本地字幕功能

音频转字幕功能在模型下载完成后于本机运行。Whisper 模型、Candle、`rwhisper` 及其他第三方组件分别适用其自身许可证；这些许可证与在线 TTS 服务权利是相互独立的问题。

## Qwen3-TTS 本地语音功能

本项目可由用户单独选择下载并在本机运行
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` 或
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`。两个模型由 Qwen 团队发布，模型页面均标注为 Apache License 2.0；模型文件不包含在本项目安装包中。应用只加载用户当前选择的版本，0.6B 首次约需下载 2.4 GB，1.7B 首次约需下载 4.5 GB。`speakers-qwen3-tts` 是基于 Candle 的社区 Rust 推理实现，并非 Qwen 官方 Rust SDK。

Qwen3-TTS 模型、模型名称、预设音色、推理组件和生成内容分别受其适用许可证、模型条款及法律约束。本项目的 PolyForm Noncommercial 许可证仍然只允许将本项目作者代码用于个人学习和非商业研究；第三方组件采用更宽松许可证，并不会自动扩大本项目许可证授予的用途。

用户不得使用 Qwen3-TTS 或任何预设音色冒充真实个人、制作欺骗性内容、侵犯声音权、人格权或著作权，或从事法律禁止的行为。对外发布前应人工试听并核对文本、发音、语种和字幕时间轴。

参考资料：

- Qwen3-TTS 官方项目：<https://github.com/QwenLM/Qwen3-TTS>
- Qwen3-TTS 0.6B CustomVoice 模型页：<https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice>
- Qwen3-TTS 1.7B CustomVoice 模型页：<https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice>
- Apache License 2.0：<https://www.apache.org/licenses/LICENSE-2.0>

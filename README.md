# MiniCPM-V Local

Local visual preprocessing using MiniCPM-V 4.6 (1.3B). Captions images and
video timelines locally without sending pixels to the main model.

⚠️ **自动卸载**：本地模型 server 在 5 分钟无请求后自动从内存/显存卸载。
下次调用会自动重新加载（cold start ≈ 3–15s）。

See `docs/specs/2026-05-13-minicpm-v-local-design.md` for full design.

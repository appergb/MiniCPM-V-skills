"""Single-image pipeline. Spec §10.3."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path

from minicpm_v_local.client import VLMClient

DEFAULT_PROMPT = (
    "请详细描述这张图片的内容。按以下顺序组织你的回答：\n"
    "1. 整体场景：这是什么样的画面（截图/照片/图表/文档）？\n"
    "2. 文字内容：逐一列出图片中可见的所有文字，包括中文和英文。\n"
    "3. 物体与布局：描述可见的主要物体及其空间排列。\n"
    "4. 如有 UI 界面元素（按钮、菜单、标签），请分别说明。"
)

PROMPT_PRESETS: dict[str, str] = {
    "ui": "请描述这个用户界面：列出可见的按钮、菜单、输入框、标签和所有文字内容，"
          "说明它们的层级与布局。",
    "photo": "请详细描述这张照片：场景、主体、氛围、可见的人物或物体、光线和构图。",
    "doc": "请逐字列出这份文档的所有文字内容，保持原有的段落、标题、列表结构。"
           "如有表格请按行列输出。如有插图请单独说明。",
    "chart": "请分析这个图表：类型（柱状图/折线图/饼图等）、坐标轴含义、数据系列、"
             "关键数值、整体趋势和结论。",
}


def resolve_prompt(prompt: str | None, preset: str | None) -> str:
    """`prompt` overrides everything. `preset` picks from PROMPT_PRESETS.
    Otherwise returns DEFAULT_PROMPT."""
    if prompt:
        return prompt
    if preset:
        if preset not in PROMPT_PRESETS:
            raise ValueError(
                f"unknown prompt preset: {preset!r}. "
                f"Choose from: {sorted(PROMPT_PRESETS)}"
            )
        return PROMPT_PRESETS[preset]
    return DEFAULT_PROMPT


def caption_image(client: VLMClient, image_path: Path, *,
                  model: str, prompt: str = DEFAULT_PROMPT,
                  served_model: str | None = None) -> dict:
    """`model` = identifier reported in the JSON envelope (e.g. HF repo ID).
    `served_model` = exact name to send in the HTTP `model` field; must match
    what the server preloaded. If None, falls back to `model`."""
    t0 = time.monotonic()
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    text = client.caption(image_path, prompt=prompt, model=served_model or model)
    dt = int((time.monotonic() - t0) * 1000)
    return {
        "version": 1,
        "input": {"path": str(image_path), "sha256": sha},
        "model": model,
        "result": {"caption": text, "objects": [], "ocr_text": None},
        "timing_ms": {"load": 0, "infer": dt},
    }

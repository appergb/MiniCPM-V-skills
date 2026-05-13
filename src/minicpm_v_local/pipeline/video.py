"""Video pipeline. Spec §10.4, §12.4."""
from __future__ import annotations
import json
import re
import subprocess
import time
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path

from minicpm_v_local import paths
from minicpm_v_local.client import VLMClient
from minicpm_v_local.config import VideoConfig

DEFAULT_PROMPT = "Describe what's happening in this frame in one sentence."


@dataclass
class Frame:
    t: float
    path: Path
    caption: str | None = None
    error: str | None = None


def probe(video: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration:stream=r_frame_rate",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    duration = float(data["format"]["duration"])
    fps_str = data["streams"][0].get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else 30.0
    return {"duration": duration, "fps": fps}


def extract_keyframes(video: Path, *, cfg: VideoConfig) -> list[Frame]:
    run_id = uuid.uuid4().hex[:8]
    frames_dir = paths.frames_tmp_dir(run_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # scene-change frames
    pattern = str(frames_dir / "scene_%04d.jpg")
    log = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video),
         "-vf", f"select='gt(scene,{cfg.scene_threshold})',showinfo",
         "-vsync", "vfr", pattern],
        capture_output=True, text=True,
    )

    times = []
    for line in log.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            times.append(float(m.group(1)))

    files = sorted(frames_dir.glob("scene_*.jpg"))
    frames = [Frame(t=t, path=f) for t, f in zip(times, files)]

    # uniform fallback (确保每 fallback_interval 至少一帧)
    info = probe(video)
    duration = info["duration"]
    interval = cfg.fallback_interval
    target_count = int(duration / interval)
    if len(frames) < target_count:
        for i in range(target_count):
            t = i * interval
            out = frames_dir / f"uniform_{i:04d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                 "-frames:v", "1", str(out)],
                capture_output=True, check=False,
            )
            if out.exists():
                frames.append(Frame(t=t, path=out))

    frames.sort(key=lambda f: f.t)
    if len(frames) > cfg.max_frames:
        # 等距下采样
        step = len(frames) / cfg.max_frames
        frames = [frames[int(i * step)] for i in range(cfg.max_frames)]
    return frames


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def merge_scenes(frames: list[Frame], *, threshold: float) -> list[dict]:
    if not frames:
        return []
    scenes = []
    cur = {"start": frames[0].t, "end": frames[0].t, "summary": frames[0].caption or "",
           "frame_indices": [0], "_tokens": _token_set(frames[0].caption or "")}
    for i, fr in enumerate(frames[1:], start=1):
        cap = fr.caption or ""
        toks = _token_set(cap)
        sim = _jaccard(cur["_tokens"], toks)
        if sim >= threshold:
            cur["end"] = fr.t
            cur["frame_indices"].append(i)
            cur["_tokens"] |= toks
        else:
            scenes.append({k: v for k, v in cur.items() if not k.startswith("_")})
            cur = {"start": fr.t, "end": fr.t, "summary": cap, "frame_indices": [i],
                   "_tokens": toks}
    scenes.append({k: v for k, v in cur.items() if not k.startswith("_")})
    return scenes


def process_video(
    client: VLMClient, video: Path, *,
    model: str, cfg: VideoConfig, prompt: str = DEFAULT_PROMPT,
    served_model: str | None = None,
    on_frame_done=None,
) -> dict:
    """`model` = identifier reported in the JSON envelope (e.g. HF repo ID).
    `served_model` = exact name to send in the HTTP `model` field; must match
    what the server preloaded. If None, falls back to `model`."""
    t0 = time.monotonic()
    info = probe(video)
    sha = hashlib.sha256(video.read_bytes()).hexdigest()
    t_ffmpeg_start = time.monotonic()
    frames = extract_keyframes(video, cfg=cfg)
    ffmpeg_ms = int((time.monotonic() - t_ffmpeg_start) * 1000)

    wire_model = served_model or model
    for fr in frames:
        try:
            fr.caption = client.caption(fr.path, prompt=prompt, model=wire_model)
        except Exception as e:
            fr.error = str(e)
        if on_frame_done:
            on_frame_done(fr)

    scenes = merge_scenes(frames, threshold=cfg.scene_merge_similarity)
    infer_ms = int((time.monotonic() - t0) * 1000) - ffmpeg_ms

    return {
        "version": 1,
        "input": {"path": str(video), "sha256": sha, "duration_s": info["duration"], "fps": info["fps"]},
        "model": model,
        "frames": [
            {"t": fr.t, "path": str(fr.path), "caption": fr.caption, "error": fr.error}
            for fr in frames
        ],
        "scenes": scenes,
        "timing_ms": {"ffmpeg": ffmpeg_ms, "infer_total": infer_ms},
    }

"""CLI entry. Spec §10."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from minicpm_v_local import paths, doctor
from minicpm_v_local.config import load
from minicpm_v_local.runtime import detect
from minicpm_v_local.runtime.factory import get_backend
from minicpm_v_local.server import manager
from minicpm_v_local.server.state import read_state
from minicpm_v_local.download import ensure_model
from minicpm_v_local.client import VLMClient
from minicpm_v_local.pipeline.image import caption_image
from minicpm_v_local.pipeline.video import process_video


def _add_common(p):
    p.add_argument("--backend", choices=["auto", "mlx", "cuda", "cpu"], default=None)
    p.add_argument("--quant", default=None)
    p.add_argument("--ttl", type=int, default=None, help="保活秒数；0 = 立即销毁")
    p.add_argument("--max-lifetime", type=int, default=None)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--isolated", action="store_true")
    p.add_argument("--output", choices=["json", "jsonl"], default="json")
    p.add_argument("--prompt", default=None)


def build_parser():
    parser = argparse.ArgumentParser(prog="minicpm-v")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_doc = sub.add_parser("doctor")
    p_doc.add_argument("--reset", action="store_true")
    p_doc.add_argument("--backend", choices=["auto", "mlx", "cuda", "cpu"], default=None,
                       help="Override platform autodetection")

    p_img = sub.add_parser("image")
    p_img.add_argument("path", type=Path)
    _add_common(p_img)

    p_vid = sub.add_parser("video")
    p_vid.add_argument("path", type=Path)
    _add_common(p_vid)

    sub.add_parser("status")
    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--force", action="store_true")
    return parser


def _cfg_with_overrides(args) -> "Config":
    overrides = {}
    if args.backend: overrides["backend"] = args.backend
    if args.quant: overrides["quant"] = args.quant
    if args.ttl is not None: overrides["idle_timeout"] = args.ttl
    if args.max_lifetime is not None: overrides["max_lifetime"] = args.max_lifetime
    if args.isolated: overrides["isolation"] = True
    return load(paths.config_file(), cli_overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "doctor":
        if args.reset and paths.config_file().exists():
            paths.config_file().unlink()
        force = args.backend if args.backend and args.backend != "auto" else None
        if force is None:
            return doctor.run()
        return doctor.run(force_backend=force)

    if args.cmd == "status":
        s = read_state(paths.state_file())
        print(json.dumps(s.to_dict() if s else {"alive": False}, indent=2))
        return 0

    if args.cmd == "stop":
        manager.stop(force=args.force)
        return 0

    # image / video
    if not paths.config_file().exists():
        return doctor.run()

    cfg = _cfg_with_overrides(args)
    tag = detect.resolve(cfg.backend)
    backend = get_backend(tag, quant=cfg.quant)

    model_dir = ensure_model(backend.artifact_id(), backend=tag)

    state = manager.ensure_warm(
        backend, model_dir,
        port_range=cfg.server.port_range,
        health_timeout=cfg.server.health_timeout,
        ttl_seconds=args.ttl if args.ttl is not None else cfg.idle_timeout,
        max_lifetime=cfg.max_lifetime,
        keep=args.keep,
        isolation_mode=(cfg.isolation_mode if cfg.isolation else "none"),
    )

    client = VLMClient(base_url=f"http://127.0.0.1:{state.port}")
    try:
        if args.cmd == "image":
            result = caption_image(client, args.path, model=backend.artifact_id(),
                                   prompt=args.prompt or _default_image_prompt())
        else:
            result = process_video(client, args.path, model=backend.artifact_id(),
                                   cfg=cfg.video, prompt=args.prompt or _default_video_prompt())
    finally:
        client.close()

    if args.ttl == 0:
        manager.stop()

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _default_image_prompt() -> str:
    from minicpm_v_local.pipeline.image import DEFAULT_PROMPT
    return DEFAULT_PROMPT


def _default_video_prompt() -> str:
    from minicpm_v_local.pipeline.video import DEFAULT_PROMPT
    return DEFAULT_PROMPT

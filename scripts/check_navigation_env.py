#!/usr/bin/env python3
"""
Standalone smoke test for the Navigation task environment (AI2-THOR).

What this checks:
1) A navigation task JSON is parseable (by default, uses the repo's built-in JSON)
2) AI2-THOR Controller can start
3) A scene can be loaded + agent can be teleported to the episode start pose
4) A few actions can be executed (Move/Rotate/Look)
5) (optional) Saves rendered frames for visual debugging

Example:
  python scripts/check_navigation_env.py --eval-set base --seed 0 --steps 5 --save-dir /tmp/nav_smoke
  python scripts/check_navigation_env.py --dataset-json /path/to/base.json --task-idx 0 --steps 5

Notes:
- This script does NOT install any dependencies. If imports fail, it prints a useful hint.
- This script is intentionally **independent of `vagen`** (no imports from `vagen`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_ACTIONS = [
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
]


def _repo_root() -> Path:
    # This file lives at <repo>/scripts/check_navigation_env.py
    return Path(__file__).resolve().parents[1]


def _dataset_path(eval_set: str) -> Path:
    return _repo_root() / "vagen" / "env" / "navigation" / "datasets" / f"{eval_set}.json"


def _print_header(args: argparse.Namespace) -> None:
    print("=== Navigation Env Smoke Test ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Repo root: {_repo_root()}")
    print(f"eval_set: {args.eval_set}")
    print(f"dataset_json: {args.dataset_json or '(auto)'}")
    print(f"seed: {args.seed}")
    print(f"task_idx: {args.task_idx if args.task_idx is not None else '(seed-based)'}")
    print(f"steps: {args.steps}")
    print(f"resolution: {args.resolution}")
    print(f"fov: {args.fov}")
    print(f"gpu_device: {args.gpu_device}")
    print(f"multiview: {args.multiview}")
    print(f"down_sample_ratio: {args.down_sample_ratio}")
    print(f"step_length: {args.step_length}")
    print(f"success_threshold: {args.success_threshold}")
    print(f"cloud_rendering: {not args.no_cloud_rendering}")
    print(f"save_dir: {args.save_dir or '(disabled)'}")
    print()


def _load_tasks_from_json(ds_path: Path, down_sample_ratio: float) -> List[Dict[str, Any]]:
    with ds_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("JSON does not contain a non-empty 'tasks' list.")

    if 0 <= down_sample_ratio < 1:
        select_every = max(1, round(1 / down_sample_ratio))
        tasks = tasks[0 : len(tasks) : select_every]

    return tasks


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_frame(frame: Any, out_path: Path) -> Optional[str]:
    try:
        # frame is expected to be a HxWx3 uint8 numpy array (from ai2thor event.frame)
        pil_image = importlib.import_module("PIL.Image")  # pillow is optional; only needed if --save-dir is used
        pil_image.fromarray(frame).save(out_path)
        return None
    except ImportError:
        return "PIL not installed (pip install pillow) - disable --save-dir or install pillow"
    except Exception as e:
        return f"failed to save frame: {e}"


def _compute_distance_to_target(agent_pos: Dict[str, float], target_pos: Dict[str, float]) -> float:
    return math.sqrt((agent_pos["x"] - target_pos["x"]) ** 2 + (agent_pos["z"] - target_pos["z"]) ** 2)


def _thor_step(controller: Any, action: str, step_length: float) -> Any:
    if action == "moveahead":
        return controller.step(action="MoveAhead", moveMagnitude=step_length)
    if action == "moveback":
        return controller.step(action="MoveBack", moveMagnitude=step_length)
    if action == "moveright":
        return controller.step(action="MoveRight", moveMagnitude=step_length)
    if action == "moveleft":
        return controller.step(action="MoveLeft", moveMagnitude=step_length)
    if action == "rotateright":
        return controller.step(action="RotateRight", degrees=90)
    if action == "rotateleft":
        return controller.step(action="RotateLeft", degrees=90)
    if action == "lookup":
        return controller.step(action="LookUp", degrees=30)
    if action == "lookdown":
        return controller.step(action="LookDown", degrees=30)
    raise ValueError(f"Unknown action: {action}")


def _resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_json:
        return Path(args.dataset_json).expanduser().resolve()
    return _dataset_path(args.eval_set)


def _check_and_select_task(args: argparse.Namespace) -> Tuple[Path, List[Dict[str, Any]], int, Dict[str, Any]]:
    ds_path = _resolve_dataset_path(args)
    print(f"[1/3] Checking task JSON: {ds_path}")
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)

    tasks = _load_tasks_from_json(ds_path, args.down_sample_ratio)
    idx = args.task_idx if args.task_idx is not None else (args.seed % len(tasks))
    if idx < 0 or idx >= len(tasks):
        raise IndexError(f"task_idx {idx} out of range (0..{len(tasks)-1})")
    task = tasks[idx]
    print(f"  OK: tasks={len(tasks)}; selected_idx={idx}; keys={sorted(task.keys())}")
    return ds_path, tasks, idx, task


def _run_env_smoke(args: argparse.Namespace) -> int:
    print("[2/3] Starting AI2-THOR controller...")
    try:
        ai2thor_controller = importlib.import_module("ai2thor.controller")
        ai2thor_platform = importlib.import_module("ai2thor.platform")
        CloudRendering = getattr(ai2thor_platform, "CloudRendering", None)
    except Exception as e:
        print("  FAIL: could not import ai2thor.")
        print(f"  Error: {e}")
        print()
        print("  Common causes:")
        print("  - ai2thor not installed (see vagen/env/README.md -> Navigation)")
        print("  - missing system deps (Linux often needs libvulkan1 / vulkan-tools)")
        print("  - platform/rendering backend mismatch (try --no-cloud-rendering)")
        return 3

    controller = None
    try:
        thor_config: Dict[str, Any] = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": args.resolution,
            "height": args.resolution,
            "fieldOfView": args.fov,
            "gpu_device": args.gpu_device,
            "server_timeout": args.server_timeout,
            "server_start_timeout": args.server_start_timeout,
        }
        if not args.no_cloud_rendering:
            if CloudRendering is None:
                raise RuntimeError("ai2thor.platform.CloudRendering not found (try --no-cloud-rendering)")
            thor_config["platform"] = CloudRendering

        controller = ai2thor_controller.Controller(**thor_config)
        print("  OK: AI2-THOR Controller constructed.")
    except Exception as e:
        print("  FAIL: AI2-THOR Controller construction failed.")
        print(f"  Error: {e}")
        if args.print_traceback:
            traceback.print_exc()
        return 4

    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    if save_dir is not None:
        _safe_mkdir(save_dir)

    print("[3/3] Resetting + stepping...")
    try:
        ds_path, tasks, idx, task = _check_and_select_task(args)
        scene_name = task.get("scene")
        if not isinstance(scene_name, str) or not scene_name:
            raise ValueError("task missing 'scene'")

        event = controller.reset(scene=scene_name)
        if args.multiview:
            cam_event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = cam_event.metadata["actionReturn"].copy()
            pose["orthographic"] = True
            controller.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        # Teleport to episode start pose (if present)
        agent_pose = task.get("agentPose")
        if isinstance(agent_pose, dict) and "position" in agent_pose:
            pos = agent_pose["position"]
            rot_y = agent_pose.get("rotation", 0.0)
            horizon = agent_pose.get("horizon", 0.0)
            controller.step(
                action="Teleport",
                position={"x": pos["x"], "y": pos["y"], "z": pos["z"]},
                rotation={"x": 0, "y": rot_y, "z": 0},
                horizon=horizon,
                standing=True,
            )

        instruction = task.get("instruction")
        target_type = task.get("targetObjectType")
        print(f"  OK: scene={scene_name} target={target_type} instruction={instruction!r}")

        # Save reset frame
        if save_dir is not None:
            err = _save_frame(controller.last_event.frame, save_dir / "step_000_reset.png")
            if err:
                print(f"  WARN: could not save reset frame: {err}")

        for i in range(args.steps):
            action = random.choice(VALID_ACTIONS)
            _thor_step(controller, action, args.step_length)

            last_ok = controller.last_event.metadata.get("lastActionSuccess")
            agent_pos = controller.last_event.metadata.get("agent", {}).get("position", {})

            target_pos = task.get("target_position")
            distance = None
            success = None
            done = False
            if isinstance(target_pos, dict) and "x" in target_pos and "z" in target_pos and isinstance(agent_pos, dict):
                distance = _compute_distance_to_target(agent_pos, target_pos)
                success = distance <= args.success_threshold
                done = bool(success)

            print(
                f"  step={i+1:03d}",
                f"action={action}",
                f"last_action_success={last_ok}",
                f"distance={distance}",
                f"success={success}",
                f"done={done}",
            )

            if save_dir is not None:
                err = _save_frame(controller.last_event.frame, save_dir / f"step_{i+1:03d}.png")
                if err:
                    print(f"  WARN: could not save frame for step {i+1}: {err}")

            if done:
                break

        return 0
    except Exception as e:
        print("  FAIL: runtime error during reset/step.")
        print(f"  Error: {e}")
        if args.print_traceback:
            traceback.print_exc()
        return 5
    finally:
        try:
            if controller is not None:
                controller.stop()
        except Exception:
            # Best-effort cleanup; avoid masking the real failure.
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone smoke test for the navigation task environment (AI2-THOR). No vagen imports."
    )
    parser.add_argument(
        "--eval-set",
        default="base",
        choices=["base", "common_sense", "complex_instruction", "visual_appearance", "long_horizon"],
        help="Which built-in navigation split to use (only used if --dataset-json is not provided).",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default="",
        help="Path to a navigation tasks JSON (expects top-level key 'tasks'). Overrides --eval-set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode selector (seed %% num_tasks).")
    parser.add_argument("--task-idx", type=int, default=None, help="Pick an explicit task index (overrides --seed).")
    parser.add_argument("--steps", type=int, default=5, help="How many random steps to run.")
    parser.add_argument("--resolution", type=int, default=255, help="Rendered image width/height.")
    parser.add_argument("--fov", type=int, default=100, help="Camera field of view.")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device id (passed to AI2-THOR).")
    parser.add_argument("--multiview", action="store_true", help="Enable multiview camera.")
    parser.add_argument("--down-sample-ratio", type=float, default=1.0, help="Downsample tasks list (0..1].")
    parser.add_argument("--step-length", type=float, default=0.5, help="Move magnitude for MoveAhead/Back/Left/Right.")
    parser.add_argument("--success-threshold", type=float, default=1.5, help="Success if distance <= threshold.")
    parser.add_argument(
        "--no-cloud-rendering",
        action="store_true",
        help="Do not force CloudRendering platform (use AI2-THOR default).",
    )
    parser.add_argument("--server-timeout", type=int, default=300, help="AI2-THOR server_timeout (seconds).")
    parser.add_argument("--server-start-timeout", type=int, default=300, help="AI2-THOR server_start_timeout (seconds).")
    parser.add_argument("--save-dir", type=str, default="", help="If set, saves frames into this directory.")
    parser.add_argument(
        "--print-traceback",
        action="store_true",
        help="Print full Python traceback on failures (useful for debugging deps).",
    )
    args = parser.parse_args()

    args.save_dir = args.save_dir.strip() or None
    args.dataset_json = args.dataset_json.strip() or None

    _print_header(args)

    return _run_env_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())



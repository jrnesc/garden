"""
Record animation clips from the locomotion server.
Produces bone data in the exact format GetBoneData() outputs (world-space).
Run: cd locomotion-server && uv run python record_clips.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
AI4ANIM_ROOT = REPO_ROOT / "ai4animationpy"
sys.path.insert(0, str(AI4ANIM_ROOT))

from server import LocomotionController, Vector3

OUTPUT = REPO_ROOT / "app" / "public" / "animations.json"
FPS = 30
DT = 1.0 / FPS


def record_clip(controller, name, velocity, direction, seconds=3):
    """Record bone data for N seconds at given velocity/direction."""
    controller.SetIdle()
    controller.ClientVelocity = np.array(velocity, dtype=np.float64)
    controller.ClientDirection = np.array(direction, dtype=np.float64)

    # Warm up for 1 second
    for _ in range(FPS):
        controller.Update(DT)

    frames = []
    for _ in range(int(seconds * FPS)):
        controller.Update(DT)
        data = controller.GetBoneData()
        # Strip down to just what we need
        frames.append({
            "bones": [
                {
                    "name": b["name"],
                    "position": b["position"],
                    "quaternion": b["quaternion"],
                }
                for b in data["bones"]
            ],
            "root": data["root"],
        })

    print(f"  [{name}] {len(frames)} frames recorded")
    return frames


print("[record] Loading controller...")
ctrl = LocomotionController()

# Reset to origin
ctrl.ClientVelocity = Vector3.Zero()
ctrl.ClientDirection = np.array([0, 0, 1], dtype=np.float64)

clips = {}

# Idle
print("[record] Recording idle...")
clips["idle"] = record_clip(ctrl, "idle", [0, 0, 0], [0, 0, 1], seconds=3)

# Walk forward (several styles)
for style in ["Neutral", "Zombie", "Dinosaur", "Chicken", "BigSteps"]:
    print(f"[record] Recording {style}...")
    ctrl.CurrentStyle = style
    if style in ctrl.GuidanceTemplates:
        ctrl.GuidancePositions = ctrl.GuidanceTemplates[style].copy()
    clips[style.lower()] = record_clip(ctrl, style, [0, 0, 1.2], [0, 0, 1], seconds=3)

output = {"fps": FPS, "clips": clips}

with open(OUTPUT, "w") as f:
    json.dump(output, f)

size = OUTPUT.stat().st_size / (1024 * 1024)
print(f"[record] Saved {OUTPUT} ({size:.1f} MB)")
print(f"[record] Clips: {list(clips.keys())}")

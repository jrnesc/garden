"""
Validate ONNX output against Python model output.
Constructs the same T-pose input the TS engine would use on first frame,
runs both Python model and ONNX, and compares outputs.

Run: cd locomotion-server && uv run python validate_onnx.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
AI4ANIM_ROOT = REPO_ROOT / "ai4animationpy"
sys.path.insert(0, str(AI4ANIM_ROOT))

BIPED_DIR = AI4ANIM_ROOT / "Demos" / "Locomotion" / "Biped"
ASSETS_DIR = AI4ANIM_ROOT / "Demos" / "_ASSETS_" / "Geno"
sys.path.insert(0, str(ASSETS_DIR))

from ai4animation.Math import Transform, Vector3, Rotation, Quaternion
from ai4animation.AI.FeedTensor import FeedTensor
from ai4animation.AI.ReadTensor import ReadTensor
import Definitions

# Load companion data (same as TS uses)
with open(REPO_ROOT / "app" / "public" / "locomotion-data.json") as f:
    companion = json.load(f)

bone_names = companion["boneNames"]
n = companion["boneCount"]
tpose = np.array(companion["tposeTransforms"])  # [23, 4, 4]
guidances = companion["guidances"]

# Load Python model
model = torch.load(str(BIPED_DIR / "Network.pt"), weights_only=False, map_location="cpu")
model.eval()

# Build the SAME input the TS engine builds on first frame
# root = identity
root = np.eye(4)
inv_root = np.eye(4)

# Bone transforms in root-local space: inv(root) @ transforms = transforms (root=identity)
transforms = tpose.copy()
local_transforms = np.matmul(inv_root, transforms)  # = transforms since root=identity

inputs = FeedTensor("X", model.InputDim)
inputs.Feed(Transform.GetPosition(local_transforms))
inputs.Feed(Transform.GetAxisZ(local_transforms))
inputs.Feed(Transform.GetAxisY(local_transforms))

# Velocities = zero
velocities = Vector3.Zero(n)
inputs.Feed(Vector3.DirectionTo(velocities, root))

# Future trajectory = all at origin (rootControl uninitialized)
root_control = np.tile(np.eye(4), (16, 1, 1))
future_local = np.matmul(inv_root, root_control)
inputs.FeedVector3(Transform.GetPosition(future_local), x=True, y=False, z=True)
inputs.FeedVector3(Transform.GetAxisZ(future_local), x=True, y=False, z=True)

root_vel = Vector3.Zero(16)
root_vel_local = Vector3.DirectionTo(root_vel, root)
inputs.FeedVector3(root_vel_local, x=True, y=False, z=True)

# Guidance = idle
idle_guidance = np.array(guidances.get("Idle", [[0,0,0]] * n))
inputs.Feed(idle_guidance)

input_tensor = inputs.GetTensor().reshape(1, -1)
print(f"Input shape: {input_tensor.shape}, dim={model.InputDim}")
print(f"Input first 10: {input_tensor[0, :10].numpy()}")

# Run Python model
with torch.no_grad():
    py_out, _, _, _ = model(
        input_tensor,
        noise=0.5 * torch.ones(1, model.LatentDim),
        iterations=3,
        seed=torch.zeros(1, model.LatentDim),
    )
py_out = py_out.reshape(16, -1).numpy()
print(f"Python output shape: {py_out.shape}")

# Run ONNX model
import onnxruntime as ort
onnx_path = REPO_ROOT / "app" / "public" / "locomotion.onnx"
session = ort.InferenceSession(str(onnx_path))
onnx_input = input_tensor.numpy().astype(np.float32)
onnx_out = session.run(None, {"input": onnx_input})[0]
print(f"ONNX output shape: {onnx_out.shape}")

# Reshape ONNX to match
if onnx_out.ndim == 2 and onnx_out.shape[1] == 16 * 352:
    onnx_out = onnx_out.reshape(16, 352)
elif onnx_out.ndim == 3:
    onnx_out = onnx_out.reshape(16, -1)

# Compare outputs
print(f"\nMax abs diff Python vs ONNX: {np.max(np.abs(py_out - onnx_out)):.6f}")

# Parse frame 0 output using ReadTensor
reader = ReadTensor("Y", py_out)
root_vectors = reader.ReadVector3()  # [16, 3]
bone_positions = reader.ReadVector3(n)  # [16, 23, 3]
bone_rotations = reader.ReadRotation3D(n)  # [16, 23, 3, 3]
bone_velocities = reader.ReadVector3(n)  # [16, 23, 3]
contacts = reader.Read(4)  # [16, 4]

print(f"\n=== FRAME 0 (Python model) ===")
print(f"rootVector[0]: {root_vectors[0]}")
print()

for b in range(5):
    pos = bone_positions[0, b]
    rot = bone_rotations[0, b]
    print(f"bone[{b}] {bone_names[b]}:")
    print(f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  rot diag: [{rot[0,0]:.3f}, {rot[1,1]:.3f}, {rot[2,2]:.3f}]")
    print(f"  rot:\n{rot}")

# Also parse from ONNX output for comparison
onnx_reader = ReadTensor("Y", onnx_out)
onnx_root_vectors = onnx_reader.ReadVector3()
onnx_bone_positions = onnx_reader.ReadVector3(n)
onnx_bone_rotations = onnx_reader.ReadRotation3D(n)

print(f"\n=== FRAME 0 (ONNX model) ===")
print(f"rootVector[0]: {onnx_root_vectors[0]}")
for b in range(5):
    pos = onnx_bone_positions[0, b]
    rot = onnx_bone_rotations[0, b]
    print(f"bone[{b}] {bone_names[b]}:")
    print(f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  rot diag: [{rot[0,0]:.3f}, {rot[1,1]:.3f}, {rot[2,2]:.3f}]")

# Now show what the TS would read if it reads the flat output sequentially (frame-major)
print(f"\n=== TS-style sequential read from ONNX flat output ===")
flat = onnx_out.flatten()
print(f"Total output values: {len(flat)}")
p = 0
rv = flat[p:p+3]; p += 3
print(f"rootVector[0]: [{rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}]")
for b in range(5):
    bp = flat[p:p+3]; p += 3
    # After all 23 bone positions, then rotations
# Actually let me read it properly
p = 0
# Frame 0:
rv = flat[p:p+3]; p += 3
print(f"\nFrame 0 sequential read:")
print(f"  rootVec: [{rv[0]:.6f}, {rv[1]:.6f}, {rv[2]:.6f}]")
positions_f0 = []
for b in range(n):
    bp = flat[p:p+3]; p += 3
    positions_f0.append(bp)
for b in range(5):
    print(f"  pos[{b}] {bone_names[b]}: [{positions_f0[b][0]:.4f}, {positions_f0[b][1]:.4f}, {positions_f0[b][2]:.4f}]")

# Rotations: 6 values per bone (z then y)
for b in range(5):
    z = flat[p:p+3]; p_after_z = p+3
    y = flat[p_after_z:p_after_z+3]
    print(f"  rot[{b}] {bone_names[b]}: z=[{z[0]:.4f},{z[1]:.4f},{z[2]:.4f}] y=[{y[0]:.4f},{y[1]:.4f},{y[2]:.4f}]")
    p += 6

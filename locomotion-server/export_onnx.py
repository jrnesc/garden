"""
Export the CodebookMatching locomotion model to ONNX + companion data JSON.
Run: cd locomotion-server && uv run python export_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
AI4ANIM_ROOT = REPO_ROOT / "ai4animationpy"
sys.path.insert(0, str(AI4ANIM_ROOT))

BIPED_DIR = AI4ANIM_ROOT / "Demos" / "Locomotion" / "Biped"
ASSETS_DIR = AI4ANIM_ROOT / "Demos" / "_ASSETS_" / "Geno"
OUTPUT_DIR = REPO_ROOT / "app" / "public"

sys.path.insert(0, str(ASSETS_DIR))
import Definitions

# Load the model
print("[export] Loading model...")
model = torch.load(str(BIPED_DIR / "Network.pt"), weights_only=False, map_location="cpu")
model.eval()

print(f"[export] InputDim={model.InputDim}, OutputDim={model.OutputDim}, LatentDim={model.LatentDim}")
print(f"[export] SequenceLength={model.SequenceLength}, SequenceWindow={model.SequenceWindow}")


# Wrapper that bakes in noise/iterations/seed for clean ONNX export
class LocoInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("noise", 0.0 * torch.ones(1, model.LatentDim))
        self.register_buffer("seed", torch.zeros(1, model.LatentDim))

    def forward(self, x):
        # Unrolled forward with 3 denoising iterations, no results tracking
        inp = self.model.InputStats.Normalize(x)
        timestamps = self.model.timing().to(x.device)
        timestamps = self.model.TimeStats.Normalize(timestamps)

        z, p = self.model.Estimator(inp, noise=self.noise)
        p = p + self.seed
        z, p = self.model.Denoiser(p, inp, noise=self.noise)
        z, p = self.model.Denoiser(p, inp, noise=self.noise)
        z, p = self.model.Denoiser(p, inp, noise=self.noise)

        y = self.model.Decoder(z, inp, timestamps)
        y = self.model.OutputStats.Denormalize(y)
        return y


wrapper = LocoInference(model)
wrapper.eval()

# Test forward pass
dummy_input = torch.zeros(1, model.InputDim)
with torch.no_grad():
    test_output = wrapper(dummy_input)
print(f"[export] Test output shape: {test_output.shape}")

# Export ONNX
onnx_path = OUTPUT_DIR / "locomotion.onnx"
print(f"[export] Exporting ONNX to {onnx_path}...")
torch.onnx.export(
    wrapper,
    dummy_input,
    str(onnx_path),
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    dynamo=False,
)
onnx_size = onnx_path.stat().st_size / (1024 * 1024)
print(f"[export] ONNX saved: {onnx_size:.1f} MB")

# === Export companion data ===

print("[export] Exporting companion data...")

# Bone names
bone_names = Definitions.FULL_BODY_NAMES

# Load GLB for T-pose and parent info
from ai4animation.Import.GLBImporter import GLB
from ai4animation.Math import Transform, Vector3

glb = GLB.Create(str(ASSETS_DIR / "Model.glb"))
joint_name_to_idx = {n: i for i, n in enumerate(glb.JointNames)}

# Build parent indices for our bone set
parent_indices = []
tpose_transforms = []
for i, name in enumerate(bone_names):
    glb_idx = joint_name_to_idx.get(name, -1)
    if glb_idx >= 0:
        tpose_transforms.append(glb.JointMatrices[glb_idx].tolist())
    else:
        tpose_transforms.append(np.eye(4).tolist())

    # Find parent in our bone set
    if glb_idx < 0:
        parent_indices.append(i)
        continue
    parent_name = glb.JointParents[glb_idx]
    if parent_name in bone_names:
        parent_indices.append(bone_names.index(parent_name))
    else:
        # Walk up until we find a bone in our set
        found = False
        pn = parent_name
        while pn in joint_name_to_idx:
            pidx = joint_name_to_idx[pn]
            pn = glb.JointParents[pidx]
            if pn in bone_names:
                parent_indices.append(bone_names.index(pn))
                found = True
                break
        if not found:
            parent_indices.append(i)

# Guidance templates
guidances = {}
guidance_dir = BIPED_DIR / "Guidances"
for path in sorted(guidance_dir.iterdir()):
    if path.suffix == ".npz":
        with np.load(str(path), allow_pickle=True) as data:
            guidances[path.stem] = data["Positions"].tolist()

# Compute zero transforms and default lengths
bone_count = len(bone_names)
tpose_np = np.array(tpose_transforms)
zero_transforms = []
default_lengths = []
for i in range(bone_count):
    pi = parent_indices[i]
    if pi == i:
        zero_transforms.append(np.eye(4).tolist())
        default_lengths.append(0.0)
    else:
        try:
            zt = np.linalg.inv(tpose_np[pi]) @ tpose_np[i]
            zero_transforms.append(zt.tolist())
            pos = zt[:3, 3]
            default_lengths.append(float(np.linalg.norm(pos)))
        except np.linalg.LinAlgError:
            zero_transforms.append(np.eye(4).tolist())
            default_lengths.append(0.0)

# Build children map
children = [[] for _ in range(bone_count)]
for i in range(bone_count):
    pi = parent_indices[i]
    if pi != i:
        children[pi].append(i)

companion = {
    "boneNames": bone_names,
    "boneCount": bone_count,
    "parentIndices": parent_indices,
    "children": children,
    "tposeTransforms": tpose_transforms,
    "zeroTransforms": zero_transforms,
    "defaultLengths": default_lengths,
    "guidances": guidances,
    "sequenceLength": model.SequenceLength,
    "sequenceWindow": model.SequenceWindow,
    "inputDim": model.InputDim,
    "outputDim": model.OutputDim,
    "latentDim": model.LatentDim,
}

json_path = OUTPUT_DIR / "locomotion-data.json"
with open(json_path, "w") as f:
    json.dump(companion, f)
json_size = json_path.stat().st_size / 1024
print(f"[export] Companion JSON saved: {json_size:.1f} KB")
print(f"[export] Guidances: {list(guidances.keys())}")
print("[export] Done!")

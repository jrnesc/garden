"""
WebSocket locomotion server for ai4animationpy.
Runs the neural biped locomotion controller and streams bone transforms to the browser.
Click targets replace WASD input — the character walks to where you click.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import websockets

# Add ai4animation to path (avoid installing the full package which needs raylib)
REPO_ROOT = Path(__file__).parent.parent
AI4ANIM_ROOT = REPO_ROOT / "ai4animationpy"
sys.path.insert(0, str(AI4ANIM_ROOT))

# Import only the modules we need (no raylib dependency)
from ai4animation.Math import Quaternion, Rotation, Tensor, Transform, Vector3
from ai4animation.AI.FeedTensor import FeedTensor
from ai4animation.AI.ReadTensor import ReadTensor
from ai4animation.IK.FABRIK import FABRIK

# Paths
BIPED_DIR = AI4ANIM_ROOT / "Demos" / "Locomotion" / "Biped"
ASSETS_DIR = AI4ANIM_ROOT / "Demos" / "_ASSETS_" / "Geno"

sys.path.insert(0, str(ASSETS_DIR))
import Definitions

# Constants from Program.py
SEQUENCE_WINDOW = 0.5
SEQUENCE_LENGTH = 16
SEQUENCE_FPS = 30
PREDICTION_FPS = 10
CONTACT_POWER = 3.0
CONTACT_THRESHOLD = 2.0 / 3.0
MIN_TIMESCALE = 1.0
MAX_TIMESCALE = 1.5
SYNCHRONIZATION_SENSITIVITY = 5
TIMESCALE_SENSITIVITY = 5

BONE_NAMES = Definitions.FULL_BODY_NAMES
NUM_BONES = len(BONE_NAMES)

TARGET_FPS = 30
ARRIVE_THRESHOLD = 0.25  # stop walking when this close to target


def to_float(v):
    """Safely convert numpy scalar/array to Python float."""
    return float(np.asarray(v).flat[0])


# ── Minimal Actor ──
# Holds bone transforms and velocities without the full ECS framework.

class MinimalActor:
    """Lightweight Actor that holds bone state without Entity/Scene dependencies."""

    def __init__(self, glb_path: str, bone_names: list[str]):
        from ai4animation.Import.GLBImporter import GLB
        model = GLB.Create(str(glb_path))

        self.BoneNames = bone_names
        self.BoneCount = len(bone_names)

        # Build bone index map from GLB joint names
        joint_name_to_idx = {n: i for i, n in enumerate(model.JointNames)}
        joint_parents = model.JointParents
        joint_matrices = model.JointMatrices  # T-pose transforms

        # Initialize transforms from GLB T-pose
        self.Transforms = np.zeros((self.BoneCount, 4, 4), dtype=np.float64)
        self.Velocities = Vector3.Zero(self.BoneCount)
        self.Root = Transform.Identity()

        # Map our bone_names to GLB joint indices
        self._glb_indices = []
        for name in bone_names:
            idx = joint_name_to_idx.get(name, -1)
            self._glb_indices.append(idx)
            if idx >= 0:
                self.Transforms[bone_names.index(name)] = joint_matrices[idx]
            else:
                self.Transforms[bone_names.index(name)] = Transform.Identity()

        # Build parent map for our bones
        self.ParentIndex = np.zeros(self.BoneCount, dtype=int)
        for i, name in enumerate(bone_names):
            glb_idx = self._glb_indices[i]
            if glb_idx < 0:
                self.ParentIndex[i] = i
                continue
            parent_name = joint_parents[glb_idx]
            if parent_name in bone_names:
                self.ParentIndex[i] = bone_names.index(parent_name)
            else:
                # Walk up until we find a bone in our set
                found = False
                pn = parent_name
                while pn in joint_name_to_idx:
                    pidx = joint_name_to_idx[pn]
                    pn = joint_parents[pidx]
                    if pn in bone_names:
                        self.ParentIndex[i] = bone_names.index(pn)
                        found = True
                        break
                if not found:
                    self.ParentIndex[i] = i  # root

        # Compute zero-pose transforms (bone-to-parent)
        self.ZeroTransforms = np.zeros_like(self.Transforms)
        for i in range(self.BoneCount):
            pi = self.ParentIndex[i]
            if pi == i:
                self.ZeroTransforms[i] = Transform.Identity()
            else:
                try:
                    self.ZeroTransforms[i] = Transform.TransformationTo(
                        self.Transforms[i:i+1], self.Transforms[pi:pi+1]
                    )[0]
                except np.linalg.LinAlgError:
                    self.ZeroTransforms[i] = Transform.Identity()

        # Default bone lengths
        self.DefaultLengths = np.array([
            Vector3.Length(Transform.GetPosition(self.ZeroTransforms[i:i+1]))[0]
            for i in range(self.BoneCount)
        ])

        # Children map for alignment restoration
        self.Children = [[] for _ in range(self.BoneCount)]
        for i in range(self.BoneCount):
            pi = self.ParentIndex[i]
            if pi != i:
                self.Children[pi].append(i)

    def GetBoneNames(self): return self.BoneNames
    def GetBoneCount(self): return self.BoneCount
    def GetTransforms(self): return self.Transforms.copy()
    def GetPositions(self): return Transform.GetPosition(self.Transforms)
    def GetRotations(self): return Transform.GetRotation(self.Transforms)
    def GetVelocities(self): return self.Velocities.copy()
    def GetRootPosition(self): return Transform.GetPosition(self.Root)
    def GetRootDirection(self): return Transform.GetAxisZ(self.Root)

    def SetTransforms(self, values): self.Transforms[:] = values
    def SetVelocities(self, values): self.Velocities[:] = values

    def GetBoneIndex(self, name):
        return self.BoneNames.index(name) if name in self.BoneNames else -1

    def GetBoneIndices(self, names):
        return [self.GetBoneIndex(n) for n in names]

    def RestoreBoneLengths(self):
        parents = self.ParentIndex
        a = self.Transforms[parents]
        b = self.Transforms
        a_pos = Transform.GetPosition(a)
        b_pos = Transform.GetPosition(b)
        lengths = self.DefaultLengths.reshape(-1, 1)
        direction = b_pos - a_pos
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        d = a_pos + lengths * direction / norm
        # Don't move root bone
        mask = (parents != np.arange(self.BoneCount))
        Transform.SetPosition(self.Transforms, np.where(mask.reshape(-1, 1), d, b_pos))

    def RestoreBoneAlignments(self):
        for i in range(self.BoneCount):
            if len(self.Children[i]) == 1:
                try:
                    ci = self.Children[i][0]
                    bone_tf = self.Transforms[i:i+1]
                    child_pos = Transform.GetPosition(self.Transforms[ci:ci+1])[0]
                    zero_child_pos = Vector3.PositionFrom(
                        Transform.GetPosition(self.ZeroTransforms[ci:ci+1]),
                        bone_tf
                    )[0]
                    bone_pos = Transform.GetPosition(bone_tf)[0]
                    from_dir = zero_child_pos - bone_pos
                    to_dir = child_pos - bone_pos
                    from_norm = float(np.linalg.norm(from_dir))
                    to_norm = float(np.linalg.norm(to_dir))
                    if from_norm > 1e-8 and to_norm > 1e-8:
                        rot = Quaternion.ToMatrix(
                            Quaternion.FromTo(from_dir, to_dir)
                        )
                        cur_rot = Transform.GetRotation(bone_tf)[0]
                        self.Transforms[i, :3, :3] = (rot @ cur_rot).reshape(3, 3)
                except Exception:
                    pass  # alignment is polish, don't break the loop


# ── Minimal FABRIK wrapper ──

class MinimalBone:
    """Lightweight bone for FABRIK — wraps an index into the actor's transforms."""
    def __init__(self, actor, index):
        self.Actor = actor
        self.Index = index
        self.Parent = None
        self.Children = []

    def GetPosition(self):
        return Transform.GetPosition(self.Actor.Transforms[self.Index:self.Index+1])[0]

    def SetPosition(self, value):
        Transform.SetPosition(self.Actor.Transforms, value, self.Index)

    def GetRotation(self):
        return Transform.GetRotation(self.Actor.Transforms[self.Index:self.Index+1])[0]

    def SetRotation(self, value):
        self.Actor.Transforms[self.Index, :3, :3] = value.reshape(3, 3)

    def GetTransform(self):
        return Transform.GetTransform(self.Actor.Transforms, self.Index)


# ── Minimal TimeSeries ──

class MinimalTimeSeries:
    def __init__(self, start, end, sample_count):
        self.Start = start
        self.End = end
        self.SampleCount = sample_count
        self.Timestamps = Tensor.LinSpace(start, end, sample_count)
        self.DeltaTime = (end - start) / max(sample_count - 1, 1)


# ── Root Series ──

class RootSeries(MinimalTimeSeries):
    def __init__(self, base_series, transforms=None, velocities=None):
        super().__init__(base_series.Start, base_series.End, base_series.SampleCount)
        self.Transforms = Transform.Identity(self.SampleCount) if transforms is None else transforms
        self.Velocities = Vector3.Zero(self.SampleCount) if velocities is None else velocities

    def GetPosition(self, index):
        return Transform.GetPosition(self.Transforms, index)

    def SetPosition(self, value, index):
        Transform.SetPosition(self.Transforms, value, index)

    def GetDirection(self, index):
        return Transform.GetAxisZ(self.Transforms, index)

    def SetDirection(self, value, index):
        Transform.SetRotation(self.Transforms, Rotation.LookPlanar(value), index)

    def GetVelocity(self, index):
        return Vector3.GetVector(self.Velocities, index)

    def SetVelocity(self, value, index):
        Vector3.SetVector(self.Velocities, value, index)

    def GetLength(self):
        prev = Transform.GetPosition(self.Transforms)[:-1]
        nxt = Transform.GetPosition(self.Transforms)[1:]
        return float(np.sum(np.linalg.norm(nxt - prev, axis=-1)))

    def Control(self, position, direction, velocity, delta_time,
                move_sensitivity=10.0, turn_sensitivity=10.0):
        pivot = 0
        direction = Vector3.Normalize(direction)
        if Vector3.Length(direction) == 0.0:
            if Vector3.Length(velocity) != 0.0:
                direction = Vector3.Normalize(velocity)
            else:
                direction = self.GetDirection(pivot)

        self.SetVelocity(
            Vector3.LerpDt(self.GetVelocity(pivot), velocity, delta_time, move_sensitivity),
            pivot
        )
        self.SetPosition(position + self.GetVelocity(pivot) * delta_time, pivot)
        self.SetDirection(
            Vector3.SlerpDt(self.GetDirection(pivot), direction, delta_time, turn_sensitivity),
            pivot
        )

        for index in range(pivot + 1, self.SampleCount):
            ratio = index / max(self.SampleCount - 1, 1)
            self.SetVelocity(
                Vector3.LerpDt(
                    self.GetVelocity(index - 1), velocity,
                    self.DeltaTime, ratio * move_sensitivity
                ),
                index
            )
            self.SetPosition(
                self.GetPosition(index - 1) + self.GetVelocity(index) * self.DeltaTime,
                index
            )
            self.SetDirection(
                Vector3.Slerp(self.GetDirection(pivot), direction, ratio),
                index
            )


# ── Sequence (predicted future motion) ──

class Sequence:
    def __init__(self):
        self.Timestamps = None
        self.Trajectory = None
        self.MotionTransforms = None
        self.MotionVelocities = None
        self.Contacts = None
        self.Guidances = None

    def _get_index_pair(self, timestamp):
        if self.Timestamps is None:
            return 0, 0, 0.0
        ratio = np.interp(timestamp, [self.Timestamps[0], self.Timestamps[-1]], [0, len(self.Timestamps) - 1])
        ratio = np.clip(ratio, 0, len(self.Timestamps) - 1)
        a = int(np.floor(ratio))
        b = int(np.ceil(ratio))
        w = (ratio - a) if b != a else 0.0
        return a, b, float(w)

    def SampleRoot(self, ts):
        a, b, w = self._get_index_pair(ts)
        return Transform.Interpolate(self.Trajectory.Transforms[a], self.Trajectory.Transforms[b], w)

    def SamplePositions(self, ts):
        a, b, w = self._get_index_pair(ts)
        return Vector3.Lerp(
            Transform.GetPosition(self.MotionTransforms[a]),
            Transform.GetPosition(self.MotionTransforms[b]), w
        )

    def SampleRotations(self, ts):
        a, b, w = self._get_index_pair(ts)
        return Rotation.Interpolate(
            Transform.GetRotation(self.MotionTransforms[a]),
            Transform.GetRotation(self.MotionTransforms[b]), w
        )

    def SampleVelocities(self, ts):
        a, b, w = self._get_index_pair(ts)
        return Vector3.Lerp(self.MotionVelocities[a], self.MotionVelocities[b], w)

    def SampleContacts(self, ts):
        a, b, w = self._get_index_pair(ts)
        return Tensor.Interpolate(self.Contacts[a], self.Contacts[b], w)

    def GetRootLock(self):
        return 1.0 if float(np.mean(self.Contacts)) > 0.75 else 0.0

    def GetLength(self):
        return self.Trajectory.GetLength()


# ── Smooth Step (contact processing) ──

def smooth_step(x, threshold, power):
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x < threshold, 0.0, (x - threshold) / (1.0 - threshold))
    return np.power(y, power)


# ── LegIK ──

class LegIK:
    def __init__(self, actor, hip_name, knee_name, ankle_name, ball_name):
        self.Actor = actor
        self.HipIdx = actor.GetBoneIndex(hip_name)
        self.KneeIdx = actor.GetBoneIndex(knee_name)
        self.AnkleIdx = actor.GetBoneIndex(ankle_name)
        self.BallIdx = actor.GetBoneIndex(ball_name)

        ankle_pos = actor.GetPositions()[self.AnkleIdx]
        ball_pos = actor.GetPositions()[self.BallIdx]
        self.AnkleBaseline = float(ankle_pos[1])
        self.BallBaseline = float(ball_pos[1])
        self.AnkleBallDistance = float(np.linalg.norm(ankle_pos - ball_pos))
        self.AnkleTargetPosition = ankle_pos.copy()
        self.BallTargetPosition = ball_pos.copy()

    def Solve(self, ankle_contact, ball_contact):
        try:
            # Ankle
            w = float(ankle_contact)
            locked = self.AnkleTargetPosition.copy()
            current_ankle = self.Actor.GetPositions()[self.AnkleIdx]
            locked[1] = max(locked[1] * (1 - w) + self.AnkleBaseline * w, self.AnkleBaseline)
            self.AnkleTargetPosition = current_ankle * (1 - w) + locked * w
            self.Actor.Transforms[self.AnkleIdx, :3, 3] = self.AnkleTargetPosition[:3]

            # Ball
            w = float(ball_contact)
            locked = self.BallTargetPosition.copy()
            current_ball = self.Actor.GetPositions()[self.BallIdx]
            locked[1] = max(locked[1] * (1 - w) + self.BallBaseline * w, self.BallBaseline)
            self.BallTargetPosition = current_ball * (1 - w) + locked * w
            self.Actor.Transforms[self.BallIdx, :3, 3] = self.BallTargetPosition[:3]
        except Exception:
            pass


# ── Locomotion Controller ──

class LocomotionController:
    def __init__(self):
        print("[loco] Loading model...")
        self.Model = torch.load(str(BIPED_DIR / "Network.pt"), weights_only=False, map_location="cpu")
        self.Model.eval()

        print("[loco] Loading character...")
        self.Actor = MinimalActor(str(ASSETS_DIR / "Model.glb"), BONE_NAMES)

        # Control state
        self.ControlSeries = MinimalTimeSeries(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.SimulationObject = RootSeries(self.ControlSeries)
        self.RootControl = RootSeries(self.ControlSeries)

        # Load guidance templates
        self.GuidanceTemplates = {}
        guidance_dir = BIPED_DIR / "Guidances"
        for path in sorted(guidance_dir.iterdir()):
            if path.suffix == ".npz":
                with np.load(str(path), allow_pickle=True) as data:
                    self.GuidanceTemplates[path.stem] = data["Positions"].copy()
        print(f"[loco] Loaded {len(self.GuidanceTemplates)} guidance styles: {list(self.GuidanceTemplates.keys())}")

        self.GuidancePositions = self.GuidanceTemplates.get("Idle", self.Actor.GetPositions()).copy()

        # Prediction state
        self.Previous = None
        self.Sequence = None
        self.Synchronization = 0.0
        self.Timescale = 1.0
        self.TrajectoryCorrection = 0.05
        self.Timestamp = 0.0
        self.TotalTime = 0.0

        # IK
        self.LeftLegIK = LegIK(
            self.Actor,
            Definitions.LeftHipName, Definitions.LeftKneeName,
            Definitions.LeftAnkleName, Definitions.LeftBallName
        )
        self.RightLegIK = LegIK(
            self.Actor,
            Definitions.RightHipName, Definitions.RightKneeName,
            Definitions.RightAnkleName, Definitions.RightBallName
        )
        self.ContactIndices = self.Actor.GetBoneIndices([
            Definitions.LeftAnkleName, Definitions.LeftBallName,
            Definitions.RightAnkleName, Definitions.RightBallName,
        ])

        # Client-driven velocity (direction + speed)
        self.ClientVelocity = Vector3.Zero()
        self.ClientDirection = Vector3.Create(0, 0, 1)
        self.CurrentStyle = "Idle"

        print("[loco] Ready.")

    def SetMovement(self, velocity, direction):
        """Set movement from client — velocity and facing direction."""
        self.ClientVelocity = np.array(velocity, dtype=np.float64)
        self.ClientDirection = np.array(direction, dtype=np.float64)

    def SetIdle(self):
        self.ClientVelocity = Vector3.Zero()

    def Update(self, dt):
        """Run one frame of the locomotion loop."""
        self.TotalTime += dt

        # Control
        self._control(dt)

        # Predict (at PREDICTION_FPS)
        if self.Timestamp == 0.0 or self.TotalTime - self.Timestamp > 1.0 / PREDICTION_FPS:
            self.Timestamp = self.TotalTime
            self._predict()

        # Animate
        self._animate(dt)

    def _control(self, dt):
        """Apply client-provided velocity and direction."""
        root_pos = self.Actor.GetRootPosition()
        velocity = self.ClientVelocity
        direction = self.ClientDirection

        if Vector3.Length(direction) < 0.01:
            direction = self.Actor.GetRootDirection()

        # Update guidance style
        speed = float(np.linalg.norm(velocity))
        style = "Idle" if speed < 0.1 else self.CurrentStyle
        if style in self.GuidanceTemplates:
            self.GuidancePositions = self.GuidanceTemplates[style].copy()

        # Simulation
        position = Vector3.Lerp(
            self.SimulationObject.GetPosition(0),
            root_pos,
            self.Synchronization
        )
        self.SimulationObject.Control(position, direction, velocity, dt)

        # Trajectory correction
        if self.Sequence is not None:
            self.RootControl.Transforms = Transform.Interpolate(
                self.SimulationObject.Transforms,
                self.Sequence.Trajectory.Transforms,
                self.TrajectoryCorrection,
            )
            for i in range(self.RootControl.SampleCount):
                target = Transform.GetPosition(self.RootControl.Transforms)[i:]
                current = root_pos.reshape(-1, 3)
                time_vals = self.RootControl.Timestamps[i:].reshape(-1, 1)
                self.RootControl.Velocities[i] = Tensor.Sum(
                    target - current, axis=0, keepDim=False
                ) / np.maximum(Tensor.Sum(time_vals, axis=0, keepDim=False), 1e-6)
            self.RootControl.Velocities = Vector3.Lerp(
                self.RootControl.Velocities,
                self.Sequence.Trajectory.Velocities,
                self.TrajectoryCorrection,
            )

    def _predict(self):
        """Run the neural network to predict future motion."""
        inputs = FeedTensor("X", self.Model.InputDim)
        root = self.Actor.Root

        transforms = Transform.TransformationTo(self.Actor.GetTransforms(), root)
        velocities = Vector3.DirectionTo(self.Actor.GetVelocities(), root)
        inputs.Feed(Transform.GetPosition(transforms))
        inputs.Feed(Transform.GetAxisZ(transforms))
        inputs.Feed(Transform.GetAxisY(transforms))
        inputs.Feed(velocities)

        future_root_tf = Transform.TransformationTo(self.RootControl.Transforms, root)
        future_root_vel = Vector3.DirectionTo(self.RootControl.Velocities, root)
        inputs.FeedVector3(Transform.GetPosition(future_root_tf), x=True, y=False, z=True)
        inputs.FeedVector3(Transform.GetAxisZ(future_root_tf), x=True, y=False, z=True)
        inputs.FeedVector3(future_root_vel, x=True, y=False, z=True)

        inputs.Feed(self.GuidancePositions)

        with torch.no_grad():
            outputs, _, _, _ = self.Model(
                inputs.GetTensor().reshape(1, -1),
                noise=0.5 * torch.ones(1, self.Model.LatentDim),
                iterations=3,
                seed=torch.zeros(1, self.Model.LatentDim),
            )
        outputs = outputs.reshape(SEQUENCE_LENGTH, -1)
        outputs = ReadTensor("Y", Tensor.ToNumPy(outputs))

        # Unpack outputs
        future_root_vectors = outputs.ReadVector3()
        future_root_delta = Tensor.ZerosLike(future_root_vectors)
        for i in range(1, SEQUENCE_LENGTH):
            future_root_delta[i] = future_root_delta[i - 1] + future_root_vectors[i]
        future_root_transforms = Transform.TransformationFrom(
            Transform.DeltaXZ(future_root_delta), root
        )
        future_root_velocities = Tensor.ZerosLike(future_root_vectors)
        future_root_velocities[..., [0, 2]] = future_root_vectors[..., [0, 2]] * SEQUENCE_FPS
        future_root_velocities = Vector3.DirectionFrom(future_root_velocities, future_root_transforms)

        future_motion_transforms = Transform.TransformationFrom(
            Transform.TR(
                outputs.ReadVector3(self.Actor.GetBoneCount()),
                outputs.ReadRotation3D(self.Actor.GetBoneCount()),
            ),
            future_root_transforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )
        future_motion_velocities = Vector3.DirectionFrom(
            outputs.ReadVector3(self.Actor.GetBoneCount()),
            future_root_transforms.reshape(SEQUENCE_LENGTH, 1, 4, 4),
        )

        raw_contacts = outputs.Read(4)
        future_contacts = smooth_step(raw_contacts, CONTACT_THRESHOLD, CONTACT_POWER)
        future_guidances = outputs.ReadVector3(self.Actor.GetBoneCount())

        self.Previous = self.Sequence
        self.Sequence = Sequence()
        if self.Previous is None:
            self.Previous = self.Sequence
        self.Sequence.Timestamps = Tensor.LinSpace(0.0, SEQUENCE_WINDOW, SEQUENCE_LENGTH)
        self.Sequence.Trajectory = RootSeries(
            self.ControlSeries, future_root_transforms, future_root_velocities
        )
        self.Sequence.MotionTransforms = future_motion_transforms
        self.Sequence.MotionVelocities = future_motion_velocities
        self.Sequence.Contacts = future_contacts
        self.Sequence.Guidances = future_guidances

    def _animate(self, dt):
        """Interpolate and apply predicted motion."""
        if self.Sequence is None or self.Previous is None:
            return

        # Timescale synchronization
        req_speed = to_float(
            np.linalg.norm(self.Actor.GetRootPosition() - self.SimulationObject.GetPosition(0))
            + self.SimulationObject.GetLength()
        ) / SEQUENCE_WINDOW
        pred_speed = self.Sequence.GetLength() / SEQUENCE_WINDOW
        if req_speed > 0.1 and pred_speed > 0.1:
            ts = req_speed / pred_speed
            sync = 1.0
        else:
            ts = 1.0
            sync = 0.0
        self.Timescale = to_float(np.clip(
            Tensor.InterpolateDt(self.Timescale, ts, dt, TIMESCALE_SENSITIVITY),
            MIN_TIMESCALE, 1.15  # tighter cap than original 1.5
        ))
        self.Synchronization = to_float(
            Tensor.InterpolateDt(self.Synchronization, sync, dt, SYNCHRONIZATION_SENSITIVITY)
        )

        sdt = dt * self.Timescale
        blend = (self.TotalTime - self.Timestamp) * PREDICTION_FPS

        root = Transform.Interpolate(self.Previous.SampleRoot(sdt), self.Sequence.SampleRoot(sdt), blend)
        positions = Vector3.Lerp(self.Previous.SamplePositions(sdt), self.Sequence.SamplePositions(sdt), blend)
        rotations = Rotation.Interpolate(self.Previous.SampleRotations(sdt), self.Sequence.SampleRotations(sdt), blend)
        velocities = Vector3.Lerp(self.Previous.SampleVelocities(sdt), self.Sequence.SampleVelocities(sdt), blend)
        contacts = Tensor.Interpolate(self.Previous.SampleContacts(sdt), self.Sequence.SampleContacts(sdt), blend)

        self.Actor.Root = Transform.Interpolate(root, self.Actor.Root, self.Sequence.GetRootLock())
        self.Actor.SetTransforms(
            Transform.TR(
                Vector3.Lerp(self.Actor.GetPositions() + velocities * sdt, positions, 0.5),
                rotations,
            )
        )
        self.Actor.SetVelocities(velocities)

        self.Actor.RestoreBoneLengths()
        self.Actor.RestoreBoneAlignments()

        # IK
        self.LeftLegIK.Solve(float(contacts[0]), float(contacts[1]))
        self.RightLegIK.Solve(float(contacts[2]), float(contacts[3]))

        # Advance sequences
        self.Previous.Timestamps -= sdt
        self.Sequence.Timestamps -= sdt

    def GetBoneData(self):
        """Return WORLD-space bone transforms for Three.js.

        The client converts to local-space using its own skeleton hierarchy,
        avoiding any parent-chain mismatch between our 22-bone subset and
        the full 62-bone GLB skeleton.
        """
        bones = []
        root = self.Actor.Root

        for i, name in enumerate(self.Actor.BoneNames):
            world_tf = root @ self.Actor.Transforms[i]
            pos = Transform.GetPosition(world_tf.reshape(1, 4, 4))[0]
            rot = Transform.GetRotation(world_tf.reshape(1, 4, 4))[0]
            quat = Quaternion.FromMatrix(rot)
            bones.append({
                "name": name,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "quaternion": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            })

        root_pos = self.Actor.GetRootPosition()
        root_rot = Transform.GetRotation(root.reshape(1, 4, 4))[0]
        root_quat = Quaternion.FromMatrix(root_rot)
        return {
            "type": "frame",
            "root": [float(root_pos[0]), float(root_pos[1]), float(root_pos[2])],
            "rootQuat": [float(root_quat[0]), float(root_quat[1]), float(root_quat[2]), float(root_quat[3])],
            "bones": bones,
        }


# ── WebSocket Server ──

controller = None

async def handle_client(websocket):
    global controller
    print(f"[ws] Client connected")
    controller.SetIdle()
    controller.CurrentStyle = "Idle"
    try:
        loop_task = asyncio.create_task(animation_loop(websocket))

        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "move":
                vel = data.get("velocity", [0, 0, 0])
                dir_ = data.get("direction", [0, 0, 1])
                controller.SetMovement(vel, dir_)
            elif data.get("type") == "idle":
                controller.SetIdle()
            elif data.get("type") == "style":
                style = data.get("style", "Idle")
                if style in controller.GuidanceTemplates:
                    controller.CurrentStyle = style
                    print(f"[ws] Style: {style}")

        loop_task.cancel()
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("[ws] Client disconnected")


async def animation_loop(websocket):
    """Run the locomotion loop and stream bone data."""
    frame_time = 1.0 / TARGET_FPS
    last = time.time()

    while True:
        now = time.time()
        dt = min(now - last, 0.1)
        last = now

        if dt > 0:
            controller.Update(dt)

        try:
            data = controller.GetBoneData()
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            break

        elapsed = time.time() - now
        sleep_time = max(0, frame_time - elapsed)
        await asyncio.sleep(sleep_time)


async def main():
    global controller
    controller = LocomotionController()

    async with websockets.serve(handle_client, "localhost", 8765):
        print("[ws] Locomotion server running on ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())

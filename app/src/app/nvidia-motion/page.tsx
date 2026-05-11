"use client";

import type * as THREE from "three";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import LoadingParticles from "@/components/LoadingParticles";
import {
  HUD_BOX_BASE,
  HUD_BOX_SQUARE,
  HUD_FONT,
  IconArrowLeft,
  IconClose,
  IconKeyboard,
} from "@/components/hud-icons";

type NvidiaMotionProbe = {
  mode: string;
  fps: number;
  coordinateSystem: string;
  numPredFrames: number;
  qpos: number[][];
  live?: boolean;
  request?: {
    mode: string;
    movement: [number, number, number];
    facing: [number, number, number];
    seed: number;
  };
};

type MotionTarget = {
  position: [number, number];
  heading: number;
};

type G1Body = {
  name: string;
  parent: string | null;
  pos: [number, number, number];
  quat: [number, number, number, number];
  geoms: Array<{
    mesh: string;
    pos: [number, number, number];
    quat: [number, number, number, number];
    rgba: [number, number, number, number];
  }>;
  joints: Array<{
    name: string;
    axis: [number, number, number];
    qposIndex: number;
  }>;
};

type G1Kinematics = {
  coordinateSystem: string;
  qposDofs: number;
  meshBaseUrl: string;
  bodies: G1Body[];
};

const BODY_SEGMENTS: Array<[string, string]> = [
  ["pelvis", "left_hip_pitch_link"],
  ["left_hip_pitch_link", "left_hip_roll_link"],
  ["left_hip_roll_link", "left_hip_yaw_link"],
  ["left_hip_yaw_link", "left_knee_link"],
  ["left_knee_link", "left_ankle_pitch_link"],
  ["left_ankle_pitch_link", "left_ankle_roll_link"],
  ["pelvis", "right_hip_pitch_link"],
  ["right_hip_pitch_link", "right_hip_roll_link"],
  ["right_hip_roll_link", "right_hip_yaw_link"],
  ["right_hip_yaw_link", "right_knee_link"],
  ["right_knee_link", "right_ankle_pitch_link"],
  ["right_ankle_pitch_link", "right_ankle_roll_link"],
  ["pelvis", "waist_yaw_link"],
  ["waist_yaw_link", "waist_roll_link"],
  ["waist_roll_link", "torso_link"],
  ["torso_link", "left_shoulder_pitch_link"],
  ["left_shoulder_pitch_link", "left_shoulder_roll_link"],
  ["left_shoulder_roll_link", "left_shoulder_yaw_link"],
  ["left_shoulder_yaw_link", "left_elbow_link"],
  ["left_elbow_link", "left_wrist_roll_link"],
  ["left_wrist_roll_link", "left_wrist_pitch_link"],
  ["left_wrist_pitch_link", "left_wrist_yaw_link"],
  ["torso_link", "right_shoulder_pitch_link"],
  ["right_shoulder_pitch_link", "right_shoulder_roll_link"],
  ["right_shoulder_roll_link", "right_shoulder_yaw_link"],
  ["right_shoulder_yaw_link", "right_elbow_link"],
  ["right_elbow_link", "right_wrist_roll_link"],
  ["right_wrist_roll_link", "right_wrist_pitch_link"],
  ["right_wrist_pitch_link", "right_wrist_yaw_link"],
];

const NVIDIA_MOTION_MODES = [
  { label: "Walk", value: 1.0, clip: "walk" },
  { label: "Slow", value: 0.6, clip: "slow_walk" },
  { label: "Stealth", value: 0.7, clip: "walk_stealth" },
  { label: "Injured", value: 0.65, clip: "injured_walk" },
  { label: "Hand crawl", value: 0.45, clip: "hand_crawling" },
  { label: "Elbow crawl", value: 0.4, clip: "elbow_crawling" },
  { label: "Boxing", value: 1.0, clip: "walk_boxing" },
  { label: "Happy dance", value: 0.8, clip: "walk_happy_dance" },
  { label: "Zombie", value: 0.75, clip: "walk_zombie" },
  { label: "Gun", value: 0.75, clip: "walk_gun" },
  { label: "Scared", value: 0.75, clip: "walk_scared" },
];

function isMotionBricksProbe(probe: NvidiaMotionProbe) {
  return probe.live && probe.coordinateSystem === "motionbricks-mujoco-qpos";
}

function createLiveBootstrapProbe(): NvidiaMotionProbe {
  const qpos = Array.from({ length: 36 }, () => 0);
  qpos[2] = 0.755;
  qpos[3] = 1;
  return {
    mode: "idle",
    fps: 30,
    coordinateSystem: "motionbricks-mujoco-qpos",
    numPredFrames: 1,
    qpos: [qpos],
    live: true,
  };
}

function mujocoToThree(
  THREE: typeof import("three"),
  x: number,
  y: number,
  z: number,
) {
  return new THREE.Vector3(y, z, -x);
}

function wxyzToQuaternion(THREE: typeof import("three"), q: number[]) {
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]).normalize();
}

function multiplyWxyz(a: number[], b: number[]) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function yawWxyz(angle: number) {
  return [Math.cos(angle / 2), 0, 0, Math.sin(angle / 2)];
}

function normalizeAngle(angle: number) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function rootYawWxyz(q: number[]) {
  const [w, x, y, z] = q;
  const forwardX = 1 - 2 * (y * y + z * z);
  const forwardY = 2 * (x * y + w * z);
  return Math.atan2(forwardY, forwardX);
}

function transformControlledQpos(
  q: number[],
  origin: number[],
  rootX: number,
  rootY: number,
  yaw: number,
) {
  const next = q.slice();
  const dx = q[0] - origin[0];
  const dy = q[1] - origin[1];
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  next[0] = rootX + dx * c - dy * s;
  next[1] = rootY + dx * s + dy * c;
  const rootQuat = multiplyWxyz(yawWxyz(yaw), next.slice(3, 7));
  next[3] = rootQuat[0];
  next[4] = rootQuat[1];
  next[5] = rootQuat[2];
  next[6] = rootQuat[3];
  return next;
}

function matrixPositionToThree(
  THREE: typeof import("three"),
  m: THREE.Matrix4,
) {
  const e = m.elements;
  return mujocoToThree(THREE, e[12], e[13], e[14]);
}

function mujocoToThreeBasis(THREE: typeof import("three")) {
  return new THREE.Matrix4().makeBasis(
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
  );
}

function mujocoMatrixToThreeMatrix(
  THREE: typeof import("three"),
  m: THREE.Matrix4,
) {
  const basis = mujocoToThreeBasis(THREE);
  return basis.clone().multiply(m).multiply(basis.clone().invert());
}

function buildBodyPoints(
  THREE: typeof import("three"),
  kinematics: G1Kinematics,
  q: number[],
) {
  const matrices = new Map<string, THREE.Matrix4>();
  const points = new Map<string, THREE.Vector3>();
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3(1, 1, 1);

  for (const body of kinematics.bodies) {
    let local: THREE.Matrix4;
    if (body.parent == null) {
      pos.set(q[0], q[1], q[2]);
      quat.copy(wxyzToQuaternion(THREE, q.slice(3, 7)));
      local = new THREE.Matrix4().compose(pos, quat, scale);
    } else {
      pos.set(body.pos[0], body.pos[1], body.pos[2]);
      quat.copy(wxyzToQuaternion(THREE, body.quat));
      local = new THREE.Matrix4().compose(pos, quat, scale);
      for (const joint of body.joints) {
        const axis = new THREE.Vector3(
          joint.axis[0],
          joint.axis[1],
          joint.axis[2],
        ).normalize();
        const rot = new THREE.Matrix4().makeRotationAxis(
          axis,
          q[joint.qposIndex] ?? 0,
        );
        local.multiply(rot);
      }
    }

    const parent = body.parent ? matrices.get(body.parent) : null;
    const world = parent ? parent.clone().multiply(local) : local;
    matrices.set(body.name, world);
    points.set(body.name, matrixPositionToThree(THREE, world));
  }

  const torso = matrices.get("torso_link");
  if (torso) {
    const headM = torso
      .clone()
      .multiply(new THREE.Matrix4().makeTranslation(0.03, 0, 0.28));
    points.set("head", matrixPositionToThree(THREE, headM));
  }
  return points;
}

function buildBodyMatrices(
  THREE: typeof import("three"),
  kinematics: G1Kinematics,
  q: number[],
) {
  const matrices = new Map<string, THREE.Matrix4>();
  const pos = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3(1, 1, 1);

  for (const body of kinematics.bodies) {
    let local: THREE.Matrix4;
    if (body.parent == null) {
      pos.set(q[0], q[1], q[2]);
      quat.copy(wxyzToQuaternion(THREE, q.slice(3, 7)));
      local = new THREE.Matrix4().compose(pos, quat, scale);
    } else {
      pos.set(body.pos[0], body.pos[1], body.pos[2]);
      quat.copy(wxyzToQuaternion(THREE, body.quat));
      local = new THREE.Matrix4().compose(pos, quat, scale);
      for (const joint of body.joints) {
        const axis = new THREE.Vector3(
          joint.axis[0],
          joint.axis[1],
          joint.axis[2],
        ).normalize();
        local.multiply(
          new THREE.Matrix4().makeRotationAxis(axis, q[joint.qposIndex] ?? 0),
        );
      }
    }
    const parent = body.parent ? matrices.get(body.parent) : null;
    matrices.set(body.name, parent ? parent.clone().multiply(local) : local);
  }

  return matrices;
}

export default function NvidiaMotionPage() {
  const mountRef = useRef<HTMLDivElement>(null);
  const probeRef = useRef<NvidiaMotionProbe | null>(null);
  const playingRef = useRef(true);
  const speedRef = useRef(1);
  const scrubFrameRef = useRef<number | null>(null);
  const showPathRef = useRef(true);
  const showMeshRef = useRef(true);
  const showSkeletonRef = useRef(true);
  const playheadRef = useRef(0);
  const liveModeRef = useRef("walk");
  const walkSpeedRef = useRef(1.0);
  const controlledRootRef = useRef({ x: 0, y: 0, yaw: 0, initialized: false });
  const qposContextRef = useRef<number[][]>([]);
  const streamStartedRef = useRef(false);
  const [probe, setProbe] = useState<NvidiaMotionProbe | null>(null);
  const [kinematics, setKinematics] = useState<G1Kinematics | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showPath, setShowPath] = useState(true);
  const [showMesh, setShowMesh] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [frame, setFrame] = useState(0);
  const [liveMode, setLiveMode] = useState("walk");
  const [activeMotionMode, setActiveMotionMode] = useState("Walk");
  const [menuOpen, setMenuOpen] = useState(false);
  const [plannerStatus, setPlannerStatus] = useState("connecting");
  const hasProbe = probe != null;

  const toggleMenu = useCallback(() => setMenuOpen((open) => !open), []);

  useEffect(() => {
    probeRef.current = probe;
  }, [probe]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  useEffect(() => {
    liveModeRef.current = liveMode;
  }, [liveMode]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    showPathRef.current = showPath;
  }, [showPath]);

  useEffect(() => {
    showMeshRef.current = showMesh;
  }, [showMesh]);

  useEffect(() => {
    showSkeletonRef.current = showSkeleton;
  }, [showSkeleton]);

  useEffect(() => {
    document.body.classList.add("no-grain");
    return () => document.body.classList.remove("no-grain");
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/nvidia-motion/g1_kinematics.json")
      .then((res) => res.json())
      .then((data: G1Kinematics) => {
        if (cancelled) return;
        setKinematics(data);
        setProbe(createLiveBootstrapProbe());
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pickMotionMode = useCallback(
    (label: string, value: number, clip: string) => {
      walkSpeedRef.current = value;
      liveModeRef.current = clip;
      setActiveMotionMode(label);
      setLiveMode(clip);
    },
    [],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Escape") {
        event.preventDefault();
        toggleMenu();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleMenu]);

  useEffect(() => {
    const mount = mountRef.current;
    const initialProbe = probeRef.current;
    if (!mount || !initialProbe || !kinematics) return;

    let disposed = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      const THREE = await import("three");
      if (disposed) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#090a0c");
      const camera = new THREE.PerspectiveCamera(
        50,
        mount.clientWidth / mount.clientHeight,
        0.01,
        100,
      );
      camera.position.set(1.6, 1.25, 3.0);
      camera.lookAt(0, 0.8, -0.8);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      mount.appendChild(renderer.domElement);

      scene.add(new THREE.HemisphereLight(0xf5f5ff, 0x222222, 1.5));
      const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
      keyLight.position.set(2, 4, 3);
      scene.add(keyLight);
      const grid = new THREE.GridHelper(6, 24, 0x3b4655, 0x1d2630);
      scene.add(grid);

      const pathGeom = new THREE.BufferGeometry();
      const path = new THREE.Line(
        pathGeom,
        new THREE.LineBasicMaterial({
          color: 0x7dd3fc,
          transparent: true,
          opacity: 0.7,
        }),
      );
      scene.add(path);

      const debugSegments = [
        ...BODY_SEGMENTS,
        ["torso_link", "head"] as [string, string],
      ];
      const positions = new Float32Array(debugSegments.length * 2 * 3);
      const skeletonGeom = new THREE.BufferGeometry();
      skeletonGeom.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      const skeleton = new THREE.LineSegments(
        skeletonGeom,
        new THREE.LineBasicMaterial({ color: 0xf8fafc, linewidth: 2 }),
      );
      scene.add(skeleton);

      const rootMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xfacc15 }),
      );
      scene.add(rootMarker);

      const targetMarker = new THREE.Group();
      const targetRingMaterial = new THREE.MeshBasicMaterial({
        color: 0x22c55e,
        transparent: true,
        opacity: 0.9,
      });
      const targetDotMaterial = new THREE.MeshBasicMaterial({
        color: 0xbbf7d0,
      });
      const targetRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.16, 0.008, 8, 36),
        targetRingMaterial,
      );
      targetRing.rotation.x = Math.PI / 2;
      targetMarker.add(targetRing);
      const targetDot = new THREE.Mesh(
        new THREE.SphereGeometry(0.028, 16, 16),
        targetDotMaterial,
      );
      targetDot.position.y = 0.025;
      targetMarker.add(targetDot);
      targetMarker.visible = false;
      scene.add(targetMarker);

      type ArtifactState = "ground" | "pickup" | "carried" | "settle";
      type ArtifactItem = {
        id: string;
        group: THREE.Group;
        coreMaterial: THREE.MeshStandardMaterial;
        halo: THREE.Mesh;
        haloMaterial: THREE.MeshBasicMaterial;
        focusRing: THREE.Mesh;
        focusRingMaterial: THREE.MeshBasicMaterial;
        groundPosition: THREE.Vector3;
        state: ArtifactState;
        pickupStartedAt: number;
        pickupStartMatrix: THREE.Matrix4;
        settleStartedAt: number;
        settleStartMatrix: THREE.Matrix4;
        settleTargetPosition: THREE.Vector3;
        settleTargetQuaternion: THREE.Quaternion;
      };
      const artifactCoreGeometry = new THREE.BoxGeometry(0.13, 0.13, 0.13);
      const artifactHaloGeometry = new THREE.TorusGeometry(0.11, 0.007, 8, 36);
      const artifactFocusGeometry = new THREE.TorusGeometry(0.16, 0.005, 8, 44);
      const artifactCapGeometry = new THREE.SphereGeometry(0.035, 18, 18);
      const artifacts: ArtifactItem[] = [];
      const artifactById = new Map<string, ArtifactItem>();
      const createArtifact = (
        id: string,
        position: [number, number, number],
        color: number,
        accent: number,
        rotationY: number,
      ) => {
        const group = new THREE.Group();
        const coreMaterial = new THREE.MeshStandardMaterial({
          color,
          emissive: accent,
          emissiveIntensity: 0.12,
          roughness: 0.28,
          metalness: 0.22,
        });
        const core = new THREE.Mesh(artifactCoreGeometry, coreMaterial);
        group.add(core);
        const haloMaterial = new THREE.MeshBasicMaterial({
          color: accent,
          transparent: true,
          opacity: 0.9,
        });
        const halo = new THREE.Mesh(artifactHaloGeometry, haloMaterial);
        halo.rotation.x = Math.PI / 2;
        group.add(halo);
        const focusRingMaterial = new THREE.MeshBasicMaterial({
          color: 0xfacc15,
          transparent: true,
          opacity: 0,
        });
        const focusRing = new THREE.Mesh(
          artifactFocusGeometry,
          focusRingMaterial,
        );
        focusRing.rotation.x = Math.PI / 2;
        group.add(focusRing);
        const cap = new THREE.Mesh(
          artifactCapGeometry,
          new THREE.MeshBasicMaterial({ color: 0xf8fafc }),
        );
        cap.position.y = 0.078;
        group.add(cap);
        group.position.set(...position);
        group.rotation.set(0, rotationY, 0);
        scene.add(group);

        const item: ArtifactItem = {
          id,
          group,
          coreMaterial,
          halo,
          haloMaterial,
          focusRing,
          focusRingMaterial,
          groundPosition: group.position.clone(),
          state: "ground",
          pickupStartedAt: 0,
          pickupStartMatrix: new THREE.Matrix4(),
          settleStartedAt: 0,
          settleStartMatrix: new THREE.Matrix4(),
          settleTargetPosition: new THREE.Vector3(),
          settleTargetQuaternion: new THREE.Quaternion(),
        };
        artifacts.push(item);
        artifactById.set(id, item);
      };
      createArtifact(
        "blue",
        [1.15, 0.12, -1.65],
        0x93c5fd,
        0x38bdf8,
        Math.PI * 0.18,
      );
      createArtifact(
        "amber",
        [-0.95, 0.12, -1.15],
        0xfbbf24,
        0xf59e0b,
        -Math.PI * 0.12,
      );
      createArtifact(
        "rose",
        [0.4, 0.12, 1.25],
        0xfda4af,
        0xfb7185,
        Math.PI * 0.34,
      );

      const meshRoot = new THREE.Group();
      scene.add(meshRoot);
      const meshGroups = new Map<string, THREE.Group>();
      const keys: Record<string, boolean> = {};
      const moveDir = new THREE.Vector3();
      const sendVector = new THREE.Vector3();
      const cameraTarget = new THREE.Vector3(0, 0.8, 0);
      const currentRootThree = new THREE.Vector3();
      const currentRootMujoco = { x: 0, y: 0, yaw: 0 };
      let activeTarget: MotionTarget | null = null;
      let activeTargetKind: "ground" | "artifact" | "drop" | null = null;
      let activeTargetArtifactId: string | null = null;
      let activeTargetDistance = Infinity;
      let hoveredArtifactId: string | null = null;
      let selectedArtifactId: string | null = null;
      let carriedArtifactId: string | null = null;
      const pickupStartPosition = new THREE.Vector3();
      const pickupStartQuaternion = new THREE.Quaternion();
      const pickupTargetPosition = new THREE.Vector3();
      const pickupTargetQuaternion = new THREE.Quaternion();
      const pickupScale = new THREE.Vector3();
      const carriedArtifactMatrix = new THREE.Matrix4();
      const settleStartPosition = new THREE.Vector3();
      const settleStartQuaternion = new THREE.Quaternion();
      const settlePosition = new THREE.Vector3();
      const settleScale = new THREE.Vector3(1, 1, 1);
      let lastSentTargetKey = "";
      let lastSentMode = "";
      let lastSentMovement: [number, number, number] = [999, 999, 999];
      let targetMarkerBaseScale = 1;
      const targetKey = (target: MotionTarget | null) =>
        target
          ? `${target.position[0].toFixed(2)},${target.position[1].toFixed(2)},${target.heading.toFixed(2)}`
          : "";
      const isInteractionTarget = () =>
        activeTargetKind === "artifact" || activeTargetKind === "drop";
      const interactionApproachBlend = () => {
        if (!isInteractionTarget()) return 1;
        return Math.max(0.34, Math.min(1, activeTargetDistance / 0.95));
      };
      const targetMode = () =>
        isInteractionTarget() && activeTargetDistance < 1.15
          ? "slow_walk"
          : liveModeRef.current;
      const setTargetMarkerKind = (kind: "ground" | "artifact" | "drop") => {
        if (kind === "artifact") {
          targetRingMaterial.color.setHex(0x38bdf8);
          targetDotMaterial.color.setHex(0xe0f2fe);
          targetMarkerBaseScale = 0.86;
          targetMarker.scale.setScalar(targetMarkerBaseScale);
          return;
        }
        if (kind === "drop") {
          targetRingMaterial.color.setHex(0xf59e0b);
          targetDotMaterial.color.setHex(0xfef3c7);
          targetMarkerBaseScale = 1.08;
          targetMarker.scale.setScalar(targetMarkerBaseScale);
          return;
        }
        targetRingMaterial.color.setHex(0x22c55e);
        targetDotMaterial.color.setHex(0xbbf7d0);
        targetMarkerBaseScale = 1;
        targetMarker.scale.setScalar(targetMarkerBaseScale);
      };
      const sendControl = (
        movement: [number, number, number],
        mode: string,
        target: MotionTarget | null = activeTarget,
      ) => {
        const nextTargetKey = targetKey(target);
        const changed =
          mode !== lastSentMode ||
          nextTargetKey !== lastSentTargetKey ||
          Math.abs(movement[0] - lastSentMovement[0]) > 0.08 ||
          Math.abs(movement[1] - lastSentMovement[1]) > 0.08 ||
          Math.abs(movement[2] - lastSentMovement[2]) > 0.08;
        if (!changed) return;
        lastSentMode = mode;
        lastSentMovement = movement;
        lastSentTargetKey = nextTargetKey;
        void fetch("/api/motionbricks/control", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ mode, movement, target }),
        }).catch((error) => {
          setPlannerStatus(
            error instanceof Error ? error.message : "control failed",
          );
        });
      };
      const readMovement = (): [number, number, number] => {
        sendVector.set(0, 0, 0);
        const worldFwdX = -Math.sin(camYaw);
        const worldFwdZ = -Math.cos(camYaw);
        const worldRightX = Math.cos(camYaw);
        const worldRightZ = -Math.sin(camYaw);
        if (keys.KeyW || keys.ArrowUp) {
          sendVector.x += worldFwdX;
          sendVector.z += worldFwdZ;
        }
        if (keys.KeyS || keys.ArrowDown) {
          sendVector.x -= worldFwdX;
          sendVector.z -= worldFwdZ;
        }
        if (keys.KeyA || keys.ArrowLeft) {
          sendVector.x -= worldRightX;
          sendVector.z -= worldRightZ;
        }
        if (keys.KeyD || keys.ArrowRight) {
          sendVector.x += worldRightX;
          sendVector.z += worldRightZ;
        }
        if (sendVector.lengthSq() <= 0.01) return [0, 0, 0];
        sendVector.normalize();
        return [-sendVector.z, sendVector.x, 0];
      };
      const sendCurrentControl = () => {
        const manualMovement = readMovement();
        if (manualMovement[0] !== 0 || manualMovement[1] !== 0) {
          if (activeTarget) {
            activeTarget = null;
            activeTargetKind = null;
            activeTargetArtifactId = null;
            activeTargetDistance = Infinity;
            selectedArtifactId = carriedArtifactId;
            targetMarker.visible = false;
          }
          sendControl(manualMovement, liveModeRef.current, null);
          return;
        }
        if (activeTarget) {
          const dx = activeTarget.position[0] - currentRootMujoco.x;
          const dy = activeTarget.position[1] - currentRootMujoco.y;
          const distance = Math.hypot(dx, dy);
          activeTargetDistance = distance;
          if (distance < 0.28) {
            const targetArtifact = activeTargetArtifactId
              ? artifactById.get(activeTargetArtifactId)
              : null;
            if (
              activeTargetKind === "artifact" &&
              targetArtifact?.state === "ground"
            ) {
              targetArtifact.state = "pickup";
              targetArtifact.pickupStartedAt = performance.now();
              targetArtifact.group.updateMatrixWorld(true);
              targetArtifact.pickupStartMatrix.copy(
                targetArtifact.group.matrixWorld,
              );
              targetArtifact.group.matrixAutoUpdate = false;
              selectedArtifactId = targetArtifact.id;
            }
            if (
              activeTargetKind === "drop" &&
              targetArtifact?.state === "carried"
            ) {
              targetArtifact.state = "settle";
              targetArtifact.settleStartedAt = performance.now();
              hoveredArtifactId = null;
              renderer.domElement.style.cursor = "";
              targetArtifact.group.updateMatrixWorld(true);
              targetArtifact.settleStartMatrix.copy(
                targetArtifact.group.matrixWorld,
              );
              targetArtifact.group.matrixAutoUpdate = false;
              targetArtifact.settleTargetPosition.copy(
                targetArtifact.groundPosition,
              );
              targetArtifact.settleTargetQuaternion.setFromEuler(
                new THREE.Euler(0, currentRootMujoco.yaw + Math.PI * 0.18, 0),
              );
              carriedArtifactId = null;
              selectedArtifactId = targetArtifact.id;
            }
            activeTarget = null;
            activeTargetKind = null;
            activeTargetArtifactId = null;
            activeTargetDistance = Infinity;
            targetMarker.visible = false;
            sendControl([0, 0, 0], "idle", null);
            return;
          }
          const approachBlend = interactionApproachBlend();
          const movement: [number, number, number] = [
            (dx / distance) * approachBlend,
            (dy / distance) * approachBlend,
            0,
          ];
          activeTarget.heading = Math.atan2(dy, dx);
          sendControl(movement, targetMode(), activeTarget);
          return;
        }
        activeTargetDistance = Infinity;
        sendControl([0, 0, 0], "idle", null);
      };
      const isTypingTarget = (target: EventTarget | null) =>
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable);
      const onKeyDown = (event: KeyboardEvent) => {
        if (isTypingTarget(event.target)) return;
        keys[event.code] = true;
        if (
          [
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD",
            "ArrowUp",
            "ArrowLeft",
            "ArrowDown",
            "ArrowRight",
          ].includes(event.code)
        ) {
          sendCurrentControl();
        }
      };
      const onKeyUp = (event: KeyboardEvent) => {
        keys[event.code] = false;
        if (
          [
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD",
            "ArrowUp",
            "ArrowLeft",
            "ArrowDown",
            "ArrowRight",
          ].includes(event.code)
        ) {
          sendCurrentControl();
        }
      };
      window.addEventListener("keydown", onKeyDown);
      window.addEventListener("keyup", onKeyUp);

      let dragging = false;
      let dragStarted = false;
      let pointerDownX = 0;
      let pointerDownY = 0;
      let camYaw = 0;
      let camPitch = 0.36;
      let camDist = 4.1;
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const groundHit = new THREE.Vector3();
      const updatePointerRay = (event: MouseEvent) => {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
        raycaster.setFromCamera(pointer, camera);
      };
      const findArtifactForObject = (object: THREE.Object3D) => {
        let cursor: THREE.Object3D | null = object;
        while (cursor) {
          const item = artifacts.find(
            (candidate) => candidate.group === cursor,
          );
          if (item) return item;
          cursor = cursor.parent;
        }
        return null;
      };
      const groundArtifactGroups = () =>
        artifacts
          .filter((item) => item.state === "ground")
          .map((item) => item.group);
      const pickGroundArtifact = () => {
        const hits = raycaster.intersectObjects(groundArtifactGroups(), true);
        return hits.length > 0 ? findArtifactForObject(hits[0].object) : null;
      };
      const updateArtifactHover = (event: MouseEvent) => {
        updatePointerRay(event);
        const interactionBusy = artifacts.some(
          (item) => item.state === "pickup" || item.state === "settle",
        );
        hoveredArtifactId = interactionBusy
          ? null
          : (pickGroundArtifact()?.id ?? null);
        renderer.domElement.style.cursor = hoveredArtifactId
          ? "pointer"
          : carriedArtifactId
            ? "copy"
            : "";
      };
      const setTargetFromPointer = (event: MouseEvent) => {
        updatePointerRay(event);
        const carriedArtifact = carriedArtifactId
          ? artifactById.get(carriedArtifactId)
          : null;
        if (carriedArtifact?.state === "carried") {
          if (!raycaster.ray.intersectPlane(groundPlane, groundHit)) return;
          carriedArtifact.groundPosition.copy(groundHit);
          carriedArtifact.groundPosition.y = 0.12;
          activeTarget = {
            position: [-groundHit.z, groundHit.x],
            heading: currentRootMujoco.yaw,
          };
          activeTargetKind = "drop";
          activeTargetDistance = Infinity;
          activeTargetArtifactId = carriedArtifact.id;
          selectedArtifactId = carriedArtifact.id;
          setTargetMarkerKind("drop");
          targetMarker.position.copy(groundHit);
          targetMarker.position.y = 0.02;
          targetMarker.visible = true;
          sendCurrentControl();
          return;
        }
        if (
          artifacts.some(
            (item) => item.state === "pickup" || item.state === "settle",
          )
        )
          return;
        const targetArtifact = pickGroundArtifact();
        if (targetArtifact) {
          activeTarget = {
            position: [
              -targetArtifact.group.position.z,
              targetArtifact.group.position.x,
            ],
            heading: Math.atan2(
              targetArtifact.group.position.x - currentRootMujoco.y,
              -targetArtifact.group.position.z - currentRootMujoco.x,
            ),
          };
          activeTargetKind = "artifact";
          activeTargetDistance = Infinity;
          activeTargetArtifactId = targetArtifact.id;
          selectedArtifactId = targetArtifact.id;
          setTargetMarkerKind("artifact");
          targetMarker.position.set(
            targetArtifact.group.position.x,
            0.02,
            targetArtifact.group.position.z,
          );
          targetMarker.visible = true;
          sendCurrentControl();
          return;
        }
        if (!raycaster.ray.intersectPlane(groundPlane, groundHit)) return;
        activeTarget = {
          position: [-groundHit.z, groundHit.x],
          heading: currentRootMujoco.yaw,
        };
        activeTargetKind = "ground";
        activeTargetArtifactId = null;
        activeTargetDistance = Infinity;
        selectedArtifactId = null;
        setTargetMarkerKind("ground");
        targetMarker.position.copy(groundHit);
        targetMarker.position.y = 0.02;
        targetMarker.visible = true;
        sendCurrentControl();
      };
      const onMouseDown = (event: MouseEvent) => {
        if (event.button !== 0) return;
        dragging = true;
        dragStarted = false;
        pointerDownX = event.clientX;
        pointerDownY = event.clientY;
      };
      const onMouseUp = (event: MouseEvent) => {
        if (event.button !== 0) return;
        const wasClick =
          !dragStarted &&
          Math.hypot(
            event.clientX - pointerDownX,
            event.clientY - pointerDownY,
          ) < 4;
        dragging = false;
        if (wasClick) setTargetFromPointer(event);
      };
      const onMouseMove = (event: MouseEvent) => {
        if (!dragging) {
          updateArtifactHover(event);
          return;
        }
        if (
          Math.hypot(
            event.clientX - pointerDownX,
            event.clientY - pointerDownY,
          ) > 4
        ) {
          dragStarted = true;
        }
        camYaw -= event.movementX * 0.008;
        camPitch += event.movementY * 0.008;
        camPitch = Math.max(0.1, Math.min(1.2, camPitch));
      };
      const onWheel = (event: WheelEvent) => {
        event.preventDefault();
        camDist *= 1 + event.deltaY * 0.001;
        camDist = Math.max(1.4, Math.min(10, camDist));
      };
      renderer.domElement.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mouseup", onMouseUp);
      window.addEventListener("mousemove", onMouseMove);
      renderer.domElement.addEventListener("wheel", onWheel, {
        passive: false,
      });

      const { STLLoader } =
        await import("three/examples/jsm/loaders/STLLoader.js");
      const loader = new STLLoader();
      const geomCache = new Map<string, THREE.BufferGeometry>();
      const materialCache = new Map<string, THREE.MeshStandardMaterial>();
      const getMaterial = (rgba: [number, number, number, number]) => {
        const key = rgba.join(",");
        const cached = materialCache.get(key);
        if (cached) return cached;
        const isDarkPart = rgba[0] < 0.35 && rgba[1] < 0.35 && rgba[2] < 0.35;
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(isDarkPart ? 0x39070a : 0xd91f2d),
          roughness: isDarkPart ? 0.64 : 0.48,
          metalness: isDarkPart ? 0.14 : 0.22,
          transparent: rgba[3] < 1,
          opacity: rgba[3],
        });
        materialCache.set(key, mat);
        return mat;
      };
      const geometryBasis = mujocoToThreeBasis(THREE);

      await Promise.all(
        kinematics.bodies.map(async (body) => {
          if (body.geoms.length === 0) return;
          const group = new THREE.Group();
          group.matrixAutoUpdate = false;
          meshGroups.set(body.name, group);
          meshRoot.add(group);

          for (const geom of body.geoms) {
            let geometry = geomCache.get(geom.mesh);
            if (!geometry) {
              try {
                geometry = await loader.loadAsync(
                  `${kinematics.meshBaseUrl}${geom.mesh}.STL`,
                );
              } catch {
                console.warn(`[nvidia-motion] missing G1 mesh: ${geom.mesh}.STL`);
                continue;
              }
              geometry.applyMatrix4(geometryBasis);
              geometry.computeVertexNormals();
              geomCache.set(geom.mesh, geometry);
            }
            const mesh = new THREE.Mesh(geometry, getMaterial(geom.rgba));
            const local = new THREE.Matrix4().compose(
              new THREE.Vector3(geom.pos[0], geom.pos[1], geom.pos[2]),
              wxyzToQuaternion(THREE, geom.quat),
              new THREE.Vector3(1, 1, 1),
            );
            mesh.matrixAutoUpdate = false;
            mesh.matrix.copy(mujocoMatrixToThreeMatrix(THREE, local));
            group.add(mesh);
          }
        }),
      );

      const onResize = () => {
        camera.aspect = mount.clientWidth / mount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mount.clientWidth, mount.clientHeight);
      };
      window.addEventListener("resize", onResize);

      let last = performance.now();
      let lastProbe = initialProbe;
      let lastFrameUiUpdate = 0;
      let lastPathUpdate = 0;
      let lastPathFrameCount = 0;
      let lastControlUpdate = 0;
      const liveRootOffset = { x: 0, y: 0 };
      const liveYawOffset = { value: 0 };
      const controlledRoot = controlledRootRef.current;
      if (!controlledRoot.initialized) {
        controlledRoot.x = initialProbe.qpos[0]?.[0] ?? 0;
        controlledRoot.y = initialProbe.qpos[0]?.[1] ?? 0;
        controlledRoot.yaw = 0;
        controlledRoot.initialized = true;
      }
      const updatePath = (nextProbe: NvidiaMotionProbe) => {
        const origin = nextProbe.qpos[0];
        const start = isMotionBricksProbe(nextProbe)
          ? Math.max(0, nextProbe.qpos.length - 140)
          : 0;
        const pathPoints = nextProbe.qpos.slice(start).map((q) => {
          const controlled = isMotionBricksProbe(nextProbe)
            ? q
            : transformControlledQpos(
                q,
                origin,
                controlledRoot.x,
                controlledRoot.y,
                controlledRoot.yaw,
              );
          return mujocoToThree(
            THREE,
            controlled[0],
            controlled[1],
            controlled[2] - 0.73,
          );
        });
        const pathPositions = new Float32Array(pathPoints.length * 3);
        pathPoints.forEach((point, index) => {
          pathPositions[index * 3] = point.x;
          pathPositions[index * 3 + 1] = point.y;
          pathPositions[index * 3 + 2] = point.z;
        });
        pathGeom.setAttribute(
          "position",
          new THREE.BufferAttribute(pathPositions, 3),
        );
        pathGeom.computeBoundingSphere();
      };
      let eventSource: EventSource | null = null;
      if (!streamStartedRef.current) {
        streamStartedRef.current = true;
        eventSource = new EventSource("/api/motionbricks/stream");
        eventSource.addEventListener("ready", () =>
          setPlannerStatus("streaming"),
        );
        eventSource.onmessage = (event) => {
          if (disposed) return;
          const frameData = JSON.parse(event.data) as {
            mode: string;
            fps: number;
            qpos: number[];
          };
          const incomingQpos = frameData.qpos.slice();
          const current = probeRef.current;
          if (current && isMotionBricksProbe(current)) {
            const lastQpos = current.qpos[current.qpos.length - 1];
            if (lastQpos) {
              const displayX = incomingQpos[0] + liveRootOffset.x;
              const displayY = incomingQpos[1] + liveRootOffset.y;
              const dx = displayX - lastQpos[0];
              const dy = displayY - lastQpos[1];
              const jump = Math.hypot(dx, dy);
              const displayYaw = normalizeAngle(
                rootYawWxyz(incomingQpos.slice(3, 7)) + liveYawOffset.value,
              );
              const lastYaw = rootYawWxyz(lastQpos.slice(3, 7));
              const yawJump = normalizeAngle(displayYaw - lastYaw);
              if (jump > 0.35) {
                liveRootOffset.x += lastQpos[0] - displayX;
                liveRootOffset.y += lastQpos[1] - displayY;
              }
              if (Math.abs(yawJump) > Math.PI * 0.55) {
                liveYawOffset.value = normalizeAngle(
                  liveYawOffset.value - yawJump,
                );
              }
            }
            incomingQpos[0] += liveRootOffset.x;
            incomingQpos[1] += liveRootOffset.y;
            if (Math.abs(liveYawOffset.value) > 1e-4) {
              const rootQuat = multiplyWxyz(
                yawWxyz(liveYawOffset.value),
                incomingQpos.slice(3, 7),
              );
              incomingQpos[3] = rootQuat[0];
              incomingQpos[4] = rootQuat[1];
              incomingQpos[5] = rootQuat[2];
              incomingQpos[6] = rootQuat[3];
            }
            current.qpos.push(incomingQpos);
            if (current.qpos.length > 360) {
              const trim = current.qpos.length - 360;
              current.qpos.splice(0, trim);
              playheadRef.current = Math.max(0, playheadRef.current - trim);
            }
            current.mode = frameData.mode;
            current.fps = frameData.fps;
            current.numPredFrames = current.qpos.length;
          } else {
            const liveProbe: NvidiaMotionProbe = {
              mode: frameData.mode,
              fps: frameData.fps,
              coordinateSystem: "motionbricks-mujoco-qpos",
              numPredFrames: 1,
              qpos: [incomingQpos],
              live: true,
            };
            probeRef.current = liveProbe;
            playheadRef.current = 0;
            scrubFrameRef.current = null;
            setFrame(0);
            setProbe(liveProbe);
            updatePath(liveProbe);
          }
        };
        eventSource.onerror = () => setPlannerStatus("stream error");
      }
      updatePath(initialProbe);

      const updateSkeleton = (frameIndex: number) => {
        const currentProbe = probeRef.current;
        if (!currentProbe) return;
        const origin = currentProbe.qpos[0] ?? currentProbe.qpos[frameIndex];
        const controlledQpos = isMotionBricksProbe(currentProbe)
          ? currentProbe.qpos[frameIndex].slice()
          : transformControlledQpos(
              currentProbe.qpos[frameIndex],
              origin,
              controlledRoot.x,
              controlledRoot.y,
              controlledRoot.yaw,
            );
        currentRootMujoco.x = controlledQpos[0];
        currentRootMujoco.y = controlledQpos[1];
        currentRootMujoco.yaw = rootYawWxyz(controlledQpos.slice(3, 7));
        qposContextRef.current.push(controlledQpos.slice());
        if (qposContextRef.current.length > 24) {
          qposContextRef.current.splice(0, qposContextRef.current.length - 24);
        }
        const points = buildBodyPoints(THREE, kinematics, controlledQpos);
        let offset = 0;
        for (const [a, b] of debugSegments) {
          const pa = points.get(a);
          const pb = points.get(b);
          if (!pa || !pb) continue;
          positions[offset++] = pa.x;
          positions[offset++] = pa.y;
          positions[offset++] = pa.z;
          positions[offset++] = pb.x;
          positions[offset++] = pb.y;
          positions[offset++] = pb.z;
        }
        skeletonGeom.attributes.position.needsUpdate = true;
        const pelvis = points.get("pelvis");
        if (pelvis) {
          rootMarker.position.copy(pelvis);
          currentRootThree.copy(pelvis);
        }

        const matrices = buildBodyMatrices(THREE, kinematics, controlledQpos);
        for (const [name, group] of meshGroups) {
          const matrix = matrices.get(name);
          if (!matrix) continue;
          group.matrix.copy(mujocoMatrixToThreeMatrix(THREE, matrix));
          group.matrixWorldNeedsUpdate = true;
        }
        const wrist =
          matrices.get("right_wrist_yaw_link") ??
          matrices.get("right_wrist_pitch_link");
        if (wrist) {
          const localGrip = new THREE.Matrix4().makeTranslation(0, -0.01, -0.2);
          carriedArtifactMatrix.copy(
            mujocoMatrixToThreeMatrix(THREE, wrist).multiply(localGrip),
          );
          carriedArtifactMatrix.decompose(
            pickupTargetPosition,
            pickupTargetQuaternion,
            pickupScale,
          );
          pickupTargetQuaternion.setFromEuler(
            new THREE.Euler(0, currentRootMujoco.yaw + Math.PI * 0.18, 0),
          );
          carriedArtifactMatrix.compose(
            pickupTargetPosition,
            pickupTargetQuaternion,
            pickupScale,
          );
          for (const item of artifacts) {
            if (item.state !== "pickup" && item.state !== "carried") continue;
            if (item.state === "pickup") {
              const progress = Math.min(
                1,
                (performance.now() - item.pickupStartedAt) / 520,
              );
              const eased = 1 - Math.pow(1 - progress, 3);
              item.pickupStartMatrix.decompose(
                pickupStartPosition,
                pickupStartQuaternion,
                pickupScale,
              );
              pickupStartPosition.lerp(pickupTargetPosition, eased);
              pickupStartQuaternion.slerp(pickupTargetQuaternion, eased);
              item.group.matrix.compose(
                pickupStartPosition,
                pickupStartQuaternion,
                pickupScale,
              );
              if (progress >= 1) {
                item.state = "carried";
                carriedArtifactId = item.id;
                selectedArtifactId = item.id;
              }
            } else {
              item.group.matrix.copy(carriedArtifactMatrix);
            }
            item.group.matrixWorldNeedsUpdate = true;
          }
        }
        for (const item of artifacts) {
          if (item.state !== "settle") continue;
          const progress = Math.min(
            1,
            (performance.now() - item.settleStartedAt) / 420,
          );
          const eased = 1 - Math.pow(1 - progress, 3);
          item.settleStartMatrix.decompose(
            settleStartPosition,
            settleStartQuaternion,
            settleScale,
          );
          settlePosition.lerpVectors(
            settleStartPosition,
            item.settleTargetPosition,
            eased,
          );
          settlePosition.y += Math.sin(progress * Math.PI) * 0.08;
          settleStartQuaternion.slerp(item.settleTargetQuaternion, eased);
          item.group.matrix.compose(
            settlePosition,
            settleStartQuaternion,
            settleScale,
          );
          item.group.matrixWorldNeedsUpdate = true;
          if (progress >= 1) {
            item.state = "ground";
            item.group.matrixAutoUpdate = true;
            item.group.position.copy(item.groundPosition);
            item.group.rotation.set(
              0,
              currentRootMujoco.yaw + Math.PI * 0.18,
              0,
            );
            item.group.scale.setScalar(1);
            if (selectedArtifactId === item.id) selectedArtifactId = null;
          }
        }
      };

      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        const currentProbe = probeRef.current;
        if (!currentProbe) {
          renderer.render(scene, camera);
          return;
        }

        if (currentProbe !== lastProbe) {
          lastProbe = currentProbe;
          if (!isMotionBricksProbe(currentProbe)) {
            playheadRef.current = 0;
          }
          updatePath(currentProbe);
          lastPathFrameCount = currentProbe.qpos.length;
        }
        if (
          isMotionBricksProbe(currentProbe) &&
          currentProbe.qpos.length !== lastPathFrameCount &&
          now - lastPathUpdate > 250
        ) {
          updatePath(currentProbe);
          lastPathFrameCount = currentProbe.qpos.length;
          lastPathUpdate = now;
        }
        path.visible = showPathRef.current;
        meshRoot.visible = showMeshRef.current;
        skeleton.visible = showSkeletonRef.current;
        const artifactPulse = 1 + Math.sin(now * 0.008) * 0.08;
        for (const item of artifacts) {
          const artifactSelected =
            selectedArtifactId === item.id ||
            item.state === "pickup" ||
            item.state === "carried" ||
            item.state === "settle";
          const artifactFocus =
            hoveredArtifactId === item.id || artifactSelected;
          item.coreMaterial.emissiveIntensity = artifactFocus ? 0.45 : 0.12;
          item.haloMaterial.opacity = artifactFocus ? 0.95 : 0.58;
          item.halo.scale.setScalar((artifactFocus ? 1.18 : 1) * artifactPulse);
          item.focusRing.visible = item.state !== "carried";
          item.focusRingMaterial.opacity = artifactFocus ? 0.72 : 0;
          item.focusRing.scale.setScalar(
            (artifactFocus ? 1.08 : 0.9) * artifactPulse,
          );
        }
        targetRing.rotation.z = now * 0.0025;
        if (targetMarker.visible) {
          const approachPulse =
            isInteractionTarget() && activeTargetDistance < 1.15
              ? 1 + (1 - Math.min(1, activeTargetDistance / 1.15)) * 0.28
              : 1;
          targetMarker.scale.setScalar(targetMarkerBaseScale * approachPulse);
        }

        moveDir.set(0, 0, 0);
        const worldFwdX = -Math.sin(camYaw);
        const worldFwdZ = -Math.cos(camYaw);
        const worldRightX = Math.cos(camYaw);
        const worldRightZ = -Math.sin(camYaw);

        if (keys.KeyW || keys.ArrowUp) {
          moveDir.x += worldFwdX;
          moveDir.z += worldFwdZ;
        }
        if (keys.KeyS || keys.ArrowDown) {
          moveDir.x -= worldFwdX;
          moveDir.z -= worldFwdZ;
        }
        if (keys.KeyA || keys.ArrowLeft) {
          moveDir.x -= worldRightX;
          moveDir.z -= worldRightZ;
        }
        if (keys.KeyD || keys.ArrowRight) {
          moveDir.x += worldRightX;
          moveDir.z += worldRightZ;
        }

        if (moveDir.lengthSq() > 0.01) {
          moveDir.normalize();
          controlledRoot.x += -moveDir.z * walkSpeedRef.current * dt;
          controlledRoot.y += moveDir.x * walkSpeedRef.current * dt;
          controlledRoot.yaw = Math.atan2(moveDir.x, -moveDir.z);
          if (!playingRef.current) {
            playingRef.current = true;
            setPlaying(true);
          }
        }
        if (isMotionBricksProbe(currentProbe) && now - lastControlUpdate > 80) {
          sendCurrentControl();
          lastControlUpdate = now;
        }

        const liveCushion = 0;
        let frameIndex: number;
        if (isMotionBricksProbe(currentProbe)) {
          frameIndex = Math.max(0, currentProbe.qpos.length - 1 - liveCushion);
          playheadRef.current = frameIndex;
        } else {
          const scrub = scrubFrameRef.current;
          if (scrub != null) {
            playheadRef.current = scrub;
          } else if (playingRef.current) {
            playheadRef.current += dt * currentProbe.fps * speedRef.current;
          }
          frameIndex =
            ((Math.floor(playheadRef.current) % currentProbe.qpos.length) +
              currentProbe.qpos.length) %
            currentProbe.qpos.length;
        }
        updateSkeleton(frameIndex);
        if (now - lastFrameUiUpdate > 100) {
          setFrame(frameIndex);
          lastFrameUiUpdate = now;
        }
        cameraTarget.lerp(currentRootThree, 1 - Math.exp(-dt * 10));
        const camOffX = camDist * Math.cos(camPitch) * Math.sin(camYaw);
        const camOffY = camDist * Math.sin(camPitch);
        const camOffZ = camDist * Math.cos(camPitch) * Math.cos(camYaw);
        camera.position.set(
          cameraTarget.x + camOffX,
          cameraTarget.y + camOffY,
          cameraTarget.z + camOffZ,
        );
        camera.lookAt(cameraTarget.x, cameraTarget.y + 0.6, cameraTarget.z);
        renderer.render(scene, camera);
      });

      cleanup = () => {
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup", onKeyUp);
        renderer.domElement.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mouseup", onMouseUp);
        window.removeEventListener("mousemove", onMouseMove);
        renderer.domElement.removeEventListener("wheel", onWheel);
        window.removeEventListener("resize", onResize);
        eventSource?.close();
      streamStartedRef.current = false;
        renderer.setAnimationLoop(null);
        renderer.domElement.style.cursor = "";
        renderer.dispose();
        mount.removeChild(renderer.domElement);
      };
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [hasProbe, kinematics]);

  const handleScrub = (nextFrame: number) => {
    scrubFrameRef.current = nextFrame;
    playheadRef.current = nextFrame;
    setFrame(nextFrame);
  };

  const releaseScrub = () => {
    scrubFrameRef.current = null;
  };

  return (
    <div className="fixed inset-0 bg-black text-zinc-100">
      <div ref={mountRef} className="absolute inset-0" />

      {!probe || !kinematics ? (
        <LoadingParticles label="loading nvidia motion" className="z-30" />
      ) : null}

      {!menuOpen && probe && kinematics ? (
        <div style={HUD_FONT} className="pointer-events-none">
          <Link
            href="/"
            aria-label="back"
            className={`pointer-events-auto absolute left-5 top-5 z-10 ${HUD_BOX_SQUARE}`}
          >
            <IconArrowLeft />
          </Link>
          <div className="pointer-events-auto absolute right-5 top-5 z-10 flex gap-2">
            <button
              onClick={toggleMenu}
              aria-label="settings"
              className={HUD_BOX_SQUARE}
            >
              <IconKeyboard />
            </button>
          </div>
          <div className="absolute bottom-5 left-5 z-10 text-[11px] uppercase tracking-wider text-zinc-500">
            {`${probe.mode} · frame ${frame} · ${plannerStatus}`}
          </div>
        </div>
      ) : null}

      {menuOpen ? (
        <div
          style={HUD_FONT}
          className="absolute inset-0 z-20 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(event) => {
            if (event.target === event.currentTarget) setMenuOpen(false);
          }}
        >
          <div className="max-h-[80vh] w-[460px] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <span className="text-[13px] text-white">NVIDIA Motion Controls</span>
              <button
                onClick={toggleMenu}
                aria-label="close"
                className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-white/10 hover:text-white"
              >
                <IconClose />
              </button>
            </div>

            <div className="space-y-5 p-5">
              <div className="space-y-2.5 text-[13px]">
                <div className="flex justify-between">
                  <span className="text-white">Move</span>
                  <span className="text-zinc-400">wasd</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Orbit camera</span>
                  <span className="text-zinc-400">drag</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Zoom</span>
                  <span className="text-zinc-400">scroll</span>
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
                  Motion
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {NVIDIA_MOTION_MODES.map((tier) => (
                    <button
                      key={tier.label}
                      onClick={() =>
                        pickMotionMode(tier.label, tier.value, tier.clip)
                      }
                      className={`${HUD_BOX_BASE} h-8 px-2 text-[12px] ${
                        activeMotionMode === tier.label
                          ? "border-white/30 bg-white/10 text-white"
                          : ""
                      }`}
                    >
                      {tier.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
                  Playback
                </div>
                <div className="grid gap-3">
                  <div className="flex gap-2">
                    <button
                      onClick={() => setPlaying((value) => !value)}
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px]`}
                    >
                      {playing ? "Pause" : "Play"}
                    </button>
                    <select
                      value={speed}
                      onChange={(event) =>
                        setSpeed(Number(event.currentTarget.value))
                      }
                      className={`${HUD_BOX_BASE} h-8 px-2 text-[12px]`}
                    >
                      <option value={0.5}>0.5x</option>
                      <option value={1}>1x</option>
                      <option value={1.5}>1.5x</option>
                      <option value={2}>2x</option>
                    </select>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={Math.max(0, (probe?.qpos.length ?? 1) - 1)}
                    value={frame}
                    onChange={(event) =>
                      handleScrub(Number(event.currentTarget.value))
                    }
                    onPointerUp={releaseScrub}
                    onKeyUp={releaseScrub}
                    className="w-full"
                  />
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
                  Debug
                </div>
                <div className="flex flex-wrap gap-2">
                  {[
                    ["Path", showPath, setShowPath],
                    ["Mesh", showMesh, setShowMesh],
                    ["Bones", showSkeleton, setShowSkeleton],
                  ].map(([label, checked, setter]) => (
                    <button
                      key={label as string}
                      onClick={() =>
                        (setter as (value: boolean) => void)(
                          !(checked as boolean),
                        )
                      }
                      className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                        checked ? "border-white/30 bg-white/10 text-white" : ""
                      }`}
                    >
                      {label as string}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

"use client";

import type * as THREE from "three";
import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/api";
import LoadingParticles from "@/components/LoadingParticles";
import { loadColliderMesh } from "@/lib/collider";
import { LocomotionEngine, type LocoData } from "@/lib/locomotion";
import {
  createPhysicsWorld,
  type PhysicsArtifactBody,
  type PhysicsRampBody,
  type PhysicsWorld,
} from "@/lib/physics";
import Link from "next/link";
import {
  IconArrowLeft,
  IconBrush,
  IconCamera,
  IconClose,
  IconGrid,
  IconKeyboard,
  IconTrash,
  IconUndo,
  HUD_FONT,
  HUD_BOX_BASE,
  HUD_BOX_SQUARE,
} from "./hud-icons";

type CharacterDef = {
  label: string;
  glb: string;
  onnx: string;
  data: string;
  defaultStyle: string;
};

const ONNX_BASE =
  process.env.NEXT_PUBLIC_API_URL ? `${API_BASE}/assets` : "";

const CHARACTERS: CharacterDef[] = [
  { label: "Geno", glb: "/character.glb", onnx: `${ONNX_BASE}/locomotion.onnx`, data: "/locomotion-data.json", defaultStyle: "Neutral" },
  { label: "Dog", glb: "/dog.glb", onnx: `${ONNX_BASE}/quadruped.onnx`, data: "/quadruped-data.json", defaultStyle: "Walk" },
  { label: "Wolf", glb: "/wolf.glb", onnx: `${ONNX_BASE}/quadruped.onnx`, data: "/quadruped-data.json", defaultStyle: "Walk" },
];

const SPEED_TIERS = [
  { label: "Walk", value: 0.3 },
  { label: "Stride", value: 1.0 },
  { label: "Run", value: 1.5 },
  { label: "Sprint", value: 2.0 },
];

type Props = {
  splatUrl: string;
  splatPaged?: boolean;
  colliderMeshUrl?: string | null;
  metricScale?: number | null;
  groundPlaneOffset?: number | null;
  userPrompt?: string | null;
  sceneCaption?: string | null;
  backHref: string;
};

type EditOp = {
  color: string | null;
  op: string;
  size: string;
};

type WalkArtifactState = "ground" | "selected" | "pickup" | "carried" | "settle";

type WalkArtifact = {
  id: string;
  label: string;
  group: THREE.Group;
  coreMaterial: THREE.MeshStandardMaterial;
  haloMaterial: THREE.MeshBasicMaterial;
  focusMaterial: THREE.MeshBasicMaterial;
  body: PhysicsArtifactBody | null;
  state: WalkArtifactState;
  transitionFrom: THREE.Vector3;
  transitionTo: THREE.Vector3;
  transitionStart: number;
  transitionDuration: number;
  settleUntil: number;
  delivered: boolean;
};

type WalkRamp = {
  group: THREE.Group;
  body: PhysicsRampBody | null;
  selected: boolean;
};

type WalkerHandle = {
  snapshot: () => { dataUrl: string; width: number; height: number };
  paintAt: (ndcX: number, ndcY: number, edit: EditOp) => number;
};

const GROK_INPUT_PER_M = 0.2;
const GROK_OUTPUT_PER_M = 0.5;

const COLOR_MAP: Record<string, [number, number, number]> = {
  red: [1.4, 0.3, 0.3],
  blue: [0.3, 0.3, 1.4],
  green: [0.3, 1.4, 0.3],
  yellow: [1.4, 1.4, 0.3],
  orange: [1.4, 0.8, 0.3],
  purple: [1.2, 0.3, 1.2],
  pink: [1.4, 0.5, 1.0],
  cyan: [0.3, 1.4, 1.4],
  white: [1.4, 1.4, 1.4],
  black: [0.1, 0.1, 0.1],
  dark: [0.3, 0.3, 0.3],
  light: [1.4, 1.4, 1.4],
};

const SIZE_MAP: Record<string, number> = {
  small: 0.35,
  medium: 0.8,
  large: 2.0,
};

const SPARK_SORT_INTERVAL_MS = 140;
const SPARK_MIN_ALPHA = 2 / 255;
const SPARK_MAX_STD_DEV = Math.sqrt(4);
const SPARK_MAX_PIXEL_RADIUS = 384;
const WALKER_MAX_PIXEL_RATIO = 1.0;
const CAMERA_FOLLOW_HZ = 5;
const CAMERA_FOLLOW_DEADBAND_M = 0.0025;
const CAMERA_NECK_PITCH = 0.24;
const CAMERA_NECK_DISTANCE = 0.82;
const CAMERA_MIN_DISTANCE = 0.28;
const CAMERA_MAX_DISTANCE = 8;
const CAMERA_ANCHOR_FALLBACK_Y = 0.42;
const CAMERA_ANCHOR_BONES = ["Neck", "Head", "HeadSite", "Spine1", "Spine", "Hips"];

const COLOR_CSS: Record<string, string> = {
  red: "#e04040", blue: "#4040e0", green: "#40e040", yellow: "#e0e040",
  orange: "#e09040", purple: "#c040c0", pink: "#e070b0", cyan: "#40e0e0",
  white: "#e0e0e0", black: "#1a1a1a", dark: "#404040", light: "#d0d0d0",
};

function resolveEditColor(edit: EditOp): [number, number, number] {
  switch (edit.op) {
    case "erase":
      return [0.05, 0.05, 0.05];
    case "brighten":
      return [1.8, 1.8, 1.8];
    case "darken":
      return [0.3, 0.3, 0.3];
    case "recolor":
    default: {
      const key = (edit.color ?? "red").toLowerCase();
      return COLOR_MAP[key] ?? COLOR_MAP.red;
    }
  }
}

function formatUsage(usage: Record<string, unknown>): string {
  const inTok = asNum(usage.input_tokens ?? usage.prompt_tokens);
  const outTok = asNum(usage.output_tokens ?? usage.completion_tokens);
  if (inTok == null && outTok == null) return JSON.stringify(usage);
  const parts: string[] = [];
  if (inTok != null) parts.push(`${inTok} in`);
  if (outTok != null) parts.push(`${outTok} out`);
  if (inTok != null && outTok != null) {
    const cost =
      (inTok * GROK_INPUT_PER_M) / 1_000_000 +
      (outTok * GROK_OUTPUT_PER_M) / 1_000_000;
    parts.push(`~$${cost.toFixed(5)}`);
  }
  return parts.join(" · ");
}

function asNum(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}

export default function Walker({
  splatUrl,
  splatPaged = false,
  colliderMeshUrl,
  metricScale,
  userPrompt,
  sceneCaption,
  backHref,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const apiRef = useRef<WalkerHandle | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const editActionsRef = useRef<{ undo: () => void; clear: () => void } | null>(null);
  const splatRef = useRef<THREE.Object3D | null>(null);
  const viewModeRef = useRef<0 | 1 | 2>(0); // 0=mesh, 1=splat+mesh, 2=splat
  const [viewMode, setViewMode] = useState<0 | 1 | 2>(0);
  const cameraFollowRef = useRef(false);
  const [cameraFollowUI, setCameraFollowUI] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuTab, setMenuTab] = useState<"edit" | "game" | "controls">("edit");
  const [activeChar, setActiveChar] = useState("Dog");
  const [activeStyle, setActiveStyle] = useState("Walk");
  const [activeSpeed, setActiveSpeed] = useState("Walk");
  const [styles, setStyles] = useState<string[]>([]);
  const [charLoading, setCharLoading] = useState(false);
  const [splatLoading, setSplatLoading] = useState(true);
  const walkSpeedRef = useRef(1.0);
  const engineRef = useRef<LocomotionEngine | null>(null);
  const loadCharRef = useRef<((def: CharacterDef) => Promise<void>) | null>(null);
  const [intent, setIntent] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEdit, setLastEdit] = useState<{
    reason: string | null;
    usage: Record<string, unknown> | null;
  } | null>(null);
  const [paintMode, setPaintMode] = useState(false);
  const [deliveryMode, setDeliveryMode] = useState(false);
  const [deliveryDelivered, setDeliveryDelivered] = useState(0);
  const [rampSelectedUI, setRampSelectedUI] = useState(false);
  const [paintColor, setPaintColor] = useState("red");
  const [paintOp, setPaintOp] = useState("recolor");
  const [paintSize, setPaintSize] = useState("medium");
  const paintModeRef = useRef(false);
  const deliveryModeRef = useRef(false);
  const deliveryRuntimeRef = useRef<{ setEnabled: (enabled: boolean) => void } | null>(null);
  const paintOpRef = useRef("recolor");
  const paintColorRef = useRef("red");
  const paintSizeRef = useRef("medium");

  // Sync paint refs so canvas event handlers can read current values
  useEffect(() => { paintModeRef.current = paintMode; }, [paintMode]);
  useEffect(() => {
    deliveryModeRef.current = deliveryMode;
    deliveryRuntimeRef.current?.setEnabled(deliveryMode);
  }, [deliveryMode]);
  useEffect(() => { paintOpRef.current = paintOp; }, [paintOp]);
  useEffect(() => { paintColorRef.current = paintColor; }, [paintColor]);
  useEffect(() => { paintSizeRef.current = paintSize; }, [paintSize]);

  const toggleMenu = useCallback(() => setMenuOpen((o) => !o), []);

  const pickStyle = useCallback((s: string) => {
    engineRef.current?.setStyle(s);
    setActiveStyle(s);
  }, []);

  const pickSpeed = useCallback((label: string, value: number) => {
    walkSpeedRef.current = value;
    setActiveSpeed(label);
  }, []);

  const loadCharacter = useCallback(async (def: CharacterDef) => {
    await loadCharRef.current?.(def);
  }, []);

  // disable grain overlay on the walk page
  useEffect(() => {
    document.body.classList.add("no-grain");
    return () => document.body.classList.remove("no-grain");
  }, []);

  // ESC toggles menu
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Escape") {
        e.preventDefault();
        toggleMenu();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggleMenu]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let disposed = false;
    let cleanup: (() => void) | null = null;

    (async () => {
      const THREE = await import("three");
      const {
        SplatMesh,
        SparkRenderer,
        SplatEdit,
        SplatEditSdf,
        SplatEditSdfType,
        SplatEditRgbaBlendMode,
      } = await import("@sparkjsdev/spark");
      if (disposed) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#000");

      // Lighting for textured models (Dog/Wolf). MeshBasicMaterial (Geno) ignores these.
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(5, 10, 5);
      scene.add(dirLight);

      const camera = new THREE.PerspectiveCamera(
        70,
        mount.clientWidth / mount.clientHeight,
        0.01,
        1000
      );
      camera.position.set(0, 0, 2);
      camera.rotation.order = "YXZ";

      const renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, WALKER_MAX_PIXEL_RATIO));
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      mount.appendChild(renderer.domElement);

      const spark = new SparkRenderer({
        renderer,
        maxPixelRadius: SPARK_MAX_PIXEL_RADIUS,
        maxStdDev: SPARK_MAX_STD_DEV,
        minAlpha: SPARK_MIN_ALPHA,
        falloff: 0,
        lodRenderScale: 2.5,
        minSortIntervalMs: SPARK_SORT_INTERVAL_MS,
        sortRadial: true,
      });
      scene.add(spark);

      const splat = new SplatMesh({
        url: splatUrl,
        paged: splatPaged,
        editable: true,
        raycastable: true,
        minRaycastOpacity: 0.5,
      });
      splat.rotation.x = Math.PI;
      splat.visible = false;
      scene.add(splat);
      splatRef.current = splat;

      splat.initialized
        .then(() => {
          if (disposed) return;
          spark.falloff = 1;
          setSplatLoading(false);
          // splat loaded
        })
        .catch(() => {
          if (!disposed) setSplatLoading(false);
        });

      // === Collider Mesh ===

      let colliderMesh: THREE.Mesh | null = null;
      const colliderWorldBounds = new THREE.Box3();

      if (colliderMeshUrl) {
        try {
          const result = await loadColliderMesh(colliderMeshUrl);
          if (result) {
            colliderMesh = result.mesh;
            colliderMesh.material = new THREE.MeshBasicMaterial({
              color: 0x00ff00,
              wireframe: true,
              opacity: 0.4,
              transparent: true,
              depthWrite: false,
              depthTest: false,
            });
            colliderMesh.renderOrder = 999;
            colliderMesh.visible = true;
            // Add the full GLTF scene root — preserves parent transforms
            scene.add(result.root);
            result.root.updateMatrixWorld(true);
            colliderWorldBounds.setFromObject(colliderMesh);
            meshRef.current = colliderMesh;
          }
        } catch (e) {
          console.error("[walker] collider mesh failed:", e);
        }
      }

      // === Character (ONNX neural locomotion) ===

      const mScale = metricScale ?? 2;
      const CHAR_SCALE = 1.0 / mScale; // engine→scene position mapping
      const VIS_SCALE = 1 / 3; // visual scale for character bones

      // Character container — no rotation; character lives in world space
      const characterContainer = new THREE.Group();
      scene.add(characterContainer);

      let characterRoot: THREE.Group | null = null;
      let locoEngine: LocomotionEngine | null = null;
      let locoData: LocoData | null = null;
      let physics: PhysicsWorld | null = null;
      const boneByName = new Map<string, THREE.Bone>();
      const boneIndexByName = new Map<string, number>();
      // Anchor in world space: engine origin maps here
      const localAnchor = new THREE.Vector3();
      // Physics-tracked character position (world space, at feet)
      let physX = 0, physY = 0, physZ = 0;
      // Previous engine root XZ — to compute per-frame delta
      let prevEngineX = 0, prevEngineZ = 0;
      // Smoothed hips anchor — absorbs per-frame jitter from neural net predictions
      let smoothHipsX = 0, smoothHipsY = 0, smoothHipsZ = 0;
      let smoothHipsInit = false;

      // Camera state
      let camYaw = 0;
      let camPitch = CAMERA_NECK_PITCH;
      let camDist = CAMERA_NECK_DISTANCE;
      const orbitTarget = new THREE.Vector3();
      let orbitTargetInit = false;

      // World→local bone transform (same as working /character demo)
      const _wp = new THREE.Vector3();
      const _wq = new THREE.Quaternion();
      const _ws = new THREE.Vector3(1, 1, 1);
      const _wm = new THREE.Matrix4();
      const _pi2 = new THREE.Matrix4();
      const _lm = new THREE.Matrix4();
      const _lp = new THREE.Vector3();
      const _lq = new THREE.Quaternion();
      const _ls = new THREE.Vector3();

      const applyWorldToBone = (
        bone: THREE.Bone,
        wx: number, wy: number, wz: number,
        qx: number, qy: number, qz: number, qw: number,
      ) => {
        _wp.set(wx, wy, wz);
        _wq.set(qx, qy, qz, qw);
        _wm.compose(_wp, _wq, _ws);
        if (bone.parent) {
          _pi2.copy(bone.parent.matrixWorld).invert();
          _lm.multiplyMatrices(_pi2, _wm);
        } else {
          _lm.copy(_wm);
        }
        _lm.decompose(_lp, _lq, _ls);
        bone.position.copy(_lp);
        bone.quaternion.copy(_lq);
        bone.updateMatrix();
        bone.updateMatrixWorld(true);
      };

      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      const gltfLoader = new GLTFLoader();

      // --- Physics (once, tied to world not character) ---
      if (colliderMesh) {
        colliderMesh.geometry.computeBoundingBox();
        const bb = colliderMesh.geometry.boundingBox!;
        const center = new THREE.Vector3();
        bb.getCenter(center);
        colliderMesh.updateMatrixWorld(true);
        colliderWorldBounds.setFromObject(colliderMesh);
        colliderMesh.localToWorld(center);
        localAnchor.copy(center);
        physX = center.x;
        physY = center.y;
        physZ = center.z;

        const capsuleR = 0.05 * CHAR_SCALE;
        const capsuleH = 0.2 * CHAR_SCALE;
        physics = await createPhysicsWorld(
          colliderMesh, capsuleR, capsuleH, center.x, center.y, center.z,
        );
        console.log("[walker] rapier physics ready, spawn:", center.x.toFixed(3), center.y.toFixed(3), center.z.toFixed(3));
      }

      // === Portable artifacts ===

      const ARTIFACT_HALF = 0.075;
      const ARTIFACT_PICKUP_RADIUS = 0.5;
      const ARTIFACT_PICKUP_MS = 420;
      const ARTIFACT_DROP_MS = 320;
      const DELIVERY_RADIUS = 0.38;
      const artifactGeometry = new THREE.BoxGeometry(
        ARTIFACT_HALF * 2,
        ARTIFACT_HALF * 2,
        ARTIFACT_HALF * 2,
      );
      const artifactHaloGeometry = new THREE.TorusGeometry(ARTIFACT_HALF * 1.65, 0.006, 8, 36);
      const artifactFocusGeometry = new THREE.TorusGeometry(ARTIFACT_HALF * 2.25, 0.004, 8, 44);
      const artifactCapGeometry = new THREE.SphereGeometry(ARTIFACT_HALF * 0.45, 16, 16);
      const artifactPalette = [
        { id: "relay", label: "Relay", color: 0x36d399, offset: new THREE.Vector3(0.55, 0.28, -0.4) },
        { id: "core", label: "Core", color: 0x60a5fa, offset: new THREE.Vector3(-0.5, 0.32, -0.62) },
        { id: "seed", label: "Seed", color: 0xfbbf24, offset: new THREE.Vector3(0.15, 0.36, 0.58) },
      ];
      const artifacts: WalkArtifact[] = [];
      const artifactById = new Map<string, WalkArtifact>();
      let selectedArtifactId: string | null = null;
      let carriedArtifactId: string | null = null;
      let deliveredCount = 0;
      let lastDeliveredCount = -1;
      let carryBoneCandidates = ["HeadSite", "Head", "RightHandSite", "RightHand"];

      const groundProbeRaycaster = new THREE.Raycaster();
      const groundProbeOrigin = new THREE.Vector3();
      const groundProbeDirection = new THREE.Vector3(0, -1, 0);
      const groundProbeTargets: THREE.Object3D[] = colliderMesh ? [colliderMesh] : [];
      const projectToColliderGround = (
        position: THREE.Vector3,
        objectHalfHeight = 0,
        lift = 0.01,
      ) => {
        if (!colliderMesh || colliderWorldBounds.isEmpty()) return position.clone();
        const probeTop = Number.isFinite(colliderWorldBounds.max.y)
          ? colliderWorldBounds.max.y + 4
          : position.y + 8;
        const probeDepth = Math.max(
          12,
          Number.isFinite(colliderWorldBounds.max.y) && Number.isFinite(colliderWorldBounds.min.y)
            ? colliderWorldBounds.max.y - colliderWorldBounds.min.y + 8
            : 12,
        );
        const origSide = (colliderMesh.material as THREE.Material).side;
        (colliderMesh.material as THREE.Material).side = THREE.DoubleSide;
        groundProbeOrigin.set(position.x, probeTop, position.z);
        groundProbeRaycaster.set(groundProbeOrigin, groundProbeDirection);
        groundProbeRaycaster.far = probeDepth;
        const hits = groundProbeRaycaster.intersectObjects(groundProbeTargets, true);
        (colliderMesh.material as THREE.Material).side = origSide;
        if (hits.length === 0) return position.clone();
        const hitY = groundProbeOrigin.y - hits[0].distance;
        const grounded = new THREE.Vector3(position.x, hitY + objectHalfHeight + lift, position.z);
        return grounded;
      };

      const deliveryPosition = projectToColliderGround(
        localAnchor.clone().add(new THREE.Vector3(0.85, 0, 0.75)),
        0,
        0.018,
      );
      const deliveryMaterial = new THREE.MeshBasicMaterial({
        color: 0x36d399,
        transparent: true,
        opacity: 0.58,
        depthWrite: false,
        depthTest: false,
      });
      const deliveryFillMaterial = new THREE.MeshBasicMaterial({
        color: 0x36d399,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
        depthTest: false,
      });
      const deliveryBeaconMaterial = new THREE.MeshBasicMaterial({
        color: 0x36d399,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
        depthTest: false,
      });
      const deliveryZone = new THREE.Group();
      const deliveryRing = new THREE.Mesh(
        new THREE.TorusGeometry(DELIVERY_RADIUS, 0.01, 8, 64),
        deliveryMaterial,
      );
      deliveryRing.rotation.x = Math.PI / 2;
      const deliveryFill = new THREE.Mesh(
        new THREE.CircleGeometry(DELIVERY_RADIUS * 0.95, 64),
        deliveryFillMaterial,
      );
      deliveryFill.rotation.x = -Math.PI / 2;
      const deliveryBeacon = new THREE.Mesh(
        new THREE.CylinderGeometry(DELIVERY_RADIUS * 0.72, DELIVERY_RADIUS * 0.72, 0.5, 40, 1, true),
        deliveryBeaconMaterial,
      );
      deliveryBeacon.position.y = 0.25;
      deliveryRing.renderOrder = 1200;
      deliveryFill.renderOrder = 1199;
      deliveryBeacon.renderOrder = 1198;
      deliveryZone.position.copy(deliveryPosition);
      deliveryZone.add(deliveryFill, deliveryRing, deliveryBeacon);
      scene.add(deliveryZone);

      const setArtifactUserData = (object: THREE.Object3D, id: string) => {
        object.userData.walkArtifactId = id;
        for (const child of object.children) setArtifactUserData(child, id);
      };

      for (const spec of artifactPalette) {
        const group = new THREE.Group();
        const initial = projectToColliderGround(
          localAnchor.clone().add(new THREE.Vector3(spec.offset.x, 0, spec.offset.z)),
          ARTIFACT_HALF,
          0.025,
        );
        group.position.copy(initial);

        const coreMaterial = new THREE.MeshStandardMaterial({
          color: spec.color,
          emissive: spec.color,
          emissiveIntensity: 0.3,
          roughness: 0.35,
          metalness: 0.2,
        });
        const haloMaterial = new THREE.MeshBasicMaterial({
          color: spec.color,
          transparent: true,
          opacity: 0.55,
          depthWrite: false,
        });
        const focusMaterial = new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0,
          depthWrite: false,
        });
        const capMaterial = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          emissive: 0xffffff,
          emissiveIntensity: 0.12,
          roughness: 0.25,
        });

        const core = new THREE.Mesh(artifactGeometry, coreMaterial);
        const halo = new THREE.Mesh(artifactHaloGeometry, haloMaterial);
        halo.rotation.x = Math.PI / 2;
        const focus = new THREE.Mesh(artifactFocusGeometry, focusMaterial);
        focus.rotation.x = Math.PI / 2;
        const cap = new THREE.Mesh(artifactCapGeometry, capMaterial);
        cap.position.y = ARTIFACT_HALF * 1.15;
        group.add(core, halo, focus, cap);
        setArtifactUserData(group, spec.id);
        scene.add(group);

        const body = physics?.createArtifactBox({
          x: initial.x,
          y: initial.y,
          z: initial.z,
          halfExtents: { x: ARTIFACT_HALF, y: ARTIFACT_HALF, z: ARTIFACT_HALF },
        }) ?? null;

        const artifact: WalkArtifact = {
          id: spec.id,
          label: spec.label,
          group,
          coreMaterial,
          haloMaterial,
          focusMaterial,
          body,
          state: "ground",
          transitionFrom: new THREE.Vector3(),
          transitionTo: new THREE.Vector3(),
          transitionStart: 0,
          transitionDuration: 1,
          settleUntil: 0,
          delivered: false,
        };
        artifacts.push(artifact);
        artifactById.set(spec.id, artifact);
      }

      // === Movable ramp ===

      const RAMP_LENGTH = 0.72;
      const RAMP_WIDTH = 0.34;
      const RAMP_HEIGHT = 0.08;
      const rampPosition = projectToColliderGround(
        localAnchor.clone().add(new THREE.Vector3(-0.85, 0, 0.55)),
        0,
        0.018,
      );
      const rampGroup = new THREE.Group();
      rampGroup.position.copy(rampPosition);
      rampGroup.userData.walkRamp = true;

      const rampGeometry = new THREE.BoxGeometry(RAMP_WIDTH, RAMP_HEIGHT, RAMP_LENGTH);
      const rampMaterial = new THREE.MeshStandardMaterial({
        color: 0xf8fafc,
        emissive: 0x0ea5e9,
        emissiveIntensity: 0.08,
        roughness: 0.58,
        metalness: 0.12,
      });
      const rampMesh = new THREE.Mesh(rampGeometry, rampMaterial);
      const rampOutline = new THREE.Mesh(
        new THREE.BoxGeometry(RAMP_WIDTH * 1.08, 0.018, RAMP_LENGTH * 1.04),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.16,
          depthWrite: false,
        }),
      );
      rampOutline.position.y = RAMP_HEIGHT / 2 + 0.012;
      rampOutline.rotation.x = Math.PI / 2;
      rampOutline.visible = false;
      rampGroup.add(rampMesh, rampOutline);
      scene.add(rampGroup);

      const rampQuat = new THREE.Quaternion().setFromEuler(rampGroup.rotation);
      const ramp: WalkRamp = {
        group: rampGroup,
        body: physics?.createRamp({
          x: rampPosition.x,
          y: rampPosition.y + RAMP_HEIGHT / 2,
          z: rampPosition.z,
          halfExtents: { x: RAMP_WIDTH / 2, y: RAMP_HEIGHT / 2, z: RAMP_LENGTH / 2 },
          rotation: { x: rampQuat.x, y: rampQuat.y, z: rampQuat.z, w: rampQuat.w },
        }) ?? null,
        selected: false,
      };

      const placeRamp = (point: THREE.Vector3) => {
        const grounded = projectToColliderGround(point, 0, 0.018);
        ramp.group.position.copy(grounded);
        ramp.group.updateMatrixWorld(true);
        const q = new THREE.Quaternion().setFromEuler(ramp.group.rotation);
        ramp.body?.dropAt(
          grounded.x,
          grounded.y + RAMP_HEIGHT + 0.08,
          grounded.z,
          { x: q.x, y: q.y, z: q.z, w: q.w },
        );
      };

      const setDeliveryGameEnabled = (enabled: boolean) => {
        deliveryZone.visible = enabled;
        for (const artifact of artifacts) {
          artifact.group.visible = enabled;
          artifact.body?.setPhysicsEnabled(enabled && artifact.state !== "carried");
        }
        ramp.group.visible = enabled;
        ramp.body?.setPhysicsEnabled(enabled);
        if (!enabled) {
          const selected = selectedArtifactId ? artifactById.get(selectedArtifactId) : null;
          if (selected?.state === "selected") selected.state = "ground";
          selectedArtifactId = null;
          carriedArtifactId = null;
          ramp.selected = false;
          rampOutline.visible = false;
          setRampSelectedUI(false);
        }
      };
      setDeliveryGameEnabled(deliveryModeRef.current);
      deliveryRuntimeRef.current = { setEnabled: setDeliveryGameEnabled };

      // --- Character loader (same pattern as character/page.tsx) ---
      const doLoadCharacter = async (def: CharacterDef) => {
        setCharLoading(true);

        // Remove old character
        if (characterRoot) {
          characterContainer.remove(characterRoot);
          characterRoot = null;
        }
        boneByName.clear();
        boneIndexByName.clear();

        const gltf = await gltfLoader.loadAsync(def.glb);
        characterRoot = new THREE.Group();
        characterRoot.add(gltf.scene);

        // Geno has no textures — white. Dog/Wolf have baked textures.
        // Force all materials opaque (GLB exports can have transparency enabled).
        gltf.scene.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            // Frustum culling breaks with boneInverse scaling — the bounding
            // sphere is computed from the original bind pose, not where the
            // character actually renders. Disable it.
            mesh.frustumCulled = false;
            if (def.label === "Geno") {
              mesh.material = new THREE.MeshBasicMaterial({ color: 0xffffff });
            } else {
              const mat = mesh.material as THREE.Material;
              mat.transparent = false;
              mat.opacity = 1;
              mat.depthWrite = true;
            }
          }
        });

        // Collect bones + scale skeleton via boneInverses
        gltf.scene.traverse((child) => {
          if ((child as THREE.SkinnedMesh).isSkinnedMesh) {
            const sm = child as THREE.SkinnedMesh;
            for (const bone of sm.skeleton.bones) {
              boneByName.set(bone.name, bone);
            }
            const scaleMat = new THREE.Matrix4().makeScale(VIS_SCALE, VIS_SCALE, VIS_SCALE);
            for (let i = 0; i < sm.skeleton.boneInverses.length; i++) {
              sm.skeleton.boneInverses[i].premultiply(scaleMat);
            }
          }
        });

        characterContainer.add(characterRoot);

        // Position at current physics location
        if (physics) {
          characterRoot.position.set(physX, physY, physZ);
          characterRoot.visible = true;
        } else {
          characterRoot.visible = false;
        }

        // Load ONNX locomotion engine
        const dataResp = await fetch(def.data);
        locoData = await dataResp.json() as LocoData;
        locoEngine = new LocomotionEngine(locoData);
        await locoEngine.loadModel(def.onnx);
        locoEngine.setStyle(def.defaultStyle);
        engineRef.current = locoEngine;

        // Reset engine tracking
        prevEngineX = 0;
        prevEngineZ = 0;

        for (let i = 0; i < locoData.boneNames.length; i++) {
          boneIndexByName.set(locoData.boneNames[i], i);
        }

        setStyles(locoEngine.getStyles());
        setActiveStyle(def.defaultStyle);
        setActiveChar(def.label);
        carryBoneCandidates =
          def.label === "Geno"
            ? ["RightHand", "RightHandSite", "Head", "HeadSite"]
            : ["HeadSite", "Head", "RightHandSite", "RightHand"];
        setCharLoading(false);
        console.log(`[walker] loaded ${def.label}`);
      };

      loadCharRef.current = doLoadCharacter;

      try {
        await doLoadCharacter(CHARACTERS[1]); // Dog
      } catch (e) {
        console.error("[walker] character/locomotion load failed:", e);
      }

      // === Raycasting ===

      const raycaster = new THREE.Raycaster();
      const ndc = new THREE.Vector2();

      const artifactHitRoots = artifacts.map((item) => item.group);

      const findArtifactId = (object: THREE.Object3D): string | null => {
        let current: THREE.Object3D | null = object;
        while (current) {
          const id = current.userData.walkArtifactId;
          if (typeof id === "string") return id;
          current = current.parent;
        }
        return null;
      };

      const raycastArtifact = (
        ndcX: number,
        ndcY: number
      ): WalkArtifact | null => {
        if (!deliveryModeRef.current) return null;
        ndc.set(ndcX, ndcY);
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObjects(artifactHitRoots, true);
        for (const hit of hits) {
          const id = findArtifactId(hit.object);
          if (!id) continue;
          const artifact = artifactById.get(id);
          if (artifact && artifact.state !== "carried" && !artifact.delivered) return artifact;
        }
        return null;
      };

      const raycastRamp = (
        ndcX: number,
        ndcY: number,
      ): WalkRamp | null => {
        if (!deliveryModeRef.current) return null;
        ndc.set(ndcX, ndcY);
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(ramp.group, true);
        return hits.length > 0 ? ramp : null;
      };

      const raycastMesh = (
        ndcX: number,
        ndcY: number
      ): THREE.Vector3 | null => {
        if (!colliderMesh) return null;
        const origSide = (colliderMesh.material as THREE.Material).side;
        (colliderMesh.material as THREE.Material).side = THREE.DoubleSide;
        ndc.set(ndcX, ndcY);
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObject(colliderMesh);
        (colliderMesh.material as THREE.Material).side = origSide;
        if (hits.length === 0) return null;
        // Don't trust hits[0].point — the mesh transform makes it wrong.
        // Use the hit distance along the camera ray instead.
        const dist = hits[0].distance;
        const pt = raycaster.ray.origin.clone().addScaledVector(raycaster.ray.direction, dist);
        return pt;
      };

      const raycastSplat = (
        ndcX: number,
        ndcY: number
      ): THREE.Vector3 | null => {
        ndc.set(ndcX, ndcY);
        raycaster.setFromCamera(ndc, camera);
        const hits: {
          distance: number;
          point: THREE.Vector3;
          object: THREE.Object3D;
        }[] = [];
        splat.raycast(raycaster, hits);
        if (hits.length === 0) return null;
        hits.sort((a, b) => a.distance - b.distance);
        return hits[0].point.clone();
      };

      // === Edit System ===

      const editHistory: THREE.Object3D[][] = [];

      const applyEdit = (
        worldCenter: THREE.Vector3,
        radius: number,
        edit: EditOp
      ): THREE.Object3D => {
        const [r, g, b] = resolveEditColor(edit);
        const sdf = new SplatEditSdf({
          type: SplatEditSdfType.SPHERE,
          radius,
          color: new THREE.Color(r, g, b),
        });
        sdf.position.copy(splat.worldToLocal(worldCenter.clone()));
        const splatEdit = new SplatEdit({
          rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
          sdfs: [sdf],
          softEdge: Math.min(0.4, radius * 0.4),
        });
        splatEdit.add(sdf);
        splat.add(splatEdit);
        return splatEdit;
      };

      const editAtHit = (
        hitPoint: THREE.Vector3 | null,
        edit: EditOp
      ): number => {
        if (!hitPoint) return 0;
        const radius = SIZE_MAP[edit.size] ?? SIZE_MAP.medium;
        const splatEdit = applyEdit(hitPoint, radius, edit);
        editHistory.push([splatEdit]);
        return 1;
      };

      const paintAt = (ndcX: number, ndcY: number, edit: EditOp): number => {
        const hit = raycastMesh(ndcX, ndcY) ?? raycastSplat(ndcX, ndcY);
        return editAtHit(hit, edit);
      };

      const undoLastEdit = () => {
        const group = editHistory.pop();
        if (!group) return;
        for (const ed of group) ed.removeFromParent();
      };
      const clearAllEdits = () => {
        while (editHistory.length > 0) {
          const group = editHistory.pop();
          if (group) for (const ed of group) ed.removeFromParent();
        }
      };

      const canvas = renderer.domElement;

      apiRef.current = {
        snapshot: () => {
          renderer.render(scene, camera);
          return {
            dataUrl: canvas.toDataURL("image/png"),
            width: canvas.width,
            height: canvas.height,
          };
        },
        paintAt,
      };

      editActionsRef.current = {
        undo: undoLastEdit,
        clear: clearAllEdits,
      };

      // === Input Handlers (WASD character + orbit-follow camera) ===

      const ORBIT_SENS = 0.008;
      const PITCH_MIN = 0.1;
      const PITCH_MAX = Math.PI / 2 - 0.05;
      let dragging = false;
      let painting = false;
      let pointerDownX = 0;
      let pointerDownY = 0;
      let pointerMoved = false;

      const doPaint = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const cx = (e.clientX - rect.left) / rect.width;
        const cy = (e.clientY - rect.top) / rect.height;
        const edit: EditOp = {
          color: paintColorRef.current,
          op: paintOpRef.current,
          size: paintSizeRef.current,
        };
        paintAt(cx * 2 - 1, -(cy * 2 - 1), edit);
      };

      const getCharacterPosition = () => new THREE.Vector3(physX, physY, physZ);

      const getCarryPosition = () => {
        const carry = new THREE.Vector3(physX, physY + ARTIFACT_HALF * 2.4, physZ);
        for (const name of carryBoneCandidates) {
          const bone = boneByName.get(name);
          if (!bone) continue;
          bone.updateMatrixWorld(true);
          bone.getWorldPosition(carry);
          carry.y += ARTIFACT_HALF * 0.65;
          return carry;
        }
        return carry;
      };

      const pickUpArtifact = (artifact: WalkArtifact) => {
        selectedArtifactId = artifact.id;
        carriedArtifactId = null;
        artifact.state = "pickup";
        artifact.settleUntil = 0;
        const carry = getCarryPosition();
        artifact.body?.setPhysicsEnabled(false);
        artifact.transitionFrom.copy(artifact.group.position);
        artifact.transitionTo.copy(carry);
        artifact.transitionStart = performance.now();
        artifact.transitionDuration = ARTIFACT_PICKUP_MS;
      };

      const placeCarriedArtifact = (point: THREE.Vector3) => {
        if (!carriedArtifactId) return;
        const artifact = artifactById.get(carriedArtifactId);
        if (!artifact) return;
        const releaseY = point.y + ARTIFACT_HALF + 0.025;
        artifact.state = "settle";
        artifact.transitionFrom.copy(artifact.group.position);
        artifact.transitionTo.set(point.x, releaseY, point.z);
        artifact.transitionStart = performance.now();
        artifact.transitionDuration = ARTIFACT_DROP_MS;
        artifact.settleUntil = artifact.transitionStart + ARTIFACT_DROP_MS + 500;
        selectedArtifactId = artifact.id;
        carriedArtifactId = null;
      };

      const selectArtifact = (artifact: WalkArtifact) => {
        if (carriedArtifactId || artifact.delivered) return;
        selectedArtifactId = artifact.id;
        artifact.state = "selected";
        const dist = artifact.group.position.distanceTo(getCharacterPosition());
        if (dist <= ARTIFACT_PICKUP_RADIUS) pickUpArtifact(artifact);
      };

      const handleSceneClick = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const cx = (e.clientX - rect.left) / rect.width;
        const cy = (e.clientY - rect.top) / rect.height;
        const ndcX = cx * 2 - 1;
        const ndcY = -(cy * 2 - 1);

        if (!deliveryModeRef.current) return;

        if (carriedArtifactId) {
          const hit = raycastMesh(ndcX, ndcY) ?? raycastSplat(ndcX, ndcY);
          if (hit) placeCarriedArtifact(hit);
          return;
        }

        if (ramp.selected) {
          const hit = raycastMesh(ndcX, ndcY) ?? raycastSplat(ndcX, ndcY);
          if (hit) {
            placeRamp(hit);
            ramp.selected = false;
            rampOutline.visible = false;
            setRampSelectedUI(false);
          }
          return;
        }

        if (raycastRamp(ndcX, ndcY)) {
          ramp.selected = true;
          rampOutline.visible = true;
          setRampSelectedUI(true);
          return;
        }

        const artifact = raycastArtifact(ndcX, ndcY);
        if (artifact) {
          selectArtifact(artifact);
          return;
        }

        const selected = selectedArtifactId ? artifactById.get(selectedArtifactId) : null;
        if (selected?.state === "selected") selected.state = "ground";
        selectedArtifactId = null;
        ramp.selected = false;
        rampOutline.visible = false;
        setRampSelectedUI(false);
      };

      const onMouseDown = (e: MouseEvent) => {
        if (e.button !== 0) return;
        if (paintModeRef.current) {
          painting = true;
          doPaint(e);
          return;
        }
        pointerDownX = e.clientX;
        pointerDownY = e.clientY;
        pointerMoved = false;
        dragging = true;
      };
      const onMouseUp = (e: MouseEvent) => {
        if (e.button !== 0) return;
        const wasClick = dragging && !pointerMoved;
        painting = false;
        dragging = false;
        if (wasClick) handleSceneClick(e);
      };
      const onMouseMove = (e: MouseEvent) => {
        if (painting && paintModeRef.current) {
          doPaint(e);
          return;
        }
        if (!dragging) return;
        if (
          Math.abs(e.clientX - pointerDownX) > 4 ||
          Math.abs(e.clientY - pointerDownY) > 4
        ) {
          pointerMoved = true;
        }
        camYaw -= e.movementX * ORBIT_SENS;
        camPitch += e.movementY * ORBIT_SENS;
        if (camPitch < PITCH_MIN) camPitch = PITCH_MIN;
        if (camPitch > PITCH_MAX) camPitch = PITCH_MAX;
      };
      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        camDist *= 1 + e.deltaY * 0.001;
        if (camDist < CAMERA_MIN_DISTANCE) camDist = CAMERA_MIN_DISTANCE;
        if (camDist > CAMERA_MAX_DISTANCE) camDist = CAMERA_MAX_DISTANCE;
      };

      canvas.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mouseup", onMouseUp);
      window.addEventListener("mousemove", onMouseMove);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      const isTypingTarget = (target: EventTarget | null) =>
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable);

      const keys: Record<string, boolean> = {};
      const onKeyDown = (e: KeyboardEvent) => {
        if (e.code === "Escape") return; // handled by React
        if (isTypingTarget(e.target)) return;
        if (e.code === "KeyZ") {
          if (e.shiftKey) clearAllEdits();
          else undoLastEdit();
          e.preventDefault();
          return;
        }
        if (e.code === "KeyM") {
          const next = ((viewModeRef.current + 1) % 3) as 0 | 1 | 2;
          viewModeRef.current = next;
          if (meshRef.current) meshRef.current.visible = next === 0 || next === 1;
          if (splatRef.current) splatRef.current.visible = next === 1 || next === 2;
          setViewMode(next);
          e.preventDefault();
          return;
        }
        keys[e.code] = true;
      };
      const onKeyUp = (e: KeyboardEvent) => {
        if (isTypingTarget(e.target)) {
          keys[e.code] = false;
          return;
        }
        keys[e.code] = false;
      };
      window.addEventListener("keydown", onKeyDown);
      window.addEventListener("keyup", onKeyUp);

      const onResize = () => {
        if (!mount) return;
        camera.aspect = mount.clientWidth / mount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, WALKER_MAX_PIXEL_RATIO));
        renderer.setSize(mount.clientWidth, mount.clientHeight);
      };
      window.addEventListener("resize", onResize);

      // === Animation Loop ===

      const _charWorldPos = new THREE.Vector3();
      const _artifactPos = new THREE.Vector3();
      const _rampQuat = new THREE.Quaternion();
      let last = performance.now();

      const getCameraAnchorWorldPosition = (target: THREE.Vector3) => {
        for (const name of CAMERA_ANCHOR_BONES) {
          const bone = boneByName.get(name);
          if (!bone) continue;
          bone.getWorldPosition(target);
          return target;
        }
        target.set(physX, physY + CAMERA_ANCHOR_FALLBACK_Y, physZ);
        return target;
      };

      const easeInOut = (t: number) =>
        t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

      const finishDeliveryIfNeeded = (artifact: WalkArtifact) => {
        if (artifact.delivered) return;
        const flatDist = Math.hypot(
          artifact.transitionTo.x - deliveryPosition.x,
          artifact.transitionTo.z - deliveryPosition.z,
        );
        if (flatDist > DELIVERY_RADIUS) return;
        artifact.delivered = true;
        deliveredCount += 1;
        setDeliveryDelivered(deliveredCount);
        artifact.coreMaterial.color.setHex(0xffffff);
        artifact.coreMaterial.emissive.setHex(0x36d399);
        artifact.haloMaterial.color.setHex(0xffffff);
        artifact.focusMaterial.color.setHex(0x36d399);
      };

      const syncArtifacts = (now: number, dt: number) => {
        if (!deliveryModeRef.current) return;
        const selected = selectedArtifactId ? artifactById.get(selectedArtifactId) : null;
        if (selected && selected.state === "selected") {
          const dist = selected.group.position.distanceTo(getCharacterPosition());
          if (dist <= ARTIFACT_PICKUP_RADIUS) pickUpArtifact(selected);
        }

        const carry = carriedArtifactId ? artifactById.get(carriedArtifactId) : null;
        const carryPosition = carry ? getCarryPosition() : null;
        const followAlpha = 1 - Math.exp(-dt * 18);

        for (const artifact of artifacts) {
          if (artifact.state === "pickup") {
            const t = Math.min(
              1,
              (now - artifact.transitionStart) / artifact.transitionDuration,
            );
            artifact.transitionTo.copy(getCarryPosition());
            artifact.group.position.lerpVectors(
              artifact.transitionFrom,
              artifact.transitionTo,
              easeInOut(t),
            );
            if (t >= 1) {
              artifact.state = "carried";
              carriedArtifactId = artifact.id;
            }
          } else if (artifact.state === "settle") {
            if (artifact.transitionDuration > 0) {
              const t = Math.min(
                1,
                (now - artifact.transitionStart) / artifact.transitionDuration,
              );
              artifact.group.position.lerpVectors(
                artifact.transitionFrom,
                artifact.transitionTo,
                easeInOut(t),
              );
              if (t >= 1 && artifact.body) {
                artifact.body.setPhysicsEnabled(true);
                artifact.body.setDynamicPosition(
                  artifact.transitionTo.x,
                  artifact.transitionTo.y,
                  artifact.transitionTo.z,
                );
                artifact.body.setDynamic({ x: 0, y: -0.12, z: 0 });
                artifact.transitionDuration = 0;
                finishDeliveryIfNeeded(artifact);
              }
            } else if (artifact.body) {
              const p = artifact.body.getPosition();
              _artifactPos.set(p.x, p.y, p.z);
              artifact.group.position.lerp(_artifactPos, followAlpha);
            }
          } else if (artifact === carry && carryPosition) {
            artifact.group.position.lerp(carryPosition, followAlpha);
          } else if (artifact.body) {
            const p = artifact.body.getPosition();
            _artifactPos.set(p.x, p.y, p.z);
            artifact.group.position.copy(_artifactPos);
          }

          if (artifact.state === "settle" && now >= artifact.settleUntil) {
            artifact.state =
              selectedArtifactId === artifact.id && !artifact.delivered ? "selected" : "ground";
            artifact.settleUntil = 0;
          }

          const isSelected =
            selectedArtifactId === artifact.id ||
            carriedArtifactId === artifact.id ||
            artifact.state === "selected";
          const pulse = 1 + Math.sin(now * 0.007) * 0.08;
          artifact.coreMaterial.emissiveIntensity = isSelected ? 0.62 : 0.3;
          artifact.haloMaterial.opacity = isSelected ? 0.9 : 0.55;
          artifact.focusMaterial.opacity = isSelected ? 0.6 : 0;
          artifact.group.rotation.y += dt * (artifact.state === "carried" ? 1.5 : 0.45);
          artifact.group.scale.setScalar(isSelected ? pulse : 1);
        }

        const completion = deliveredCount / artifacts.length;
        if (deliveredCount !== lastDeliveredCount) {
          lastDeliveredCount = deliveredCount;
          setDeliveryDelivered(deliveredCount);
        }
        deliveryMaterial.opacity = 0.45 + completion * 0.45;
        deliveryFillMaterial.opacity = 0.08 + completion * 0.2;
        deliveryBeaconMaterial.opacity = 0.14 + completion * 0.24;
        deliveryZone.scale.setScalar(1 + Math.sin(now * 0.005) * (0.025 + completion * 0.03));
      };

      const syncRamp = () => {
        if (!deliveryModeRef.current) return;
        const pose = ramp.body?.getPosition();
        if (!pose || ramp.selected) return;
        ramp.group.position.set(
          pose.translation.x,
          pose.translation.y - RAMP_HEIGHT / 2,
          pose.translation.z,
        );
        _rampQuat.set(
          pose.rotation.x,
          pose.rotation.y,
          pose.rotation.z,
          pose.rotation.w,
        );
        ramp.group.quaternion.copy(_rampQuat);
      };

      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        let physicsMoved = false;

        // === WASD → drive locomotion engine ===
        if (locoEngine && locoEngine.ready && characterRoot?.visible) {
          const speed = walkSpeedRef.current;

          // Compute movement direction from camera yaw (world-space XZ)
          // Camera yaw orbits around the character; W = toward where camera faces
          const worldFwdX = -Math.sin(camYaw);
          const worldFwdZ = -Math.cos(camYaw);
          const worldRightX = Math.cos(camYaw);
          const worldRightZ = -Math.sin(camYaw);

          let moveX = 0, moveZ = 0;
          if (keys.KeyW || keys.ArrowUp)    { moveX += worldFwdX;   moveZ += worldFwdZ; }
          if (keys.KeyS || keys.ArrowDown)  { moveX -= worldFwdX;   moveZ -= worldFwdZ; }
          if (keys.KeyA || keys.ArrowLeft)  { moveX -= worldRightX; moveZ -= worldRightZ; }
          if (keys.KeyD || keys.ArrowRight) { moveX += worldRightX; moveZ += worldRightZ; }

          const moveMag = Math.sqrt(moveX * moveX + moveZ * moveZ);
          if (moveMag > 0.01) {
            const nx = moveX / moveMag;
            const nz = moveZ / moveMag;
            locoEngine.setMovement(
              [nx * speed, 0, nz * speed],
              [nx, 0, nz]
            );
          } else {
            locoEngine.setIdle();
          }

          locoEngine.update(dt);

          // Use Hips bone (bone 0) as anchor instead of engine root.
          // Engine root and bone positions are updated by different paths in animate()
          // (root from trajectory, bones from 50/50 velocity/position blend), so
          // subtracting engine root from bone positions amplifies their desync → oscillation.
          // Using the Hips bone keeps all offsets bone-relative and coherent.
          const bones = locoEngine.getBoneData();
          const rawHips = bones[0].position;

          // Smooth the hips anchor — raw hips jitter from neural net predictions
          // is invisible on character page but amplified by the (bone - hips) subtraction here
          if (!smoothHipsInit) {
            smoothHipsX = rawHips[0]; smoothHipsY = rawHips[1]; smoothHipsZ = rawHips[2];
            smoothHipsInit = true;
          } else {
            const s = 1 - Math.exp(-dt * 15); // ~15 Hz — tracks movement, kills tremor
            smoothHipsX += (rawHips[0] - smoothHipsX) * s;
            smoothHipsY += (rawHips[1] - smoothHipsY) * s;
            smoothHipsZ += (rawHips[2] - smoothHipsZ) * s;
          }
          const hipsPos: [number, number, number] = [smoothHipsX, smoothHipsY, smoothHipsZ];

          // Compute hips delta since last frame (in scene units)
          const engineDX = (smoothHipsX - prevEngineX) * CHAR_SCALE;
          const engineDZ = (smoothHipsZ - prevEngineZ) * CHAR_SCALE;
          prevEngineX = smoothHipsX;
          prevEngineZ = smoothHipsZ;

          // Feed delta through Rapier KCC — handles ground + walls
          if (physics) {
            const result = physics.move(
              physX, physY, physZ,
              engineDX, 0, engineDZ,
              dt,
            );
            physX = result.x;
            physY = result.y;
            physZ = result.z;
            physicsMoved = true;
          } else {
            // Fallback: no physics, just track hips position directly
            physX = localAnchor.x + hipsPos[0] * CHAR_SCALE;
            physZ = localAnchor.z + hipsPos[2] * CHAR_SCALE;
          }

          // Scale bone positions relative to Hips, then place at physics position
          for (const b of bones) {
            const bone = boneByName.get(b.name);
            if (!bone) continue;
            const bx = physX + (b.position[0] - hipsPos[0]) * VIS_SCALE;
            const by = physY + b.position[1] * VIS_SCALE;
            const bz = physZ + (b.position[2] - hipsPos[2]) * VIS_SCALE;
            applyWorldToBone(
              bone,
              bx, by, bz,
              b.quaternion[0], b.quaternion[1], b.quaternion[2], b.quaternion[3],
            );
          }

          // Track scene position for camera follow
          characterRoot.position.set(physX, physY, physZ);

          // === Camera ===
          characterContainer.updateMatrixWorld(false);
          getCameraAnchorWorldPosition(_charWorldPos);
          if (!orbitTargetInit) {
            orbitTarget.copy(_charWorldPos);
            orbitTargetInit = true;
          } else if (cameraFollowRef.current) {
            const followDeltaSq = orbitTarget.distanceToSquared(_charWorldPos);
            if (followDeltaSq > CAMERA_FOLLOW_DEADBAND_M * CAMERA_FOLLOW_DEADBAND_M) {
              orbitTarget.lerp(_charWorldPos, 1 - Math.exp(-dt * CAMERA_FOLLOW_HZ));
            }
          }

          const camOffX = camDist * Math.cos(camPitch) * Math.sin(camYaw);
          const camOffY = camDist * Math.sin(camPitch);
          const camOffZ = camDist * Math.cos(camPitch) * Math.cos(camYaw);
          camera.position.set(
            orbitTarget.x + camOffX,
            orbitTarget.y + camOffY,
            orbitTarget.z + camOffZ,
          );
          camera.lookAt(orbitTarget.x, orbitTarget.y, orbitTarget.z);
        }

        if (physics && !physicsMoved) physics.step();
        syncArtifacts(now, dt);
        syncRamp();

        renderer.render(scene, camera);
      });

      // === Cleanup ===

      cleanup = () => {
        apiRef.current = null;
        meshRef.current = null;
        window.removeEventListener("resize", onResize);
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup", onKeyUp);
        canvas.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mouseup", onMouseUp);
        window.removeEventListener("mousemove", onMouseMove);
        canvas.removeEventListener("wheel", onWheel);
        if (deliveryRuntimeRef.current?.setEnabled === setDeliveryGameEnabled) {
          deliveryRuntimeRef.current = null;
        }
        for (const artifact of artifacts) artifact.body?.dispose();
        ramp.body?.dispose();
        physics?.dispose();
        splat.dispose();
        renderer.setAnimationLoop(null);
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [splatUrl, splatPaged, colliderMeshUrl, metricScale]);

  const cycleViewMode = useCallback(() => {
    const next = ((viewModeRef.current + 1) % 3) as 0 | 1 | 2;
    viewModeRef.current = next;
    if (meshRef.current) meshRef.current.visible = next === 0 || next === 1;
    if (splatRef.current) splatRef.current.visible = next === 1 || next === 2;
    setViewMode(next);
  }, []);

  // ── Edit intent submit ──

  const onIntentSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = intent.trim();
    const api = apiRef.current;
    if (!text || submitting || !api) return;
    setSubmitting(true);
    setError(null);
    try {
      const snap = api.snapshot();
      const res = await fetch(`${API_BASE}/edit-intent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          screenshot: snap.dataUrl,
          intent: text,
          width: snap.width,
          height: snap.height,
          userPrompt: userPrompt ?? null,
          sceneCaption: sceneCaption ?? null,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = (await res.json()) as {
        x?: number;
        y?: number;
        box?: { nx1: number; ny1: number; nx2: number; ny2: number };
        color?: string | null;
        op?: string;
        size?: string;
        reason?: string | null;
        usage?: Record<string, unknown> | null;
        error?: string;
      };
      if (data.error) throw new Error(data.error);
      if (
        typeof data.x !== "number" ||
        typeof data.y !== "number" ||
        !data.box
      ) {
        throw new Error("invalid response");
      }
      const edit: EditOp = {
        color: data.color ?? null,
        op: data.op ?? "recolor",
        size: data.size ?? "medium",
      };
      const cx = (data.box.nx1 + data.box.nx2) / 2;
      const cy = (data.box.ny1 + data.box.ny2) / 2;
      api.paintAt(cx * 2 - 1, -(cy * 2 - 1), edit);
      setLastEdit({ reason: data.reason ?? null, usage: data.usage ?? null });
      setIntent("");
      setMenuOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <div
        ref={mountRef}
        className="h-full w-full"
        style={paintMode ? { cursor: "crosshair" } : undefined}
      />

      {(splatLoading || charLoading) && (
        <LoadingParticles
          label={splatLoading ? "loading splat" : "loading character"}
          className="z-30"
        />
      )}

      {/* HUD — always visible when menu is closed */}
      {!menuOpen && !splatLoading && (
        <div style={HUD_FONT} className="pointer-events-none">
          {/* top-left: back */}
          <Link
            href={backHref}
            aria-label="back"
            className={`pointer-events-auto absolute left-5 top-5 z-10 ${HUD_BOX_SQUARE}`}
          >
            <IconArrowLeft />
          </Link>

          {/* top-right: delivery objective */}
          {deliveryMode && (
            <div className="absolute right-5 top-5 z-10">
              <div className="rounded-2xl border border-white/10 bg-black/55 px-4 py-3 text-right backdrop-blur-md">
                <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-400">Delivered</div>
                <div className="mt-1 text-[22px] leading-none text-white">
                  {deliveryDelivered}/3
                </div>
                <div
                  className={`mt-2 h-1.5 w-28 overflow-hidden rounded-full bg-white/10 ${
                    deliveryDelivered === 3 ? "ring-1 ring-emerald-300/50" : ""
                  }`}
                >
                  <div
                    className="h-full rounded-full bg-emerald-300 transition-all duration-300"
                    style={{ width: `${(deliveryDelivered / 3) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* left-center: ramp placement state */}
          {deliveryMode && rampSelectedUI && (
            <div className="absolute left-5 top-1/2 z-10 -translate-y-1/2">
              <div className="rounded-2xl border border-cyan-200/20 bg-black/55 px-4 py-3 backdrop-blur-md">
                <div className="text-[10px] uppercase tracking-[0.16em] text-cyan-200">Ramp</div>
                <div className="mt-1 text-[13px] text-white">Click floor to place</div>
              </div>
            </div>
          )}

          {deliveryMode && deliveryDelivered === 3 && (
            <div className="absolute left-1/2 top-[88px] z-10 -translate-x-1/2">
              <div className="rounded-2xl border border-emerald-200/25 bg-black/60 px-5 py-3 text-center backdrop-blur-md">
                <div className="text-[10px] uppercase tracking-[0.18em] text-emerald-200">Complete</div>
                <div className="mt-1 text-[14px] text-white">All artifacts delivered</div>
              </div>
            </div>
          )}

          {/* bottom-center: action tray */}
          <div className="pointer-events-auto absolute bottom-6 left-1/2 z-10 -translate-x-1/2">
            <div className="flex items-center gap-1 rounded-2xl border border-white/10 bg-black/55 p-1.5 backdrop-blur-md">
              <button
                onClick={() => editActionsRef.current?.undo()}
                aria-label="undo"
                title="undo (z)"
                className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
              >
                <IconUndo />
              </button>
              <button
                onClick={() => editActionsRef.current?.clear()}
                aria-label="clear edits"
                title="clear all (shift+z)"
                className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
              >
                <IconTrash />
              </button>
              <div className="mx-1 h-5 w-px bg-white/10" />
              <button
                onClick={() => { setMenuTab("edit"); setMenuOpen(true); }}
                aria-label="edit"
                title="edit world"
                className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                  paintMode
                    ? "bg-white/15 text-white"
                    : "text-zinc-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                <IconBrush />
              </button>
              <button
                onClick={cycleViewMode}
                aria-label="view mode"
                title={
                  viewMode === 0
                    ? "view: mesh — press m for splat+mesh"
                    : viewMode === 1
                    ? "view: splat + mesh — press m for splat"
                    : "view: splat — press m for mesh"
                }
                className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                  viewMode !== 0
                    ? "bg-white/15 text-white"
                    : "text-zinc-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                <IconGrid />
              </button>
              <button
                onClick={() => {
                  const next = !cameraFollowRef.current;
                  cameraFollowRef.current = next;
                  setCameraFollowUI(next);
                }}
                aria-label="camera follow"
                title={cameraFollowUI ? "camera: follow" : "camera: locked"}
                className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                  cameraFollowUI
                    ? "bg-white/15 text-white"
                    : "text-zinc-300 hover:bg-white/10 hover:text-white"
                }`}
              >
                <IconCamera />
              </button>
              <button
                onClick={() => { setMenuTab("controls"); setMenuOpen(true); }}
                aria-label="controls"
                title="controls"
                className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
              >
                <IconKeyboard />
              </button>
            </div>
          </div>
        </div>
      )}


      {/* Menu */}
      {menuOpen && (
        <div
          style={HUD_FONT}
          className="absolute inset-0 z-20 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) setMenuOpen(false);
          }}
        >
          <div className="h-[560px] w-[500px] overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl">
            {/* tab bar */}
            <div className="flex items-center justify-between border-b border-white/10 px-2 py-2">
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setMenuTab("edit")}
                  className={`rounded-md px-3 py-1.5 text-[13px] ${
                    menuTab === "edit"
                      ? "bg-white/10 text-white"
                      : "text-zinc-400 hover:text-white"
                  }`}
                >
                  Edit
                </button>
                <button
                  onClick={() => setMenuTab("game")}
                  className={`rounded-md px-3 py-1.5 text-[13px] ${
                    menuTab === "game"
                      ? "bg-white/10 text-white"
                      : "text-zinc-400 hover:text-white"
                  }`}
                >
                  Game
                </button>
                <button
                  onClick={() => setMenuTab("controls")}
                  className={`rounded-md px-3 py-1.5 text-[13px] ${
                    menuTab === "controls"
                      ? "bg-white/10 text-white"
                      : "text-zinc-400 hover:text-white"
                  }`}
                >
                  Controls
                </button>
              </div>
              <button
                onClick={toggleMenu}
                aria-label="close"
                className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-white/10 hover:text-white"
              >
                <IconClose />
              </button>
            </div>

            {/* content */}
            <div className="h-[507px] p-5">
              {menuTab === "edit" && (
                <div className="space-y-5">
                  {/* Paint mode toggle */}
                  <button
                    onClick={() => { setPaintMode((p) => !p); setMenuOpen(false); }}
                    className={`w-full rounded-lg border px-4 py-2.5 text-[13px] transition ${
                      paintMode
                        ? "border-white/30 bg-white/15 text-white"
                        : "border-white/10 bg-white/5 text-zinc-400 hover:border-white/20 hover:text-white"
                    }`}
                  >
                    {paintMode ? "Painting — click & drag on world" : "Enable paintbrush"}
                  </button>

                  {/* Color */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Color</div>
                    <div className="flex flex-wrap gap-2">
                      {Object.keys(COLOR_CSS).map((c) => (
                        <button
                          key={c}
                          onClick={() => { setPaintColor(c); setPaintOp("recolor"); }}
                          title={c}
                          className={`h-7 w-7 rounded-full border-2 transition ${
                            paintColor === c && paintOp === "recolor"
                              ? "border-white scale-110"
                              : "border-transparent hover:border-white/40"
                          }`}
                          style={{ backgroundColor: COLOR_CSS[c] }}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Tool */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Tool</div>
                    <div className="flex gap-2">
                      {(["recolor", "erase", "brighten", "darken"] as const).map((op) => (
                        <button
                          key={op}
                          onClick={() => setPaintOp(op)}
                          className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                            paintOp === op
                              ? "border-white/30 bg-white/10 text-white"
                              : ""
                          }`}
                        >
                          {op}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Size */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Size</div>
                    <div className="flex gap-2">
                      {Object.keys(SIZE_MAP).map((s) => (
                        <button
                          key={s}
                          onClick={() => setPaintSize(s)}
                          className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                            paintSize === s
                              ? "border-white/30 bg-white/10 text-white"
                              : ""
                          }`}
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* AI edit */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">AI Edit</div>
                    <form onSubmit={onIntentSubmit}>
                      <input
                        type="text"
                        value={intent}
                        onChange={(e) => setIntent(e.target.value)}
                        placeholder={submitting ? "thinking…" : "make the table blue"}
                        disabled={submitting}
                        className="w-full rounded-lg border border-white/10 bg-white/5 px-3.5 py-2.5 text-[13px] text-white placeholder:text-zinc-500 focus:border-white/30 focus:outline-none disabled:opacity-50"
                      />
                      {error && menuTab === "edit" && (
                        <div className="mt-3 text-[12px] text-red-400">{error}</div>
                      )}
                      {lastEdit?.reason && !submitting && !error && (
                        <div className="mt-3 text-[12px] leading-relaxed text-zinc-400">
                          {lastEdit.reason}
                        </div>
                      )}
                    </form>
                  </div>
                </div>
              )}

              {menuTab === "game" && (
                <div>
                  <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Games</div>
                  <button
                    onClick={() => setDeliveryMode((enabled) => !enabled)}
                    className={`flex w-full items-center justify-between rounded-lg border px-4 py-3 text-left transition ${
                      deliveryMode
                        ? "border-emerald-200/30 bg-emerald-300/15 text-emerald-100"
                        : "border-white/10 bg-white/5 text-zinc-300 hover:border-white/20 hover:text-white"
                    }`}
                  >
                    <span className="text-[14px] text-white">Artifact Delivery</span>
                    <span className="text-[12px]">{deliveryMode ? "On" : "Off"}</span>
                  </button>
                </div>
              )}

              {menuTab === "controls" && (
                <div className="space-y-4">
                  {/* Keys */}
                  <div className="space-y-2 text-[13px]">
                    <div className="flex justify-between">
                      <span className="text-white">Walk</span>
                      <span className="text-zinc-400">wasd</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">Orbit camera</span>
                      <span className="text-zinc-400">click + drag</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">Zoom</span>
                      <span className="text-zinc-400">scroll</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">Undo / clear</span>
                      <span className="text-zinc-400">z / shift+z</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">Mesh overlay</span>
                      <span className="text-zinc-400">m</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">Game options</span>
                      <span className="text-zinc-400">Game tab</span>
                    </div>
                  </div>

                  {/* Character */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Character</div>
                    <div className="flex gap-2">
                      {CHARACTERS.map((c) => (
                        <button
                          key={c.label}
                          onClick={() => loadCharacter(c)}
                          disabled={charLoading}
                          className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                            activeChar === c.label
                              ? "border-white/30 bg-white/10 text-white"
                              : ""
                          }`}
                        >
                          {c.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Speed */}
                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Speed</div>
                    <div className="flex gap-2">
                      {SPEED_TIERS.map((t) => (
                        <button
                          key={t.label}
                          onClick={() => pickSpeed(t.label, t.value)}
                          className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                            activeSpeed === t.label
                              ? "border-white/30 bg-white/10 text-white"
                              : ""
                          }`}
                        >
                          {t.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Style */}
                  {styles.length > 0 && (
                    <div>
                      <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Style</div>
                      <div className="flex flex-wrap gap-2">
                        {styles.map((s) => (
                          <button
                            key={s}
                            onClick={() => pickStyle(s)}
                            className={`${HUD_BOX_BASE} h-8 px-3 text-[12px] ${
                              activeStyle === s
                                ? "border-white/30 bg-white/10 text-white"
                                : ""
                            }`}
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

    </>
  );
}

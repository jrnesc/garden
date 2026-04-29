"use client";

import type * as THREE from "three";
import { useEffect, useRef, useState, useCallback } from "react";
import { API_BASE } from "@/lib/api";
import { loadColliderMesh } from "@/lib/collider";
import { LocomotionEngine, type LocoData } from "@/lib/locomotion";
import { createPhysicsWorld, type PhysicsWorld } from "@/lib/physics";
import Link from "next/link";
import {
  IconArrowLeft,
  IconBrush,
  IconCamera,
  IconClose,
  IconGrid,
  IconKeyboard,
  IconMessage,
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
  const cameraFollowRef = useRef(true);
  const [cameraFollowUI, setCameraFollowUI] = useState(true);
  const [quickIntent, setQuickIntent] = useState("");
  const [quickSubmitting, setQuickSubmitting] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuTab, setMenuTab] = useState<"edit" | "ask" | "controls">("edit");
  const [activeChar, setActiveChar] = useState("Dog");
  const [activeStyle, setActiveStyle] = useState("Walk");
  const [activeSpeed, setActiveSpeed] = useState("Walk");
  const [styles, setStyles] = useState<string[]>([]);
  const [charLoading, setCharLoading] = useState(false);
  const walkSpeedRef = useRef(1.0);
  const engineRef = useRef<LocomotionEngine | null>(null);
  const loadCharRef = useRef<((def: CharacterDef) => Promise<void>) | null>(null);
  const [intent, setIntent] = useState("");
  const [question, setQuestion] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastEdit, setLastEdit] = useState<{
    reason: string | null;
    usage: Record<string, unknown> | null;
  } | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [askSubmitting, setAskSubmitting] = useState(false);
  const [paintMode, setPaintMode] = useState(false);
  const [paintColor, setPaintColor] = useState("red");
  const [paintOp, setPaintOp] = useState("recolor");
  const [paintSize, setPaintSize] = useState("medium");
  const paintModeRef = useRef(false);
  const paintOpRef = useRef("recolor");
  const paintColorRef = useRef("red");
  const paintSizeRef = useRef("medium");

  // Sync paint refs so canvas event handlers can read current values
  useEffect(() => { paintModeRef.current = paintMode; }, [paintMode]);
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
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      mount.appendChild(renderer.domElement);

      const spark = new SparkRenderer({
        renderer,
        maxPixelRadius: 2,
        maxStdDev: 1.0,
        falloff: 0,
        lodRenderScale: 3,
        minSortIntervalMs: 16,
      });
      scene.add(spark);

      const splat = new SplatMesh({
        url: splatUrl,
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
          spark.maxPixelRadius = 512;
          spark.maxStdDev = Math.sqrt(8);
          spark.falloff = 1;
          spark.lodRenderScale = 2.5;
          // splat loaded
        })
        .catch(() => {});

      // === Collider Mesh ===

      let colliderMesh: THREE.Mesh | null = null;

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
      let camPitch = 0.4;
      let camDist = 3.0;

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

      const onMouseDown = (e: MouseEvent) => {
        if (e.button !== 0) return;
        if (paintModeRef.current) {
          painting = true;
          doPaint(e);
          return;
        }
        dragging = true;
      };
      const onMouseUp = (e: MouseEvent) => {
        if (e.button !== 0) return;
        painting = false;
        dragging = false;
      };
      const onMouseMove = (e: MouseEvent) => {
        if (painting && paintModeRef.current) {
          doPaint(e);
          return;
        }
        if (!dragging) return;
        camYaw -= e.movementX * ORBIT_SENS;
        camPitch += e.movementY * ORBIT_SENS;
        if (camPitch < PITCH_MIN) camPitch = PITCH_MIN;
        if (camPitch > PITCH_MAX) camPitch = PITCH_MAX;
      };
      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        camDist *= 1 + e.deltaY * 0.001;
        if (camDist < 0.5) camDist = 0.5;
        if (camDist > 20) camDist = 20;
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
        renderer.setSize(mount.clientWidth, mount.clientHeight);
      };
      window.addEventListener("resize", onResize);

      // === Animation Loop ===

      const _charWorldPos = new THREE.Vector3();
      let last = performance.now();

      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;

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
          if (cameraFollowRef.current) {
            characterContainer.updateMatrixWorld(false);
            _charWorldPos.copy(characterRoot.position);
            characterContainer.localToWorld(_charWorldPos);

            const camOffX = camDist * Math.cos(camPitch) * Math.sin(camYaw);
            const camOffY = camDist * Math.sin(camPitch);
            const camOffZ = camDist * Math.cos(camPitch) * Math.cos(camYaw);
            camera.position.set(
              _charWorldPos.x + camOffX,
              _charWorldPos.y + camOffY,
              _charWorldPos.z + camOffZ,
            );
            camera.lookAt(_charWorldPos.x, _charWorldPos.y, _charWorldPos.z);
          }
        }

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
  }, [splatUrl, colliderMeshUrl, metricScale]);

  const cycleViewMode = useCallback(() => {
    const next = ((viewModeRef.current + 1) % 3) as 0 | 1 | 2;
    viewModeRef.current = next;
    if (meshRef.current) meshRef.current.visible = next === 0 || next === 1;
    if (splatRef.current) splatRef.current.visible = next === 1 || next === 2;
    setViewMode(next);
  }, []);

  const onQuickSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = quickIntent.trim();
    const api = apiRef.current;
    if (!text || quickSubmitting || !api) return;
    setQuickSubmitting(true);
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
      if (typeof data.x !== "number" || typeof data.y !== "number" || !data.box) {
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
      setQuickIntent("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setQuickSubmitting(false);
    }
  };

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

  // ── Ask the world ──

  const onAskSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = question.trim();
    const api = apiRef.current;
    if (!text || askSubmitting || !api) return;
    setAskSubmitting(true);
    setAnswer(null);
    setError(null);
    try {
      const snap = api.snapshot();
      const res = await fetch(`${API_BASE}/ask-world`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          screenshot: snap.dataUrl,
          question: text,
          width: snap.width,
          height: snap.height,
          sceneCaption: sceneCaption ?? null,
        }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
      const data = (await res.json()) as {
        answer?: string;
        error?: string;
      };
      if (data.error) throw new Error(data.error);
      setAnswer(data.answer ?? "no answer");
      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAskSubmitting(false);
    }
  };

  return (
    <>
      <div
        ref={mountRef}
        className="h-full w-full"
        style={paintMode ? { cursor: "crosshair" } : undefined}
      />

      {/* HUD — always visible when menu is closed */}
      {!menuOpen && (
        <div style={HUD_FONT} className="pointer-events-none">
          {/* top-left: back */}
          <Link
            href={backHref}
            aria-label="back"
            className={`pointer-events-auto absolute left-5 top-5 z-10 ${HUD_BOX_SQUARE}`}
          >
            <IconArrowLeft />
          </Link>

          {/* top-center: edit prompt */}
          <form
            onSubmit={onQuickSubmit}
            className="pointer-events-auto absolute left-1/2 top-5 z-10 -translate-x-1/2"
          >
            <div className="flex h-12 w-[min(560px,calc(100vw-280px))] items-center gap-2 rounded-2xl border border-white/10 bg-black/55 px-4 backdrop-blur-md focus-within:border-white/25">
              <input
                type="text"
                value={quickIntent}
                onChange={(e) => setQuickIntent(e.target.value)}
                placeholder={quickSubmitting ? "thinking…" : "describe a change…"}
                disabled={quickSubmitting}
                className="h-full flex-1 bg-transparent text-[14px] text-white placeholder:text-zinc-400 focus:outline-none disabled:opacity-50"
              />
            </div>
          </form>

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
              <div className="mx-1 h-5 w-px bg-white/10" />
              <button
                onClick={() => { setMenuTab("ask"); setMenuOpen(true); }}
                aria-label="ask"
                title="ask the world"
                className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-300 hover:bg-white/10 hover:text-white"
              >
                <IconMessage />
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
          <div className="w-[460px] overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl">
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
                  onClick={() => setMenuTab("ask")}
                  className={`rounded-md px-3 py-1.5 text-[13px] ${
                    menuTab === "ask"
                      ? "bg-white/10 text-white"
                      : "text-zinc-400 hover:text-white"
                  }`}
                >
                  Ask
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
            <div className="min-h-[200px] p-5">
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

              {menuTab === "ask" && (
                <form onSubmit={onAskSubmit}>
                  <p className="mb-3 text-[12px] leading-relaxed text-zinc-400">
                    Ask the world a question about what you see. It will look at your current view and answer.
                  </p>
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder={askSubmitting ? "looking…" : "what is that structure?"}
                    disabled={askSubmitting}
                    autoFocus
                    className="w-full rounded-lg border border-white/10 bg-white/5 px-3.5 py-2.5 text-[13px] text-white placeholder:text-zinc-500 focus:border-white/30 focus:outline-none disabled:opacity-50"
                  />
                  {error && menuTab === "ask" && (
                    <div className="mt-3 text-[12px] text-red-400">{error}</div>
                  )}
                  {answer && !askSubmitting && (
                    <div className="mt-4 text-[13px] leading-relaxed text-zinc-200">
                      {answer}
                    </div>
                  )}
                </form>
              )}

              {menuTab === "controls" && (
                <div className="space-y-5">
                  {/* Keys */}
                  <div className="space-y-2.5 text-[13px]">
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

"use client";

import type * as THREE from "three";
import { useEffect, useRef, useState, useCallback } from "react";
import Link from "next/link";
import { LocomotionEngine, type LocoData } from "@/lib/locomotion";
import LoadingParticles from "@/components/LoadingParticles";
import {
  IconArrowLeft,
  IconClose,
  IconKeyboard,
  HUD_FONT,
  HUD_BOX_BASE,
  HUD_BOX_SQUARE,
} from "@/components/hud-icons";
import { API_BASE } from "@/lib/api";

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
  { label: "Walk", value: 1.0 },
  { label: "Stride", value: 1.5 },
  { label: "Run", value: 2.5 },
  { label: "Sprint", value: 4.0 },
];

export default function CharacterPage() {
  const mountRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<LocomotionEngine | null>(null);
  const walkSpeedRef = useRef(1.0);
  const sceneRef = useRef<{
    THREE: typeof import("three");
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    boneByName: Map<string, THREE.Bone>;
    characterRoot: THREE.Group;
  } | null>(null);

  const [menuOpen, setMenuOpen] = useState(false);
  const [styles, setStyles] = useState<string[]>([]);
  const [activeStyle, setActiveStyle] = useState("Neutral");
  const [activeSpeed, setActiveSpeed] = useState("Walk");
  const [activeChar, setActiveChar] = useState("Geno");
  const [loading, setLoading] = useState(true);

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
    const s = sceneRef.current;
    if (!s) return;
    setLoading(true);

    // Remove old character
    if (s.characterRoot.children.length > 0) {
      s.characterRoot.clear();
    }
    s.boneByName.clear();

    const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");

    // Load new GLB
    const gltf = await new GLTFLoader().loadAsync(def.glb);
    s.characterRoot.add(gltf.scene);
    // Geno has no textures — use white. Dog/Wolf have baked textures — keep them.
    if (def.label === "Geno") {
      gltf.scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          (child as THREE.Mesh).material = new s.THREE.MeshBasicMaterial({ color: 0xffffff });
        }
      });
    }
    gltf.scene.traverse((child) => {
      if ((child as THREE.SkinnedMesh).isSkinnedMesh) {
        for (const bone of (child as THREE.SkinnedMesh).skeleton.bones) {
          s.boneByName.set(bone.name, bone);
        }
      }
    });

    // Load engine
    const dataResp = await fetch(def.data);
    const locoData: LocoData = await dataResp.json();
    const engine = new LocomotionEngine(locoData);
    await engine.loadModel(def.onnx);
    engine.setStyle(def.defaultStyle);
    engineRef.current = engine;

    setStyles(engine.getStyles());
    setActiveStyle(def.defaultStyle);
    setActiveChar(def.label);
    setLoading(false);
    console.log(`[char] Loaded ${def.label}`);
  }, []);

  useEffect(() => {
    document.body.classList.add("no-grain");
    return () => document.body.classList.remove("no-grain");
  }, []);

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
      if (disposed) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#111");
      const camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.01, 100);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      mount.appendChild(renderer.domElement);

      // Lighting for textured models
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(5, 10, 5);
      scene.add(dirLight);

      // Ground + grid
      const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(20, 20),
        new THREE.MeshBasicMaterial({ color: 0x222222 })
      );
      ground.rotation.x = -Math.PI / 2;
      scene.add(ground);
      scene.add(new THREE.GridHelper(20, 20, 0x444444, 0x333333));

      // Character container
      const characterRoot = new THREE.Group();
      scene.add(characterRoot);
      const boneByName = new Map<string, THREE.Bone>();

      sceneRef.current = { THREE, scene, camera, renderer, boneByName, characterRoot };

      // World→local bone transform temps
      const _wp = new THREE.Vector3();
      const _wq = new THREE.Quaternion();
      const _ws = new THREE.Vector3(1, 1, 1);
      const _wm = new THREE.Matrix4();
      const _pi = new THREE.Matrix4();
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
          _pi.copy(bone.parent.matrixWorld).invert();
          _lm.multiplyMatrices(_pi, _wm);
        } else {
          _lm.copy(_wm);
        }
        _lm.decompose(_lp, _lq, _ls);
        bone.position.copy(_lp);
        bone.quaternion.copy(_lq);
        bone.updateMatrix();
        bone.updateMatrixWorld(true);
      };

      // Load default character
      await loadCharacter(CHARACTERS[0]);

      // WASD
      const keys: Record<string, boolean> = {};
      const onKeyDown = (e: KeyboardEvent) => { keys[e.code] = true; };
      const onKeyUp = (e: KeyboardEvent) => { keys[e.code] = false; };
      window.addEventListener("keydown", onKeyDown);
      window.addEventListener("keyup", onKeyUp);

      // Click-to-walk
      let walkTarget: THREE.Vector3 | null = null;
      const ARRIVE_THRESHOLD = 0.25;
      const raycaster = new THREE.Raycaster();
      const ndcMouse = new THREE.Vector2();
      const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const _hitPoint = new THREE.Vector3();

      const markerGeo = new THREE.RingGeometry(0.08, 0.14, 24);
      markerGeo.rotateX(-Math.PI / 2);
      const marker = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({
        color: 0x888888, transparent: true, opacity: 0.5, depthWrite: false,
      }));
      marker.visible = false;
      scene.add(marker);

      // Camera orbit
      let dragging = false;
      let dragDist = 0;
      let prevX = 0, prevY = 0;
      let camAngle = 0, camPitch = 0.4, camDist = 5;
      const onDown = (e: MouseEvent) => {
        dragging = true; dragDist = 0;
        prevX = e.clientX; prevY = e.clientY;
      };
      const onUp = (e: MouseEvent) => {
        if (dragDist < 4) {
          const rect = renderer.domElement.getBoundingClientRect();
          ndcMouse.set(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1
          );
          raycaster.setFromCamera(ndcMouse, camera);
          const t = raycaster.ray.distanceToPlane(groundPlane);
          if (t !== null && t > 0) {
            raycaster.ray.at(t, _hitPoint);
            walkTarget = _hitPoint.clone();
            marker.position.set(walkTarget.x, 0.01, walkTarget.z);
            marker.visible = true;
          }
        }
        dragging = false;
      };
      const onMove = (e: MouseEvent) => {
        if (!dragging) return;
        const dx = e.clientX - prevX, dy = e.clientY - prevY;
        dragDist += Math.abs(dx) + Math.abs(dy);
        camAngle -= dx * 0.01;
        camPitch = Math.max(0.1, Math.min(1.2, camPitch + dy * 0.01));
        prevX = e.clientX; prevY = e.clientY;
      };
      const onWheel = (e: WheelEvent) => {
        camDist = Math.max(1, Math.min(15, camDist + e.deltaY * 0.01));
      };
      renderer.domElement.addEventListener("mousedown", onDown);
      window.addEventListener("mouseup", onUp);
      window.addEventListener("mousemove", onMove);
      renderer.domElement.addEventListener("wheel", onWheel);

      const onResize = () => {
        camera.aspect = mount.clientWidth / mount.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mount.clientWidth, mount.clientHeight);
      };
      window.addEventListener("resize", onResize);

      // Render loop
      let last = performance.now();
      const moveDir = new THREE.Vector3();

      renderer.setAnimationLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;

        const engine = engineRef.current;
        if (!engine || !engine.ready) {
          renderer.render(scene, camera);
          return;
        }

        const speed = walkSpeedRef.current;

        moveDir.set(0, 0, 0);
        if (keys.KeyW || keys.ArrowUp) moveDir.z -= 1;
        if (keys.KeyS || keys.ArrowDown) moveDir.z += 1;
        if (keys.KeyA || keys.ArrowLeft) moveDir.x -= 1;
        if (keys.KeyD || keys.ArrowRight) moveDir.x += 1;

        if (moveDir.lengthSq() > 0) {
          walkTarget = null;
          marker.visible = false;
          moveDir.normalize();
          engine.setMovement(
            [moveDir.x * speed, 0, moveDir.z * speed],
            [moveDir.x, 0, moveDir.z]
          );
        } else if (walkTarget) {
          const rootPos = engine.getRootPosition();
          const dx = walkTarget.x - rootPos[0];
          const dz = walkTarget.z - rootPos[2];
          const dist = Math.sqrt(dx * dx + dz * dz);
          if (dist < ARRIVE_THRESHOLD) {
            walkTarget = null;
            marker.visible = false;
            engine.setIdle();
          } else {
            const nx = dx / dist;
            const nz = dz / dist;
            engine.setMovement(
              [nx * speed, 0, nz * speed],
              [nx, 0, nz]
            );
          }
        } else {
          engine.setIdle();
        }

        engine.update(dt);

        const bones = engine.getBoneData();
        for (const b of bones) {
          const bone = boneByName.get(b.name);
          if (!bone) continue;
          applyWorldToBone(
            bone,
            b.position[0], b.position[1], b.position[2],
            b.quaternion[0], b.quaternion[1], b.quaternion[2], b.quaternion[3],
          );
        }

        camera.position.set(
          Math.sin(camAngle) * camDist,
          camDist * camPitch,
          Math.cos(camAngle) * camDist
        );
        camera.lookAt(0, 0.8, 0);

        renderer.render(scene, camera);
      });

      cleanup = () => {
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup", onKeyUp);
        renderer.domElement.removeEventListener("mousedown", onDown);
        window.removeEventListener("mouseup", onUp);
        window.removeEventListener("mousemove", onMove);
        renderer.domElement.removeEventListener("wheel", onWheel);
        window.removeEventListener("resize", onResize);
        renderer.setAnimationLoop(null);
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    return () => { disposed = true; cleanup?.(); };
  }, [loadCharacter]);

  return (
    <div className="fixed inset-0 bg-black">
      <div ref={mountRef} className="h-full w-full" />

      {loading && (
        <LoadingParticles label="loading character" className="z-30" />
      )}

      {!menuOpen && !loading && (
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
        </div>
      )}

      {menuOpen && (
        <div
          style={HUD_FONT}
          className="absolute inset-0 z-20 flex items-center justify-center bg-black/50 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) setMenuOpen(false);
          }}
        >
          <div className="w-[460px] max-h-[80vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-950/95 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <span className="text-[13px] text-white">Controls</span>
              <button
                onClick={toggleMenu}
                aria-label="close"
                className="flex h-7 w-7 items-center justify-center rounded-md text-zinc-400 hover:bg-white/10 hover:text-white"
              >
                <IconClose />
              </button>
            </div>

            <div className="space-y-5 p-5">
              {/* Keys */}
              <div className="space-y-2.5 text-[13px]">
                <div className="flex justify-between">
                  <span className="text-white">Move</span>
                  <span className="text-zinc-400">wasd</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white">Walk to point</span>
                  <span className="text-zinc-400">click</span>
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

              {/* Character */}
              <div>
                <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">Character</div>
                <div className="flex gap-2">
                  {CHARACTERS.map((c) => (
                    <button
                      key={c.label}
                      onClick={() => loadCharacter(c)}
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
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

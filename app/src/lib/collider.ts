import type * as THREE from "three";

type ColliderResult = {
  /** The full GLTF scene (add this to your scene to preserve transforms) */
  root: THREE.Group;
  /** Reference to the first mesh inside — use for raycasting */
  mesh: THREE.Mesh;
};

/**
 * Load a Marble collider mesh GLB via GLTFLoader.
 * Returns the full GLTF scene root (with PI rotation) AND a reference
 * to the mesh inside it. Adding the root preserves the GLTF's parent
 * transforms so raycasting and rendering stay in the same space.
 */
export async function loadColliderMesh(
  url: string
): Promise<ColliderResult | null> {
  try {
    const { GLTFLoader } = await import(
      "three/examples/jsm/loaders/GLTFLoader.js"
    );
    const gltf = await new GLTFLoader().loadAsync(url);

    let mesh: THREE.Mesh | null = null;
    gltf.scene.traverse((child: THREE.Object3D) => {
      if ((child as THREE.Mesh).isMesh && !mesh) {
        mesh = child as THREE.Mesh;
      }
    });

    if (!mesh) return null;

    // Apply PI rotation on the root to match the splat orientation
    gltf.scene.rotation.x = Math.PI;

    return { root: gltf.scene, mesh };
  } catch (e) {
    console.warn("[collider] failed to load mesh:", e);
    return null;
  }
}

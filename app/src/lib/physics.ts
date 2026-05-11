/**
 * Rapier physics for splat world — trimesh collider + capsule KCC.
 * Replaces Three.js raycast ground detection with proper collision.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import type * as THREE from "three";

let rapierInit: Promise<void> | null = null;

async function ensureRapier() {
  if (!rapierInit) rapierInit = RAPIER.init();
  await rapierInit;
}

export type PhysicsWorld = {
  /** Step the physics world (call once per frame) */
  step(): void;
  /** Create a dynamic box body for an object in the splat world. */
  createArtifactBox(opts: {
    x: number;
    y: number;
    z: number;
    halfExtents: { x: number; y: number; z: number };
  }): PhysicsArtifactBody;
  /** Create a movable kinematic ramp collider. */
  createRamp(opts: {
    x: number;
    y: number;
    z: number;
    halfExtents: { x: number; y: number; z: number };
    rotation: { x: number; y: number; z: number; w: number };
  }): PhysicsRampBody;
  /**
   * Move the character capsule by a desired delta.
   * Returns the corrected world position after collision resolution.
   */
  move(
    currentX: number,
    currentY: number,
    currentZ: number,
    deltaX: number,
    deltaY: number,
    deltaZ: number,
    dt: number,
  ): { x: number; y: number; z: number; grounded: boolean };
  /** Dispose all physics resources */
  dispose(): void;
};

export type PhysicsArtifactBody = {
  /** Pin the body to a kinematic target, useful for carrying/placing. */
  setKinematicPosition(x: number, y: number, z: number): void;
  /** Toggle whether the artifact participates in the physics solve. */
  setPhysicsEnabled(enabled: boolean): void;
  /** Release the body back to dynamic simulation. */
  setDynamic(linearVelocity?: { x: number; y: number; z: number }): void;
  /** Teleport while keeping the body dynamic. */
  setDynamicPosition(x: number, y: number, z: number): void;
  getPosition(): { x: number; y: number; z: number };
  dispose(): void;
};

export type PhysicsRampBody = {
  setPhysicsEnabled(enabled: boolean): void;
  dropAt(
    x: number,
    y: number,
    z: number,
    rotation: { x: number; y: number; z: number; w: number },
  ): void;
  getPosition(): {
    translation: { x: number; y: number; z: number };
    rotation: { x: number; y: number; z: number; w: number };
  };
  dispose(): void;
};

/**
 * Build a Rapier physics world from a Three.js collider mesh.
 *
 * @param mesh       The Three.js mesh whose geometry becomes a trimesh collider.
 *                   Must have its matrixWorld up to date (call updateMatrixWorld first).
 * @param capsuleRadius  Half-width of the character capsule (world units).
 * @param capsuleHeight  Total height of the capsule (world units). The half-height
 *                       of the cylindrical segment is (capsuleHeight/2 - capsuleRadius).
 * @param spawnX/Y/Z    Initial capsule position in world space.
 */
export async function createPhysicsWorld(
  mesh: THREE.Mesh,
  capsuleRadius: number,
  capsuleHeight: number,
  spawnX: number,
  spawnY: number,
  spawnZ: number,
): Promise<PhysicsWorld> {
  await ensureRapier();

  // -- World with gravity --
  const gravity = new RAPIER.Vector3(0, -9.81, 0);
  const world = new RAPIER.World(gravity);

  // -- Trimesh collider from Three.js mesh --
  const geo = mesh.geometry;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (!geo.index && (geo as any).computeBoundsTree) (geo as any).computeBoundsTree();
  const posAttr = geo.getAttribute("position");
  const indexAttr = geo.index;

  // Bake mesh's world transform into vertices so the trimesh
  // lives in the same coordinate system as the character.
  mesh.updateMatrixWorld(true);
  const worldMatrix = mesh.matrixWorld;

  const vertCount = posAttr.count;
  const vertices = new Float32Array(vertCount * 3);
  const _v = { x: 0, y: 0, z: 0 }; // scratch
  for (let i = 0; i < vertCount; i++) {
    _v.x = posAttr.getX(i);
    _v.y = posAttr.getY(i);
    _v.z = posAttr.getZ(i);
    // Apply world transform manually (avoid importing THREE.Vector3)
    const e = worldMatrix.elements; // column-major
    const ox = e[0] * _v.x + e[4] * _v.y + e[8] * _v.z + e[12];
    const oy = e[1] * _v.x + e[5] * _v.y + e[9] * _v.z + e[13];
    const oz = e[2] * _v.x + e[6] * _v.y + e[10] * _v.z + e[14];
    vertices[i * 3] = ox;
    vertices[i * 3 + 1] = oy;
    vertices[i * 3 + 2] = oz;
  }

  let indices: Uint32Array;
  if (indexAttr) {
    indices = new Uint32Array(indexAttr.count);
    for (let i = 0; i < indexAttr.count; i++) indices[i] = indexAttr.getX(i);
  } else {
    // Non-indexed: every 3 verts is a triangle
    indices = new Uint32Array(vertCount);
    for (let i = 0; i < vertCount; i++) indices[i] = i;
  }

  // Debug: log trimesh bounds
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < vertCount; i++) {
    const vx = vertices[i * 3], vy = vertices[i * 3 + 1], vz = vertices[i * 3 + 2];
    if (vx < minX) minX = vx; if (vx > maxX) maxX = vx;
    if (vy < minY) minY = vy; if (vy > maxY) maxY = vy;
    if (vz < minZ) minZ = vz; if (vz > maxZ) maxZ = vz;
  }
  console.log("[physics] trimesh verts:", vertCount, "tris:", indices.length / 3);
  console.log("[physics] trimesh bounds:",
    minX.toFixed(3), minY.toFixed(3), minZ.toFixed(3), "→",
    maxX.toFixed(3), maxY.toFixed(3), maxZ.toFixed(3));
  console.log("[physics] capsule spawn:", spawnX.toFixed(3), spawnY.toFixed(3), spawnZ.toFixed(3));

  const trimeshDesc = RAPIER.ColliderDesc.trimesh(vertices, indices);
  world.createCollider(trimeshDesc);

  // -- Character capsule (kinematic position-based) --
  const halfHeight = Math.max(0.01, capsuleHeight / 2 - capsuleRadius);
  const bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
    spawnX,
    spawnY + capsuleHeight / 2, // center of capsule above ground
    spawnZ,
  );
  const body = world.createRigidBody(bodyDesc);

  const capsuleDesc = RAPIER.ColliderDesc.capsule(halfHeight, capsuleRadius);
  world.createCollider(capsuleDesc, body);

  // -- Kinematic Character Controller --
  const kcc = world.createCharacterController(0.02); // 2cm offset
  kcc.enableAutostep(0.15, 0.1, true); // max step height, min width, include dynamic
  kcc.enableSnapToGround(0.3); // snap distance
  kcc.setSlideEnabled(true);
  kcc.setMaxSlopeClimbAngle((80 * Math.PI) / 180); // steep — splat worlds have uneven floors

  const collider = body.collider(0);
  let grounded = false;

  return {
    step() {
      world.step();
    },

    createArtifactBox(opts) {
      const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(opts.x, opts.y, opts.z)
        .setLinearDamping(1.8)
        .setAngularDamping(1.4)
        .setCanSleep(false);
      const artifactBody = world.createRigidBody(bodyDesc);
      const colliderDesc = RAPIER.ColliderDesc.cuboid(
        opts.halfExtents.x,
        opts.halfExtents.y,
        opts.halfExtents.z,
      )
        .setDensity(0.7)
        .setFriction(0.9)
        .setRestitution(0.05);
      const artifactCollider = world.createCollider(colliderDesc, artifactBody);

      return {
        setKinematicPosition(x, y, z) {
          if (artifactBody.bodyType() !== RAPIER.RigidBodyType.KinematicPositionBased) {
            artifactBody.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
          }
          artifactBody.setNextKinematicTranslation(new RAPIER.Vector3(x, y, z));
        },

        setPhysicsEnabled(enabled) {
          artifactCollider.setEnabled(enabled);
          artifactBody.setEnabled(enabled);
        },

        setDynamic(linearVelocity) {
          if (artifactBody.bodyType() !== RAPIER.RigidBodyType.Dynamic) {
            artifactBody.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
          }
          artifactBody.setLinvel(
            new RAPIER.Vector3(
              linearVelocity?.x ?? 0,
              linearVelocity?.y ?? 0,
              linearVelocity?.z ?? 0,
            ),
            true,
          );
          artifactBody.setAngvel(new RAPIER.Vector3(0, 0, 0), true);
        },

        setDynamicPosition(x, y, z) {
          if (artifactBody.bodyType() !== RAPIER.RigidBodyType.Dynamic) {
            artifactBody.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
          }
          artifactBody.setTranslation(new RAPIER.Vector3(x, y, z), true);
          artifactBody.setLinvel(new RAPIER.Vector3(0, 0, 0), true);
          artifactBody.setAngvel(new RAPIER.Vector3(0, 0, 0), true);
        },

        getPosition() {
          const p = artifactBody.translation();
          return { x: p.x, y: p.y, z: p.z };
        },

        dispose() {
          if (artifactBody.isValid()) world.removeRigidBody(artifactBody);
        },
      };
    },

    createRamp(opts) {
      const rampBodyDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(opts.x, opts.y, opts.z)
        .setRotation(opts.rotation)
        .setLinearDamping(1.6)
        .setAngularDamping(2.1)
        .setCanSleep(false);
      const rampBody = world.createRigidBody(rampBodyDesc);
      const rampColliderDesc = RAPIER.ColliderDesc.cuboid(
        opts.halfExtents.x,
        opts.halfExtents.y,
        opts.halfExtents.z,
      )
        .setDensity(1.4)
        .setFriction(1)
        .setRestitution(0.02);
      world.createCollider(rampColliderDesc, rampBody);
      const rampCollider = rampBody.collider(0);

      return {
        setPhysicsEnabled(enabled) {
          rampCollider.setEnabled(enabled);
          rampBody.setEnabled(enabled);
        },

        dropAt(x, y, z, rotation) {
          if (rampBody.bodyType() !== RAPIER.RigidBodyType.Dynamic) {
            rampBody.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
          }
          rampBody.setTranslation(new RAPIER.Vector3(x, y, z), true);
          rampBody.setRotation(rotation, true);
          rampBody.setLinvel(new RAPIER.Vector3(0, -0.15, 0), true);
          rampBody.setAngvel(new RAPIER.Vector3(0, 0, 0), true);
        },

        getPosition() {
          const translation = rampBody.translation();
          const rotation = rampBody.rotation();
          return {
            translation: { x: translation.x, y: translation.y, z: translation.z },
            rotation: { x: rotation.x, y: rotation.y, z: rotation.z, w: rotation.w },
          };
        },

        dispose() {
          if (rampBody.isValid()) world.removeRigidBody(rampBody);
        },
      };
    },

    move(
      currentX: number,
      currentY: number,
      currentZ: number,
      deltaX: number,
      deltaY: number,
      deltaZ: number,
      dt: number,
    ) {
      // Position the body at current character location (center of capsule)
      body.setTranslation(
        new RAPIER.Vector3(currentX, currentY + capsuleHeight / 2, currentZ),
        true,
      );
      // Step the world so collision structures are up to date for KCC
      world.step();

      // Zero gravity when grounded — any downward force on a slope causes sliding
      const gravY = grounded ? 0 : -9.81 * dt;

      const desiredMovement = new RAPIER.Vector3(
        deltaX,
        deltaY + gravY,
        deltaZ,
      );
      kcc.computeColliderMovement(collider, desiredMovement);

      const corrected = kcc.computedMovement();
      grounded = kcc.computedGrounded();

      const finalX = currentX + corrected.x;
      const finalY = currentY + corrected.y;
      const finalZ = currentZ + corrected.z;

      return { x: finalX, y: finalY, z: finalZ, grounded };
    },

    dispose() {
      world.free();
    },
  };
}

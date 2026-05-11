/**
 * Neural locomotion engine — runs the CodebookMatching model via ONNX in the browser.
 * Ported from locomotion-server/server.py + ai4animationpy math modules.
 *
 * All 4x4 matrices are stored as Float64Array(16) in ROW-MAJOR order (matching numpy).
 * Convert to Three.js column-major only at the boundary (getBoneData → applyWorldToBone).
 */

// Load onnxruntime-web from CDN — avoids Turbopack bundling issues
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _ort: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sessionCache = new Map<string, Promise<any>>();
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sessionRunQueue = new WeakMap<any, Promise<unknown>>();
const getOrt = async () => {
  if (_ort) return _ort;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((globalThis as any).ort) { _ort = (globalThis as any).ort; return _ort; }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.min.js";
    script.onload = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      _ort = (globalThis as any).ort;
      if (_ort) {
        _ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/";
        resolve(_ort);
      } else reject(new Error("onnxruntime-web failed to load"));
    };
    script.onerror = () => reject(new Error("Failed to load onnxruntime-web from CDN"));
    document.head.appendChild(script);
  });
};

const runSessionQueued = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session: any,
  feeds: Record<string, unknown>,
) => {
  const previous = sessionRunQueue.get(session) ?? Promise.resolve();
  const run = previous
    .catch(() => undefined)
    .then(() => session.run(feeds));
  sessionRunQueue.set(session, run.catch(() => undefined));
  return run;
};

// ─── Constants ───

const SEQ_LEN = 16;
const SEQ_WINDOW = 0.5;
const SEQ_FPS = 30;
const PREDICTION_FPS = 10;
const CONTACT_POWER = 3.0;
const CONTACT_THRESHOLD = 2.0 / 3.0;
const MIN_TIMESCALE = 1.0;
const MAX_TIMESCALE = 1.15;
const SYNC_SENSITIVITY = 5;
const TIMESCALE_SENSITIVITY = 5;
const TRAJ_CORRECTION = 0.25;

// ─── Companion data type ───

export type LocoData = {
  boneNames: string[];
  boneCount: number;
  parentIndices: number[];
  children: number[][];
  tposeTransforms: number[][][];
  zeroTransforms: number[][][];
  defaultLengths: number[];
  guidances: Record<string, number[][]>;
  sequenceLength: number;
  sequenceWindow: number;
  inputDim: number;
  outputDim: number;
  latentDim: number;
  feedBoneAxes?: boolean; // biped feeds Z+Y axes, quadruped doesn't
};

// ─── Mat4 utilities (row-major Float64Array(16)) ───

function mat4Identity(): Float64Array {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Copy(a: Float64Array): Float64Array {
  return new Float64Array(a);
}

function mat4Multiply(a: Float64Array, b: Float64Array): Float64Array {
  const o = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      o[r * 4 + c] = s;
    }
  }
  return o;
}

function mat4Inverse(m: Float64Array): Float64Array {
  const inv = new Float64Array(16);
  // Cofactor expansion for 4x4
  const m00=m[0],m01=m[1],m02=m[2],m03=m[3];
  const m10=m[4],m11=m[5],m12=m[6],m13=m[7];
  const m20=m[8],m21=m[9],m22=m[10],m23=m[11];
  const m30=m[12],m31=m[13],m32=m[14],m33=m[15];

  const b00=m00*m11-m01*m10, b01=m00*m12-m02*m10, b02=m00*m13-m03*m10;
  const b03=m01*m12-m02*m11, b04=m01*m13-m03*m11, b05=m02*m13-m03*m12;
  const b06=m20*m31-m21*m30, b07=m20*m32-m22*m30, b08=m20*m33-m23*m30;
  const b09=m21*m32-m22*m31, b10=m21*m33-m23*m31, b11=m22*m33-m23*m32;

  let det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (Math.abs(det) < 1e-12) return mat4Identity();
  det = 1.0 / det;

  inv[0]=(m11*b11-m12*b10+m13*b09)*det;
  inv[1]=(m02*b10-m01*b11-m03*b09)*det;
  inv[2]=(m31*b05-m32*b04+m33*b03)*det;
  inv[3]=(m22*b04-m21*b05-m23*b03)*det;
  inv[4]=(m12*b08-m10*b11-m13*b07)*det;
  inv[5]=(m00*b11-m02*b08+m03*b07)*det;
  inv[6]=(m32*b02-m30*b05-m33*b01)*det;
  inv[7]=(m20*b05-m22*b02+m23*b01)*det;
  inv[8]=(m10*b10-m11*b08+m13*b06)*det;
  inv[9]=(m01*b08-m00*b10-m03*b06)*det;
  inv[10]=(m30*b04-m31*b02+m33*b00)*det;
  inv[11]=(m21*b02-m20*b04-m23*b00)*det;
  inv[12]=(m11*b07-m10*b09-m12*b06)*det;
  inv[13]=(m00*b09-m01*b07+m02*b06)*det;
  inv[14]=(m31*b01-m30*b03-m32*b00)*det;
  inv[15]=(m20*b03-m21*b01+m22*b00)*det;
  return inv;
}

// Position = column 3, rows 0-2 → indices [3, 7, 11]
function mat4GetPos(m: Float64Array): [number, number, number] {
  return [m[3], m[7], m[11]];
}

function mat4SetPos(m: Float64Array, x: number, y: number, z: number) {
  m[3] = x; m[7] = y; m[11] = z;
}

// Z axis = column 2 → indices [2, 6, 10]
function mat4GetAxisZ(m: Float64Array): [number, number, number] {
  return [m[2], m[6], m[10]];
}

// Y axis = column 1 → indices [1, 5, 9]
function mat4GetAxisY(m: Float64Array): [number, number, number] {
  return [m[1], m[5], m[9]];
}

// Get 3x3 rotation as flat array [r00,r01,r02,r10,r11,r12,r20,r21,r22]
function mat4GetRot3x3(m: Float64Array): Float64Array {
  const r = new Float64Array(9);
  r[0]=m[0]; r[1]=m[1]; r[2]=m[2];
  r[3]=m[4]; r[4]=m[5]; r[5]=m[6];
  r[6]=m[8]; r[7]=m[9]; r[8]=m[10];
  return r;
}

// Compose translation + rotation into 4x4
function mat4TR(pos: [number,number,number], rot3x3: Float64Array): Float64Array {
  const m = mat4Identity();
  m[0]=rot3x3[0]; m[1]=rot3x3[1]; m[2]=rot3x3[2]; m[3]=pos[0];
  m[4]=rot3x3[3]; m[5]=rot3x3[4]; m[6]=rot3x3[5]; m[7]=pos[1];
  m[8]=rot3x3[6]; m[9]=rot3x3[7]; m[10]=rot3x3[8]; m[11]=pos[2];
  return m;
}

// Interpolate two transforms
function mat4Lerp(a: Float64Array, b: Float64Array, w: number): Float64Array {
  const o = new Float64Array(16);
  const w1 = 1 - w;
  for (let i = 0; i < 16; i++) o[i] = w1 * a[i] + w * b[i];
  // Re-orthogonalize rotation
  rotNormalize(o);
  return o;
}

// DeltaXZ: position with y zeroed, rotation from y component
// delta[1] is in DEGREES (ai4animation convention — Rotation.RotationY uses Deg2Rad internally)
function mat4DeltaXZ(delta: [number,number,number]): Float64Array {
  const angleRad = delta[1] * Math.PI / 180;
  const rot = rotLookPlanar([Math.sin(angleRad), 0, Math.cos(angleRad)]);
  return mat4TR([delta[0], 0, delta[2]], rot);
}

// ─── Vec3 utilities ───

type V3 = [number, number, number];

function v3Zero(): V3 { return [0, 0, 0]; }
function v3Copy(a: V3): V3 { return [a[0], a[1], a[2]]; }
function v3Add(a: V3, b: V3): V3 { return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function v3Sub(a: V3, b: V3): V3 { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function v3Scale(a: V3, s: number): V3 { return [a[0]*s, a[1]*s, a[2]*s]; }
function v3Dot(a: V3, b: V3): number { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function v3Cross(a: V3, b: V3): V3 {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function v3Len(a: V3): number { return Math.sqrt(v3Dot(a, a)); }
function v3Normalize(a: V3): V3 {
  const l = v3Len(a);
  return l < 1e-8 ? [0,0,0] : [a[0]/l, a[1]/l, a[2]/l];
}
function v3Lerp(a: V3, b: V3, w: number): V3 {
  const w1 = 1-w;
  return [w1*a[0]+w*b[0], w1*a[1]+w*b[1], w1*a[2]+w*b[2]];
}
function v3LerpDt(a: V3, b: V3, dt: number, rate: number): V3 {
  return v3Lerp(a, b, 1 - Math.exp(-dt * rate));
}
function v3SlerpDt(a: V3, b: V3, dt: number, rate: number): V3 {
  // Simplified: just lerp normalized vectors
  const na = v3Normalize(a), nb = v3Normalize(b);
  const r = v3Normalize(v3Lerp(na, nb, 1 - Math.exp(-dt * rate)));
  return r;
}

// Rotate vec3 by 3x3 rotation matrix
function rot3x3MulVec(rot: Float64Array, v: V3): V3 {
  return [
    rot[0]*v[0]+rot[1]*v[1]+rot[2]*v[2],
    rot[3]*v[0]+rot[4]*v[1]+rot[5]*v[2],
    rot[6]*v[0]+rot[7]*v[1]+rot[8]*v[2],
  ];
}

// DirectionTo: inverse(rotation) @ vec
function v3DirectionTo(v: V3, space: Float64Array): V3 {
  // Transpose of rotation = inverse for orthonormal
  const r = mat4GetRot3x3(space);
  const rt = new Float64Array(9);
  rt[0]=r[0]; rt[1]=r[3]; rt[2]=r[6];
  rt[3]=r[1]; rt[4]=r[4]; rt[5]=r[7];
  rt[6]=r[2]; rt[7]=r[5]; rt[8]=r[8];
  return rot3x3MulVec(rt, v);
}

// DirectionFrom: rotation @ vec
function v3DirectionFrom(v: V3, space: Float64Array): V3 {
  return rot3x3MulVec(mat4GetRot3x3(space), v);
}

// ─── Rotation utilities ───

function rotLookPlanar(z: V3): Float64Array {
  const nz = v3Normalize(z);
  const y: V3 = [0, 1, 0];
  const x = v3Normalize(v3Cross(y, nz));
  // columns: x, y, z (row-major 3x3)
  const r = new Float64Array(9);
  r[0]=x[0]; r[1]=y[0]; r[2]=nz[0];
  r[3]=x[1]; r[4]=y[1]; r[5]=nz[1];
  r[6]=x[2]; r[7]=y[2]; r[8]=nz[2];
  return r;
}

function rotNormalize(m: Float64Array) {
  // Re-orthogonalize the 3x3 rotation part of a 4x4 using Z and Y axes
  const z = v3Normalize(mat4GetAxisZ(m));
  const y = v3Normalize(mat4GetAxisY(m));
  const x = v3Normalize(v3Cross(y, z));
  const ny = v3Cross(z, x);
  m[0]=x[0]; m[1]=ny[0]; m[2]=z[0];
  m[4]=x[1]; m[5]=ny[1]; m[6]=z[1];
  m[8]=x[2]; m[9]=ny[2]; m[10]=z[2];
}

function rotInterpolate(a: Float64Array, b: Float64Array, w: number): Float64Array {
  const o = new Float64Array(9);
  const w1 = 1 - w;
  for (let i = 0; i < 9; i++) o[i] = w1 * a[i] + w * b[i];
  // Normalize: look(z, y)
  const z = v3Normalize([o[2], o[5], o[8]]);
  const y = v3Normalize([o[1], o[4], o[7]]);
  const x = v3Normalize(v3Cross(y, z));
  const ny = v3Cross(z, x);
  o[0]=x[0]; o[1]=ny[0]; o[2]=z[0];
  o[3]=x[1]; o[4]=ny[1]; o[5]=z[1];
  o[6]=x[2]; o[7]=ny[2]; o[8]=z[2];
  return o;
}

// Quaternion from 3x3 rotation (row-major) → [x, y, z, w]
// Uses the largest diagonal element for stability
function quatFromRot3x3(r: Float64Array): [number,number,number,number] {
  const m00=r[0],m01=r[1],m02=r[2];
  const m10=r[3],m11=r[4],m12=r[5];
  const m20=r[6],m21=r[7],m22=r[8];
  const trace = m00 + m11 + m22;
  let x: number, y: number, z: number, w: number;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    w = 0.25 / s;
    x = (m21 - m12) * s;
    y = (m02 - m20) * s;
    z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  return [x, y, z, w];
}

// Rotation3D from two axes (z-forward, y-up)
function rotLook(z: V3, y: V3): Float64Array {
  const nz = v3Normalize(z);
  const x = v3Normalize(v3Cross(v3Normalize(y), nz));
  const ny = v3Cross(nz, x);
  const r = new Float64Array(9);
  r[0]=x[0]; r[1]=ny[0]; r[2]=nz[0];
  r[3]=x[1]; r[4]=ny[1]; r[5]=nz[1];
  r[6]=x[2]; r[7]=ny[2]; r[8]=nz[2];
  return r;
}

// ─── Smooth step for contacts ───

function smoothStep(x: number): number {
  x = Math.max(0, Math.min(1, x));
  if (x < CONTACT_THRESHOLD) return 0;
  const y = (x - CONTACT_THRESHOLD) / (1 - CONTACT_THRESHOLD);
  return Math.pow(y, CONTACT_POWER);
}

// ─── Scalar interpolation ───

function lerpDt(a: number, b: number, dt: number, rate: number): number {
  return a + (b - a) * (1 - Math.exp(-dt * rate));
}

// ─── RootSeries ───

class RootSeries {
  count: number;
  timestamps: Float64Array;
  transforms: Float64Array[]; // count × mat4
  velocities: V3[];           // count × vec3

  constructor(count: number, window: number) {
    this.count = count;
    this.timestamps = new Float64Array(count);
    for (let i = 0; i < count; i++) this.timestamps[i] = (window * i) / Math.max(count - 1, 1);
    this.transforms = Array.from({ length: count }, () => mat4Identity());
    this.velocities = Array.from({ length: count }, () => v3Zero());
  }

  getPosition(i: number): V3 { return mat4GetPos(this.transforms[i]); }
  setPosition(i: number, p: V3) { mat4SetPos(this.transforms[i], p[0], p[1], p[2]); }
  getDirection(i: number): V3 { return mat4GetAxisZ(this.transforms[i]); }
  setDirection(i: number, d: V3) {
    const rot = rotLookPlanar(d);
    const m = this.transforms[i];
    m[0]=rot[0]; m[1]=rot[1]; m[2]=rot[2];
    m[4]=rot[3]; m[5]=rot[4]; m[6]=rot[5];
    m[8]=rot[6]; m[9]=rot[7]; m[10]=rot[8];
  }

  getLength(): number {
    let total = 0;
    for (let i = 1; i < this.count; i++) {
      const a = this.getPosition(i-1), b = this.getPosition(i);
      total += v3Len(v3Sub(b, a));
    }
    return total;
  }

  control(position: V3, direction: V3, velocity: V3, dt: number) {
    const moveSens = 10, turnSens = 10;
    let dir = v3Normalize(direction);
    if (v3Len(dir) < 0.01) {
      dir = v3Len(velocity) > 0.01 ? v3Normalize(velocity) : this.getDirection(0);
    }

    this.velocities[0] = v3LerpDt(this.velocities[0], velocity, dt, moveSens);
    this.setPosition(0, v3Add(position, v3Scale(this.velocities[0], dt)));
    this.setDirection(0, v3SlerpDt(this.getDirection(0), dir, dt, turnSens));

    const deltaT = this.timestamps[1] - this.timestamps[0];
    for (let i = 1; i < this.count; i++) {
      const ratio = i / Math.max(this.count - 1, 1);
      this.velocities[i] = v3LerpDt(this.velocities[i-1], velocity, deltaT, ratio * moveSens);
      this.setPosition(i, v3Add(this.getPosition(i-1), v3Scale(this.velocities[i], deltaT)));
      this.setDirection(i, v3Lerp(this.getDirection(0), dir, ratio) as V3);
      // Normalize direction after lerp
      const d = v3Normalize(this.getDirection(i));
      this.setDirection(i, d);
    }
  }
}

// ─── Sequence (predicted motion) ───

class Sequence {
  timestamps: Float64Array;
  trajectory: RootSeries;
  motionTransforms: Float64Array[][]; // [seqLen][boneCount] of mat4
  motionVelocities: V3[][];           // [seqLen][boneCount]
  contacts: Float64Array;             // [seqLen * 4]
  meanContact = 0;

  constructor(boneCount: number) {
    this.timestamps = new Float64Array(SEQ_LEN);
    for (let i = 0; i < SEQ_LEN; i++) this.timestamps[i] = (SEQ_WINDOW * i) / (SEQ_LEN - 1);
    this.trajectory = new RootSeries(SEQ_LEN, SEQ_WINDOW);
    this.motionTransforms = Array.from({ length: SEQ_LEN }, () =>
      Array.from({ length: boneCount }, () => mat4Identity())
    );
    this.motionVelocities = Array.from({ length: SEQ_LEN }, () =>
      Array.from({ length: boneCount }, () => v3Zero())
    );
    this.contacts = new Float64Array(SEQ_LEN * 4);
  }

  private indexPair(ts: number): [number, number, number] {
    const ratio = Math.max(0, Math.min(SEQ_LEN - 1,
      ((ts - this.timestamps[0]) / (this.timestamps[SEQ_LEN-1] - this.timestamps[0] || 1)) * (SEQ_LEN - 1)
    ));
    const a = Math.floor(ratio), b = Math.ceil(ratio);
    return [a, b, a === b ? 0 : ratio - a];
  }

  sampleRoot(ts: number): Float64Array {
    const [a, b, w] = this.indexPair(ts);
    return mat4Lerp(this.trajectory.transforms[a], this.trajectory.transforms[b], w);
  }

  samplePositions(ts: number, boneCount: number): V3[] {
    const [a, b, w] = this.indexPair(ts);
    return Array.from({ length: boneCount }, (_, i) =>
      v3Lerp(mat4GetPos(this.motionTransforms[a][i]), mat4GetPos(this.motionTransforms[b][i]), w)
    );
  }

  sampleRotations(ts: number, boneCount: number): Float64Array[] {
    const [a, b, w] = this.indexPair(ts);
    return Array.from({ length: boneCount }, (_, i) =>
      rotInterpolate(mat4GetRot3x3(this.motionTransforms[a][i]), mat4GetRot3x3(this.motionTransforms[b][i]), w)
    );
  }

  sampleVelocities(ts: number, boneCount: number): V3[] {
    const [a, b, w] = this.indexPair(ts);
    return Array.from({ length: boneCount }, (_, i) =>
      v3Lerp(this.motionVelocities[a][i], this.motionVelocities[b][i], w)
    );
  }

  sampleContacts(ts: number): [number, number, number, number] {
    const [a, b, w] = this.indexPair(ts);
    const w1 = 1 - w;
    return [
      w1*this.contacts[a*4]+w*this.contacts[b*4],
      w1*this.contacts[a*4+1]+w*this.contacts[b*4+1],
      w1*this.contacts[a*4+2]+w*this.contacts[b*4+2],
      w1*this.contacts[a*4+3]+w*this.contacts[b*4+3],
    ];
  }

  getRootLock(): number {
    return this.meanContact > 0.75 ? 1.0 : 0.0;
  }
}

// ─── Output bone data ───

export type BoneFrame = {
  name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number];
};

// ─── Leg IK (foot planting) ───

// Biped leg bone names (Geno skeleton)
const BIPED_LEFT_LEG  = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"];
const BIPED_RIGHT_LEG = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"];

class LegIK {
  private hipIdx: number;
  private ankleIdx: number;
  private ballIdx: number;
  private ankleBaseline: number;
  private ballBaseline: number;
  private ankleTarget: V3;
  private ballTarget: V3;

  constructor(
    private transforms: Float64Array[],
    boneNames: string[],
    hipName: string, _kneeName: string, ankleName: string, ballName: string,
  ) {
    this.hipIdx = boneNames.indexOf(hipName);
    this.ankleIdx = boneNames.indexOf(ankleName);
    this.ballIdx = boneNames.indexOf(ballName);

    if (!this.valid) {
      this.ankleBaseline = 0;
      this.ballBaseline = 0;
      this.ankleTarget = v3Zero();
      this.ballTarget = v3Zero();
      return;
    }

    const anklePos = mat4GetPos(transforms[this.ankleIdx]);
    const ballPos = mat4GetPos(transforms[this.ballIdx]);
    this.ankleBaseline = anklePos[1];
    this.ballBaseline = ballPos[1];
    this.ankleTarget = v3Copy(anklePos);
    this.ballTarget = v3Copy(ballPos);
  }

  get valid() { return this.hipIdx >= 0 && this.ankleIdx >= 0 && this.ballIdx >= 0; }

  solve(ankleContact: number, ballContact: number) {
    if (!this.valid) return;

    // Ankle
    {
      const w = ankleContact;
      const current = mat4GetPos(this.transforms[this.ankleIdx]);
      const locked: V3 = [this.ankleTarget[0], this.ankleTarget[1], this.ankleTarget[2]];
      locked[1] = Math.max(locked[1] * (1 - w) + this.ankleBaseline * w, this.ankleBaseline);
      this.ankleTarget = [
        current[0] * (1 - w) + locked[0] * w,
        current[1] * (1 - w) + locked[1] * w,
        current[2] * (1 - w) + locked[2] * w,
      ];
      mat4SetPos(this.transforms[this.ankleIdx], this.ankleTarget[0], this.ankleTarget[1], this.ankleTarget[2]);
    }

    // Ball (toe)
    {
      const w = ballContact;
      const current = mat4GetPos(this.transforms[this.ballIdx]);
      const locked: V3 = [this.ballTarget[0], this.ballTarget[1], this.ballTarget[2]];
      locked[1] = Math.max(locked[1] * (1 - w) + this.ballBaseline * w, this.ballBaseline);
      this.ballTarget = [
        current[0] * (1 - w) + locked[0] * w,
        current[1] * (1 - w) + locked[1] * w,
        current[2] * (1 - w) + locked[2] * w,
      ];
      mat4SetPos(this.transforms[this.ballIdx], this.ballTarget[0], this.ballTarget[1], this.ballTarget[2]);
    }
  }
}

// ─── Main engine ───

export class LocomotionEngine {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  private data: LocoData;

  // Actor state
  private root: Float64Array;
  private transforms: Float64Array[];
  private velocities: V3[];

  // Simulation
  private simulation: RootSeries;
  private rootControl: RootSeries;

  // Prediction
  private previous: Sequence | null = null;
  private sequence: Sequence | null = null;

  // Timing
  private timescale = 1.0;
  private synchronization = 0.0;
  private timestamp = 0;
  private blendTimestamp = 0; // when prediction result actually landed (for crossfade)
  private totalTime = 0;

  // Input
  private inputVelocity: V3 = v3Zero();
  private inputDirection: V3 = [0, 0, 1];
  private guidancePositions: number[][] = [];
  currentStyle = "Idle";

  // IK
  private leftLegIK: LegIK | null = null;
  private rightLegIK: LegIK | null = null;

  ready = false;

  constructor(data: LocoData) {
    this.data = data;
    const n = data.boneCount;

    // Initialize actor from T-pose
    this.root = mat4Identity();
    this.transforms = data.tposeTransforms.map(rows => {
      const m = new Float64Array(16);
      for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m[r*4+c] = rows[r][c];
      return m;
    });
    this.velocities = Array.from({ length: n }, () => v3Zero());

    // Simulation
    this.simulation = new RootSeries(SEQ_LEN, SEQ_WINDOW);
    this.rootControl = new RootSeries(SEQ_LEN, SEQ_WINDOW);

    // Guidance
    this.guidancePositions = data.guidances["Idle"] || Array.from({ length: n }, () => [0,0,0]);

    // IK — biped only (quadrupeds have LeftFoot but not LeftToeBase)
    if (data.boneNames.includes("LeftToeBase")) {
      this.leftLegIK = new LegIK(this.transforms, data.boneNames, ...BIPED_LEFT_LEG as [string, string, string, string]);
      this.rightLegIK = new LegIK(this.transforms, data.boneNames, ...BIPED_RIGHT_LEG as [string, string, string, string]);
    }
  }

  async loadModel(onnxUrl: string) {
    const ORT = await getOrt();
    const cachedSession = sessionCache.get(onnxUrl);
    if (cachedSession) {
      this.session = await cachedSession;
    } else {
      const sessionPromise = ORT.InferenceSession.create(onnxUrl, {
        executionProviders: ["wasm"],
      });
      sessionCache.set(onnxUrl, sessionPromise);
      this.session = await sessionPromise;
    }
    this.ready = true;
    console.log("[loco] ONNX model loaded");
  }

  setMovement(velocity: V3, direction: V3) {
    this.inputVelocity = velocity;
    this.inputDirection = direction;
  }

  setIdle() {
    this.inputVelocity = v3Zero();
  }

  setStyle(style: string) {
    if (this.data.guidances[style]) {
      this.currentStyle = style;
      this.guidancePositions = this.data.guidances[style];
    }
  }

  getStyles(): string[] {
    return Object.keys(this.data.guidances);
  }

  private predicting = false;

  update(dt: number) {
    if (!this.ready) return;
    this.totalTime += dt;
    this.control(dt);

    if (!this.predicting && (this.timestamp === 0 || this.totalTime - this.timestamp > 1.0 / PREDICTION_FPS)) {
      this.timestamp = this.totalTime;
      this.predicting = true;
      this.predict().finally(() => { this.predicting = false; });
    }

    this.animate(dt);
  }

  private control(dt: number) {
    const rootPos = mat4GetPos(this.root);
    const velocity = this.inputVelocity;
    let direction = v3Copy(this.inputDirection);
    if (v3Len(direction) < 0.01) direction = mat4GetAxisZ(this.root);

    // Guidance style — use Idle when stopped, otherwise respect user's chosen style
    const speed = v3Len(velocity);
    const style = speed < 0.1 ? "Idle" : this.currentStyle;
    if (this.data.guidances[style]) {
      this.guidancePositions = this.data.guidances[style];
    }

    // Simulation
    const simPos = this.simulation.getPosition(0);
    const position = v3Lerp(simPos, rootPos, this.synchronization);
    this.simulation.control(position, direction, velocity, dt);

    // Trajectory correction — blend transforms toward prediction, compute velocity from positions
    if (this.sequence) {
      // Blend transforms: 95% simulation, 5% prediction
      for (let i = 0; i < this.rootControl.count; i++) {
        this.rootControl.transforms[i] = mat4Lerp(
          this.simulation.transforms[i],
          this.sequence.trajectory.transforms[i],
          TRAJ_CORRECTION
        );
      }
      // Compute velocities from position offsets / time (matches Python _control)
      for (let i = 0; i < this.rootControl.count; i++) {
        let sumX = 0, sumY = 0, sumZ = 0, sumT = 0;
        for (let j = i; j < this.rootControl.count; j++) {
          const p = mat4GetPos(this.rootControl.transforms[j]);
          sumX += p[0] - rootPos[0];
          sumY += p[1] - rootPos[1];
          sumZ += p[2] - rootPos[2];
          sumT += this.rootControl.timestamps[j];
        }
        const invT = sumT > 1e-6 ? 1 / sumT : 0;
        this.rootControl.velocities[i] = [sumX * invT, sumY * invT, sumZ * invT];
      }
      // Blend computed velocities with predicted velocities
      for (let i = 0; i < this.rootControl.count; i++) {
        this.rootControl.velocities[i] = v3Lerp(
          this.rootControl.velocities[i],
          this.sequence.trajectory.velocities[i],
          TRAJ_CORRECTION
        );
      }
    }
  }

  private async predict() {
    if (!this.session) return;
    const n = this.data.boneCount;
    const root = this.root;
    const invRoot = mat4Inverse(root);

    // Build input tensor
    const input = new Float32Array(this.data.inputDim);
    let pivot = 0;

    const feed = (values: number[]) => {
      for (const v of values) input[pivot++] = v;
    };
    const feedV3 = (v: V3) => { feed([v[0], v[1], v[2]]); };
    const feedV3XZ = (v: V3) => { feed([v[0], v[2]]); };

    // Bone transforms in root-local space: inv(root) @ world-space transforms
    for (let i = 0; i < n; i++) {
      const local = mat4Multiply(invRoot, this.transforms[i]);
      feedV3(mat4GetPos(local));
    }
    // Biped feeds bone Z and Y axes; quadruped doesn't
    if (this.data.feedBoneAxes !== false) {
      for (let i = 0; i < n; i++) {
        const local = mat4Multiply(invRoot, this.transforms[i]);
        feedV3(mat4GetAxisZ(local));
      }
      for (let i = 0; i < n; i++) {
        const local = mat4Multiply(invRoot, this.transforms[i]);
        feedV3(mat4GetAxisY(local));
      }
    }
    // Velocities in root-local
    for (let i = 0; i < n; i++) {
      feedV3(v3DirectionTo(this.velocities[i], root));
    }

    // Future trajectory in root-local
    for (let i = 0; i < SEQ_LEN; i++) {
      const local = mat4Multiply(invRoot, this.rootControl.transforms[i]);
      feedV3XZ(mat4GetPos(local));
    }
    for (let i = 0; i < SEQ_LEN; i++) {
      const local = mat4Multiply(invRoot, this.rootControl.transforms[i]);
      feedV3XZ(mat4GetAxisZ(local));
    }
    for (let i = 0; i < SEQ_LEN; i++) {
      const vel = v3DirectionTo(this.rootControl.velocities[i], root);
      feedV3XZ(vel);
    }

    // Guidance positions
    for (let i = 0; i < n; i++) {
      const g = this.guidancePositions[i] || [0,0,0];
      feed(g);
    }

    // Run inference synchronously (WASM is sync-capable)
    const ORT = await getOrt();
    const inputTensor = new ORT.Tensor("float32", input, [1, this.data.inputDim]);
    try {
      const results = await runSessionQueued(this.session, { input: inputTensor });
      const outputData = results.output.data as Float32Array;
      this.unpackPrediction(outputData, root);
    } catch (e) {
      console.warn("[loco] inference failed:", e);
    }
  }

  private unpackPrediction(raw: Float32Array, root: Float64Array) {
    const n = this.data.boneCount;

    // Output shape: [1, SEQ_LEN, outputDim] flattened
    // Read sequentially
    let pivot = 0;
    const read = (count: number): number[] => {
      const out: number[] = [];
      for (let i = 0; i < count; i++) out.push(raw[pivot++]);
      return out;
    };
    const readV3 = (): V3 => {
      const d = read(3);
      return [d[0], d[1], d[2]];
    };
    const readV3Batch = (count: number): V3[] => {
      return Array.from({ length: count }, () => readV3());
    };
    const readRot3D = (): Float64Array => {
      const z = readV3();
      const y = readV3();
      return rotLook(z, y);
    };

    // Per-sequence-frame data
    const rootVectors: V3[] = [];
    const bonePositions: V3[][] = [];
    const boneRotations: Float64Array[][] = [];
    const boneVelocities: V3[][] = [];
    const contacts: number[][] = [];
    const guidances: V3[][] = [];

    for (let s = 0; s < SEQ_LEN; s++) {
      rootVectors.push(readV3());
      bonePositions.push(readV3Batch(n));
      // Rotations: Python ReadRotation3D(23) reads ALL z-axes (23×3=69) then ALL y-axes (69)
      // NOT interleaved per bone
      const zAxes = readV3Batch(n);
      const yAxes = readV3Batch(n);
      const rots: Float64Array[] = [];
      for (let b = 0; b < n; b++) rots.push(rotLook(zAxes[b], yAxes[b]));
      boneRotations.push(rots);
      boneVelocities.push(readV3Batch(n));
      contacts.push(read(4));
      guidances.push(readV3Batch(n));
    }

    // Compute root trajectory from cumulative deltas
    const rootDeltas: V3[] = Array.from({ length: SEQ_LEN }, () => v3Zero());
    for (let i = 1; i < SEQ_LEN; i++) {
      rootDeltas[i] = v3Add(rootDeltas[i-1], rootVectors[i]);
    }

    // Build sequence
    this.previous = this.sequence;
    const seq = new Sequence(n);

    // Root trajectory
    for (let i = 0; i < SEQ_LEN; i++) {
      const delta = mat4DeltaXZ(rootDeltas[i]);
      seq.trajectory.transforms[i] = mat4Multiply(root, delta);
      const vel: V3 = [rootVectors[i][0] * SEQ_FPS, 0, rootVectors[i][2] * SEQ_FPS];
      seq.trajectory.velocities[i] = v3DirectionFrom(vel, seq.trajectory.transforms[i]);
    }

    // Motion transforms + velocities in WORLD space (matches Python TransformationFrom)
    // Each seq frame's bone pose is transformed from root-local to world via that frame's root
    for (let s = 0; s < SEQ_LEN; s++) {
      const rootTf = seq.trajectory.transforms[s];
      for (let b = 0; b < n; b++) {
        const local = mat4TR(bonePositions[s][b], boneRotations[s][b]);
        seq.motionTransforms[s][b] = mat4Multiply(rootTf, local);
        seq.motionVelocities[s][b] = v3DirectionFrom(boneVelocities[s][b], rootTf);
      }
    }

    // Contacts
    let totalContact = 0;
    for (let s = 0; s < SEQ_LEN; s++) {
      for (let c = 0; c < 4; c++) {
        const val = smoothStep(contacts[s][c]);
        seq.contacts[s*4+c] = val;
        totalContact += val;
      }
    }
    seq.meanContact = totalContact / (SEQ_LEN * 4);

    this.sequence = seq;
    if (!this.previous) this.previous = seq;
    this.blendTimestamp = this.totalTime; // crossfade starts when result lands, not when predict() fired
    this.hasPrediction = true;
  }

  private animate(dt: number) {
    if (!this.sequence || !this.previous) return;
    const n = this.data.boneCount;

    // Timescale
    const simPos = this.simulation.getPosition(0);
    const rootPos = mat4GetPos(this.root);
    const reqSpeed = (v3Len(v3Sub(rootPos, simPos)) + this.simulation.getLength()) / SEQ_WINDOW;
    const predSpeed = this.sequence.trajectory.getLength() / SEQ_WINDOW;
    let ts = 1.0, sync = 0.0;
    if (reqSpeed > 0.1 && predSpeed > 0.1) {
      ts = reqSpeed / predSpeed;
      sync = 1.0;
    }
    this.timescale = Math.max(MIN_TIMESCALE, Math.min(MAX_TIMESCALE,
      lerpDt(this.timescale, ts, dt, TIMESCALE_SENSITIVITY)));
    this.synchronization = lerpDt(this.synchronization, sync, dt, SYNC_SENSITIVITY);

    const sdt = dt * this.timescale;
    const blend = Math.max(0, Math.min(1, (this.totalTime - this.timestamp) * PREDICTION_FPS));

    // Interpolate between previous and current prediction
    const rootTf = mat4Lerp(this.previous.sampleRoot(sdt), this.sequence.sampleRoot(sdt), blend);
    const positions = this.previous.samplePositions(sdt, n).map((p, i) =>
      v3Lerp(p, this.sequence!.samplePositions(sdt, n)[i], blend)
    );
    const rotations = this.previous.sampleRotations(sdt, n).map((r, i) =>
      rotInterpolate(r, this.sequence!.sampleRotations(sdt, n)[i], blend)
    );
    const velocities = this.previous.sampleVelocities(sdt, n).map((v, i) =>
      v3Lerp(v, this.sequence!.sampleVelocities(sdt, n)[i], blend)
    );

    // Update root
    this.root = mat4Lerp(rootTf, this.root, this.sequence.getRootLock());

    // Update bone transforms
    for (let i = 0; i < n; i++) {
      const curPos = mat4GetPos(this.transforms[i]);
      const newPos = v3Lerp(v3Add(curPos, v3Scale(velocities[i], sdt)), positions[i], 0.5);
      this.transforms[i] = mat4TR(newPos, rotations[i]);
    }
    this.velocities = velocities;

    // Foot IK — plant feet when contact signal is high
    if (this.leftLegIK && this.rightLegIK) {
      const prevContacts = this.previous.sampleContacts(sdt);
      const seqContacts = this.sequence.sampleContacts(sdt);
      const contacts: [number, number, number, number] = [
        prevContacts[0] * (1 - blend) + seqContacts[0] * blend,
        prevContacts[1] * (1 - blend) + seqContacts[1] * blend,
        prevContacts[2] * (1 - blend) + seqContacts[2] * blend,
        prevContacts[3] * (1 - blend) + seqContacts[3] * blend,
      ];
      this.leftLegIK.solve(contacts[0], contacts[1]);
      this.rightLegIK.solve(contacts[2], contacts[3]);
    }

    // Advance sequence timestamps only when not waiting on a prediction
    // (Python runs predict synchronously — no gap. Without this guard,
    // animate keeps consuming the stale sequence during async ONNX,
    // pulling the root back toward the prediction's start position.)
    if (!this.predicting) {
      for (let i = 0; i < SEQ_LEN; i++) {
        this.previous.timestamps[i] -= sdt;
        this.sequence.timestamps[i] -= sdt;
      }
    }
  }

  /** Snap each bone to the correct distance from its parent (matches Python RestoreBoneLengths). */
  private restoreBoneLengths() {
    const parents = this.data.parentIndices;
    for (let i = 0; i < this.data.boneCount; i++) {
      const pi = parents[i];
      if (pi === i || pi < 0) continue; // root bone
      const aPos = mat4GetPos(this.transforms[pi]);
      const bPos = mat4GetPos(this.transforms[i]);
      const dir = v3Sub(bPos, aPos);
      const norm = v3Len(dir);
      if (norm < 1e-8) continue;
      const targetLen = this.data.defaultLengths[i];
      const newPos = v3Add(aPos, v3Scale(dir, targetLen / norm));
      mat4SetPos(this.transforms[i], newPos[0], newPos[1], newPos[2]);
    }
  }

  /** Rotate single-child bones to point toward their child (matches Python RestoreBoneAlignments). */
  private restoreBoneAlignments() {
    for (let i = 0; i < this.data.boneCount; i++) {
      const children = this.data.children[i];
      if (children.length !== 1) continue;
      const ci = children[0];
      const bonePos = mat4GetPos(this.transforms[i]);
      const childPos = mat4GetPos(this.transforms[ci]);

      // Zero-pose child local position (PositionFrom: transform by current bone)
      const zt = this.data.zeroTransforms[ci];
      const zeroChildLocalPos: V3 = [zt[0][3], zt[1][3], zt[2][3]];
      const boneRot = mat4GetRot3x3(this.transforms[i]);
      const zeroChildWorld = v3Add(bonePos, rot3x3MulVec(boneRot, zeroChildLocalPos));

      // fromDir = where child should be (zero-pose), toDir = where child actually is
      const fromDir = v3Sub(zeroChildWorld, bonePos);
      const toDir = v3Sub(childPos, bonePos);
      const fromNorm = v3Len(fromDir);
      const toNorm = v3Len(toDir);
      if (fromNorm < 1e-8 || toNorm < 1e-8) continue;

      // Rodrigues rotation from fromDir to toDir
      const f = v3Scale(fromDir, 1 / fromNorm);
      const t = v3Scale(toDir, 1 / toNorm);
      const dot = v3Dot(f, t);
      if (dot > 0.9999) continue; // already aligned
      const cross = v3Cross(f, t);
      const crossLen = v3Len(cross);
      if (crossLen < 1e-8) continue;

      const axis = v3Scale(cross, 1 / crossLen);
      const sinA = crossLen;
      const cosA = dot;
      const kx = axis[0], ky = axis[1], kz = axis[2];

      // Apply rotation to each column of the current 3x3 rotation
      const m = this.transforms[i];
      for (let col = 0; col < 3; col++) {
        const cx = m[col], cy = m[4 + col], cz = m[8 + col];
        const kcx = ky * cz - kz * cy;
        const kcy = kz * cx - kx * cz;
        const kcz = kx * cy - ky * cx;
        const kdotc = kx * cx + ky * cy + kz * cz;
        m[col]     = cx + sinA * kcx + (1 - cosA) * (kx * kdotc - cx);
        m[4 + col] = cy + sinA * kcy + (1 - cosA) * (ky * kdotc - cy);
        m[8 + col] = cz + sinA * kcz + (1 - cosA) * (kz * kdotc - cz);
      }
    }
  }

  private frameCount = 0;
  private hasPrediction = false;

  /** Get world-space bone data for Three.js rendering */
  getBoneData(): BoneFrame[] {
    const bones: BoneFrame[] = [];
    for (let i = 0; i < this.data.boneCount; i++) {
      // transforms are already world-space (initialized from GLB global matrices,
      // updated by animate() which blends with world-space MotionTransforms).
      // Do NOT multiply by root — that double-applies the root offset.
      const pos = mat4GetPos(this.transforms[i]);
      const rot = mat4GetRot3x3(this.transforms[i]);
      const quat = quatFromRot3x3(rot);
      bones.push({
        name: this.data.boneNames[i],
        position: pos,
        quaternion: quat,
      });
    }
    this.frameCount++;
    return bones;
  }

  /** Get bone transforms in parent-local space (using engine parentIndices).
   *  Root bones (parent=-1) are relative to engine root.
   *  Use these with a scaled container — bind matrices stay happy because
   *  local rotations are correct, and the container scale handles visual sizing. */
  getBoneLocalData(): BoneFrame[] {
    const bones: BoneFrame[] = [];
    for (let i = 0; i < this.data.boneCount; i++) {
      const pi = this.data.parentIndices[i];
      const parentWorld = pi < 0 ? this.root : this.transforms[pi];
      const invParent = mat4Inverse(parentWorld);
      const local = mat4Multiply(invParent, this.transforms[i]);
      const pos = mat4GetPos(local);
      const rot = mat4GetRot3x3(local);
      const quat = quatFromRot3x3(rot);
      bones.push({
        name: this.data.boneNames[i],
        position: pos,
        quaternion: quat,
      });
    }
    return bones;
  }

  getRootPosition(): V3 {
    const pos = mat4GetPos(this.root);
    return [pos[0], 0, pos[2]];
  }

  /** Override engine root XZ — used by physics to correct after collision */
  setRootPosition(x: number, z: number) {
    mat4SetPos(this.root, x, this.root[7], z);
  }
}

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import readline from "node:readline";

export type MotionBricksFrame = {
  frame: number;
  mode: string;
  fps: number;
  qpos: number[];
};

type Listener = (frame: MotionBricksFrame) => void;

class MotionBricksStreamWorker {
  private child: ChildProcessWithoutNullStreams;
  private listeners = new Set<Listener>();
  private readyPromise: Promise<void>;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((error: Error) => void) | null = null;
  private stderr = "";

  constructor() {
    const appRoot = process.cwd();
    const repoRoot = path.resolve(appRoot, "..");
    const motionbricksRoot = path.join(repoRoot, "GR00T-WholeBodyControl", "motionbricks");
    const workerScript = path.join(repoRoot, "GR00T-WholeBodyControl", "tools", "motionbricks_stream_worker.py");

    this.child = spawn(path.join(motionbricksRoot, ".venv", "bin", "python"), [workerScript], {
      cwd: motionbricksRoot,
      env: {
        ...process.env,
        PYTHONPATH: path.join(motionbricksRoot, "motionbricks"),
        MPLCONFIGDIR: "/private/tmp/mplconfig",
        MOTIONBRICKS_CONTROL_FILE: "/private/tmp/motionbricks-control.json",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.readyPromise = new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    readline.createInterface({ input: this.child.stdout }).on("line", (line) => this.handleLine(line));

    this.child.stderr.on("data", (chunk) => {
      this.stderr += String(chunk);
      if (this.stderr.length > 8000) this.stderr = this.stderr.slice(-8000);
    });
    this.child.on("error", (error) => this.fail(error));
    this.child.on("close", (code) => this.fail(new Error(this.stderr.trim() || `MotionBricks stream exited with ${code}`)));
  }

  async ready() {
    await this.readyPromise;
  }

  sendControl(payload: Record<string, unknown>) {
    this.child.stdin.write(`${JSON.stringify(payload)}\n`);
  }

  addListener(listener: Listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private handleLine(line: string) {
    let message: unknown;
    try {
      message = JSON.parse(line);
    } catch {
      return;
    }
    if (typeof message !== "object" || message == null) return;
    if ("ready" in message) {
      this.readyResolve?.();
      this.readyResolve = null;
      this.readyReject = null;
      return;
    }
    if ("qpos" in message) {
      for (const listener of this.listeners) listener(message as MotionBricksFrame);
    }
  }

  private fail(error: Error) {
    this.readyReject?.(error);
    this.readyResolve = null;
    this.readyReject = null;
    this.listeners.clear();
  }
}

const globalState = globalThis as typeof globalThis & {
  motionBricksStreamWorker?: MotionBricksStreamWorker;
};

export function getMotionBricksStreamWorker() {
  globalState.motionBricksStreamWorker ??= new MotionBricksStreamWorker();
  return globalState.motionBricksStreamWorker;
}

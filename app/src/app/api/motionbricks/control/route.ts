import { getMotionBricksStreamWorker } from "@/lib/motionbricks-stream-worker";
import { writeFile } from "node:fs/promises";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function parseVec3(value: unknown): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3) return [0, 0, 0];
  const next = value.map((item) => Number(item));
  if (next.some((item) => !Number.isFinite(item))) return [0, 0, 0];
  return next.map((item) => Math.max(-1, Math.min(1, item))) as [number, number, number];
}

function parseTarget(value: unknown) {
  if (value == null) return null;
  if (typeof value !== "object") return null;
  const target = value as Record<string, unknown>;
  const position = Array.isArray(target.position) ? target.position.map((item) => Number(item)) : null;
  const heading = Number(target.heading);
  if (!position || position.length !== 2 || position.some((item) => !Number.isFinite(item))) return null;
  if (!Number.isFinite(heading)) return null;
  return {
    position: position as [number, number],
    heading,
  };
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as Record<string, unknown>;
  const payload = {
    mode: typeof body.mode === "string" ? body.mode : "idle",
    movement: parseVec3(body.movement),
    target: parseTarget(body.target),
    updatedAt: Date.now(),
  };
  await writeFile("/private/tmp/motionbricks-control.json", JSON.stringify(payload));

  const worker = getMotionBricksStreamWorker();
  worker.sendControl(payload);
  return Response.json({ ok: true });
}

import { getMotionBricksStreamWorker } from "@/lib/motionbricks-stream-worker";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const worker = getMotionBricksStreamWorker();
  await worker.ready();

  const encoder = new TextEncoder();
  let removeListener: (() => void) | null = null;

  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("event: ready\ndata: {}\n\n"));
      removeListener = worker.addListener((frame) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(frame)}\n\n`));
      });
    },
    cancel() {
      removeListener?.();
    },
  });

  return new Response(stream, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}

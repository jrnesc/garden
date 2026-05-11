import { Hono } from "hono";
import type { Context } from "hono";
import { cors } from "hono/cors";
import { ulid } from "ulid";

type Env = {
  JOURNEYS_BUCKET: R2Bucket;
  WLT_API_KEY: string;
  XAI_API_KEY: string;
};

type AppContext = Context<{ Bindings: Env }>;

const MARBLE_BASE = "https://api.worldlabs.ai/marble/v1";
const XAI_IMAGES = "https://api.x.ai/v1/images/generations";
const XAI_EDITS = "https://api.x.ai/v1/images/edits";
const XAI_RESPONSES = "https://api.x.ai/v1/responses";
const XAI_MODEL = "grok-imagine-image";
const XAI_VISION_MODEL = "grok-4-1-fast-non-reasoning";

// ── Types ──

type MarbleOperation = {
  operation_id: string;
  done: boolean;
  error: unknown | null;
  metadata: unknown | null;
  response: MarbleWorldResponse | null;
};

type MarbleWorldResponse = {
  id?: string;
  display_name?: string;
  world_marble_url?: string;
  assets?: {
    caption?: string;
    thumbnail_url?: string;
    splats?: {
      spz_urls?: Record<string, string>;
      semantics_metadata?: {
        metric_scale_factor?: number;
        ground_plane_offset?: number;
      };
    };
    mesh?: { collider_mesh_url?: string };
    imagery?: { pano_url?: string };
  };
};

type JourneyStatus = "pending" | "done" | "error";

type JourneyRecord = {
  ulid: string;
  operationId: string;
  prompt: string;
  style: string | null;
  imageUrl: string;
  createdAt: string;
  status: JourneyStatus;
  error?: string;
  world?: MarbleWorldResponse | null;
};

// ── R2 helpers ──

const journeyKey = (id: string) => `journeys/${id}.json`;

async function saveJourney(bucket: R2Bucket, rec: JourneyRecord): Promise<void> {
  await bucket.put(journeyKey(rec.ulid), JSON.stringify(rec), {
    httpMetadata: { contentType: "application/json" },
  });
}

async function loadJourney(
  bucket: R2Bucket,
  id: string
): Promise<JourneyRecord | null> {
  const obj = await bucket.get(journeyKey(id));
  if (!obj) return null;
  return (await obj.json()) as JourneyRecord;
}

async function listJourneys(bucket: R2Bucket): Promise<JourneyRecord[]> {
  const listing = await bucket.list({ prefix: "journeys/", limit: 200 });
  const records = await Promise.all(
    listing.objects.map(async (o) => {
      const obj = await bucket.get(o.key);
      if (!obj) return null;
      return (await obj.json()) as JourneyRecord;
    })
  );
  return records
    .filter((r): r is JourneyRecord => r !== null)
    // ULIDs are lexicographically time-sortable — newest first.
    .sort((a, b) => b.ulid.localeCompare(a.ulid));
}

// ── Marble + Grok helpers ──

async function marbleGenerate(
  env: Env,
  body: Record<string, unknown>
): Promise<MarbleOperation> {
  const res = await fetch(`${MARBLE_BASE}/worlds:generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "WLT-Api-Key": env.WLT_API_KEY,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`marble generate ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as MarbleOperation;
}

async function marblePoll(env: Env, operationId: string): Promise<MarbleOperation> {
  const res = await fetch(`${MARBLE_BASE}/operations/${operationId}`, {
    headers: { "WLT-Api-Key": env.WLT_API_KEY },
  });
  if (!res.ok) {
    throw new Error(`marble poll ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as MarbleOperation;
}

async function grokImage(env: Env, prompt: string): Promise<string> {
  const res = await fetch(XAI_IMAGES, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.XAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: XAI_MODEL,
      prompt,
      n: 1,
      response_format: "url",
      aspect_ratio: "16:9",
      resolution: "2k",
    }),
  });
  if (!res.ok) {
    throw new Error(`grok image ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as { data?: { url?: string }[] };
  const url = data.data?.[0]?.url;
  if (!url) throw new Error("grok image returned no url");
  return url;
}

async function grokEditImage(
  env: Env,
  prompt: string,
  imageDataUrl: string
): Promise<string> {
  const res = await fetch(XAI_EDITS, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.XAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: XAI_MODEL,
      prompt,
      image: { url: imageDataUrl, type: "image_url" },
      n: 1,
      response_format: "url",
      resolution: "2k",
    }),
  });
  if (!res.ok) {
    throw new Error(`grok edit ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as { data?: { url?: string }[] };
  const url = data.data?.[0]?.url;
  if (!url) throw new Error("grok edit returned no url");
  return url;
}

// ── App ──

const app = new Hono<{ Bindings: Env }>();

app.use("*", cors({ origin: "*", allowMethods: ["GET", "POST", "DELETE", "OPTIONS"] }));

app.get("/", (c) => c.text("inside-api ok"));

// GET /assets/* — serve static assets from R2 (ONNX models, etc.)
app.get("/assets/:key{.+}", async (c) => {
  const key = `assets/${c.req.param("key")}`;
  const obj = await c.env.JOURNEYS_BUCKET.get(key);
  if (!obj) return c.json({ error: "not found" }, 404);
  const headers = new Headers();
  headers.set("Content-Type", obj.httpMetadata?.contentType ?? "application/octet-stream");
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("Access-Control-Allow-Origin", "*");
  return new Response(obj.body, { headers });
});

// POST /splats — run the full chain and persist a pending record.
// Body: { prompt: string, style?: string, displayName?: string }
// Returns: JourneyRecord (status "pending")
async function createSplatResponse(c: AppContext) {
  const body = await c.req.json<{
    prompt?: string;
    style?: string;
    displayName?: string;
  }>();
  const prompt = body.prompt?.trim();
  if (!prompt) return c.json({ error: "prompt required" }, 400);

  const imagePrompt = body.style ? `${body.style}. ${prompt}` : prompt;

  try {
    const imageUrl = await grokImage(c.env, imagePrompt);

    const op = await marbleGenerate(c.env, {
      display_name: body.displayName ?? prompt.slice(0, 60),
      model: "marble-1.1-plus",
      world_prompt: {
        type: "image",
        image_prompt: { source: "uri", uri: imageUrl },
        text_prompt: prompt,
      },
    });

    const rec: JourneyRecord = {
      ulid: ulid(),
      operationId: op.operation_id,
      prompt,
      style: body.style ?? null,
      imageUrl,
      createdAt: new Date().toISOString(),
      status: "pending",
    };
    await saveJourney(c.env.JOURNEYS_BUCKET, rec);
    return c.json(rec);
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
}

app.post("/splats", createSplatResponse);

// POST /journeys — legacy alias for /splats
app.post("/journeys", createSplatResponse);

async function listSplatsResponse(c: {
  env: Env;
  json: (data: unknown, status?: number) => Response;
}) {
  const records = await listJourneys(c.env.JOURNEYS_BUCKET);
  return c.json({ splats: records, journeys: records });
}

// GET /splats — list all records, newest first
app.get("/splats", async (c) => listSplatsResponse(c));

// GET /journeys — legacy alias for /splats
app.get("/journeys", async (c) => listSplatsResponse(c));

async function getSplatResponse(c: {
  env: Env;
  req: { param: (name: string) => string };
  json: (data: unknown, status?: number) => Response;
}) {
  const id = c.req.param("id");
  const rec = await loadJourney(c.env.JOURNEYS_BUCKET, id);
  if (!rec) return c.json({ error: "not found" }, 404);

  if (rec.status !== "pending") return c.json(rec);

  try {
    const op = await marblePoll(c.env, rec.operationId);
    if (op.error) {
      rec.status = "error";
      rec.error = typeof op.error === "string" ? op.error : JSON.stringify(op.error);
      await saveJourney(c.env.JOURNEYS_BUCKET, rec);
    } else if (op.done && op.response) {
      rec.status = "done";
      rec.world = op.response;
      await saveJourney(c.env.JOURNEYS_BUCKET, rec);
    }
    return c.json(rec);
  } catch (err) {
    return c.json({ ...rec, error: String(err) }, 502);
  }
}

// GET /splats/:ulid — fetch one record. If pending, poll Marble and update if done.
app.get("/splats/:id", async (c) => getSplatResponse(c));

// GET /journeys/:ulid — legacy alias for /splats/:ulid
app.get("/journeys/:id", async (c) => getSplatResponse(c));

async function deleteSplatResponse(c: {
  env: Env;
  req: { param: (name: string) => string };
  json: (data: unknown, status?: number) => Response;
}) {
  const id = c.req.param("id");
  await c.env.JOURNEYS_BUCKET.delete(journeyKey(id));
  return c.json({ ok: true });
}

// DELETE /splats/:ulid — remove from R2
app.delete("/splats/:id", async (c) => deleteSplatResponse(c));

// DELETE /journeys/:ulid — legacy alias for /splats/:ulid
app.delete("/journeys/:id", async (c) => deleteSplatResponse(c));

// POST /edit-intent — Wave 2 spike.
// Body: { screenshot: base64 data URL, intent: string, width: number, height: number,
//         userPrompt?: string, sceneCaption?: string }
// Asks Grok vision to identify WHERE to apply the edit (normalized pixel) AND how
// to apply it (color, op, size). userPrompt + sceneCaption are free semantic context
// to help disambiguate ambiguous architectural/stylistic vocab ("column" vs "arch").
app.post("/edit-intent", async (c) => {
  const body = await c.req.json<{
    screenshot?: string;
    intent?: string;
    width?: number;
    height?: number;
    userPrompt?: string | null;
    sceneCaption?: string | null;
  }>();
  const { screenshot, intent, width, height, userPrompt, sceneCaption } = body;
  if (!screenshot || !intent || !width || !height) {
    return c.json({ error: "screenshot, intent, width, height all required" }, 400);
  }
  if (!screenshot.startsWith("data:image/")) {
    return c.json({ error: "screenshot must be a data URL" }, 400);
  }

  const contextLines: string[] = [];
  if (userPrompt) {
    contextLines.push(`The user originally asked for: "${userPrompt}"`);
  }
  if (sceneCaption) {
    contextLines.push(
      `An AI-generated description of this scene is: "${sceneCaption}"`
    );
  }
  const contextBlock =
    contextLines.length > 0
      ? `\nContext to help you disambiguate object names in this scene:\n${contextLines.join("\n")}\n`
      : "";

  const prompt = `You are helping edit a 3D gaussian splat scene. The user is standing inside the scene and can see the attached image — this is their current view. Their edit intent is:

"${intent}"
${contextBlock}
You output a BOUNDING BOX (not a single point) around the target, plus a description of how to apply the edit. The bounding box should tightly enclose ONE specific instance of the thing the user named. All coordinates are normalized [0, 1] where (0, 0) is the top-left of the image and (1, 1) is the bottom-right.

Rules for the bounding box:

- It must TIGHTLY enclose the target. Not the whole image. Not a loose "general area." If the target occupies 10% of the image, the box is 10% of the image.
- If the user names a PLURAL or COLLECTIVE ("the arches", "the columns", "the cracks"), DO NOT draw a box around all of them at once — the average of many arches in a perspective view is empty sky at the vanishing point. Instead, pick ONE specific clearly-visible instance (the nearest, leftmost, rightmost, most prominent, whichever best matches the intent) and box that one.
- Prefer literal object matches over visually similar ones — "column" is a vertical load-bearing element, not an arch; "arch" is the curved element spanning between columns; "doorway" is an opening in a wall, not the wall around it.
- The box corners must satisfy x2 > x1 and y2 > y1.
- A point inside the box will be used to raycast through the 3D scene, so the box must enclose VISIBLE PIXELS OF THE TARGET — not empty sky in front of, above, or between the target.

Also decide HOW to apply the edit:
- "color": one of "red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan", "white", "black", or null if the intent has no color
- "op": one of "recolor" (apply a color tint), "erase" (make the region vanish/go black), "brighten" (make it glow / lighter), "darken" (make it darker / moodier). Default to "recolor" if unsure.
- "size": one of "small" (a single object or small feature, ~0.35 units), "medium" (a normal object or medium region, ~0.8 units, the default), "large" (a whole wall/sky/floor-sized region, ~2 units). Pick size based on what the user is describing, not image pixels or box size.

Respond with ONLY a JSON object on a single line, no other text, no code fences:
{"x1": <float>, "y1": <float>, "x2": <float>, "y2": <float>, "color": <string|null>, "op": <string>, "size": <string>, "reason": "<one short sentence describing which specific instance you boxed and why>"}`;

  try {
    const grokRes = await fetch(XAI_RESPONSES, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${c.env.XAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: XAI_VISION_MODEL,
        // Don't let xAI persist the response: our input payload includes a large
        // base64 PNG which blows past their server-side storage limit otherwise.
        store: false,
        input: [
          {
            role: "user",
            content: [
              { type: "input_image", image_url: screenshot, detail: "high" },
              { type: "input_text", text: prompt },
            ],
          },
        ],
      }),
    });
    if (!grokRes.ok) {
      return c.json(
        { error: `grok ${grokRes.status}: ${await grokRes.text()}` },
        502
      );
    }
    const data = (await grokRes.json()) as {
      output?: { content?: { type?: string; text?: string }[] }[];
      output_text?: string;
      usage?: Record<string, unknown>;
    };
    // Tolerate a few known response shapes. xAI's Responses API returns either an
    // `output` array of messages with content blocks, or a top-level `output_text`.
    let text = data.output_text ?? "";
    if (!text && Array.isArray(data.output)) {
      for (const msg of data.output) {
        for (const block of msg.content ?? []) {
          if (block.text) text += block.text;
        }
      }
    }
    const match = text.match(/\{[\s\S]*?\}/);
    if (!match) {
      return c.json({ error: "no json in response", raw: text }, 502);
    }
    const parsed = JSON.parse(match[0]) as {
      x1?: number;
      y1?: number;
      x2?: number;
      y2?: number;
      color?: string | null;
      op?: string;
      size?: string;
      reason?: string;
    };
    if (
      typeof parsed.x1 !== "number" ||
      typeof parsed.y1 !== "number" ||
      typeof parsed.x2 !== "number" ||
      typeof parsed.y2 !== "number"
    ) {
      return c.json({ error: "missing bbox corners in response", raw: text }, 502);
    }
    // Clamp each corner to [0, 1], then normalize ordering so x1<=x2 and y1<=y2.
    const clamp = (v: number) => Math.max(0, Math.min(1, v));
    let nx1 = clamp(parsed.x1);
    let ny1 = clamp(parsed.y1);
    let nx2 = clamp(parsed.x2);
    let ny2 = clamp(parsed.y2);
    if (nx1 > nx2) [nx1, nx2] = [nx2, nx1];
    if (ny1 > ny2) [ny1, ny2] = [ny2, ny1];
    // Degenerate box (point or line) → expand slightly so the raycast still has
    // something to work with. Shouldn't happen if Grok follows the prompt.
    if (nx2 - nx1 < 0.001) {
      nx1 = Math.max(0, nx1 - 0.01);
      nx2 = Math.min(1, nx2 + 0.01);
    }
    if (ny2 - ny1 < 0.001) {
      ny1 = Math.max(0, ny1 - 0.01);
      ny2 = Math.min(1, ny2 + 0.01);
    }
    const ncx = (nx1 + nx2) / 2;
    const ncy = (ny1 + ny2) / 2;
    return c.json({
      // Center-of-box point in buffer pixels — what the client raycasts through.
      x: Math.round(ncx * width),
      y: Math.round(ncy * height),
      nx: ncx,
      ny: ncy,
      // Full bounding box in normalized coords — client renders it as a debug overlay.
      box: { nx1, ny1, nx2, ny2 },
      color: parsed.color ?? null,
      op: parsed.op ?? "recolor",
      size: parsed.size ?? "medium",
      reason: parsed.reason ?? null,
      usage: data.usage ?? null,
    });
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
});

// POST /ask-world — Ask a question about what's visible in the current view.
// Body: { screenshot: base64 data URL, question: string, width: number, height: number,
//         sceneCaption?: string }
// Returns: { answer: string }
app.post("/ask-world", async (c) => {
  const body = await c.req.json<{
    screenshot?: string;
    question?: string;
    width?: number;
    height?: number;
    sceneCaption?: string | null;
  }>();
  const { screenshot, question, sceneCaption } = body;
  if (!screenshot || !question) {
    return c.json({ error: "screenshot and question required" }, 400);
  }
  if (!screenshot.startsWith("data:image/")) {
    return c.json({ error: "screenshot must be a data URL" }, 400);
  }

  const contextBlock = sceneCaption
    ? `\nThe world model described this scene as: "${sceneCaption}"\n`
    : "";

  const prompt = `You are a narrator inside a 3D world generated from a painting. The user is standing inside this world and can see the attached image — this is their current view. They are asking you a question about what they see.
${contextBlock}
Their question: "${question}"

Answer in 1-3 sentences. Be evocative and specific about what's visible. Speak as if you are the world itself — you know your own architecture, your own light, your own mood. Don't say "in the image" or "it appears to be" — describe what IS, as if you are there. Be poetic but grounded in what's actually visible.`;

  try {
    const grokRes = await fetch(XAI_RESPONSES, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${c.env.XAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: XAI_VISION_MODEL,
        store: false,
        input: [
          {
            role: "user",
            content: [
              { type: "input_image", image_url: screenshot, detail: "high" },
              { type: "input_text", text: prompt },
            ],
          },
        ],
      }),
    });
    if (!grokRes.ok) {
      return c.json(
        { error: `grok ${grokRes.status}: ${await grokRes.text()}` },
        502
      );
    }
    const data = (await grokRes.json()) as {
      output?: { content?: { type?: string; text?: string }[] }[];
      output_text?: string;
    };
    let text = data.output_text ?? "";
    if (!text && Array.isArray(data.output)) {
      for (const msg of data.output) {
        for (const block of msg.content ?? []) {
          if (block.text) text += block.text;
        }
      }
    }
    if (!text) {
      return c.json({ error: "no response from vision model" }, 502);
    }
    return c.json({ answer: text.trim() });
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
});

// POST /splats/backfill — reconstruct a record from a known Marble world_id or operation_id.
// Body: { worldId | operationId, prompt, style?, imageUrl? }
// Uses the Marble thumbnail as the card image when no Grok imageUrl is provided.
async function backfillSplatResponse(c: AppContext) {
  const body = await c.req.json<{
    worldId?: string;
    operationId?: string;
    prompt?: string;
    style?: string;
    imageUrl?: string;
  }>();
  if ((!body.worldId && !body.operationId) || !body.prompt) {
    return c.json({ error: "(worldId or operationId) and prompt required" }, 400);
  }

  try {
    let world: MarbleWorldResponse;
    if (body.worldId) {
      const res = await fetch(`${MARBLE_BASE}/worlds/${body.worldId}`, {
        headers: { "WLT-Api-Key": c.env.WLT_API_KEY },
      });
      if (!res.ok) {
        return c.json({ error: `marble world ${res.status}: ${await res.text()}` }, 502);
      }
      world = (await res.json()) as MarbleWorldResponse;
    } else {
      const op = await marblePoll(c.env, body.operationId!);
      if (!op.done || !op.response) {
        return c.json({ error: "marble op not done", op }, 409);
      }
      world = op.response;
    }
    const rec: JourneyRecord = {
      ulid: ulid(),
      operationId: body.operationId ?? body.worldId ?? "",
      prompt: body.prompt,
      style: body.style ?? null,
      imageUrl: body.imageUrl ?? world.assets?.thumbnail_url ?? "",
      createdAt: new Date().toISOString(),
      status: "done",
      world,
    };
    await saveJourney(c.env.JOURNEYS_BUCKET, rec);
    return c.json(rec);
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
}

app.post("/splats/backfill", backfillSplatResponse);

// POST /journeys/backfill — legacy alias for /splats/backfill
app.post("/journeys/backfill", backfillSplatResponse);

// POST /splats/from-image — send an existing image directly to Marble (skip Grok).
// Body: { image: base64 data URL, prompt: string, displayName?: string }
async function createSplatFromImageResponse(c: AppContext) {
  const body = await c.req.json<{
    image?: string;
    prompt?: string;
    displayName?: string;
  }>();
  if (!body.image || !body.prompt) {
    return c.json({ error: "image (data URL) and prompt required" }, 400);
  }

  try {
    // Strip the data URL prefix to get raw base64
    const base64Match = body.image.match(/^data:[^;]+;base64,(.+)$/);
    if (!base64Match) {
      return c.json({ error: "image must be a data URL (data:...;base64,...)" }, 400);
    }
    const rawBase64 = base64Match[1];
    const mimeMatch = body.image.match(/^data:([^;]+);/);
    const mediaType = mimeMatch?.[1] ?? "image/png";

    const op = await marbleGenerate(c.env, {
      display_name: body.displayName ?? body.prompt.slice(0, 60),
      model: "marble-1.1",
      world_prompt: {
        type: "image",
        image_prompt: {
          source: "data_base64",
          data_base64: rawBase64,
          media_type: mediaType,
        },
        text_prompt: body.prompt,
      },
    });

    const rec: JourneyRecord = {
      ulid: ulid(),
      operationId: op.operation_id,
      prompt: body.prompt,
      style: null,
      imageUrl: body.image.slice(0, 200),
      createdAt: new Date().toISOString(),
      status: "pending",
    };
    await saveJourney(c.env.JOURNEYS_BUCKET, rec);
    return c.json(rec);
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
}

app.post("/splats/from-image", createSplatFromImageResponse);

// POST /journeys/from-image — legacy alias for /splats/from-image
app.post("/journeys/from-image", createSplatFromImageResponse);

// POST /generate-image — image-to-image via Grok edits API.
// Body: { image: base64 data URL, prompt: string }
// Returns: { url: string } — the generated image URL
app.post("/generate-image", async (c) => {
  const body = await c.req.json<{ image?: string; prompt?: string }>();
  if (!body.image || !body.prompt) {
    return c.json({ error: "image (data URL) and prompt required" }, 400);
  }
  try {
    const url = await grokEditImage(c.env, body.prompt, body.image);
    return c.json({ url });
  } catch (err) {
    return c.json({ error: String(err) }, 502);
  }
});

export default app;

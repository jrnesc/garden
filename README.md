# garden

A neural-animated character walking through Gaussian splat worlds, all running in a browser tab.

## What this is

Three things stitched together client-side:

1. **Neural locomotion controller** — Meta's `ai4animationpy` CodebookMatching model, exported to ONNX and run via ONNX Runtime WebAssembly. Forward pass per frame produces the next pose. No server.
2. **Physics** — Rapier (WASM) for terrain collision, wall collision, foot grounding.
3. **Scene** — Gaussian splat worlds generated with Marble (World Labs), rendered with Spark.

WASD + camera orbit. Dog / Wolf / Geno characters. The character "decides" how to walk based on intent and terrain instead of blending pre-recorded clips.

## The model

> Sebastian Starke, Paul Starke, Nicky He, Taku Komura, Yuting Ye.
> **Categorical Codebook Matching for Embodied Character Controllers.**
> *ACM Transactions on Graphics (SIGGRAPH 2024).* [doi:10.1145/3658209](https://dl.acm.org/doi/10.1145/3658209)

Instead of training a motion prior and a controller separately, the framework learns the motion manifold and the sampling policy end-to-end by matching probability distributions between two categorical codebooks (input intent ↔ output motion). Same motions map to the same codes, which avoids the blurring of MLP regressors and the mode-collapse of variational models.

- Paper: <https://dl.acm.org/doi/10.1145/3658209>
- Video: <https://www.youtube.com/watch?v=NyLRcY0c0p4>
- Reference code (Unity): <https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2024>
- Python framework: <https://github.com/facebookresearch/ai4animationpy>

## Setup

This isn't a one-click clone. You'll need API keys, you'll need to export the ONNX model from Meta's repo yourself, and you'll need to generate your own splat worlds — the ones I made aren't included.

**Prerequisites**
- Node 20+, [pnpm](https://pnpm.io)
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- [Cloudflare Wrangler](https://developers.cloudflare.com/workers/wrangler/) (`pnpm i -g wrangler`)

**API keys you need**
- **World Labs Marble** — for splat world generation. Apply at <https://marble.worldlabs.ai>. Stored as `WLT_API_KEY`.
- **xAI / Grok** — for image-to-image seed generation feeding into Marble. Get a key at <https://console.x.ai>. Stored as `XAI_API_KEY`.
- **Cloudflare account** — for the Worker and R2 bucket. Replace the `account_id` in `worker/wrangler.toml` and `app/wrangler.toml` with your own.

**1. Export the ONNX model**

Clone Meta's repo alongside this one (it's gitignored intentionally — it's huge):

```bash
git clone https://github.com/facebookresearch/ai4animationpy.git
cd locomotion-server
uv sync
uv run python export_onnx.py        # biped (CodebookMatching)
uv run python export_quadruped.py   # dog/wolf
```

Outputs go into `app/public/` as `.onnx` files plus the `locomotion-data.json` / `quadruped-data.json` skeleton metadata.

**2. Worker (Marble + Grok proxy + R2)**

```bash
cd worker
pnpm install
wrangler secret put WLT_API_KEY
wrangler secret put XAI_API_KEY
wrangler r2 bucket create inside-journeys
wrangler deploy
```

**3. Generate your own splat worlds**

The Beksiński worlds I made aren't in this repo. You'll need to generate your own — the app has a generation flow, or you can hit the worker's Marble endpoint directly. Each world takes ~5–15 minutes on Marble's side and costs credits. Resulting `.spz` files are stored in R2 and referenced by the app.

**4. Run the app**

```bash
cd app
pnpm install
pnpm dev
```

Open <http://localhost:3000>. The `/character` route is a flat-ground locomotion demo (no splat dependency, good for verifying the ONNX runtime works). `/walk` is the splat-world version and needs at least one generated world.

## Layout

- `app/` — Next.js + Three.js + Spark frontend. ONNX inference, Rapier physics, splat rendering.
- `locomotion-server/` — Python tooling: ONNX export from the Meta repo, validation, a server used during development.
- `worker/` — Cloudflare Worker (Marble / Grok proxy for splat generation).
- `ai4animationpy/` — Meta's upstream repo (gitignored; clone from <https://github.com/facebookresearch/ai4animationpy>).

See `ARCHITECTURE.md`, `HANDOFF.md`, and `ONNX_HANDOFF.md` for deeper notes.

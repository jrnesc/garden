# garden

Neural characters walking through Gaussian splat worlds in the browser.

Live app: <https://garden.jrnescoffery.workers.dev/>

## What This Is

Garden combines:

- **Neural locomotion** — Meta `ai4animationpy` CodebookMatching models exported to ONNX and run client-side with ONNX Runtime WebAssembly.
- **Physics** — Rapier WASM for terrain collision, wall collision, and foot grounding.
- **Gaussian splats** — World Labs Marble scenes rendered with Spark.
- **Experimental NVIDIA MotionBricks** — a local-only `/nvidia-motion` G1 demo driven by a Python SSE worker.

WASD + camera orbit. Dog, wolf, and Geno characters. The main walker runs in-browser; generated splat worlds are stored through the Cloudflare Worker/R2 pipeline.

## Model

> Sebastian Starke, Paul Starke, Nicky He, Taku Komura, Yuting Ye.  
> **Categorical Codebook Matching for Embodied Character Controllers.**  
> *ACM Transactions on Graphics (SIGGRAPH 2024).* [doi:10.1145/3658209](https://dl.acm.org/doi/10.1145/3658209)

- Paper: <https://dl.acm.org/doi/10.1145/3658209>
- Reference code: <https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2024>
- Python framework: <https://github.com/facebookresearch/ai4animationpy>

## Setup

This is not a one-click clone. You need to export the ONNX models yourself, provide API keys, and generate your own splat worlds.

Prerequisites:

- Node 20+
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Cloudflare Wrangler
- World Labs Marble API key as `WLT_API_KEY`
- xAI API key as `XAI_API_KEY`

Export models:

```bash
git clone https://github.com/facebookresearch/ai4animationpy.git
cd locomotion-server
uv sync
uv run python export_onnx.py
uv run python export_quadruped.py
```

Run the app:

```bash
cd app
npm install
npm run dev
```

Run the API worker locally:

```bash
cd worker
npm install
npm run dev
```

Open <http://localhost:3000>. `/character` is the flat-ground locomotion demo, `/walk` is the splat walker, and `/splats` lists generated worlds. `/history` redirects to `/splats`.

## NVIDIA MotionBricks

`/nvidia-motion` is an experimental local G1 locomotion demo. The browser renders the G1 meshes, while live motion streams from NVIDIA MotionBricks through a Python SSE worker.

To run it, provide `GR00T-WholeBodyControl/` alongside this repo and set up its MotionBricks Python environment. That directory is intentionally gitignored because it is a large external dependency. This route is not deployable to Cloudflare as-is because it depends on a local Python process.

## Layout

- `app/` — Next.js + Three.js frontend.
- `worker/` — Cloudflare Worker for Marble/Grok proxying and R2-backed splat journeys.
- `locomotion-server/` — Python export and validation tools.
- `ai4animationpy/` — external Meta framework checkout, gitignored.
- `GR00T-WholeBodyControl/` — external NVIDIA MotionBricks/G1 checkout, gitignored.

See `ARCHITECTURE.md` for deeper notes.

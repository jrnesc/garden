# Architecture Overview

## Structure

```
garden/
├── app/                          # Next.js frontend (the actual product)
│   ├── src/
│   │   ├── app/                  # Pages (file-system routing)
│   │   │   ├── page.tsx          # Home — "inside" landing
│   │   │   ├── character/        # Neural locomotion demo (ONNX in browser)
│   │   │   ├── walk/             # Gaussian splat walker (character + physics + world)
│   │   │   ├── splats/           # List of generated splat worlds
│   │   │   └── nvidia-motion/    # Local-only MotionBricks / G1 experiment
│   │   ├── components/
│   │   │   ├── Walker.tsx        # Splat world renderer + character + physics + camera
│   │   │   ├── Nav.tsx           # Navigation bar
│   │   │   └── hud-icons.tsx     # SVG icons + HUD styling constants
│   │   └── lib/
│   │       ├── locomotion.ts     # Neural animation engine (ONNX)
│   │       ├── physics.ts        # Rapier WASM — trimesh collider, capsule KCC
│   │       ├── api.ts            # Worker API client (fetch wrapper)
│   │       ├── journey.ts        # World/journey type definitions
│   │       └── collider.ts       # GLB collider mesh loader
│   └── public/                   # Static assets
│       ├── locomotion.onnx       # Biped neural net (34.7 MB)
│       ├── quadruped.onnx        # Quadruped neural net (33.7 MB)
│       ├── locomotion-data.json  # Biped companion data (bones, styles)
│       ├── quadruped-data.json   # Quadruped companion data
│       ├── character.glb         # Geno biped mesh
│       ├── dog.glb               # German Shepherd mesh + textures
│       ├── wolf.glb              # Wolf mesh + textures
│       └── paintings/            # Source images for world generation
│
├── worker/                       # Cloudflare Worker (API backend)
│   └── src/index.ts              # World Labs API proxy, image generation
│
├── locomotion-server/            # Python tools (NOT runtime — export/reference only)
│   ├── export_onnx.py            # Biped ONNX export script
│   ├── export_quadruped.py       # Quadruped ONNX export script
│   ├── server.py                 # Original Python WebSocket server (REFERENCE)
│   ├── record_clips.py           # Pre-baked animation recording
│   ├── validate_onnx.py          # Python vs ONNX output comparison
│   └── pyproject.toml + uv.lock  # Python deps (use uv, not pip)
│
├── ai4animationpy/               # Meta's animation framework (VENDORED)
│                                 # Source of truth for neural nets, math, bone definitions
│                                 # Only needed for exports and reference — not runtime
│
├── GR00T-WholeBodyControl/       # NVIDIA MotionBricks / G1 source (gitignored)
│                                 # Required only for local /nvidia-motion streaming
│
└── concept.md                    # Project concept/spec
```

### ACTIVE (used at runtime)
- `app/src/app/page.tsx` — landing page
- `app/src/app/character/` — character locomotion demo
- `app/src/app/walk/` — splat world walker (main experience)
- `app/src/app/splats/` — generated splat world list
- `app/src/app/nvidia-motion/` — local-only NVIDIA MotionBricks / G1 demo
- `app/src/app/api/motionbricks/` — local SSE/control endpoints for MotionBricks
- `app/src/components/Walker.tsx` — splat renderer + character + physics + camera
- `app/src/components/Nav.tsx` — navigation bar
- `app/src/components/hud-icons.tsx` — icons + HUD constants
- `app/src/lib/locomotion.ts` — neural animation engine
- `app/src/lib/physics.ts` — Rapier physics (trimesh collider, capsule KCC)
- `app/src/lib/api.ts` — worker API calls
- `app/src/lib/journey.ts` — type definitions
- `app/src/lib/collider.ts` — collider mesh loading
- `app/public/` — ONNX models, GLBs, companion data, paintings
- `worker/` — API backend

### LOCAL EXPERIMENTAL
- `GR00T-WholeBodyControl/` — external NVIDIA MotionBricks checkout, intentionally gitignored
- `GR00T-WholeBodyControl/tools/motionbricks_stream_worker.py` — Python stream worker used by `/api/motionbricks/stream`
- `/nvidia-motion` is not Cloudflare-deployable as-is because it depends on that local Python process.

### REFERENCE (needed for development, not runtime)
- `locomotion-server/export_onnx.py` — re-export biped model
- `locomotion-server/export_quadruped.py` — re-export quadruped model
- `locomotion-server/server.py` — Python reference implementation
- `locomotion-server/validate_onnx.py` — validation tool
- `locomotion-server/record_clips.py` — clip recording
- `locomotion-server/pyproject.toml` + `uv.lock` — Python deps
- `ai4animationpy/` — Meta's framework (source for models + math reference)

## Large Files

- `locomotion.onnx` — 34.7 MB
- `quadruped.onnx` — 33.7 MB
- `dog.glb` — 2.3 MB
- `wolf.glb` — 1.6 MB
- `ai4animationpy/` — entire vendored framework

Consider git-lfs for ONNX + GLBs, .gitignore for ai4animationpy.

## Deployment

**Cloudflare Workers** (next priority):
- `app/wrangler.toml` — Next.js via OpenNext (`inside-app` worker)
- `worker/wrangler.toml` — API worker (`inside-api`, R2 bucket `inside-journeys`)
- App has service binding to API worker
- `@opennextjs/cloudflare` already in deps
- ONNX + GLB files may exceed Workers asset size limits — may need R2 or external CDN
- WASM modules (onnxruntime-web, Rapier) load client-side, no server-side WASM needed

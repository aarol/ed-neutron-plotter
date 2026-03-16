# AGENTS.md — ed-galaxy

## Project Overview

**ed-galaxy** is an Elite Dangerous neutron star route plotter and galaxy visualizer. It renders 4.4 million stars in the browser at 60/144 FPS using Three.js + WebGPU, provides instant autocomplete search over all star names, and computes jump routes between stars using neutron star boosts.

The codebase has two parts:

1. **TypeScript web frontend** (`src/`) — rendering, UI, and orchestration
2. **Rust/WebAssembly module** (`rust-module/`) — autocomplete trie, KD-tree spatial index, and A* route plotter

---

## Architecture

```
src/                    TypeScript frontend (Vite + Three.js)
  main.tsx              Entry point; initializes WASM, Galaxy, worker, ray-pick, and mounts UI
  galaxy.ts             Three.js WebGPU scene (stars, route line, camera)
  api.ts                Star coordinate resolution (local WASM or remote EDSM API fallback)
  web-worker.ts         Comlink web worker wrapper — runs WASM pathfinding off-main-thread
  line-points.ts        Three.js helper for rendering the route line
  journal/journal.ts    File System Access API for reading ED journal files
  ui/                   Preact UI components and styles
    UI.tsx              Top-level UI orchestration (search, dialogs, target bar)
    SearchBox.tsx       Autocomplete search input component
    SearchBar.tsx       Top search row (search input + GPS button)
    TargetInfo.tsx      Bottom target info bar + route trigger
    RouteDialog.tsx     Route configuration dialog (from/to/supercharged)
    JournalDialog.tsx   Journal setup dialog UI
    *.module.css        Component-scoped styles
    shared.module.css   Shared style primitives composed by UI modules
    types.ts            Shared UI types (RouteConfig, target info)

rust-module/            Rust crate compiled to WASM with wasm-pack
  src/lib.rs            WASM bindings exposed to JS (Module struct)
  src/trie.rs           LOUDS trie — space-efficient autocomplete (4.4M stars < 17 MB gzipped)
  src/kdtree.rs         KD-tree for nearest-neighbour spatial queries
  src/plotter.rs        Beam-search A* route plotter using neutron star boosts
  src/system.rs         Core data types (Coords, System)
  src/fast_json_parser.rs  SIMD-accelerated JSON parser for bulk star ingestion
  src/bin/gen_star_data.rs Binary that converts raw spansh dumps → binary data files

public/data/            Pre-generated binary data (committed / gitignored varies)
  neutron_stars*.bin    Star position arrays split into chunks (Float32, x/y/z triplets)
  search_trie.bin       Serialised LOUDS trie
  star_kdtree.bin       Serialised KD-tree
```

---

## Key Design Decisions

- **WebGPU renderer** — Three.js `WebGPURenderer` with `PointsNodeMaterial` and additive blending. Stars are split across several `Float32Array` chunks so the GPU upload is incremental.
- **WASM on two threads** — The `Module` is instantiated twice: once on the main thread for fast synchronous autocomplete (`suggest_words`), and once in a Comlink web worker for async route computation (`find_route`) so it never blocks the UI.
- **LOUDS trie** — A level-order unary degree sequence trie serialised to `search_trie.bin`. The trie stores only searchable star names; coordinates are stored separately in sorted order so the trie doesn't need to embed them.
- **Beam-search A\*** — The plotter in `plotter.rs` uses an approximate A* with a configurable beam width to keep route finding fast over millions of nodes.
- **SharedArrayBuffer** — Star position data is placed in a `SharedArrayBuffer` so both the main thread and the worker can read star coordinates without copying.

---

## Development Commands

### Frontend

```sh
pnpm install        # Install JS dependencies
pnpm dev            # Start Vite dev server (hot reload)
pnpm build          # TypeScript check + Vite production build
pnpm preview        # Preview production build locally
```

### Rust / WASM

All commands run from the `rust-module/` directory.

```sh
# 1. Generate binary data files → public/data/
cargo run --release

# 2. Build the WASM package (output: rust-module/pkg/)
wasm-pack build --target web
```

Step 1 requires the raw spansh data dumps to be present in `rust-module/`:

- `systems_neutron.json` — from <https://spansh.co.uk/dumps>
- `systems_1day.json` (or similar) — from the same source

Step 2 must be re-run whenever Rust sources change.

---

## Data Flow

```
spansh dumps (JSON)
  └─ cargo run --release (gen_star_data.rs)
       ├─ neutron_stars*.bin  →  loaded by main.tsx → galaxy.loadStars()
       ├─ search_trie.bin     →  Module.set_trie()  → suggest_words()
       └─ star_kdtree.bin     →  Module.set_kdtree() / WasmWorker.setKDTree()

User types in search box
  └─ UI/SearchBox.onSuggest()  →  Module.suggest_words(prefix, n) (main thread, sync)

User submits route
  └─ UI/RouteDialog.onSubmit()  →  main.tsx handleGenerateRoute()
  └─ api.getStarCoords()        →  Module.get_coords_for_star() or EDSM API
  └─ WasmWorker.findRoute(start, end, callback)  →  plotter::plot() (worker thread)
       └─ routeReportCallback()  →  galaxy.setRoutePoints()  →  Three.js line update

Journal initialization
  └─ UI/JournalDialog.onInitialize()  →  main.tsx journal.init()

Star pick in 3D scene
  └─ main.tsx raycaster + Module.find_star_near_ray()  →  UI target info update
```

---

## Module API (WASM → JS)

Exposed on `Module` (from `rust-module/pkg`):

| Method | Description |
|---|---|
| `set_trie(data: Uint8Array)` | Load the LOUDS trie |
| `set_stars(stars: Float32Array)` | Load flat x/y/z star positions |
| `set_kdtree(data: Uint8Array)` | Load the KD-tree index |
| `suggest_words(prefix, n)` | Return up to `n` autocomplete suggestions |
| `get_coords_for_star(name)` | Return `{x, y, z}` or `undefined` |
| `find_route(start, end, cb)` | Run beam-search A*, calling `cb` with progress; returns flat `Float32Array` of route coords |

---

## File Conventions

- **UI location** — UI components live under `src/ui/` and are implemented as `*.tsx` Preact components.
- **CSS Modules** — Component-scoped styles use `*.module.css` (e.g. `ui/SearchBox.module.css`, `ui/RouteDialog.module.css`).
- **TypeScript strict mode** is enabled (`tsconfig.json`).
- **No test framework** is currently set up for the frontend. Rust unit tests live in the same source files (`#[cfg(test)]` blocks).
- The `vite.config.js` uses `vite-plugin-wasm` and `vite-plugin-top-level-await` to support WASM imports.

import "./style.css";
import { Galaxy } from "./galaxy";
import init, { Module } from "../rust-module/pkg/rust_module";
import { Raycaster, Vector2, Vector3 } from "three/webgpu";
import { api } from "./api";
import * as Comlink from "comlink";
import { WasmWorker as WasmWorkerAPI } from "./web-worker";
import Worker from "./web-worker?worker";
import { Journal } from "./journal/journal";
import { createRef, render } from "preact";
import { UI, type UIHandle } from "./ui/UI";
import { loadStoredFocusedSystem, saveStoredFocusedSystem } from "./ui/focusStorage";
import { ToastProvider } from "./ui/toast";
import type { RouteConfig, RouteNode, TargetInfoState } from "./ui/types";
import { RouteContext, RouteModel, type RouteState } from "./ui/state/routeModel";
import { effect } from "@preact/signals";

async function main() {
  await init();
  const galaxy = new Galaxy();
  await galaxy.init();

  // Wasm on the main thread for fast search, coordinate queries
  const primaryModule = new Module();

  // WASM pathfinding in a web worker
  const WasmWorker = Comlink.wrap<WasmWorkerAPI>(new Worker());

  // @ts-ignore
  const wasmWorker: Comlink.Remote<WasmWorkerAPI> = await new WasmWorker();
  const uiRef = createRef<UIHandle>();

  function onSuggest(prefix: string) {
    if (primaryModule) {
      console.time("trie-suggest");
      const trieResults = primaryModule.suggest_words(prefix, 10);
      console.timeEnd("trie-suggest");

      return trieResults;
    }
    return [];
  }

  function routeReportCallback(starData: Float32Array, distance: number, depth: number) {
    console.log(`Route report - Distance: ${distance}, Depth: ${depth}`);
    galaxy.setRoutePoints(starData);
  }

  const setTargetInfo = (target: TargetInfoState) => {
    uiRef.current?.setTargetInfo(target);
  };

  const focusSystem = (target: TargetInfoState) => {
    galaxy.setTarget(new Vector3(target.x, target.y, target.z));
    setTargetInfo(target);
    saveStoredFocusedSystem(target);
  };

  const handleSelectTarget = async (query: string): Promise<TargetInfoState | null> => {
    const pos = await api.getStarCoords(primaryModule, query);
    if (!pos) {
      return null;
    }

    console.log(`Found star "${query}": (${pos.x}, ${pos.y}, ${pos.z})`);
    const target = { name: query, x: pos.x, y: pos.y, z: pos.z };
    saveStoredFocusedSystem(target);
    galaxy.setTarget(new Vector3(target.x, target.y, target.z));
    return target;
  };

  const handleGenerateRoute = async (routeConfig: RouteConfig): Promise<RouteNode[]> => {
    galaxy.setRouteProgress(0) // Clear progress highlight while generating a new route

    const start = await api.getStarCoords(primaryModule, routeConfig.from);
    const end = await api.getStarCoords(primaryModule, routeConfig.to);

    if (!start || !end) {
      console.error("Could not get coordinates for stars", routeConfig.from, routeConfig.to);
      throw new Error(`Could not find coordinates for "${routeConfig.from}" or "${routeConfig.to}".`);
    }

    const res = await wasmWorker.findRoute(start, end, Comlink.proxy(routeReportCallback));
    if (res) {
      galaxy.setRoutePointsFromNodes(res);
      return res as RouteNode[];
    }

    return [];
  };

  const journal = new Journal({
    onNewLocation: async (starName, coords) => {
      console.log(`New location: ${starName} at (${coords.x}, ${coords.y}, ${coords.z})`);
      const target = {
        name: starName,
        x: coords.x / 1000,
        y: coords.y / 1000,
        z: coords.z / 1000,
      };
      focusSystem(target);
    },
  });

  const uiRoot = document.createElement("div");
  document.body.appendChild(uiRoot);

  const routeModel: RouteState = new RouteModel();

  effect(() => {
    const nodes = routeModel.nodes.value;
    const progress = routeModel.progress.value;
    galaxy.setRoutePointsFromNodes(nodes);
    galaxy.setRouteProgress(progress);
  })

  render(
    <ToastProvider>
      <RouteContext.Provider value={routeModel}>
        <UI
          onGenerateRoute={handleGenerateRoute}
          onInitializeJournal={() => journal.init()}
          onRestoreStoredRoute={(nodes, progress) => {
            galaxy.setRoutePointsFromNodes(nodes);
            galaxy.setRouteProgress(progress);
          }}
          onRouteSelectionChange={(checkedIndex) => {
            galaxy.setRouteProgress(checkedIndex);
          }}
          onStopJournalTracking={() => journal.stopTracking()}
          onSelectTarget={handleSelectTarget}
          onSuggest={onSuggest}
          ref={uiRef}
        />
      </RouteContext.Provider>
    </ToastProvider>,
    uiRoot,
  );

  const storedFocusedSystem = loadStoredFocusedSystem();
  if (storedFocusedSystem) {
    focusSystem(storedFocusedSystem);
  }

  // -----------------------------------------------------------------------
  // Star-click targeting
  // Cast a ray from the camera through the clicked pixel; project it to the
  // orbit-controls target depth and ask WASM for the nearest star.
  // A small drag-distance threshold avoids false positives while orbiting.
  // -----------------------------------------------------------------------
  const raycaster = new Raycaster();
  let pointerDownPos = new Vector2();
  let isDragging = false;

  galaxy.renderer.domElement.addEventListener('pointerdown', (e: PointerEvent) => {
    pointerDownPos.set(e.clientX, e.clientY);
    isDragging = false;
  });

  galaxy.renderer.domElement.addEventListener('pointermove', (e: PointerEvent) => {
    if (e.buttons === 0) return;
    const dx = e.clientX - pointerDownPos.x;
    const dy = e.clientY - pointerDownPos.y;
    if (dx * dx + dy * dy > 25) isDragging = true; // 5 px threshold
  });

  galaxy.renderer.domElement.addEventListener('pointerup', (e: PointerEvent) => {
    if (isDragging) return;

    // Convert to Normalised Device Coordinates [-1, 1]
    const ndc = new Vector2(
      (e.clientX / window.innerWidth) * 2 - 1,
      -(e.clientY / window.innerHeight) * 2 + 1,
    );

    raycaster.setFromCamera(ndc, galaxy.camera);

    // If the route line was clicked, target that point instead of some other nearby star
    const routeCoords = galaxy.routeLine.getHitSpriteCoords(raycaster);
    if (routeCoords) {
      const { x, y, z } = routeCoords;

      const name = primaryModule.get_star_from_coords(x, y, z);
      if (name) {
        focusSystem({ name, x, y, z });
        return;
      }
    }

    // Compute angular pick tolerance: ~8 px in screen space
    // camera.fov is vertical (degrees); convert to radians per pixel
    const fovRad = galaxy.camera.fov * (Math.PI / 180);
    const angularTolerance = (fovRad / window.innerHeight) * 8;

    const { origin, direction } = raycaster.ray;

    // WASM returns { name, coords: { x, y, z } } or null
    const hit = primaryModule.find_star_near_ray(
      origin.x, origin.y, origin.z,
      direction.x, direction.y, direction.z,
      angularTolerance,
    ) as { name: string; coords: { x: number; y: number; z: number } } | null;

    if (hit) {
      const { name, coords } = hit;
      focusSystem({ name, x: coords.x, y: coords.y, z: coords.z });
    }
  });

  fetch(`${import.meta.env.BASE_URL}data/neutron_stars0.bin`)
    .then((res) => res.arrayBuffer())
    .then(async starBuffer => {
      // Use SharedArrayBuffer for star data to share between main thread and worker
      const sab = new SharedArrayBuffer(starBuffer.byteLength);
      const sharedStarBuffer = new Uint8Array(sab);
      sharedStarBuffer.set(new Uint8Array(starBuffer));
      primaryModule.set_stars(new Float32Array(sab));
      galaxy.loadStars([new DataView(starBuffer)]);
      console.log("Star data loaded.");
      await wasmWorker.setStars(new Float32Array(sab));
    });

  fetch(`${import.meta.env.BASE_URL}data/search_trie.bin`)
    .then((res) => res.arrayBuffer())
    .then((trieBuffer) => {
      primaryModule.set_trie(new Uint8Array(trieBuffer));
      wasmWorker.setTrie(new Uint8Array(trieBuffer));
      console.log("Search trie loaded.");
    });

  fetch(`${import.meta.env.BASE_URL}data/star_kdtree.bin`)
    .then((res) => res.arrayBuffer())
    .then(async kdtreeBuffer => {
      const sab = new SharedArrayBuffer(kdtreeBuffer.byteLength);
      const sharedKdTreeBuffer = new Uint8Array(sab);
      sharedKdTreeBuffer.set(new Uint8Array(kdtreeBuffer));
      primaryModule.set_kdtree(new Uint8Array(kdtreeBuffer));
      console.log("Star KDTree loaded.");
      await wasmWorker.setKDTree(new Uint8Array(sab));
    });

  window.addEventListener("paste", (event) => {
    console.log(event.clipboardData?.types)
    const htmlData = event.clipboardData?.getData("text/html")
    if (htmlData) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(htmlData, "text/html");
      const table = doc.querySelector("table");
      console.log(table)
    }
  })
}

main();

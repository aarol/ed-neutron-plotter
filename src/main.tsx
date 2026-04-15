import "./style.css";
import { Galaxy } from "./galaxy";
import init, { Module } from "../rust-module/pkg/rust_module";
import { Vector3 } from "three/webgpu";
import * as Comlink from "comlink";
import { WasmWorker as WasmWorkerAPI } from "./web-worker";
import Worker from "./web-worker?worker";
import { render } from "preact";
import { focusedSystem, UI } from "./ui/UI";
import { ToastProvider } from "./ui/toast";
import type { RouteConfig, StarSystem } from "./ui/types";
import { RouteContext, RouteModel, type RouteState } from "./ui/state/routeModel";
import { effect } from "@preact/signals";
import { JournalContext, JournalModel, type JournalState } from "./ui/state/journalModel";
import { saveStoredFocusedSystem } from "./ui/state/localStorage";
import { api } from "./api";

async function main() {
  await init();

  // Wasm on the main thread for fast search, coordinate queries
  const primaryModule = new Module();
  const galaxy = new Galaxy();
  await galaxy.init(primaryModule);

  // WASM pathfinding in a web worker
  const WasmWorker = Comlink.wrap<WasmWorkerAPI>(new Worker());

  // @ts-ignore
  const wasmWorker: Comlink.Remote<WasmWorkerAPI> = await new WasmWorker();

  function autocompleteSearchWord(prefix: string) {
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


  const handleSelectTarget = async (query: string): Promise<StarSystem | null> => {
    const pos = await api.getStarCoords(primaryModule, query);
    if (!pos) {
      return null;
    }

    console.log(`Found star "${query}": (${pos.x}, ${pos.y}, ${pos.z})`);
    const target = { name: query, coords: pos };
    galaxy.setTarget(new Vector3(target.coords.x, target.coords.y, target.coords.z));
    return target;
  };

  const handleGenerateRoute = async (routeConfig: RouteConfig): Promise<StarSystem[]> => {
    galaxy.setRouteProgress(0) // Clear progress highlight while generating a new route

    const start = await api.getStarCoords(primaryModule, routeConfig.from);
    const end = await api.getStarCoords(primaryModule, routeConfig.to);

    if (!start || !end) {
      console.error("Could not get coordinates for stars", routeConfig.from, routeConfig.to);
      throw new Error(`Could not find coordinates for "${routeConfig.from}" or "${routeConfig.to}".`);
    }

    const res = await wasmWorker.findRoute(start, end, Comlink.proxy(routeReportCallback));
    if (res) {
      galaxy.setRoutePointsFromCoords(res.map(system => system.coords));
      return res as StarSystem[];
    }

    return [];
  };

  galaxy.onSystemFocus = (system: StarSystem) => {
    focusedSystem.value = system;
  }

  // This is a Preact Signal that will run when focusedSystem changes
  // There are a lot of these effect blocks because they bridge state
  // between the Preact and the imperative galaxy methods really well
  effect(() => {
    const system = focusedSystem.value;
    if (system) {
      const { x, y, z } = system.coords;
      console.log(`Focusing on system: ${system.name} at (${x}, ${y}, ${z})`);

      galaxy.setTarget(new Vector3(x, y, z));
      saveStoredFocusedSystem(system);
    }
  })

  const journalModel: JournalState = new JournalModel();

  effect(() => {
    const lastSystem = journalModel.lastSystem.value;
    if (lastSystem) {
      console.log(`New location from journal: ${lastSystem.name} at (${lastSystem.coords.x}, ${lastSystem.coords.y}, ${lastSystem.coords.z})`);
      galaxy.setLiveLocation(new Vector3(lastSystem.coords.x, lastSystem.coords.y, lastSystem.coords.z));
      focusedSystem.value = lastSystem;

      routeModel.markVisitedSystem(lastSystem.name);
      return;
    }

    galaxy.setLiveLocation(null);
  });

  effect(() => {
    if (!journalModel.enabled.value) {
      console.log("Journal tracking disabled, clearing live location.");
      galaxy.setLiveLocation(null);
    }
  });

  const routeModel: RouteState = new RouteModel();

  effect(() => {
    const progress = routeModel.progress.value;
    const routePoints = routeModel.nodes.value.map(node => node.system.coords);
    galaxy.setRoutePointsFromCoords(routePoints);
    galaxy.setRouteProgress(progress);
  })

  const uiRoot = document.createElement("div");
  document.body.appendChild(uiRoot);

  render(
    <ToastProvider>
      <RouteContext.Provider value={routeModel}>
        <JournalContext.Provider value={journalModel}>
          <UI
            onGenerateRoute={handleGenerateRoute}
            autocomplete={autocompleteSearchWord}
            onSelectTarget={handleSelectTarget}
          />
        </JournalContext.Provider>
      </RouteContext.Provider>
    </ToastProvider>,
    uiRoot,
  );

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
}

main();

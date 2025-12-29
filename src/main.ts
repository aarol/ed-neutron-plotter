import "./style.css";
import { SearchBox } from "./search";
import { Galaxy } from "./galaxy";
import { RouteDialog } from "./route-dialog";
import init, { Module } from "../rust-module/pkg";
import { Vector3 } from "three/webgpu";
import { api } from "./api";
import * as Comlink from "comlink";
import { WasmWorker as WasmWorkerAPI } from "./web-worker";
import Worker from "./web-worker?worker"

async function main() {
  const galaxy = new Galaxy();
  await galaxy.init();

  await init();

  // Wasm on the main thread for fast search, coordinate queries
  const primaryModule = new Module();

  // Pathfinding in a web worker
  const WasmWorker = Comlink.wrap<WasmWorkerAPI>(new Worker());

  // @ts-ignore
  const wasmWorker: Comlink.Remote<WasmWorkerAPI> = await new WasmWorker();

  function onSuggest(prefix: string) {
    if (primaryModule) {
      return primaryModule.suggest_words(prefix, 10);
    }
    return [];
  }

  // Create route dialog
  const routeDialog = new RouteDialog({
    onSuggest,
  });

  function routeReportCallback(starData: Float32Array, distance: number, depth: number) {
    console.log(`Route report - Distance: ${distance}, Depth: ${depth}`);
    galaxy.setRoutePoints(starData);
  }

  const openRoutePanel = async (word: string) => {
    // Pre-fill the "to" field with the current search term
    routeDialog.setToValue(word);

    // Open panel and wait for user configuration
    const routeConfig = await routeDialog.open();

    if (routeConfig) {
      console.log('Route configuration:', routeConfig);

      const start = await api.getStarCoords(primaryModule, routeConfig.from);
      const end = await api.getStarCoords(primaryModule, routeConfig.to);

      if (!start || !end) {
        console.error("Could not get coordinates for stars", routeConfig.from, routeConfig.to);
        return;
      }

      const res = await wasmWorker.findRoute(start, end, Comlink.proxy(routeReportCallback))
      if (res) {
        galaxy.setRoutePoints(res);
      }
    }
  };

  const searchBox = new SearchBox({
    placeholder: "Enter target star..",
    onSearch: async (query: string) => {
      const pos = await api.getStarCoords(primaryModule, query)
      if (pos) {
        console.log("Fetched star coordinates from API:", pos);
        console.log(`Found star "${query}": (${pos.x}, ${pos.y}, ${pos.z})`);
        galaxy.setTarget(new Vector3(pos.x, pos.y, pos.z));

        openRoutePanel(query);
      } else {
        window.alert("Star not found: " + query);
      }
    },
    onSuggest: onSuggest,
    onClickRoute: async (word: string) => {
      openRoutePanel(word);
    }
  });

  searchBox.mount(document.body);

  window.addEventListener("keydown", async (event) => {
    if (event.key === "p") {
      
      const start = await api.getStarCoords(primaryModule, "Sol");
      const end = await api.getStarCoords(primaryModule, "Colonia");
      const res = await wasmWorker.findRoute(start!, end!, Comlink.proxy(routeReportCallback))
      if (res) {
        galaxy.setRoutePoints(res);
      }
    }
  })

  fetch("/data/neutron_stars0.bin")
    .then(res => res.arrayBuffer())
    .then(async starBuffer => {
      // Use SharedArrayBuffer for star data to share between main thread and worker
      let sab = new SharedArrayBuffer(starBuffer.byteLength);
      let sharedStarBuffer = new Uint8Array(sab);
      sharedStarBuffer.set(new Uint8Array(starBuffer));
      primaryModule.set_stars(new Float32Array(sab))
      galaxy.loadStars([new DataView(starBuffer)]);
      console.log("Star data loaded.");
      await wasmWorker.setStars(new Float32Array(sab));
    })

  fetch("/data/search_trie.bin")
    .then(res => res.arrayBuffer())
    .then(trieBuffer => {
      primaryModule.set_trie(new Uint8Array(trieBuffer))
      console.log("Search trie loaded.");
    });

  fetch("/data/star_kdtree.bin")
    .then(res => res.arrayBuffer())
    .then(async kdtreeBuffer => {
      let sab = new SharedArrayBuffer(kdtreeBuffer.byteLength);
      let sharedKdTreeBuffer = new Uint8Array(sab);
      sharedKdTreeBuffer.set(new Uint8Array(kdtreeBuffer));
      primaryModule.set_kdtree(new Uint8Array(kdtreeBuffer))
      console.log("Star KDTree loaded.");
      await wasmWorker.setKDTree(new Uint8Array(sab));
    });
}
main();

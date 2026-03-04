import "./style.css";
import { SearchBox } from "./search";
import { Galaxy } from "./galaxy";
import { RouteDialog } from "./route-dialog";
import { JournalDialog } from "./journal/journal-dialog";
import init, { Module } from "../rust-module/pkg";
import { Raycaster, Vector2, Vector3 } from "three/webgpu";
import { api } from "./api";
import * as Comlink from "comlink";
import { WasmWorker as WasmWorkerAPI } from "./web-worker";
import Worker from "./web-worker?worker"
import { Journal } from "./journal/journal";

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

      console.time("trie-suggest");
      const trieResults = primaryModule.suggest_words(prefix, 10);
      console.timeEnd("trie-suggest");

      return trieResults;
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
        res.forEach(node => console.log(`Route node: ${node.name} at (${node.coords.x}, ${node.coords.y}, ${node.coords.z})`));
        // galaxy.setRoutePoints(res);
      }
    }
  };

  const searchBox = new SearchBox({
    placeholder: "Enter target star..",
    onSearch: async (query: string) => {
      const pos = await api.getStarCoords(primaryModule, query)
      if (pos) {
        console.log(`Found star "${query}": (${pos.x}, ${pos.y}, ${pos.z})`);
        galaxy.setTarget(new Vector3(pos.x, pos.y, pos.z));
        setTargetInfo(query, pos.x, pos.y, pos.z);

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

  const searchWrapper = document.createElement('div');
  searchWrapper.className = 'search-wrapper';

  searchBox.mount(searchWrapper);

  // Standalone GPS icon — opens the journal dialog
  const gpsIcon = document.createElement('div');
  gpsIcon.className = 'gps-icon';
  gpsIcon.title = 'Track in-game location';
  gpsIcon.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/><circle cx="12" cy="12" r="3"/></svg>`;
  gpsIcon.addEventListener('click', () => {
    const dialog = document.getElementById('journalDialog') as HTMLDialogElement | null;
    dialog?.showModal();
  });
  searchWrapper.appendChild(gpsIcon);

  document.body.appendChild(searchWrapper);

  // Target info bar
  const targetInfo = document.createElement('div');
  targetInfo.className = 'target-info';
  let currentTargetName = 'Sol';
  const setTargetInfo = (name: string, x: number, y: number, z: number) => {
    currentTargetName = name;
    targetInfo.innerHTML =
      `<span class="target-info__name">${name}</span>` +
      `<span class="target-info__coords">(${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})</span>` +
      `<button class="target-info__route-btn" title="Find route to target" aria-label="Plot route">` +
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" fill="currentColor">` +
          `<path d="M320-360h80v-120h140v100l140-140-140-140v100H360q-17 0-28.5 11.5T320-520v160ZM480-80q-15 0-29.5-6T424-104L104-424q-12-12-18-26.5T80-480q0-15 6-29.5t18-26.5l320-320q12-12 26.5-18t29.5-6q15 0 29.5 6t26.5 18l320 320q12 12 18 26.5t6 29.5q0 15-6 29.5T856-424L536-104q-12 12-26.5 18T480-80ZM320-320l160 160 320-320-320-320-320 320 160 160Zm160-160Z"/>` +
        `</svg>` +
      `</button>`;
  };
  setTargetInfo('Sol', 0, 0, 0);
  targetInfo.addEventListener('click', (e) => {
    if ((e.target as Element).closest('.target-info__route-btn')) {
      openRoutePanel(currentTargetName);
    }
  });
  document.body.appendChild(targetInfo);

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
      (e.clientX / window.innerWidth)  *  2 - 1,
      -(e.clientY / window.innerHeight) * 2 + 1,
    );

    raycaster.setFromCamera(ndc, galaxy.camera);

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
      galaxy.setTarget(new Vector3(coords.x, coords.y, coords.z));
      setTargetInfo(name, coords.x, coords.y, coords.z);
    }
  });

  const journal = new Journal({
    onNewLocation: async (starName, coords) => {
      console.log(`New location: ${starName} at (${coords.x}, ${coords.y}, ${coords.z})`);
      galaxy.setTarget(new Vector3(coords.x / 1000, coords.y / 1000, coords.z / 1000));
      setTargetInfo(starName, coords.x / 1000, coords.y / 1000, coords.z / 1000);
    }
  });

  const journalDialog = new JournalDialog(journal);
  journalDialog.mount(document.body);

  fetch(`${import.meta.env.BASE_URL}data/neutron_stars0.bin`)
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

  fetch(`${import.meta.env.BASE_URL}data/search_trie.bin`)
    .then(res => res.arrayBuffer())
    .then(trieBuffer => {
      primaryModule.set_trie(new Uint8Array(trieBuffer))
      console.log("Search trie loaded.");
    });

  fetch(`${import.meta.env.BASE_URL}data/star_kdtree.bin`)
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

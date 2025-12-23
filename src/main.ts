import "./style.css";
import { SearchBox } from "./search";
import { api } from "./api";
import { Galaxy } from "./galaxy";
import { RouteDialog } from "./route-dialog";
import * as wasm from "../rust-module/pkg";
import { Vector3 } from "three/webgpu";

async function main() {
  const galaxy = new Galaxy();
  await galaxy.init();

  wasm.init(); // set panic hook, might need to move this somewhere else

  let route_kdtree: undefined | ArrayBuffer = undefined;
  let starPosData: undefined | Float32Array = undefined;
  let searcher: wasm.Search | undefined = undefined;

  function onSuggest(prefix: string) {
    if (searcher) {
      return searcher.suggest_words(prefix, 10) as string[];
    }
    return [];
}

// Create route dialog
const routeDialog = new RouteDialog({
  onSuggest,
});

const openRoutePanel = async (word: string) => {
  // Pre-fill the "to" field with the current search term
  routeDialog.setToValue(word);

  // Open panel and wait for user configuration
  const routeConfig = await routeDialog.open();

  if (routeConfig) {
    console.log('Route configuration:', routeConfig);

    if (!route_kdtree) {
      console.log("Fetching route_kdtree.bin..")
      route_kdtree = await fetch("data/star_kdtree.bin").then(r => r.arrayBuffer())
    }

    const res = wasm.find_route(new Uint8Array(route_kdtree!), starPosData!, routeConfig.from.coords, routeConfig.to.coords,
      (report: any) => {
        console.log(report);
      })
    console.log(res)
  }
};

const searchBox = new SearchBox({
  placeholder: "Enter target star..",
  onSearch: async (query: string) => {
    // Open the route panel when Enter is pressed
    openRoutePanel(query);
    const pos = await api.getStarCoords(query);
    if (pos) {
      console.log(`Found star "${query}" at (${pos.x}, ${pos.y}, ${pos.z})`);
      galaxy.setTarget(new Vector3(pos.x, pos.y, pos.z));
    } else {
      alert(`Star "${query}" not found.`);
    }
  },
  onSuggest: onSuggest,
  onClickRoute: async (word: string) => {
    openRoutePanel(word);
  }
});

searchBox.mount(document.body);

fetch("/data/search_trie.bin").then(res => res.arrayBuffer())
  .then(buffer => {
    searcher = new wasm.Search(new Uint8Array(buffer));
    console.log("Search trie loaded.");
  })

await Promise.all([0]
  // const starPositionArrays = await Promise.all([0, 1, 2, 3, 4, 5, 6, 7, 8]
  .map(i => fetch(`/data/neutron_stars${i}.bin`)
    .then(res => res.arrayBuffer())
  ))
  .then(arrays => {
    starPosData = new Float32Array(arrays[0]) // For now, only load the first array

    galaxy.loadStars(arrays.map(arr => new DataView(arr)))
  })
}

main();

import "./style.css";
import { SearchBox } from "./search";
import { Galaxy } from "./galaxy";
import { RouteDialog } from "./route-dialog";
import * as wasm from "../rust-module/pkg";
import { Vector3 } from "three/webgpu";
import { api, wasmModule as wasmModule } from "./api";

async function main() {
  const galaxy = new Galaxy();
  await galaxy.init();

  wasm.init(); // set panic hook, might need to move this somewhere else

  function onSuggest(prefix: string) {
    if (wasmModule) {
      return wasmModule.suggest_words(prefix, 10);
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

      const res = api.findRoute(routeConfig.from.coords, routeConfig.to.coords, (report) => {
        console.log("Route finding report:", report);
      })
      console.log("Route result:", res);
    }
  };

  const searchBox = new SearchBox({
    placeholder: "Enter target star..",
    onSearch: async (query: string) => {
      const pos = await api.getStarCoords(query)
      if (pos) {
        if (!(pos instanceof wasm.Coords)) {
          console.log("Fetched star coordinates from API:", pos);
        }
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

  fetch("/data/neutron_stars0.bin")
    .then(res => res.arrayBuffer())
    .then(starBuffer => {
      wasmModule.set_stars(new Float32Array(starBuffer))
      galaxy.loadStars([new DataView(starBuffer)]);
      console.log("Star data loaded.");
    })
  fetch("/data/search_trie.bin")
    .then(res => res.arrayBuffer())
    .then(trieBuffer => {
      wasmModule.set_trie(new Uint8Array(trieBuffer))
      console.log("Search trie loaded.");
    });

  fetch("/data/star_kdtree.bin")
    .then(res => res.arrayBuffer())
    .then(kdtreeBuffer => {
      wasmModule.set_kdtree(new Uint8Array(kdtreeBuffer))
      console.log("Star KDTree loaded.");
    });
}
main();

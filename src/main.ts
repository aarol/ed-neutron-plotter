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

  let route_kdtree: undefined | ArrayBuffer = undefined;
  let starPosData: undefined | Float32Array = undefined;

  function onSuggest(prefix: string) {
    if (wasmModule) {
      return wasmModule.suggest_words(prefix, 10) as string[];
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
      const pos = wasmModule?.get_coords_for_star(query) ?? await api.getStarCoords(query);
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
}
main();

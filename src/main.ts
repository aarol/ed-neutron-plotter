import "./style.css";
import { SearchBox } from "./search";
import { api } from "./api";
import { Galaxy } from "./galaxy";
import { RouteDialog } from "./route-dialog";
import * as wasm from "../rust-module/pkg";

async function main() {
  const galaxy = new Galaxy();
  await galaxy.init();

  let trie_bin: undefined | ArrayBuffer = undefined;
  let route_kdtree: undefined | ArrayBuffer = undefined;

  // Create route dialog
  const routeDialog = new RouteDialog({
    onSuggest: (word: string) => {
      if (trie_bin) {
        return wasm.suggest_words(new Uint8Array(trie_bin), word, 10) as string[];
      }
      return [];
    }
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
        debugger
        route_kdtree = await fetch("data/star_kdtree.bin").then(r => r.arrayBuffer())
      }

      const res = wasm.find_route(new Uint8Array(route_kdtree!), routeConfig.from.coords, routeConfig.to.coords)
      console.log(res)
    }
  };

  const searchBox = new SearchBox({
    placeholder: "Enter target star..",
    onSearch: async (query: string) => {
      // Open the route panel when Enter is pressed
      openRoutePanel(query);
    },
    onSuggest: (word: string) => {
      if (trie_bin) {
        return wasm.suggest_words(new Uint8Array(trie_bin), word, 10) as string[];
      }
      return [];
    },
    onClickRoute: async (word: string) => {
      openRoutePanel(word);
    }
  });

  searchBox.mount(document.body);

  fetch("/data/search_trie.bin").then(res => res.arrayBuffer())
    .then(buffer => {
      trie_bin = buffer;
    })
}

main();

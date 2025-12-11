import "./style.css";
import { SearchBox } from "./search";
import { api } from "./api";
import { Galaxy } from "./galaxy";
import * as wasm from "../rust-module/pkg";

async function main() {
  const galaxy = new Galaxy();
  await galaxy.init();
  let trie_bin: undefined | ArrayBuffer = undefined;

  const searchBox = new SearchBox({
    placeholder: "Search stars..",
    onSearch: async (query: string) => {
      const target = await api.getStarCoords(query);
      galaxy.setTarget(target);
    },
    onSuggest: (word: string) => {
      if (trie_bin) {
        return wasm.suggest_words(new Uint8Array(trie_bin), word, 10) as string[];
      }
      return [];
    }
  });

  searchBox.mount(document.body);

  fetch("/data/search_trie.bin").then(res => res.arrayBuffer())
    .then(buffer => {
      trie_bin = buffer;
    })
}

main();

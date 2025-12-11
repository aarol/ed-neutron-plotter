mod trie;
mod utils;

use wasm_bindgen::prelude::*;

use crate::trie::CompactPatriciaTrie;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, rust-module!");
}

#[wasm_bindgen]
pub fn suggest_words(trie: &[u8], prefix: &str, num_suggestions: usize) -> Vec<JsValue> {
    let trie = CompactPatriciaTrie::from_bytes(trie);

    trie.suggest(prefix, num_suggestions)
        .into_iter()
        .map(|s| JsValue::from_str(s.as_str()))
        .collect()
}

#[wasm_bindgen]
pub fn contains(trie: &[u8], prefix: &str) -> JsValue {
    let trie = CompactPatriciaTrie::from_bytes(trie);

    JsValue::from_bool(trie.contains(prefix))
}

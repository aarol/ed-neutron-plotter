pub mod fast_json_parser;
pub mod star;
pub mod trie;
pub mod utils;
pub mod plotter;

use rkyv::rancor::Error;
use wasm_bindgen::prelude::*;

use crate::{star::Coords, trie::CompactRadixTrie};

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
    let trie = CompactRadixTrie::from_bytes(trie);

    trie.suggest(prefix, num_suggestions)
        .into_iter()
        .map(|s| JsValue::from_str(s.as_str()))
        .collect()
}

#[wasm_bindgen]
pub fn contains(trie: &[u8], prefix: &str) -> JsValue {
    let trie = CompactRadixTrie::from_bytes(trie);

    JsValue::from_bool(trie.contains(prefix))
}

#[wasm_bindgen]
pub fn find_route(kdtree_bytes: &[u8], start: JsValue, end: JsValue) -> Vec<JsValue> {
    let kdtree: kiddo::ImmutableKdTree<f32, 3> =
        rkyv::from_bytes::<kiddo::ImmutableKdTree<f32, 3>, Error>(kdtree_bytes).expect("Valid kdtree");
    
    log_u32(kdtree.size() as u32);

    let start: Coords = serde_wasm_bindgen::from_value(start).unwrap();
    let end: Coords = serde_wasm_bindgen::from_value(end).unwrap();

    log(&format!(
        "Finding route from ({}, {}, {}) to ({}, {}, {})",
        start.x, start.y, start.z, end.x, end.y, end.z
    ));



    vec![]
}
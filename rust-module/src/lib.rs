pub mod fast_json_parser;
pub mod kdtree;
pub mod plotter;
pub mod system;
pub mod trie;
pub mod utils;
mod ordered_f32;
mod louds_trie;

use wasm_bindgen::prelude::*;

use crate::{fast_json_parser::JsonCoords, system::Coords};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_u32(a: u32);
}

#[wasm_bindgen]
pub fn init() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, rust-module!");
}

// #[wasm_bindgen]
// pub fn suggest_words(trie: &[u8], prefix: &str, num_suggestions: usize) -> Vec<JsValue> {
//     let trie = CompactRadixTrie::from_bytes(trie);

//     trie.suggest(prefix, num_suggestions)
//         .into_iter()
//         .map(|s| JsValue::from_str(s.as_str()))
//         .collect()
// }

// #[wasm_bindgen]
// pub fn contains(trie: &[u8], prefix: &str) -> JsValue {
//     let trie = CompactRadixTrie::from_bytes(trie);

//     JsValue::from_bool(trie.contains(prefix))
// }



#[wasm_bindgen]
pub fn find_route(
    kdtree_bytes: &[u8],
    stars: &[f32],
    start: JsValue,
    end: JsValue,
    report_callback: &js_sys::Function,
) -> Vec<Coords> {
    let kdtree = kdtree::CompactKdTree::from_bytes(kdtree_bytes);

    log_u32(kdtree.size() as u32);

    let start: JsonCoords = serde_wasm_bindgen::from_value(start).unwrap();
    let end: JsonCoords = serde_wasm_bindgen::from_value(end).unwrap();

    let start = Coords::from(&start);
    let end = Coords::from(&end);

    log(&format!("Finding route from ({}) to ({})", start, end));

    let ship = plotter::Ship {
        fuel_tank_size: 32.0,
        jump_range: 80.0,
        max_fuel_per_jump: 4.0,
        base_mass: 100.0,
        fsd_optimized_mass: 50.0,
        fsd_boost_factor: 1.2,
        fsd_rating_val: 5.0,
        fsd_class_val: 3.0,
    };

    let stars = stars.windows(3).map(Coords::from_slice).collect::<Vec<_>>();

    let cb = |report: plotter::Report| {
        let report = serde_wasm_bindgen::to_value(&report).unwrap();
        let _ = report_callback.call1(&JsValue::NULL, &report);
    };

    plotter::plot(start, end, &stars, &ship, kdtree, cb)
}

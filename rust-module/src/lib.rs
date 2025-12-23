pub mod fast_json_parser;
pub mod kdtree;
mod ordered_f32;
pub mod plotter;
pub mod system;
pub mod trie;
pub mod utils;

use wasm_bindgen::prelude::*;

use crate::{system::Coords, trie::LoudsTrie};

/// Stores the trie and other data needed for autocomplete and plotter
#[wasm_bindgen]
pub struct Searcher {
    trie: LoudsTrie,
    stars: Vec<Coords>,
}

#[wasm_bindgen]
impl Searcher {
    #[wasm_bindgen(constructor)]
    pub fn new(trie_data: &[u8], stars: &[f32]) -> Searcher {
        // This is not zero-cost because the succinct data structures need to be initialized
        let trie = LoudsTrie::from(trie_data);
        let stars = stars.chunks(3).map(Coords::from_slice).collect::<Vec<_>>();

        Searcher { trie, stars }
    }

    pub fn suggest_words(&self, prefix: &str, num_suggestions: usize) -> Vec<JsValue> {
        self.trie
            .suggest(prefix, num_suggestions)
            .into_iter()
            .map(|s| JsValue::from_str(s.as_str()))
            .collect()
    }

    pub fn get_coords_for_star(&self, star_name: &str) -> Option<Coords> {
        self.trie.find(star_name).map(|index| {
            let coords = self.stars[index as usize];
            coords
        })
    }
}

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
pub fn find_route(
    kdtree_bytes: &[u8],
    stars: &[f32],
    start: JsValue,
    end: JsValue,
    report_callback: &js_sys::Function,
) -> Vec<Coords> {
    let kdtree = kdtree::CompactKdTree::from_bytes(kdtree_bytes);

    log_u32(kdtree.size() as u32);

    let start: Coords = serde_wasm_bindgen::from_value(start).unwrap();
    let end: Coords = serde_wasm_bindgen::from_value(end).unwrap();

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

    let report_callback = |report: plotter::Report| {
        let report = serde_wasm_bindgen::to_value(&report).unwrap();
        let _ = report_callback.call1(&JsValue::NULL, &report);
    };

    plotter::plot(start, end, &stars, &ship, kdtree, report_callback)
}

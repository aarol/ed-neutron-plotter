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
#[derive(Default)]
pub struct Module {
    trie: Option<LoudsTrie>,
    stars: Option<Box<[Coords]>>,
    kdtree: Option<kdtree::CompactKdTree>,
}

#[wasm_bindgen]
impl Module {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Module {
        Module::default()
    }

    pub fn set_trie(&mut self, trie_data: &[u8]) {
        self.trie = Some(LoudsTrie::from(trie_data));
    }

    pub fn set_stars(&mut self, stars: Box<[f32]>) {
        // Convert flat f32 array to coordinates without copying
        self.stars = Some(stars.chunks_exact(3).map(Coords::from_slice).collect());
    }

    pub fn set_kdtree(&mut self, kdtree_data: &[u8]) {
        self.kdtree = Some(kdtree::CompactKdTree::from_bytes(kdtree_data));
    }

    pub fn suggest_words(&self, prefix: &str, num_suggestions: usize) -> Vec<JsValue> {
        match self.trie {
            Some(ref trie) => trie
                .suggest(prefix, num_suggestions)
                .into_iter()
                .map(|s| JsValue::from_str(s.as_str()))
                .collect(),
            None => vec![],
        }
    }

    pub fn get_coords_for_star(&self, star_name: &str) -> Option<Coords> {
        match (&self.trie, &self.stars) {
            (Some(trie), Some(stars)) => trie.find(star_name).map(|index| stars[index as usize]),
            _ => None,
        }
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

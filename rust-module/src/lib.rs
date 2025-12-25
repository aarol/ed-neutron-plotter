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

    pub fn suggest_words(&self, prefix: &str, num_suggestions: usize) -> Vec<String> {
        match self.trie {
            Some(ref trie) => trie.suggest(prefix, num_suggestions).into_iter().collect(),
            None => vec![],
        }
    }

    pub fn get_coords_for_star(&self, star_name: &str) -> Option<Coords> {
        match (&self.trie, &self.stars) {
            (Some(trie), Some(stars)) => trie.find(star_name).map(|index| stars[index as usize]),
            _ => None,
        }
    }

    #[wasm_bindgen]
    pub fn find_route(
        &self,
        start: Coords,
        end: Coords,
        report_callback: &js_sys::Function,
    ) -> Result<Box<[f32]>, JsValue> {
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

        let report_callback = |report: plotter::Report| {
            let this = &JsValue::NULL;
            let star_coords: Box<[f32]> = report
                .curr_best_route
                .iter()
                .flat_map(|c| c.to_slice())
                .collect();
            report_callback
                .call3(
                    this,
                    &JsValue::from(star_coords),
                    &JsValue::from(report.distance),
                    &JsValue::from(report.depth),
                )
                .unwrap();
        };

        let (stars, kdtree) = match (&self.stars, &self.kdtree) {
            (Some(stars), Some(kdtree)) => (stars, kdtree),
            _ => return Err(JsValue::from_str("Stars or KDTree data not set in module")),
        };

        let result_coords = plotter::plot(start, end, stars, &ship, kdtree, report_callback);

        Ok(result_coords.iter().flat_map(|c| c.to_slice()).collect())
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

// #[wasm_bindgen]
pub fn init() {
    utils::set_panic_hook();
}

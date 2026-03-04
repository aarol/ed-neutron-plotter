pub mod fast_json_parser;
pub mod kdtree;
mod ordered_f32;
pub mod plotter;
pub mod system;
pub mod trie;
pub mod utils;
use wasm_bindgen::prelude::*;

use crate::{system::Coords, trie::LoudsTrie};

/// Binary search autocomplete over a sorted list of star names
#[wasm_bindgen]
#[derive(Default)]
pub struct BinarySearchAutocomplete {
    /// Sorted list of star names (case-folded for search)
    names: Vec<String>,
    /// Original names preserving case
    original_names: Vec<String>,
}

#[wasm_bindgen]
impl BinarySearchAutocomplete {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load names from newline-delimited string
    pub fn set_names(&mut self, data: &str) {
        let mut names: Vec<String> = data.lines().map(|s| s.to_string()).collect();
        names.sort_unstable_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
        self.original_names = names.clone();
        self.names = names.iter().map(|s| s.to_lowercase()).collect();
    }

    /// Find suggestions using binary search
    pub fn suggest(&self, prefix: &str, limit: usize) -> Vec<String> {
        log_u32(self.names.len() as u32);
        if self.names.is_empty() || limit == 0 {
            return vec![];
        }

        let prefix_lower = prefix.to_lowercase();
        
        // Binary search for the first name >= prefix
        let start_idx = self.names.partition_point(|name| name.as_str() < prefix_lower.as_str());
        
        let mut results = Vec::with_capacity(limit);
        for i in start_idx..self.names.len() {
            if !self.names[i].starts_with(&prefix_lower) {
                break;
            }
            results.push(self.original_names[i].clone());
            if results.len() >= limit {
                break;
            }
        }
        results
    }

    /// Find exact match and return its index
    pub fn find(&self, name: &str) -> Option<u32> {
        let name_lower = name.to_lowercase();
        self.names.binary_search(&name_lower).ok().map(|i| i as u32)
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }
}

/// Stores the trie and other data needed for autocomplete and plotter
#[wasm_bindgen]
#[derive(Default)]
pub struct Module {
    trie: Option<LoudsTrie<'static>>,
    stars: Option<Box<[Coords]>>,
    kdtree: Option<kdtree::CompactKdTree>,
}

#[wasm_bindgen]
impl Module {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Module {
        Module::default()
    }

    /// Load trie data from a byte array
    /// Should not be called more than once per Module instance
    pub fn set_trie(&mut self, trie_data: Box<[u8]>) {
        // Leak the boxed slice to get a 'static reference
        // Safety: This memory will not be freed until the module is dropped
        self.trie = Some(LoudsTrie::from_bytes(Box::leak(trie_data)));
    }

    pub fn set_stars(&mut self, stars: Box<[f32]>) {
        // Convert flat f32 array to coordinates without copying
        self.stars = Some(stars.chunks_exact(3).map(Coords::from_slice).collect());
    }

    pub fn set_kdtree(&mut self, kdtree_data: &[u8]) {
        self.kdtree = Some(kdtree::CompactKdTree::from_bytes(kdtree_data));
    }

    /// Suggest words using the trie
    pub fn suggest_words(&self, prefix: &str, num_suggestions: usize) -> Vec<String> {
        match self.trie {
            Some(ref trie) => trie.suggest(prefix, num_suggestions),
            None => vec![],
        }
    }

    pub fn get_coords_for_star(&self, star_name: &str) -> Option<JsValue> {
        match (&self.trie, &self.stars) {
            (Some(trie), Some(stars)) => {
                let coords = trie.find(star_name).map(|index| stars[index as usize]);
                coords.map(|c| serde_wasm_bindgen::to_value(&CoordsSerde::from(c)).unwrap())
            }
            _ => None,
        }
    }

    #[wasm_bindgen]
    pub fn find_route(
        &self,
        start: JsValue,
        end: JsValue,
        report_callback: &js_sys::Function,
    ) -> Result<Vec<JsValue>, JsValue> {
        let start = serde_wasm_bindgen::from_value::<CoordsSerde>(start)
            .map_err(|e| JsValue::from_str(&format!("Invalid start coords: {}", e)))?;
        let end = serde_wasm_bindgen::from_value::<CoordsSerde>(end)
            .map_err(|e| JsValue::from_str(&format!("Invalid end coords: {}", e)))?;
        let start: Coords = start.into();
        let end: Coords = end.into();
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
        Ok(
            result_coords.iter().map(|c| {
                let coords = stars[*c as usize];
                let key = self.trie.as_ref().expect("Trie loaded").reconstruct_key(*c as u64);

                serde_wasm_bindgen::to_value(&RouteNode {
                    coords: CoordsSerde::from(coords),
                    name: key,
                 }).expect("Failed to serialize route node")
            }).collect()
        )
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


#[derive(serde::Deserialize, serde::Serialize)]
struct RouteNode {
    coords: CoordsSerde,
    name: String,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct CoordsSerde {
    x: f32,
    y: f32,
    z: f32,
}

impl From<Coords> for CoordsSerde {
    fn from(c: Coords) -> Self {
        CoordsSerde {
            x: c.at(0),
            y: c.at(1),
            z: c.at(2),
        }
    }
}
impl From<CoordsSerde> for Coords {
    fn from(c: CoordsSerde) -> Self {
        Coords([c.x, c.y, c.z])
    }
}
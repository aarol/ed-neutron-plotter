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
        let start_idx = self
            .names
            .partition_point(|name| name.as_str() < prefix_lower.as_str());

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

    /// Find the star that appears closest to a camera ray (i.e. best for click-picking).
    ///
    /// Parameters (all in kly, the same units as star data):
    ///   ox/oy/oz          — ray origin (camera world position)
    ///   dx/dy/dz          — ray direction (does not need to be normalised)
    ///   angular_tolerance — half-angle threshold in radians; stars further than
    ///                       this from the ray axis are ignored
    ///
    /// Algorithm:
    ///   Sample N points along the ray with exponentially increasing spacing.
    ///   At each sample, query the KD-tree within a sphere whose radius equals
    ///   depth × angular_tolerance.  For every candidate star compute the true
    ///   angular separation (perp_dist / depth) and keep the global minimum.
    ///
    /// Returns `{ name: string, coords: { x, y, z } }` or `null`.
    #[wasm_bindgen]
    pub fn find_star_near_ray(
        &self,
        ox: f32,
        oy: f32,
        oz: f32,
        dx: f32,
        dy: f32,
        dz: f32,
        angular_tolerance: f32,
    ) -> JsValue {
        let (trie, stars, kdtree) = match (&self.trie, &self.stars, &self.kdtree) {
            (Some(t), Some(s), Some(k)) => (t, s, k),
            _ => return JsValue::NULL,
        };

        // Normalise direction
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-10 {
            return JsValue::NULL;
        }
        let (ndx, ndy, ndz) = (dx / len, dy / len, dz / len);

        let tolerance_sq = angular_tolerance * angular_tolerance;

        let mut best_idx: Option<u32> = None;
        let mut best_t: f32 = 0.0;
        let mut best_ang_sq = f32::INFINITY;

        // Exponentially-spaced samples
        let mut t: f32 = 0.01;
        let max_t: f32 = 240.0;

        while t <= max_t {
            let px = ox + ndx * t;
            let py = oy + ndy * t;
            let pz = oz + ndz * t;
            let point = Coords([px, py, pz]);

            // Sphere radius = angular_tolerance × depth, maximum of 50 ly
            let radius = (t * angular_tolerance).max(0.050);
            let candidates = kdtree.nearest_n_within(point, stars, radius, 32);

            for (idx, _) in candidates {
                let star = stars[idx as usize];

                // Vector from ray origin to star
                let tx = star.at(0) - ox;
                let ty = star.at(1) - oy;
                let tz = star.at(2) - oz;

                // Projection along normalised direction
                let t_proj = tx * ndx + ty * ndy + tz * ndz;
                if t_proj <= 0.0 {
                    continue; // behind camera
                }

                // Perpendicular distance squared
                let perp_x = tx - ndx * t_proj;
                let perp_y = ty - ndy * t_proj;
                let perp_z = tz - ndz * t_proj;
                let perp_sq = perp_x * perp_x + perp_y * perp_y + perp_z * perp_z;

                // Angular separation squared (perp/depth)²
                let ang_sq = perp_sq / (t_proj * t_proj);

                if ang_sq <= tolerance_sq && ang_sq < best_ang_sq {
                    best_ang_sq = ang_sq;
                    best_idx = Some(idx);
                    best_t = t_proj;
                }
            }

            t += 0.050; // take a 50ly step
        }

        match best_idx {
            Some(idx) => {
                let coords = stars[idx as usize];
                let name = trie.get_key_from_index(idx as usize).unwrap_or_else(|| "Unknown".to_string());

                let idx2 = trie.find(&name);
                
                serde_wasm_bindgen::to_value(&RouteNode {
                    coords: CoordsSerde::from(coords),
                    name,
                })
                .unwrap_or(JsValue::NULL)
            }
            None => JsValue::NULL,
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
        Ok(result_coords
            .iter()
            .map(|c| {
                let coords = stars[*c as usize];
                let key = self
                    .trie
                    .as_ref()
                    .expect("Trie loaded")
                    .get_key_from_index(*c as usize);

                serde_wasm_bindgen::to_value(&RouteNode {
                    coords: CoordsSerde::from(coords),
                    name: key.unwrap_or("Unknown".to_string()),
                })
                .expect("Failed to serialize route node")
            })
            .collect())
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

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Read};

    use super::*;

    #[allow(dead_code)]
    fn build_trie_kdtree<'a>() -> (Vec<u8>, Vec<Coords>, kdtree::CompactKdTree) {
        let stars = [
            "Alpha Centauri",
            "Barnard's Star",
            "Lalande 21185",
            "Sol",
            "Wolf 359",
        ];
        let star_coords = [
            Coords::new(4.37, 0.0, 0.0).0, // Alpha Centauri
            Coords::new(5.96, 0.0, 0.0).0, // Barnard's Star
            Coords::new(8.31, 0.0, 0.0).0, // Lalande 21185
            Coords::new(0.0, 0.0, 0.0).0,  // Sol
            Coords::new(7.78, 0.0, 0.0).0, // Wolf 359
        ];
        let mut buf = Vec::new();
        let coords_indices = LoudsTrie::build(&stars, &mut buf).unwrap();

        let mut sorted_coords = vec![[0.0; 3]; star_coords.len()];
        for (sorted_idx, &trie_idx) in coords_indices.iter().enumerate() {
            sorted_coords[trie_idx] = star_coords[sorted_idx];
        }
        let kdtree_indices = kdtree::KdTreeBuilder::from_points(&sorted_coords).build();
        let kdtree = kdtree::CompactKdTree::new(kdtree_indices.into_boxed_slice());
        let star_coords = sorted_coords.into_iter().map(Coords).collect();
        (buf, star_coords, kdtree)
    }

    #[test]
    fn test_coords_for_star() {
        let mut buf = Vec::new();
        let mut trie_file = File::open("../public/data/search_trie.bin").unwrap();
        trie_file.read_to_end(&mut buf).unwrap();
        let mut star_file = File::open("../public/data/neutron_stars0.bin").unwrap();
        let mut kdtree_file = File::open("../public/data/star_kdtree.bin").unwrap();
        let trie = LoudsTrie::from_bytes(&buf);
        let mut kdtree_buf = Vec::new();
        kdtree_file.read_to_end(&mut kdtree_buf).unwrap();
        let kdtree = kdtree::CompactKdTree::from_bytes(&kdtree_buf);
        let mut star_buf = Vec::new();
        star_file.read_to_end(&mut star_buf).unwrap();
        let star_coords: Vec<Coords> = star_buf
            .chunks_exact(12)
            .map(|chunk| {
                let x = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let y = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
                Coords([x, y, z])
            })
            .collect();

        let (index, distance) = kdtree
            .nearest(Coords::new(0., 0., 0.), &star_coords)
            .unwrap();
        // assert_eq!(index, 3);
        assert!(distance < 1e-6);
        dbg!(trie.find("Sol"));
        let name = trie.get_key_from_index(index as usize);
        assert_eq!(name.unwrap(), "Sol");
    }
}

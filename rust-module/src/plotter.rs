use core::panic;

use bitvec::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{kdtree::CompactKdTree, ordered_f32::OrderedF32, system::Coords};

pub struct Ship {
    pub fuel_tank_size: f32,
    pub jump_range: f32,
    pub max_fuel_per_jump: f32,
    pub base_mass: f32,
    pub fsd_optimized_mass: f32,
    pub fsd_boost_factor: f32,
    pub fsd_rating_val: f32,
    pub fsd_class_val: f32,
}

impl Ship {
    fn jump_range(&self, fuel: f32, overcharge_mult: f32) -> f32 {
        let mass = self.base_mass + fuel;
        let fuel_used = self.max_fuel_per_jump.min(fuel) * overcharge_mult;
        let opt_mass = self.fsd_optimized_mass * self.fsd_boost_factor;

        opt_mass
            * ((1000. * fuel_used) / self.fsd_rating_val).powf(1. / self.fsd_class_val)
            / mass
    }

    // fn fuel_cost_for_jump(
    //   &self, curr_fuel_mass: f32,
    //   dist: f32,
    //   boost: f32,
    // ) -> Option<f32> {

    //   // if dist == 0.0 {
    //   //   return Some(0.0);
    //   // }
    //   // let base_cost = dist * ((self.base_mass + curr_fuel_mass) / (self.fsd_optimized_mass * boost));
    //   // let fuel_cost = ()
    // }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Report {
    pub curr_best_route: Vec<Coords>,
    pub distance: f32,
    pub depth: u32,
}

pub fn plot(
    start_coords: Coords,
    end_coords: Coords,
    stars: &[Coords],
    ship: &Ship,
    kdtree: &CompactKdTree,
    send_report: impl Fn(Report),
) -> Vec<Coords> {
    send_report(Report {
        curr_best_route: vec![],
        distance: start_coords.dist(&end_coords),
        depth: 0,
    });
    // Fuel, system index
    let mut queue: Vec<(f32, u32)> = Vec::with_capacity(1024);
    let mut prev = FxHashMap::default();

    let start_idx = kdtree.nearest(start_coords, stars).unwrap().0;
    let end_idx = kdtree.nearest(end_coords, stars).unwrap().0;

    dbg!((start_idx, end_idx));

    queue.push((ship.fuel_tank_size, start_idx));

    let beam_width = 1024;

    let mut seen: BitVec = BitVec::repeat(false, kdtree.size());
    seen.set(start_idx as usize, true);
    let mut seen_update: BitVec = BitVec::repeat(false, kdtree.size());

    let heuristic = |range: f32, i: u32| {
        if i == end_idx {
            return f32::NEG_INFINITY;
        }
        let start_pos = stars[i as usize];
        let end_pos = stars[end_idx as usize];
        range + start_pos.dist(&end_pos).max(0.0)
    };

    let range = 0.27;
    let mut depth = 0;
    let mut dist_to_end = start_coords.dist(&end_coords);
    let mut end_found = false;
    // let mut curr_time = std::time::Instant::now();
    loop {
        if end_found || queue.is_empty() {
            break;
        }

        // if curr_time.elapsed() > Duration::from_secs(1) {
        if depth % 5 == 0 {
            // curr_time = std::time::Instant::now();
            println!(
                "Depth {}: queue size {}, dist to end {:.2}, range {:.2}",
                depth,
                queue.len(),
                dist_to_end,
                range
            );
            if let Some((_, closest_star)) = queue
                .iter()
                .max_by_key(|(_, i)| OrderedF32(heuristic(range, *i)))
            {
                let best_route = reconstruct_path(&prev, *closest_star, stars);

                send_report(Report {
                    curr_best_route: best_route,
                    distance: stars[*closest_star as usize].dist(&end_coords),
                    depth,
                });
            }
        }

        let working_queue = std::mem::take(&mut queue);
        // log_u32(working_queue.len() as u32);
        for (fuel_available, i) in working_queue {
            // let mut range = ship.jump_range(*fuel_available, 4.0);
            let coords = stars[i as usize];


            let neighbours = kdtree.nearest_n_within(coords, stars, range, beam_width);
            // log_u32(neighbours.len() as u32);
            let mut buffer = vec![];

            for (ni, _dist_sq) in neighbours {
                if !seen.get(ni as usize).is_some_and(|b| *b) {
                    seen.set(ni as usize, true);

                    buffer.push((i, ni));
                }
            }

            for (parent_i, child_i) in buffer {
                if !seen_update.get(child_i as usize).is_some_and(|b| *b) {
                    seen_update.set(child_i as usize, true);

                    prev.insert(child_i, parent_i);

                    let node_coords = stars[parent_i as usize];
                    let dist = node_coords.dist(&end_coords);
                    dist_to_end = dist_to_end.min(dist);
                    if child_i == end_idx {
                        end_found = true;
                        break;
                    }

                    queue.push((fuel_available, child_i));
                }
            }
        }

        depth += 1;
        if !queue.is_empty() {
            let count = beam_width.min(queue.len() - 1);

            queue.select_nth_unstable_by_key(count, |(_, i)| OrderedF32(heuristic(range, *i)));
            queue.truncate(count);
        }
        seen |= &seen_update;
    }

    if end_found {
        reconstruct_path(&prev, end_idx, stars)
    } else {
        panic!("No route found")
    }
}

fn reconstruct_path(prev: &FxHashMap<u32, u32>, end_idx: u32, stars: &[Coords]) -> Vec<Coords> {
    let mut path = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut current = end_idx;

    path.push(stars[current as usize]);
    seen.insert(current);

    while let Some(&p) = prev.get(&current) {
        if seen.contains(&p) {
            panic!(
                "Loop detected in path reconstruction: {} already visited",
                p
            );
        }
        seen.insert(p);
        path.push(stars[p as usize]);
        current = p;
    }

    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot() {
        let stars = std::fs::read("../public/data/neutron_stars0.bin").unwrap();
        let stars: Vec<Coords> = stars
            .chunks_exact(12)
            .map(|chunk| {
                Coords::from_slice(&[
                    f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                    f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                ])
            })
            .collect();

        let kdtree_bin = std::fs::read("../public/data/star_kdtree.bin").unwrap();

        let kdtree = CompactKdTree::from_bytes(&kdtree_bin);

        let start = Coords::new(25.044375, 0.29353125, 20.47203125);
        let end = Coords::new(9.5305, -0.91028124, 19.808125);

        let route = plot(
            start,
            end,
            &stars,
            &Ship {
                fuel_tank_size: 32.0,
                jump_range: 80.0,
                max_fuel_per_jump: 4.0,
                base_mass: 100.0,
                fsd_optimized_mass: 50.0,
                fsd_boost_factor: 1.2,
                fsd_rating_val: 5.0,
                fsd_class_val: 3.0,
            },
            &kdtree,
            |_report| {},
        );

        println!("Route length: {}", route.len());

        for pos in route {
            println!("{:?}", pos);
        }
    }
}

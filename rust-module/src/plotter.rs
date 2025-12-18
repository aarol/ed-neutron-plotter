use std::{num::NonZero, ops::ControlFlow};

use bitvec::prelude::*;
use kiddo::{SquaredEuclidean, traits::DistanceMetric};

use crate::star::Coords;

pub struct Ship {
    fuel_tank_size: f32,
    jump_range: f32,
    max_fuel_per_jump: f32,
    base_mass: f32,
    fsd_optimized_mass: f32,
    fsd_boost_factor:f32,
    fsd_rating_val: f32,
    fsd_clss_val: f32,
}

impl Ship {
  fn jump_range(&self, fuel: f32, overcharge_mult: f32) -> f32 {
    let mass = self.base_mass + fuel;
    let fuel_used = self.max_fuel_per_jump.min(fuel) * overcharge_mult;
    let opt_mass = self.fsd_optimized_mass * self.fsd_boost_factor;

    return opt_mass * ((1000. * fuel_used) / self.fsd_rating_val)
    .powf(1. / self.fsd_clss_val)
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

type KDTree = kiddo::ImmutableKdTree<f32, 3>;

pub fn plot(ship: &Ship, kdtree: KDTree, start_coords: Coords, end_coords: Coords) -> Vec<Coords> {
    // Fuel, coords
    let mut queue: Vec<(f32, Coords)> = Vec::with_capacity(1024);

    queue.push((ship.fuel_tank_size, start_coords));

    let beam_width: NonZero<usize> = NonZero::new(1024).unwrap();

    let mut seen: BitVec<> = BitVec::with_capacity(kdtree.size());

    loop {
      if queue.is_empty() {
        break;
      }

      queue.iter().for_each(|(fuel_available, coords)| {
        // let mut range = ship.jump_range(*fuel_available, 4.0);
        let mut range = 300.0;

        let neighbours = kdtree.nearest_n_within::<SquaredEuclidean>(&coords.as_slice(), range, beam_width, false);

        neighbours.iter().for_each(|nb| {
         let seen = seen.get(nb.item as usize).map_or(false, |b| *b);
          
          if !seen {

          }
        });
      });
    }

    vec![]
}

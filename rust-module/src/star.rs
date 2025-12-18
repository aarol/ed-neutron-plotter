use std::io::{self, Write};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct System {
    // id: i64,
    // id64: i64,
    pub name: String,
    pub coords: Coords,
    // date: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Coords {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Coords {
    pub fn write_to_file<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.x.to_le_bytes())?;
        writer.write_all(&self.y.to_le_bytes())?;
        writer.write_all(&self.z.to_le_bytes())?;
        Ok(())
    }

    pub fn as_slice(&self) -> [f32;3] {
        return [self.x, self.y, self.z]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Star {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Star {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn get_coord(&self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!(),
        }
    }
}

pub struct Partition<'a> {
    aabb_min: Coords,
    aabb_max: Coords,
    stars: &'a [Star],
}

impl<'a> Partition<'a> {
    pub fn new(stars: &'a [Star]) -> Self {
        let mut aabb_min = Coords {
            x: f32::INFINITY,
            y: f32::INFINITY,
            z: f32::INFINITY,
        };
        let mut aabb_max = Coords {
            x: f32::NEG_INFINITY,
            y: f32::NEG_INFINITY,
            z: f32::NEG_INFINITY,
        };
        for star in stars {
            aabb_min.x = aabb_min.x.min(star.x);
            aabb_min.y = aabb_min.y.min(star.y);
            aabb_min.z = aabb_min.z.min(star.z);

            aabb_max.x = aabb_max.x.max(star.x);
            aabb_max.y = aabb_max.y.max(star.y);
            aabb_max.z = aabb_max.z.max(star.z);
        }
        Self {
            aabb_min,
            aabb_max,
            stars,
        }
    }

    pub fn write_to_file<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.aabb_min.write_to_file(writer)?;
        self.aabb_max.write_to_file(writer)?;

        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.stars.as_ptr() as *const u8,
                self.stars.len() * std::mem::size_of::<Star>(),
            )
        };
        writer.write_all(bytes).unwrap();
        Ok(())
    }
}

/// Rearranges the `stars` slice in-place so that it can simply
/// be chopped into 2^depth chunks later.
pub fn reorder_for_partitions(stars: &mut [Star], depth: u32) {
    // Base case: If depth is 0, this segment is a finalized partition.
    if depth == 0 || stars.len() <= 1 {
        return;
    }

    // 1. Find the axis with the largest spread (Longest Axis Split)
    // We compute min/max to determine bounding box shape.
    let (min_vals, max_vals) = stars.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut mins, mut maxs), s| {
            for i in 0..3 {
                let v = s.get_coord(i);
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
            (mins, maxs)
        },
    );

    let axis = (0..3)
        .max_by(|&a, &b| (max_vals[a] - min_vals[a]).total_cmp(&(max_vals[b] - min_vals[b])))
        .unwrap_or(0);

    // 2. Split exactly in half
    let mid = stars.len() / 2;

    // 3. Quickselect: Sorts so that [0..mid] < [mid] < [mid..len]
    // This is O(N)
    stars.select_nth_unstable_by(mid, |a, b| a.get_coord(axis).total_cmp(&b.get_coord(axis)));

    // 4. Recurse in parallel
    // We split the mutable slice into two independent mutable slices
    let (left, right) = stars.split_at_mut(mid);

    // Use rayon::join to run both halves simultaneously
    rayon::join(
        || reorder_for_partitions(left, depth - 1),
        || reorder_for_partitions(right, depth - 1),
    );
}

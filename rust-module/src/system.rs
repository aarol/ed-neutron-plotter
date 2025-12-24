use std::{
    fmt::Display,
    io::{self, Write},
};

use serde::{Deserialize, Serialize};
use wasm_bindgen::{convert::WasmAbi, prelude::wasm_bindgen};

#[derive(Serialize, Deserialize, Debug)]
pub struct System {
    // id: i64,
    // id64: i64,
    pub name: String,
    pub coords: Coords,
    // date: String,
}

impl PartialOrd for System {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.name.cmp(&other.name))   
    }
}

impl PartialEq for System {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for System {
}

impl Ord for System {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name.cmp(&other.name)
    }
}

#[wasm_bindgen]
#[repr(transparent)]
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Coords([f32; 3]);

impl Coords {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Coords([x, y, z])
    }
    pub fn from_slice(slice: &[f32]) -> Self {
        Coords([slice[0], slice[1], slice[2]])
    }
    pub fn to_slice(&self) -> [f32; 3] {
        self.0
    }
    pub fn at(&self, index: usize) -> f32 {
        self.0[index]
    }
    pub fn write_to_file<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.0[0].to_le_bytes())?;
        writer.write_all(&self.0[1].to_le_bytes())?;
        writer.write_all(&self.0[2].to_le_bytes())?;
        Ok(())
    }

    pub fn dist(&self, o: &Coords) -> f32 {
        let dx = self.x() - o.x();
        let dy = self.y() - o.y();
        let dz = self.z() - o.z();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[wasm_bindgen]
impl Coords {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.0[0]
    }
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.0[1]
    }
    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f32 {
        self.0[2]
    }
}

impl WasmAbi for Coords {
    type Prim1 = f32;
    type Prim2 = f32;
    type Prim3 = f32;
    type Prim4 = ();
    fn split(self) -> (Self::Prim1, Self::Prim2, Self::Prim3, Self::Prim4) {
        (self.0[0], self.0[1], self.0[2], ())
    }
    fn join(
        prim1: Self::Prim1,
        prim2: Self::Prim2,
        prim3: Self::Prim3,
        _prim4: Self::Prim4,
    ) -> Self {
        Coords([prim1, prim2, prim3])
    }
}

impl Display for Coords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x(), self.y(), self.z())
    }
}

/// Rearranges the `stars` slice in-place so that it can simply
/// be chopped into 2^depth chunks later.
pub fn reorder_for_partitions(stars: &mut [Coords], depth: u32) {
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
                let v = s.0[i];
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
    stars.select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));

    // 4. Recurse in parallel
    // We split the mutable slice into two independent mutable slices
    let (left, right) = stars.split_at_mut(mid);

    // Use rayon::join to run both halves simultaneously
    rayon::join(
        || reorder_for_partitions(left, depth - 1),
        || reorder_for_partitions(right, depth - 1),
    );
}

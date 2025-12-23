use std::convert::TryInto;

use crate::system::Coords;

// =========================================================================
// FLAT KD-TREE FOR 3D POINTS (INDEX-ONLY)
// =========================================================================
// Uses implicit binary tree layout (like a heap):
// - Root at index 0
// - Left child of node i: 2*i + 1
// - Right child of node i: 2*i + 2
// - Split axis at depth d: d % 3
//
// The tree ONLY stores point indices. Point coordinates are provided
// externally at query time, avoiding data duplication.

const INVALID_IDX: u32 = u32::MAX;

/// Builder for constructing a flat KD-tree from 3D points.
/// Points are provided during construction but NOT stored in the final tree.
pub struct KdTreeBuilder<'a> {
    points: &'a [[f32; 3]],
}

impl<'a> KdTreeBuilder<'a> {
    pub fn new() -> Self {
        Self { points: &[] }
    }

    pub fn from_points(points: &'a [[f32; 3]]) -> Self {
        Self { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Build the flat KD-tree.
    /// Returns the tree array (implicit binary tree of point indices).
    /// The original point coordinates are discarded - you must maintain them separately.
    pub fn build(self) -> Vec<u32> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }

        // Calculate tree size for implicit binary tree
        let levels = (n as f64).log2().ceil() as u32;
        let tree_size = (1usize << levels) - 1;

        let mut tree = vec![INVALID_IDX; tree_size.max(n)];
        let mut indices: Vec<u32> = (0..n as u32).collect();

        Self::build_recursive(&self.points, &mut indices, &mut tree, 0, 0);

        tree
    }

    fn build_recursive(
        points: &[[f32; 3]],
        indices: &mut [u32],
        tree: &mut [u32],
        tree_idx: usize,
        depth: usize,
    ) {
        if indices.is_empty() || tree_idx >= tree.len() {
            return;
        }

        if indices.len() == 1 {
            tree[tree_idx] = indices[0];
            return;
        }

        let axis = depth % 3;

        // Find median using quickselect
        let mid = indices.len() / 2;
        indices.select_nth_unstable_by(mid, |&a, &b| {
            points[a as usize][axis].total_cmp(&points[b as usize][axis])
        });

        // Store median point at current node
        tree[tree_idx] = indices[mid];

        // Recurse on left and right halves
        let (left, right) = indices.split_at_mut(mid);
        let right = &mut right[1..]; // Skip the median

        let left_child = 2 * tree_idx + 1;
        let right_child = 2 * tree_idx + 2;

        Self::build_recursive(points, left, tree, left_child, depth + 1);
        Self::build_recursive(points, right, tree, right_child, depth + 1);
    }
}

impl<'a> Default for KdTreeBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// RUNTIME KD-TREE (INDEX-ONLY)
// =========================================================================

/// A compact, flat KD-tree that only stores indices.
/// Point coordinates must be provided at query time.
/// Designed for WASM usage with efficient serialization.
pub struct CompactKdTree<'a> {
    /// Implicit binary tree of point indices
    tree: &'a [u32],
}

impl<'a> CompactKdTree<'a> {
    pub fn new(tree: &'a [u32]) -> Self {
        Self { tree }
    }

    /// Deserialize from bytes.
    /// Format:
    /// - tree_len: u32
    /// - tree: [u32; tree_len]
    pub fn from_bytes(data: &'a [u8]) -> Self {
        let tree_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;

        let tree_start = 4;
        let tree_end = tree_start + tree_len * 4;
        let tree_bytes = &data[tree_start..tree_end];

        let tree: &[u32] =
            unsafe { std::slice::from_raw_parts(tree_bytes.as_ptr() as *const u32, tree_len) };

        Self { tree }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(4 + self.tree.len() * 4);

        let tree_len = self.tree.len() as u32;
        data.extend_from_slice(&tree_len.to_le_bytes());

        let tree_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.tree.as_ptr() as *const u8, self.tree.len() * 4)
        };
        data.extend_from_slice(tree_bytes);

        data
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn size(&self) -> usize {
        self.tree.len()
    }

    /// Size in bytes (just the tree structure).
    pub fn size_in_bytes(&self) -> usize {
        self.tree.len() * 4
    }

    /// Raw tree array access.
    pub fn tree(&self) -> &[u32] {
        self.tree
    }

    /// Find the nearest neighbor to the query point.
    ///
    /// `points` is a flat array of coordinates: [x0, y0, z0, x1, y1, z1, ...]
    /// Returns (point_index, squared_distance).
    pub fn nearest(&self, query: Coords, points: &[Coords]) -> Option<(u32, f32)> {
        if self.tree.is_empty() {
            return None;
        }

        let mut best_idx = INVALID_IDX;
        let mut best_dist_sq = f32::INFINITY;

        self.nearest_recursive(query, points, 0, 0, &mut best_idx, &mut best_dist_sq);

        if best_idx == INVALID_IDX {
            None
        } else {
            Some((best_idx, best_dist_sq))
        }
    }

    fn nearest_recursive(
        &self,
        query: Coords,
        points: &[Coords],
        tree_idx: usize,
        depth: usize,
        best_idx: &mut u32,
        best_dist_sq: &mut f32,
    ) {
        if tree_idx >= self.tree.len() {
            return;
        }

        let point_idx = self.tree[tree_idx];
        if point_idx == INVALID_IDX {
            return;
        }

        let point = get_point(points, point_idx);
        let dist_sq = squared_distance(&query, &point);

        if dist_sq < *best_dist_sq {
            *best_dist_sq = dist_sq;
            *best_idx = point_idx;
        }

        let axis = depth % 3;
        let diff = query.at(axis) - point.at(axis);
        let diff_sq = diff * diff;

        let (first, second) = if diff < 0.0 {
            (2 * tree_idx + 1, 2 * tree_idx + 2)
        } else {
            (2 * tree_idx + 2, 2 * tree_idx + 1)
        };

        self.nearest_recursive(query, points, first, depth + 1, best_idx, best_dist_sq);

        if diff_sq < *best_dist_sq {
            self.nearest_recursive(query, points, second, depth + 1, best_idx, best_dist_sq);
        }
    }

    /// Find up to `max_results` neighbors within a given radius.
    ///
    /// Returns vector of (point_index, squared_distance). Results are not sorted.
    pub fn nearest_n_within(
        &self,
        query: Coords,
        points: &[Coords],
        radius: f32,
        max_results: usize,
    ) -> Vec<(u32, f32)> {
        if self.tree.is_empty() {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut results = Vec::with_capacity(max_results.min(64));

        self.nearest_n_within_recursive(query, points, 0, 0, radius_sq, max_results, &mut results);

        results
    }

    fn nearest_n_within_recursive(
        &self,
        query: Coords,
        points: &[Coords],
        tree_idx: usize,
        depth: usize,
        radius_sq: f32,
        max_results: usize,
        results: &mut Vec<(u32, f32)>,
    ) {
        if tree_idx >= self.tree.len() {
            return;
        }

        let point_idx = self.tree[tree_idx];
        if point_idx == INVALID_IDX {
            return;
        }

        let point = get_point(points, point_idx);
        let dist_sq = squared_distance(&query, &point);

        if dist_sq <= radius_sq {
            results.push((point_idx, dist_sq));
        }

        let axis = depth % 3;
        let diff = query.at(axis) - point.at(axis);
        let diff_sq = diff * diff;

        let left_child = 2 * tree_idx + 1;
        let right_child = 2 * tree_idx + 2;

        let (first, second) = if diff < 0.0 {
            (left_child, right_child)
        } else {
            (right_child, left_child)
        };

        self.nearest_n_within_recursive(
            query,
            points,
            first,
            depth + 1,
            radius_sq,
            max_results,
            results,
        );

        if diff_sq <= radius_sq {
            self.nearest_n_within_recursive(
                query,
                points,
                second,
                depth + 1,
                radius_sq,
                max_results,
                results,
            );
        }
    }

    /// Find all points within a given radius.
    ///
    /// Returns vector of (point_index, squared_distance).
    pub fn within_radius(&self, query: Coords, points: &[Coords], radius: f32) -> Vec<(u32, f32)> {
        if self.tree.is_empty() {
            return Vec::new();
        }

        let radius_sq = radius * radius;
        let mut results = Vec::new();

        self.within_radius_recursive(query, points, 0, 0, radius_sq, &mut results);
        results
    }

    fn within_radius_recursive(
        &self,
        query: Coords,
        points: &[Coords],
        tree_idx: usize,
        depth: usize,
        radius_sq: f32,
        results: &mut Vec<(u32, f32)>,
    ) {
        if tree_idx >= self.tree.len() {
            return;
        }

        let point_idx = self.tree[tree_idx];
        if point_idx == INVALID_IDX {
            return;
        }

        let point = get_point(points, point_idx);
        let dist_sq = squared_distance(&query, &point);

        if dist_sq <= radius_sq {
            results.push((point_idx, dist_sq));
        }

        let axis = depth % 3;
        let diff = query.at(axis) - point.at(axis);
        let diff_sq = diff * diff;

        let left_child = 2 * tree_idx + 1;
        let right_child = 2 * tree_idx + 2;

        if diff <= 0.0 || diff_sq <= radius_sq {
            self.within_radius_recursive(query, points, left_child, depth + 1, radius_sq, results);
        }
        if diff >= 0.0 || diff_sq <= radius_sq {
            self.within_radius_recursive(query, points, right_child, depth + 1, radius_sq, results);
        }
    }
}

#[inline]
fn get_point(points: &[Coords], idx: u32) -> Coords {
    points[idx as usize]
}

#[inline]
fn squared_distance(a: &Coords, b: &Coords) -> f32 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();
    let dz = a.z() - b.z();
    dx * dx + dy * dy + dz * dz
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: f32, y: f32, z: f32) -> Coords {
        Coords::from_slice(&[x, y, z])
    }

    fn make_coords(points: &[[f32; 3]]) -> Vec<Coords> {
        points.iter().map(|p| c(p[0], p[1], p[2])).collect()
    }

    #[test]
    fn test_empty_tree() {
        let builder = KdTreeBuilder::new();
        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);

        assert!(kdtree.is_empty());
        assert!(kdtree.nearest(c(0.0, 0.0, 0.0), &[]).is_none());
    }

    #[test]
    fn test_single_point() {
        let builder = KdTreeBuilder::from_points(&[[1.0, 2.0, 3.0]]);

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);

        let points = make_coords(&[[1.0, 2.0, 3.0]]);

        let (idx, dist_sq) = kdtree.nearest(c(1.0, 2.0, 3.0), &points).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(dist_sq, 0.0);

        let (idx, dist_sq) = kdtree.nearest(c(0.0, 0.0, 0.0), &points).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(dist_sq, 14.0); // 1 + 4 + 9
    }

    #[test]
    fn test_multiple_points() {
        let builder = KdTreeBuilder::from_points(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]);
        let raw_points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);
        let points = make_coords(&raw_points);

        // Query near origin
        let (_, dist_sq) = kdtree.nearest(c(0.0, 0.0, 0.0), &points).unwrap();
        assert_eq!(dist_sq, 0.0);

        // Query near (1,1,1)
        let (idx, dist_sq) = kdtree.nearest(c(1.0, 1.0, 1.0), &points).unwrap();
        assert_eq!(dist_sq, 0.0);
        let point = get_point(&points, idx);
        assert_eq!(point.x(), 1.0);
        assert_eq!(point.y(), 1.0);
        assert_eq!(point.z(), 1.0);

        // Query equidistant from multiple points
        let (_, dist_sq) = kdtree.nearest(c(0.5, 0.0, 0.0), &points).unwrap();
        assert_eq!(dist_sq, 0.25);
    }

    #[test]
    fn test_within_radius() {
        let raw_points = &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ];
        let builder = KdTreeBuilder::from_points(raw_points);
        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);
        let points = make_coords(raw_points);

        // Radius 1.5 should include first 2 points
        let results = kdtree.within_radius(c(0.0, 0.0, 0.0), &points, 1.5);
        assert_eq!(results.len(), 2);

        // Radius 2.5 should include first 3 points
        let results = kdtree.within_radius(c(0.0, 0.0, 0.0), &points, 2.5);
        assert_eq!(results.len(), 3);

        // Radius 100 should include all
        let results = kdtree.within_radius(c(0.0, 0.0, 0.0), &points, 100.0);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_nearest_n_within() {
        let mut raw_points = Vec::new();
        for i in 0..10 {
            raw_points.push([i as f32, 0.0, 0.0]);
        }
        let builder = KdTreeBuilder::from_points(&raw_points);

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);
        let points = make_coords(&raw_points);

        // Get up to 3 within radius 5
        let results = kdtree.nearest_n_within(c(0.0, 0.0, 0.0), &points, 5.0, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_serialization() {
        let raw_points = &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let builder = KdTreeBuilder::from_points(raw_points);
        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);

        let points = make_coords(raw_points);

        // Serialize
        let bytes = kdtree.to_bytes();

        // Deserialize
        let kdtree2 = CompactKdTree::from_bytes(&bytes);

        // Verify same results
        let (idx1, dist1) = kdtree.nearest(c(0.0, 0.0, 0.0), &points).unwrap();
        let (idx2, dist2) = kdtree2.nearest(c(0.0, 0.0, 0.0), &points).unwrap();
        assert_eq!(idx1, idx2);
        assert_eq!(dist1, dist2);
    }

    #[test]
    fn test_large_tree() {
        let mut raw_points = Vec::with_capacity(1000);

        // Create a grid of points
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    raw_points.push([x as f32, y as f32, z as f32]);
                }
            }
        }
        let builder = KdTreeBuilder::from_points(&raw_points);

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);
        let points = make_coords(&raw_points);

        // Test nearest neighbor at various positions
        let (idx, dist_sq) = kdtree.nearest(c(0.0, 0.0, 0.0), &points).unwrap();
        assert_eq!(dist_sq, 0.0);
        let point = get_point(&points, idx);
        assert_eq!(point.x(), 0.0);

        let (_, dist_sq) = kdtree.nearest(c(9.0, 9.0, 9.0), &points).unwrap();
        assert_eq!(dist_sq, 0.0);

        // Test point not in grid
        let (_, dist_sq) = kdtree.nearest(c(0.5, 0.5, 0.5), &points).unwrap();
        assert_eq!(dist_sq, 0.75); // Distance to nearest corner
    }

    #[test]
    fn test_negative_coordinates() {
        let raw_points = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
        let builder = KdTreeBuilder::from_points(&raw_points);

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);
        let points = make_coords(&raw_points);

        let (idx, _) = kdtree.nearest(c(-0.9, -0.9, -0.9), &points).unwrap();
        let point = get_point(&points, idx);
        assert_eq!(point.x(), -1.0);

        let (idx, _) = kdtree.nearest(c(0.9, 0.9, 0.9), &points).unwrap();
        let point = get_point(&points, idx);
        assert_eq!(point.x(), 1.0);
    }

    #[test]
    fn test_size_in_bytes() {
        let mut raw_points = Vec::new();
        for i in 0..100 {
            raw_points.push([i as f32, 0.0, 0.0]);
        }
        let builder = KdTreeBuilder::from_points(&raw_points);

        let tree = builder.build();
        let kdtree = CompactKdTree::new(&tree);

        let size = kdtree.size_in_bytes();
        // Tree has at least 100 entries * 4 bytes
        assert!(size >= 400);
    }
}

use std::collections::VecDeque;

/// A educational LOUDS Trie implementation.
///
/// Core Components:
/// 1. bits: The LOUDS bit-string (1s and 0s).
/// 2. labels: The characters on the edges (stored in level-order).
pub struct LoudsTrie {
    bits: Vec<bool>,
    labels: Vec<u8>,
}

impl LoudsTrie {
    /// 1. CONSTRUCTION
    /// Builds a LOUDS trie from a sorted list of strings.
    pub fn new(keys: &[&str]) -> Self {
        if keys.is_empty() {
            return Self {
                bits: vec![true, false],
                labels: vec![],
            };
        }

        // We need a standard temporary Trie to build the structure first
        // because LOUDS requires Level-Order traversal (BFS) to construct.
        #[derive(Default)]
        struct TempNode {
            children: Vec<(u8, TempNode)>, // Ordered children
            is_terminal: bool,
        }

        // Build standard Trie in memory first
        let mut root = TempNode::default();
        for key in keys {
            let mut node = &mut root;
            for byte in key.bytes() {
                // Find or insert child
                if let Some(idx) = node.children.iter().position(|(b, _)| *b == byte) {
                    node = &mut node.children[idx].1;
                } else {
                    node.children.push((byte, TempNode::default()));
                    node = &mut node.children.last_mut().unwrap().1;
                }
            }
            node.is_terminal = true;
        }

        // BFS to generate LOUDS bits and Labels
        let mut bits = Vec::new();
        let mut labels = Vec::new();
        let mut queue = VecDeque::new();

        // Push fake super-root to start the sequence "10" for the actual root
        // (Standard LOUDS usually starts with "10" representing the super-root -> root edge)
        bits.push(true);
        bits.push(false);

        queue.push_back(&root);

        while let Some(node) = queue.pop_front() {
            for (char_byte, child) in &node.children {
                bits.push(true); // '1' for every child
                labels.push(*char_byte);
                queue.push_back(child);
            }
            bits.push(false); // '0' to end this node's list of children
        }

        Self { bits, labels }
    }

    /// 2. NAVIGATION (The "Magic" Math)

    // Returns the bit-index of the first child of the node at `index`.
    // Formula: select0(rank1(index + 1)) + 1
    fn first_child(&self, index: usize) -> Option<usize> {
        if index >= self.bits.len() || !self.bits[index] {
            return None;
        }

        let r1 = self.rank1(index + 1);
        let s0 = self.select0(r1)?;

        Some(s0 + 1)
    }

    // Returns the bit-index of the parent of the node at `index`.
    // Formula: select1(rank0(index))
    // Note: This effectively finds which "0" block we are inside,
    // and maps it back to the "1" that generated it.
    fn parent(&self, index: usize) -> Option<usize> {
        if index == 0 {
            return None;
        } // Super-root has no parent

        let r0 = self.rank0(index);
        let s1 = self.select1(r0)?;

        Some(s1)
    }

    // Returns the Label (character) leading to this node.
    // The label is stored at `rank1(index) - 2` because:
    // - rank1 gives us the "Node ID" (dense 0..N)
    // - Minus 1 for 0-based indexing
    // - Minus 1 because the Root has no incoming label.
    fn get_label(&self, index: usize) -> Option<u8> {
        let r1 = self.rank1(index);
        if r1 < 2 {
            return None;
        } // Root (Node 1) has no label
        Some(self.labels[r1 - 2])
    }

    /// 3. BOTTOM-UP TRAVERSAL
    /// Given a Node ID (bit index), reconstruct the full string.
    pub fn reconstruct_key(&self, mut node_index: usize) -> String {
        let mut result = Vec::new();
        // Walk up until we hit the root (index 0)
        while let Some(p_index) = self.parent(node_index) {
            if let Some(char_byte) = self.get_label(node_index) {
                result.push(char_byte);
            }
            node_index = p_index;
            if node_index == 0 {
                break;
            }
        }

        result.reverse();
        String::from_utf8(result).unwrap_or_default()
    }

    pub fn contains(&self, key: &str) -> bool {
        let mut current_index = 0; // Start at super-root

        for byte in key.bytes() {
            // Find the first child of the current node
            if let Some(mut child_index) = self.first_child(current_index) {
                let mut found = false;
                // Iterate over children
                while self.bits[child_index] {
                    if let Some(label) = self.get_label(child_index) {
                        if label == byte {
                            // Found matching child
                            current_index = child_index;
                            found = true;
                            break;
                        }
                    }
                    child_index += 1; // Move to next sibling
                }
                if !found {
                    return false; // No matching child found
                }
            } else {
                return false; // No children, key not found
            }
        }

        // Check if current_index is a terminal node (i.e., has no children)
        match self.first_child(current_index) {
            Some(_) => false, // Has children, not terminal
            None => true,     // No children, is terminal
        }
    }

    // --- Naive Rank/Select Helpers (O(N) - Slow!) ---
    // In production, use `fid-rs` or `succinct` crates here.

    fn rank1(&self, end_idx: usize) -> usize {
        self.bits[0..end_idx].iter().filter(|&&b| b).count()
    }

    fn rank0(&self, end_idx: usize) -> usize {
        self.bits[0..end_idx].iter().filter(|&&b| !b).count()
    }

    fn select0(&self, target_rank: usize) -> Option<usize> {
        let mut count = 0;
        for (i, &b) in self.bits.iter().enumerate() {
            if !b {
                count += 1;
            }
            if count == target_rank {
                return Some(i);
            }
        }
        None
    }

    fn select1(&self, target_rank: usize) -> Option<usize> {
        let mut count = 0;
        for (i, &b) in self.bits.iter().enumerate() {
            if b {
                count += 1;
            }
            if count == target_rank {
                return Some(i);
            }
        }
        None
    }
}

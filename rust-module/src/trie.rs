use std::{io::Read, mem};

use byteorder::{LittleEndian, ReadBytesExt};
use rayon::prelude::*;

/// A sentinel value representing "null" or "no node".
/// Must fit in 24 bits for the packed next_sibling field.
const NONE: u32 = 0x00FFFFFF; // 16,777,215 - maximum value for 24 bits

/// A compact node representation (16 bytes total).
///
/// Layout:
/// Each node uses 12 bytes:
/// - label_start (4 bytes)
/// - first_child (4 bytes)
/// - next_sibling_packed (4 bytes):
///   - next_sibling: bits 0-23 (24 bits, max 16M nodes)
///   - label_len: bits 24-30 (7 bits, max 127 chars)
///   - is_terminal: bit 31 (1 bit)
#[derive(Clone, Copy, Debug)]
#[repr(C)] // Ensures consistent memory layout for WASM/FFI
pub struct Node {
    pub label_start: u32,
    pub first_child: u32,
    // Packed field: next_sibling (24 bits) | label_len (7 bits) | is_terminal (1 bit)
    pub next_sibling_packed: u32,
}

impl Node {
    // Helper functions for packed next_sibling field

    pub fn next_sibling(&self) -> u32 {
        self.next_sibling_packed & 0x00FFFFFF // First 24 bits
    }

    pub fn label_len(&self) -> u16 {
        ((self.next_sibling_packed >> 24) & 0x7F) as u16 // Bits 24-30 (7 bits)
    }

    pub fn is_terminal(&self) -> bool {
        (self.next_sibling_packed >> 31) != 0 // Bit 31
    }

    pub fn set_next_sibling(&mut self, next_sibling: u32) {
        // Validate that next_sibling fits in 24 bits
        debug_assert!(
            next_sibling <= 0x00FFFFFF,
            "next_sibling {} exceeds 24-bit limit",
            next_sibling
        );
        // Clear the next_sibling bits and set new value
        self.next_sibling_packed =
            (self.next_sibling_packed & 0xFF000000) | (next_sibling & 0x00FFFFFF);
    }

    pub fn set_label_len(&mut self, label_len: u16) {
        // Validate that label_len fits in 7 bits
        debug_assert!(
            label_len <= 127,
            "label_len {} exceeds 7-bit limit",
            label_len
        );
        // Clear the label_len bits and set new value
        self.next_sibling_packed =
            (self.next_sibling_packed & 0x80FFFFFF) | ((label_len as u32 & 0x7F) << 24);
    }

    pub fn set_is_terminal(&mut self, is_terminal: bool) {
        // Clear the is_terminal bit and set new value
        self.next_sibling_packed =
            (self.next_sibling_packed & 0x7FFFFFFF) | ((is_terminal as u32) << 31);
    }

    pub fn new(
        label_start: u32,
        first_child: u32,
        next_sibling: u32,
        label_len: u16,
        is_terminal: bool,
    ) -> Self {
        let mut node = Node {
            label_start,
            first_child,
            next_sibling_packed: 0,
        };
        node.set_next_sibling(next_sibling);
        node.set_label_len(label_len);
        node.set_is_terminal(is_terminal);
        node
    }
}

/// A linear-memory Patricia Trie.
///
/// It consists of two vectors:
/// 1. `nodes`: The structural data.
/// 2. `labels`: A massive append-only byte array storing string fragments.
pub struct CompactPatriciaTrie {
    pub nodes: Vec<Node>,
    pub labels: Vec<u8>,
}

impl CompactPatriciaTrie {
    pub fn new() -> Self {
        let mut trie = Self {
            nodes: vec![],
            labels: vec![],
        };
        // Create a dummy root node.
        // The root has no label and is not terminal.
        trie.nodes.push(Node::new(0, NONE, NONE, 0, false));
        trie
    }

    /// Inserts a string into the trie.
    pub fn insert(&mut self, key: &str) {
        let key_bytes = key.as_bytes();
        let mut node_idx = 0; // Start at root
        let mut key_cursor = 0;

        // Traverse down the tree
        'outer: while key_cursor < key_bytes.len() {
            // We need to look at the children of the current node
            let mut child_idx = self.nodes[node_idx].first_child;
            let mut prev_child_idx = NONE;

            while child_idx != NONE {
                // Get common prefix length between the key remainder and this child's label
                let child_label = self.get_label(child_idx);
                let current_key_part = &key_bytes[key_cursor..];

                let common_len = Self::common_prefix(child_label, current_key_part);

                if common_len > 0 {
                    // CASE 1: Partial match or Full match

                    // If the child label matches fully, but the key has more characters,
                    // or if the child label is longer than the common match (split needed).

                    if common_len < child_label.len() {
                        // SPLIT NEEDED: The child's edge is "longer" than the match.
                        // Example: Existing="banana", Insert="ban". Common=3.
                        // We must split "banana" into "ban" -> "ana".
                        self.split_node(child_idx, common_len);
                    }

                    // Move cursor forward
                    key_cursor += common_len;
                    node_idx = child_idx as usize;
                    continue 'outer;
                }

                // No match, move to next sibling
                prev_child_idx = child_idx;
                child_idx = self.nodes[child_idx as usize].next_sibling();
            }

            // CASE 2: No matching child found.
            // We must insert a new child node with the remainder of the key.
            let remainder = &key_bytes[key_cursor..];
            let new_child_idx = self.create_node(remainder, true);

            // Link the new node into the sibling chain
            if prev_child_idx == NONE {
                // It's the first child
                self.nodes[node_idx].first_child = new_child_idx;
            } else {
                self.nodes[prev_child_idx as usize].set_next_sibling(new_child_idx);
            }

            return;
        }

        // If we exhausted the key exactly at this node, mark it terminal.
        if key_cursor == key_bytes.len() {
            self.nodes[node_idx].set_is_terminal(true);
        }
    }

    /// Returns true if the exact string exists in the trie.
    pub fn contains(&self, key: &str) -> bool {
        let key_bytes = key.as_bytes();
        let mut node_idx = 0;
        let mut key_cursor = 0;

        while key_cursor < key_bytes.len() {
            let mut child_idx = self.nodes[node_idx].first_child;
            let mut matched_child = false;

            while child_idx != NONE {
                let child_label = self.get_label(child_idx);
                let current_key_part = &key_bytes[key_cursor..];

                // For a successful search, the child label must be a prefix of the remaining key
                if current_key_part.starts_with(child_label) {
                    key_cursor += child_label.len();
                    node_idx = child_idx as usize;
                    matched_child = true;
                    break;
                }
                child_idx = self.nodes[child_idx as usize].next_sibling();
            }

            if !matched_child {
                return false;
            }
        }

        self.nodes[node_idx].is_terminal()
    }

    /// Returns up to 6 suggestions extending the given prefix.
    pub fn suggest(&self, prefix: &str, num_suggestions: usize) -> Vec<String> {
        let mut results = Vec::new();
        let prefix_bytes = prefix.as_bytes();
        let mut node_idx = 0;
        let mut key_cursor = 0;
        // Buffer contains the current prefix being built with correct capitalization
        let mut buffer = vec![];

        // 1. Traverse to the end of the prefix
        while key_cursor < prefix_bytes.len() {
            let mut child_idx = self.nodes[node_idx].first_child;
            let mut found_child = false;

            while child_idx != NONE {
                let child_label = self.get_label(child_idx);
                let current_key_part = &prefix_bytes[key_cursor..];

                // Check how much of the prefix matches this child
                let common_len = Self::common_prefix(child_label, current_key_part);

                if common_len > 0 {
                    // Match found (partial or complete)
                    // Append the matched, correctly capitalized part to the buffer
                    buffer.extend_from_slice(&child_label[..common_len]);
                    // If the match is partial (e.g. prefix="ba", child="banana"),
                    // we have found our target node. We are inside this node.
                    if common_len == current_key_part.len() {
                        // We consumed the whole prefix.
                        // We are now "inside" this child node at offset `common_len`.
                        // We start collecting from here.
                        let mut buffer = String::from_utf8(buffer).unwrap();
                        self.collect_suggestions(
                            child_idx,
                            common_len,
                            &mut buffer,
                            &mut results,
                            num_suggestions,
                        );
                        return results;
                    }

                    // If we consumed the whole child label but still have prefix left
                    // (e.g. prefix="banana", child="ban")
                    if common_len == child_label.len() {
                        key_cursor += common_len;
                        node_idx = child_idx as usize;
                        found_child = true;
                        break; // Break inner loop, continue outer 'while' to go deeper
                    }

                    // Mismatch: prefix has characters that don't match child
                    // (e.g. prefix="band", child="bar") -> No results.
                    return results;
                }

                child_idx = self.nodes[child_idx as usize].next_sibling();
            }

            if !found_child {
                // We have remaining prefix characters but no matching child.
                return results;
            }
        }

        // If we reach here, we consumed the exact prefix and landed exactly on a node boundary.
        // We act as if we are at the children of the current node.
        // However, standard traversal leaves us at `node_idx`. We need to search its children.

        // Corner case: The user typed exactly a string that is a node boundary.
        // We traverse all children of the current node_idx.
        // Note: The `node_idx` itself might be terminal, but `collect_suggestions`
        // is designed to complete a specific node.

        // We manually check if the current node (which we fully traversed) is terminal?
        // Actually, the previous logic (insert/contains) assumes we processed the node.
        // For suggestion, if we are exactly at a node, we just start DFS on its children.

        let mut buffer = String::from(prefix);

        // If the exact prefix itself is a valid word (and we are at root or a boundary), add it.
        // Note: The root is never terminal, so this check is safe.
        if self.nodes[node_idx as usize].is_terminal() {
            results.push(buffer.clone());
        }

        // Start DFS on children
        let mut child = self.nodes[node_idx as usize].first_child;
        while child != NONE {
            self.collect_suggestions(child, 0, &mut buffer, &mut results, num_suggestions);
            if results.len() >= num_suggestions {
                return results;
            }
            child = self.nodes[child as usize].next_sibling();
        }

        results
    }

    // --- Helper Methods ---

    /// Creates a new node and appends its label to the label store.
    fn create_node(&mut self, label: &[u8], is_terminal: bool) -> u32 {
        let label_start = self.labels.len() as u32;
        let label_len = label.len() as u16;

        self.labels.extend_from_slice(label);

        let idx = self.nodes.len() as u32;
        self.nodes
            .push(Node::new(label_start, NONE, NONE, label_len, is_terminal));
        idx
    }

    /// Splits an edge at `split_idx` (length of prefix).
    /// Used when an existing edge "banana" needs to become "ban" -> "ana".
    fn split_node(&mut self, node_idx: u32, split_len: usize) {
        let node_label_start = self.nodes[node_idx as usize].label_start;
        let node_label_len = self.nodes[node_idx as usize].label_len();

        // 1. Create a new node representing the suffix ("ana")
        // NOTE: We don't need to copy bytes to `labels`. We can point to the existing
        // bytes in `labels` just offset by `split_len`.
        let suffix_start = node_label_start + split_len as u32;
        let suffix_len = node_label_len - split_len as u16;

        let new_child_idx = self.nodes.len() as u32;

        // The new child inherits the children and terminal status of the original node
        let original_children = self.nodes[node_idx as usize].first_child;
        let original_terminal = self.nodes[node_idx as usize].is_terminal();

        self.nodes.push(Node::new(
            suffix_start,
            original_children, // Takes custody of existing children
            NONE,              // It will be the only child of the parent (for now)
            suffix_len,
            original_terminal,
        ));

        // 2. Update the original node to represent the prefix ("ban")
        // It now points to the new child.
        let node = &mut self.nodes[node_idx as usize];
        node.set_label_len(split_len as u16);
        node.first_child = new_child_idx;
        node.set_is_terminal(false); // "ban" is likely not terminal unless explicitly marked later
    }

    fn get_label(&self, node_idx: u32) -> &[u8] {
        let node = &self.nodes[node_idx as usize];
        let start = node.label_start as usize;
        let end = start + node.label_len() as usize;
        &self.labels[start..end]
    }
    /// Recursive helper to collect suggestions.
    ///
    /// `node_idx`: The node we are currently visiting.
    /// `offset`: How many bytes of this node's label we have already "matched" or "skipped".
    ///           (Used when the user's prefix ended in the middle of this node's label).
    pub fn collect_suggestions(
        &self,
        node_idx: u32,
        offset: usize,
        buffer: &mut String,
        results: &mut Vec<String>,
        num_suggestions: usize,
    ) {
        if results.len() >= num_suggestions {
            return;
        }

        let node = &self.nodes[node_idx as usize];
        let full_label = self.get_label(node_idx);

        // Append the *remainder* of the label (part we haven't typed yet)
        let remainder = &full_label[offset..];

        // Safety: We assume labels are valid UTF-8 since inputs are &str.
        // In a raw bytes trie, you'd handle this differently.
        let remainder_str = unsafe { std::str::from_utf8_unchecked(remainder) };

        let added_len = remainder_str.len();
        buffer.push_str(remainder_str);

        // 1. Is this node itself a valid word?
        if node.is_terminal() {
            results.push(buffer.clone());
            if results.len() >= 6 {
                // Backtrack before returning
                buffer.truncate(buffer.len() - added_len);
                return;
            }
        }

        // 2. DFS into children
        let mut child = node.first_child;
        while child != NONE {
            self.collect_suggestions(child, 0, buffer, results, num_suggestions);
            if results.len() >= num_suggestions {
                buffer.truncate(buffer.len() - added_len);
                return;
            }
            child = self.nodes[child as usize].next_sibling();
        }

        // Backtrack
        buffer.truncate(buffer.len() - added_len);
    }

    /// Memory usage estimation in bytes
    pub fn size_in_bytes(&self) -> usize {
        (self.nodes.len() * mem::size_of::<Node>()) + (self.labels.len())
    }
    /// Helper to find the length of the common prefix between two byte slices.
    fn common_prefix(a: &[u8], b: &[u8]) -> usize {
        a.iter()
            .zip(b)
            .take_while(|(x, y)| x.to_ascii_lowercase() == y.to_ascii_lowercase())
            .count()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize nodes
        let node_count = self.nodes.len() as u32;
        bytes.extend_from_slice(&node_count.to_le_bytes());
        let node_bytes = unsafe {
            std::slice::from_raw_parts(
                self.nodes.as_ptr() as *const u8,
                self.nodes.len() * mem::size_of::<Node>(),
            )
        };
        bytes.extend_from_slice(node_bytes);

        // Serialize labels
        let label_count = self.labels.len() as u32;
        bytes.extend_from_slice(&label_count.to_le_bytes());
        bytes.extend_from_slice(&self.labels);

        dbg!(self.nodes.len());
        dbg!(self.labels.len());

        bytes
    }

    pub fn from_bytes(data: &[u8]) -> Self {
        let mut cursor = std::io::Cursor::new(data);

        // Deserialize nodes
        let node_count = cursor.read_u32::<LittleEndian>().unwrap() as usize;
        let mut nodes = Vec::with_capacity(node_count);
        unsafe {
            nodes.set_len(node_count);
        }
        let node_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                nodes.as_mut_ptr() as *mut u8,
                node_count * mem::size_of::<Node>(),
            )
        };
        cursor.read_exact(node_bytes).unwrap();

        // Deserialize labels
        let label_count = cursor.read_u32::<LittleEndian>().unwrap() as usize;

        let mut labels = Vec::with_capacity(label_count);
        unsafe {
            labels.set_len(label_count);
        }
        let label_bytes =
            unsafe { std::slice::from_raw_parts_mut(labels.as_mut_ptr() as *mut u8, label_count) };
        cursor.read_exact(label_bytes).unwrap();

        Self { nodes, labels }
    }

    pub fn compress(&mut self) {
        let total_nodes = self.nodes.len();
        println!("Starting smart compression on {} nodes...", total_nodes);

        // --- Step 1: Identify Unique Strings ---
        // We map every node to a Unique ID so we can work with a smaller dataset.
        // Map: String -> Unique_ID
        let mut string_to_id = std::collections::HashMap::new();
        let mut unique_strings = Vec::new();
        // Vector mapping: Node_Index -> Unique_ID
        let mut node_to_unique_id = vec![0usize; total_nodes];

        for (i, node) in self.nodes.iter().enumerate() {
            let start = node.label_start as usize;
            let end = start + node.label_len() as usize;
            let s = String::from_utf8_lossy(&self.labels[start..end]).to_string();

            if let Some(&id) = string_to_id.get(&s) {
                node_to_unique_id[i] = id;
            } else {
                let id = unique_strings.len();
                string_to_id.insert(s.clone(), id);
                unique_strings.push(s);
                node_to_unique_id[i] = id;
            }
        }

        let num_uniques = unique_strings.len();
        println!("Reduced to {} unique strings. Continuing...", num_uniques);

        // --- Redirect Table ---
        // redirects[id] = (Target_ID, Offset)
        // Initially, everyone points to themselves (Target = Self, Offset = 0).
        let mut redirects: Vec<(usize, u32)> = (0..num_uniques).map(|i| (i, 0)).collect();
        // Tracks if a string is still a "Root" (hasn't been merged into another).
        let mut is_active = vec![true; num_uniques];

        // --- Step 2: Prefix Filter ---
        // Sort indices by string content. "Apple" comes before "ApplePie".
        let mut sorted_indices: Vec<usize> = (0..num_uniques).collect();
        sorted_indices.sort_unstable_by(|&a, &b| unique_strings[a].cmp(&unique_strings[b]));

        let mut prefix_merges = 0;
        for i in 0..num_uniques - 1 {
            let small_id = sorted_indices[i];
            let large_id = sorted_indices[i + 1];

            // If Small is prefix of Large
            if unique_strings[large_id].starts_with(&unique_strings[small_id]) {
                // Merge Small -> Large
                redirects[small_id] = (large_id, 0);
                is_active[small_id] = false;
                prefix_merges += 1;
            }
        }

        // --- Step 3: Suffix Filter ---
        // We only check items that survived the prefix pass.
        let mut active_indices: Vec<usize> = (0..num_uniques).filter(|&i| is_active[i]).collect();

        // Sort by Reversed String. "ana" comes before "ananab" (banana reversed).
        active_indices.sort_unstable_by(|&a, &b| {
            unique_strings[a].chars().rev().cmp(unique_strings[b].chars().rev())
        });

        let mut suffix_merges = 0;
        for i in 0..active_indices.len() - 1 {
            let small_id = active_indices[i];
            let large_id = active_indices[i + 1];

            let s_small = &unique_strings[small_id];
            let s_large = &unique_strings[large_id];

            // Check if small is suffix of large (by checking starts_with on reverse)
            if s_large.chars().rev().zip(s_small.chars().rev()).all(|(a, b)| a == b) {
                // Calculate offset in the forward string
                // "ana" inside "banana". Offset = 6 - 3 = 3.
                let offset = (s_large.len() - s_small.len()) as u32;

                redirects[small_id] = (large_id, offset);
                is_active[small_id] = false;
                suffix_merges += 1;
            }
        }

        // --- Step 4: Flatten Chains ---
        // We resolve the chains (A -> B -> C) so everyone points to the Final Root.
        // Map: Unique_ID -> (Final_Root_ID, Total_Offset)
        let mut final_resolution: Vec<(usize, u32)> = vec![(0, 0); num_uniques];

        for i in 0..num_uniques {
            let mut curr = i;
            let mut total_offset = 0;
            let mut depth = 0;

            // Follow the redirects until we hit a Root (active node pointing to self)
            while !is_active[curr] {
                let (next, off) = redirects[curr];
                // Safety break for cycles (shouldn't happen)
                if next == curr { break; } 
                
                total_offset += off;
                curr = next;
                
                depth += 1;
                if depth > 1000 { break; } // Prevent infinite loops
            }
            final_resolution[i] = (curr, total_offset);
        }

        // --- Step 5: Reconstruction ---
        println!("Constructing super-buffer...");
        
        let mut super_buffer = Vec::new();
        // Map: Unique_ID (of Roots) -> Absolute_Byte_Address_In_Buffer
        let mut root_addresses = vec![0u32; num_uniques];

        for i in 0..num_uniques {
            // We only write Active Roots to the file
            if is_active[i] {
                let start_addr = super_buffer.len() as u32;
                super_buffer.extend_from_slice(unique_strings[i].as_bytes());
                root_addresses[i] = start_addr;
            }
        }

        // --- Step 6: Update Nodes ---
        println!("Updating pointers for {} nodes...", total_nodes);

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let unique_id = node_to_unique_id[i];
            
            // 1. Where does this unique string live logically? (Root + Relative Offset)
            let (root_id, relative_offset) = final_resolution[unique_id];
            
            // 2. Where is that Root physically located?
            let absolute_base = root_addresses[root_id];
            
            // 3. Final Address
            node.label_start = absolute_base + relative_offset;
        }

        self.labels = super_buffer;
        println!("Smart compression complete. Final size: {} bytes.", self.labels.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion_and_search() {
        let mut trie = CompactPatriciaTrie::new();
        trie.insert("apple");
        trie.insert("app");
        trie.insert("banana");
        trie.insert("bandana");

        assert!(trie.contains("apple"));
        assert!(trie.contains("app"));
        assert!(trie.contains("banana"));
        assert!(trie.contains("bandana"));

        assert!(!trie.contains("ban")); // inserted implicitly as path, but not terminal
        assert!(!trie.contains("apples")); // too long
        assert!(!trie.contains("orange")); // missing
    }

    #[test]
    fn test_split_logic() {
        let mut trie = CompactPatriciaTrie::new();
        // Insert "test"
        trie.insert("test");
        // Insert "team" -> should split "test" into "te" -> ("st", "am")
        trie.insert("team");

        assert!(trie.contains("test"));
        assert!(trie.contains("team"));

        // Internal check (whitebox)
        // Root -> "te" (child)
        // "te" -> "st" (child), "am" (sibling of "st")
        let root_child = trie.nodes[0].first_child;
        let lbl = trie.get_label(root_child);
        assert_eq!(lbl, b"te");
    }

    #[test]
    fn test_node_memory_layout() {
        // Verify that Node is 12 bytes (3 u32s) instead of the original 16 bytes
        assert_eq!(std::mem::size_of::<Node>(), 12);

        // Test packed field functionality
        let mut node = Node::new(100, 200, 300, 50, true);

        // Test getter methods
        assert_eq!(node.label_start, 100);
        assert_eq!(node.first_child, 200);
        assert_eq!(node.next_sibling(), 300);
        assert_eq!(node.label_len(), 50);
        assert_eq!(node.is_terminal(), true);

        // Test setter methods
        node.set_next_sibling(400);
        node.set_label_len(75);
        node.set_is_terminal(false);

        assert_eq!(node.next_sibling(), 400);
        assert_eq!(node.label_len(), 75);
        assert_eq!(node.is_terminal(), false);

        // Test edge cases
        node.set_next_sibling(0x00FFFFFF); // Max 24-bit value
        node.set_label_len(127); // Max 7-bit value
        node.set_is_terminal(true);

        assert_eq!(node.next_sibling(), 0x00FFFFFF);
        assert_eq!(node.label_len(), 127);
        assert_eq!(node.is_terminal(), true);
    }
}

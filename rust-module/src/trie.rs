use std::{
    collections::{HashMap, VecDeque},
    convert::TryInto,
    mem,
};

/// Sentinel for CompactNode (23 bits)
const COMPACT_NONE: u32 = 0x007FFFFF;

/// A compact node representation (8 bytes).
/// Optimized for space and cache locality.
///
/// Layout:
/// - label_start (4 bytes)
/// - packed (4 bytes):
///   - first_child: 23 bits (8M nodes max)
///   - label_len: 7 bits (127 chars max)
///   - is_terminal: 1 bit
///   - has_next_sibling: 1 bit
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CompactNode {
    pub label_start: u32,
    pub packed: u32,
}

impl CompactNode {
    pub fn first_child(&self) -> u32 {
        self.packed & 0x007FFFFF
    }

    pub fn label_len(&self) -> u16 {
        ((self.packed >> 23) & 0x7F) as u16
    }

    pub fn is_terminal(&self) -> bool {
        ((self.packed >> 30) & 1) != 0
    }

    pub fn has_next_sibling(&self) -> bool {
        ((self.packed >> 31) & 1) != 0
    }

    pub fn new(
        label_start: u32,
        first_child: u32,
        label_len: u16,
        is_terminal: bool,
        has_next_sibling: bool,
    ) -> Self {
        debug_assert!(first_child <= 0x007FFFFF, "first_child index too large");
        debug_assert!(label_len <= 127, "label_len too large");

        let packed = (first_child & 0x007FFFFF)
            | ((label_len as u32 & 0x7F) << 23)
            | ((is_terminal as u32) << 30)
            | ((has_next_sibling as u32) << 31);

        CompactNode {
            label_start,
            packed,
        }
    }
}
#[derive(Debug, Default)]
struct Node {
    // The string segment associated with the edge leading to this node
    prefix: String,
    // Use HashMap to index children by their first character
    children: HashMap<char, Node>,
    // Marks if a word ends at this exact node
    is_leaf: bool,
}

impl Node {
    fn new(prefix: String, is_leaf: bool) -> Self {
        Self {
            prefix,
            is_leaf,
            children: HashMap::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct TrieBuilder {
    root: Node,
}

impl TrieBuilder {
    pub fn new() -> Self {
        Self {
            root: Node::new(String::from(""), false),
        }
    }

    pub fn insert(&mut self, word: &str) {
        let mut current_node = &mut self.root;
        let mut remaining_key = word;

        while !remaining_key.is_empty() {
            // 1. Look for a child that starts with the first char of our remaining key
            let first_char = remaining_key.chars().next().unwrap();

            if current_node.children.contains_key(&first_char) {
                let child_node = current_node.children.get_mut(&first_char).unwrap();
                // Calculate longest common prefix (LCP) between remaining_key and child.prefix
                let common_len = Self::common_prefix_len(&child_node.prefix, remaining_key);

                // Case 2: Full Match - We traverse deeper
                // Example: Tree has "apple", Insert "applepie" (common: "apple")
                if common_len == child_node.prefix.len() {
                    remaining_key = &remaining_key[common_len..];
                    current_node = child_node;

                    // If we consumed the whole key, mark this node as a word end
                    if remaining_key.is_empty() {
                        current_node.is_leaf = true;
                    }
                }
                // Case 3: Partial Match - We need to split the existing edge
                // Example: Tree has "apple", Insert "apply" (common: "appl")
                else {
                    // 3a. Split the existing child node
                    let child_suffix = child_node.prefix[common_len..].to_string();
                    let input_suffix = remaining_key[common_len..].to_string();

                    // Truncate the current child's prefix to the common part (e.g., "apple" -> "appl")
                    child_node.prefix.truncate(common_len);

                    // Create a new node for the split part of the original child (e.g., "e")
                    // It inherits the children and leaf status of the original node
                    let mut split_node = Node::new(child_suffix, child_node.is_leaf);
                    split_node.children = std::mem::take(&mut child_node.children);

                    // The original node is no longer a leaf (unless the new word ends exactly here)
                    child_node.is_leaf = false;

                    // Re-attach the split part
                    let split_key = split_node.prefix.chars().next().unwrap();
                    child_node.children.insert(split_key, split_node);

                    // 3b. Insert the new word's remaining part (if any)
                    if !input_suffix.is_empty() {
                        let input_key = input_suffix.chars().next().unwrap();
                        child_node
                            .children
                            .insert(input_key, Node::new(input_suffix, true));
                    } else {
                        // The inserted word ended exactly at the split point
                        child_node.is_leaf = true;
                    }

                    return;
                }
            } else {
                // No matching edge. Create a new one with the rest of the key.
                current_node
                    .children
                    .insert(first_char, Node::new(remaining_key.to_string(), true));
                return;
            }
        }
    }

    /// Converts the pointer-based RadixTree into the flat, cache-friendly CompactRadixTrie.
    /// Uses subtree sharing to compress the structure.
    pub fn build(&self) -> (Vec<CompactNode>, Vec<u8>) {
        println!("Started building compact trie...");

        let mut nodes = Vec::new();
        let mut labels = Vec::new();
        // Maps (Label, IsTerminal, FirstChildHash, NextSiblingHash) -> (Hash, NodeIndex)
        // We need mapped Hash to allow hierarchical hashing, and NodeIndex to point to it.
        // Actually the user said "map ... into the hash ... (which is an int). We'll then have another hashmap to map the hash int into an index".
        // Let's follow that.
        // Cache: (Label, IsTerminal, FirstChildHash, NextSiblingHash) -> HashID
        let mut node_hash_map: HashMap<(String, bool, i32, i32), i32> = HashMap::new();
        // Dedup: HashID -> NodeIndex
        let mut dedup_map: HashMap<i32, u32> = HashMap::new();
        
        // Counter for unique hashes
        let mut next_hash_id = 0;

        // Process root. The root is a single node list.
        // Note: The original implementation initialized root inside build.
        // We'll treat root as the start of the recursion.
        
        let root_siblings = vec![&self.root];
        self.build_recursive(
            &root_siblings,
            &mut nodes,
            &mut labels,
            &mut node_hash_map,
            &mut dedup_map,
            &mut next_hash_id
        );

        compress_labels(&mut labels, &mut nodes);

        (nodes, labels)
    }

    fn build_recursive(
        &self,
        siblings: &[&Node],
        nodes: &mut Vec<CompactNode>,
        labels: &mut Vec<u8>,
        node_hash_map: &mut HashMap<(String, bool, i32, i32), i32>,
        dedup_map: &mut HashMap<i32, u32>,
        next_hash_id: &mut i32,
    ) -> (u32, i32) {
        if siblings.is_empty() {
            return (COMPACT_NONE, -1);
        }

        let start_idx = nodes.len() as u32;
        let labels_start_len = labels.len();

        // 1. Allocate space for siblings
        // We push placeholder nodes. We'll fill them later.
        for _ in siblings {
            nodes.push(CompactNode::new(0, COMPACT_NONE, 0, false, false));
        }

        // To store computed properties for the backward pass
        let mut sibling_data = Vec::with_capacity(siblings.len());

        // 2. Recurse on children for each sibling
        for node in siblings.iter() {
            // Sort children
            let mut children: Vec<&Node> = node.children.values().collect();
            children.sort_by(|a, b| a.prefix.cmp(&b.prefix));

            // Recurse
            let (child_idx, child_hash) = self.build_recursive(
                &children,
                nodes,
                labels,
                node_hash_map,
                dedup_map,
                next_hash_id,
            );

            // Add label to main array
            let label_len = node.prefix.len();
            if label_len > 127 {
                panic!("Label '{}' too long", node.prefix);
            }
            let label_start = labels.len() as u32;
            labels.extend_from_slice(node.prefix.as_bytes());

            sibling_data.push((label_start, label_len, child_idx, child_hash));
        }

        // 3. Backward pass to compute hashes and resolve deduplication
        let mut next_sibling_hash = -1; // -1 for no sibling

        // We iterate backwards
        for i in (0..siblings.len()).rev() {
            let node = siblings[i];
            let (label_start, label_len, child_idx, child_hash) = sibling_data[i];
            
            let is_terminal = node.is_leaf;
            
            // Compute hash for this node (representing the subtree starting here)
            let key = (node.prefix.clone(), is_terminal, child_hash, next_sibling_hash);
            
            let my_hash = if let Some(&h) = node_hash_map.get(&key) {
                h
            } else {
                let h = *next_hash_id;
                *next_hash_id += 1;
                node_hash_map.insert(key, h);
                h
            };

            // Update the node in the vector
            // Note: We need to set has_next_sibling based on loop index
            let has_next = i < siblings.len() - 1;
            
            // Reconstruct the node with correct values
            nodes[(start_idx as usize) + i] = CompactNode::new(
                label_start,
                child_idx,
                label_len as u16,
                is_terminal,
                has_next,
            );

            // If this is the FIRST sibling in the chain, we check for deduplication of the WHOLE chain
            if i == 0 {
                if let Some(&existing_idx) = dedup_map.get(&my_hash) {
                    // FOUND DUPLICATE!
                    // Rollback nodes and labels
                    nodes.truncate(start_idx as usize);
                    labels.truncate(labels_start_len);
                    return (existing_idx, my_hash);
                } else {
                    // Register this new unique chain
                    dedup_map.insert(my_hash, start_idx);
                    return (start_idx, my_hash);
                }
            }

            next_sibling_hash = my_hash;
        }
        
        // This part is unreachable because the loop always runs at least once and handles i==0 return.
        (COMPACT_NONE, -1)
    }

    // Helper to find length of common prefix
    fn common_prefix_len(s1: &str, s2: &str) -> usize {
        s1.bytes()
            .zip(s2.bytes())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// An immutable, space-optimized Radix Trie.
/// Nodes are 8 bytes each (vs 12 bytes in Builder).
pub struct CompactRadixTrie<'a> {
    pub nodes: &'a [CompactNode],
    pub labels: &'a [u8],
}

impl<'a> CompactRadixTrie<'a> {
    pub fn new(nodes: &'a [CompactNode], labels: &'a [u8]) -> Self {
        Self { nodes, labels }
    }

    pub fn from_bytes(data: &'a [u8]) -> Self {
        let node_size = mem::size_of::<CompactNode>();
        let node_count = u32::from_le_bytes(data[0..4].try_into().unwrap());

        let nodes_start = 4;
        let nodes_end = nodes_start + (node_count as usize * node_size);
        let nodes_bytes = &data[nodes_start..nodes_end];

        let labels_count = u32::from_le_bytes(data[nodes_end..nodes_end + 4].try_into().unwrap());

        let labels_start = nodes_end + 4;
        let labels_end = labels_start + (labels_count as usize);

        let labels_bytes = &data[labels_start..labels_end];

        let nodes: &[CompactNode] = unsafe {
            std::slice::from_raw_parts(
                nodes_bytes.as_ptr() as *const CompactNode,
                nodes_bytes.len() / node_size,
            )
        };

        Self {
            nodes,
            labels: labels_bytes,
        }
    }

    fn get_label(&self, node_idx: u32) -> &[u8] {
        let node = &self.nodes[node_idx as usize];
        let start = node.label_start as usize;
        let end = start + node.label_len() as usize;
        &self.labels[start..end]
    }

    pub fn contains(&self, key: &str) -> bool {
        let key_bytes = key.as_bytes();
        let mut node_idx = 0;
        let mut key_cursor = 0;

        while key_cursor < key_bytes.len() {
            let mut child_idx = self.nodes[node_idx].first_child();

            if child_idx == COMPACT_NONE {
                return false;
            }

            let mut matched_child = false;

            // Iterate through sequential siblings
            loop {
                let child_label = self.get_label(child_idx);
                let current_key_part = &key_bytes[key_cursor..];

                if current_key_part.starts_with(child_label) {
                    key_cursor += child_label.len();
                    node_idx = child_idx as usize;
                    matched_child = true;
                    break;
                }

                if self.nodes[child_idx as usize].has_next_sibling() {
                    child_idx += 1;
                } else {
                    break;
                }
            }

            if !matched_child {
                return false;
            }
        }

        self.nodes[node_idx].is_terminal()
    }

    pub fn suggest(&self, prefix: &str, num_suggestions: usize) -> Vec<String> {
        let mut results = Vec::new();
        let prefix_bytes = prefix.as_bytes();
        let mut node_idx = 0;
        let mut key_cursor = 0;
        let mut buffer = vec![];

        while key_cursor < prefix_bytes.len() {
            let mut child_idx = self.nodes[node_idx].first_child();
            if child_idx == COMPACT_NONE {
                return results;
            }

            let mut found_child = false;

            loop {
                let child_label = self.get_label(child_idx);
                let current_key_part = &prefix_bytes[key_cursor..];
                let common_len = common_prefix_len(child_label, current_key_part);
                
                // Debug print
                if prefix == "APP" {
                     println!("Debug: child_idx={}, label='{}', key_part='{}', common={}", 
                         child_idx, 
                         String::from_utf8_lossy(child_label), 
                         String::from_utf8_lossy(current_key_part), 
                         common_len);
                }

                if common_len > 0 {
                    buffer.extend_from_slice(&child_label[..common_len]);

                    if common_len == current_key_part.len() {
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

                    if common_len == child_label.len() {
                        key_cursor += common_len;
                        node_idx = child_idx as usize;
                        found_child = true;
                        break;
                    }

                    return results;
                }

                if self.nodes[child_idx as usize].has_next_sibling() {
                    child_idx += 1;
                } else {
                    break;
                }
            }

            if !found_child {
                return results;
            }
        }

        let mut buffer = String::from(prefix);
        if self.nodes[node_idx as usize].is_terminal() {
            results.push(buffer.clone());
        }

        let mut child = self.nodes[node_idx as usize].first_child();
        if child != COMPACT_NONE {
            loop {
                self.collect_suggestions(child, 0, &mut buffer, &mut results, num_suggestions);
                if results.len() >= num_suggestions {
                    return results;
                }
                if self.nodes[child as usize].has_next_sibling() {
                    child += 1;
                } else {
                    break;
                }
            }
        }

        results
    }

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
        let remainder = &full_label[offset..];
        let remainder_str = unsafe { std::str::from_utf8_unchecked(remainder) };
        let added_len = remainder_str.len();
        buffer.push_str(remainder_str);

        if node.is_terminal() {
            results.push(buffer.clone());
            if results.len() >= num_suggestions {
                buffer.truncate(buffer.len() - added_len);
                return;
            }
        }

        let mut child = node.first_child();
        if child != COMPACT_NONE {
            loop {
                self.collect_suggestions(child, 0, buffer, results, num_suggestions);
                if results.len() >= num_suggestions {
                    buffer.truncate(buffer.len() - added_len);
                    return;
                }
                if self.nodes[child as usize].has_next_sibling() {
                    child += 1;
                } else {
                    break;
                }
            }
        }

        buffer.truncate(buffer.len() - added_len);
    }

    pub fn size_in_bytes(&self) -> usize {
        (self.nodes.len() * mem::size_of::<CompactNode>()) + (self.labels.len())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        let node_count = self.nodes.len() as u32;
        data.extend_from_slice(&node_count.to_le_bytes());

        let nodes_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.nodes.as_ptr() as *const u8,
                self.nodes.len() * mem::size_of::<CompactNode>(),
            )
        };
        data.extend_from_slice(nodes_bytes);

        let label_count = self.labels.len() as u32;
        data.extend_from_slice(&label_count.to_le_bytes());
        data.extend_from_slice(self.labels);

        data
    }
}

pub fn compress_labels(labels: &mut Vec<u8>, nodes: &mut Vec<CompactNode>) {
    fn calc_overlap(a: &str, b: &str) -> usize {
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        let max_ov = std::cmp::min(a_bytes.len(), b_bytes.len());

        for k in (1..=max_ov).rev() {
            if a_bytes[a_bytes.len() - k..] == b_bytes[..k] {
                return k;
            }
        }
        0
    }
    enum Action {
        None,
        Append(usize),
        Prepend(usize),
    }

    let total_nodes = nodes.len();
    println!(
        "Starting multi-stage compression on {} strings...",
        total_nodes
    );

    // ==================================================================================
    // STEP 1: Basic Deduplication
    // ==================================================================================
    let mut string_to_id = HashMap::new();
    let mut unique_strings = Vec::new();
    let mut node_to_unique_id = vec![0usize; total_nodes];

    for (i, node) in nodes.iter().enumerate() {
        let start = node.label_start as usize;
        let end = start + node.label_len() as usize;
        let slice = if end <= labels.len() {
            &labels[start..end]
        } else {
            &[]
        };
        let s = String::from_utf8_lossy(slice).to_string();

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
    println!(
        "    Reduced to {} unique strings. Analyzing substrings...",
        num_uniques
    );

    // ==================================================================================
    // STEP 2: Substring Compression (Parent/Child)
    // ==================================================================================
    // Map short strings to substrings of longer strings
    let mut redirects: Vec<(usize, u32)> = (0..num_uniques).map(|i| (i, 0)).collect();
    let mut is_active = vec![true; num_uniques];

    let mut sorted_by_len: Vec<usize> = (0..num_uniques).collect();
    sorted_by_len.sort_unstable_by(|&a, &b| unique_strings[a].len().cmp(&unique_strings[b].len()));

    let mut length_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &id in &sorted_by_len {
        let len = unique_strings[id].len();
        if len > 0 {
            length_groups.entry(len).or_default().push(id);
        }
    }

    let mut distinct_lengths: Vec<_> = length_groups.keys().cloned().collect();
    distinct_lengths.sort_unstable();

    // Rabin-Karp setup
    const P: u64 = 131;
    let max_len = if num_uniques > 0 {
        unique_strings[sorted_by_len[num_uniques - 1]].len()
    } else {
        0
    };
    let mut pow_p = vec![1u64; max_len + 1];
    for i in 1..=max_len {
        pow_p[i] = pow_p[i - 1].wrapping_mul(P);
    }

    let mut substring_hashes: HashMap<u64, (usize, u32)> = HashMap::with_capacity(50_000);
    let mut targets_start_idx = 0;

    for &len in &distinct_lengths {
        let candidates = &length_groups[&len];

        // Identify targets (strings strictly longer than current length)
        while targets_start_idx < num_uniques {
            let id = sorted_by_len[targets_start_idx];
            if unique_strings[id].len() > len {
                break;
            }
            targets_start_idx += 1;
        }
        if targets_start_idx >= num_uniques {
            break;
        }

        let target_indices = &sorted_by_len[targets_start_idx..];
        substring_hashes.clear();

        let lead_power = pow_p[len - 1];

        // Hash targets
        for &target_id in target_indices {
            let target_s = &unique_strings[target_id];
            let target_bytes = target_s.as_bytes();
            let mut current_hash: u64 = 0;

            // Initial window
            for k in 0..len {
                current_hash = current_hash
                    .wrapping_mul(P)
                    .wrapping_add(target_bytes[k] as u64);
            }
            substring_hashes
                .entry(current_hash)
                .or_insert((target_id, 0));

            // Rolling window
            for i in 1..=(target_bytes.len() - len) {
                let prev = target_bytes[i - 1] as u64;
                let new = target_bytes[i + len - 1] as u64;
                current_hash = current_hash.wrapping_sub(prev.wrapping_mul(lead_power));
                current_hash = current_hash.wrapping_mul(P).wrapping_add(new);
                substring_hashes
                    .entry(current_hash)
                    .or_insert((target_id, i as u32));
            }
        }

        // Match candidates
        for &short_id in candidates {
            let short_bytes = unique_strings[short_id].as_bytes();
            let mut h: u64 = 0;
            for &b in short_bytes {
                h = h.wrapping_mul(P).wrapping_add(b as u64);
            }

            if let Some(&(target_id, offset)) = substring_hashes.get(&h) {
                // Verify to avoid collisions
                let target_bytes = unique_strings[target_id].as_bytes();
                if short_bytes == &target_bytes[offset as usize..(offset as usize + len)] {
                    redirects[short_id] = (target_id, offset);
                    is_active[short_id] = false;
                }
            }
        }
    }

    // Resolve Step 2 pointers
    let mut step2_resolution: Vec<(usize, u32)> = vec![(0, 0); num_uniques];
    let mut active_roots = Vec::new();

    for i in 0..num_uniques {
        let mut curr = i;
        let mut total_offset = 0;
        let mut depth = 0;
        while !is_active[curr] {
            let (next, off) = redirects[curr];
            if next == curr {
                break;
            } // safety
            total_offset += off;
            curr = next;
            depth += 1;
            if depth > 1000 {
                break;
            } // cycle breaker
        }
        step2_resolution[i] = (curr, total_offset);
    }

    for i in 0..num_uniques {
        if is_active[i] {
            active_roots.push(i);
        }
    }

    println!(
        "    Step 2 complete. Merging {} root strings...",
        active_roots.len()
    );

    // ==================================================================================
    // STEP 3: Greedy Superstring Merge (Overlap Optimization)
    // ==================================================================================

    // Buckets for fast lookup: start_byte -> vec<root_id>
    let mut by_start_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];
    let mut by_end_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];

    let mut root_is_available = vec![false; num_uniques];
    let mut root_final_offsets: HashMap<usize, u32> = HashMap::with_capacity(active_roots.len());

    let mut remaining_count = 0;

    // Initialize buckets and handle empty strings immediately
    for &root_id in &active_roots {
        let s = &unique_strings[root_id];
        if s.is_empty() {
            // FIX: Empty strings have no overlap potential but must have an entry.
            // Map them to 0 (or any valid int), they read 0 bytes anyway.
            root_final_offsets.insert(root_id, 0);
            continue;
        }

        let bytes = s.as_bytes();
        by_start_byte[bytes[0] as usize].push(root_id);
        by_end_byte[bytes[bytes.len() - 1] as usize].push(root_id);

        root_is_available[root_id] = true;
        remaining_count += 1;
    }

    let mut super_buffer = Vec::new();

    while remaining_count > 0 {
        // Pick a seed
        let mut best_seed = None;

        // Quick seed selection: just pop from the active list until we find an available one
        while let Some(candidate) = active_roots.pop() {
            if root_is_available[candidate] {
                best_seed = Some(candidate);
                break;
            }
        }

        if best_seed.is_none() {
            // This happens if remaining_count > 0 but we ran out of seeds in the stack.
            // This should theoretically not happen if logic is perfect, but acts as safe exit.
            break;
        }

        let seed_id = best_seed.unwrap();

        root_is_available[seed_id] = false;
        remaining_count -= 1;

        // Chain structure: (RootID, Overlap_With_Previous)
        let mut chain: VecDeque<(usize, u32)> = VecDeque::new();
        chain.push_back((seed_id, 0));

        let mut left_edge_id = seed_id;
        let mut right_edge_id = seed_id;

        // Grow chain greedy
        loop {
            let mut best_action = Action::None;
            let mut max_savings = 0;

            // Try Append
            let r_str = &unique_strings[right_edge_id];
            // Safety check although empty strings are filtered out
            if !r_str.is_empty() {
                let r_bytes = r_str.as_bytes();
                let last_char = r_bytes[r_bytes.len() - 1] as usize;

                for &candidate_id in &by_start_byte[last_char] {
                    if !root_is_available[candidate_id] {
                        continue;
                    }
                    let c_str = &unique_strings[candidate_id];
                    let overlap = calc_overlap(r_str, c_str);
                    if overlap > max_savings {
                        max_savings = overlap;
                        best_action = Action::Append(candidate_id);
                    }
                }
            }

            // Try Prepend
            let l_str = &unique_strings[left_edge_id];
            if !l_str.is_empty() {
                let l_bytes = l_str.as_bytes();
                let first_char = l_bytes[0] as usize;

                for &candidate_id in &by_end_byte[first_char] {
                    if !root_is_available[candidate_id] {
                        continue;
                    }
                    let c_str = &unique_strings[candidate_id];
                    let overlap = calc_overlap(c_str, l_str);
                    if overlap >= max_savings && overlap > 0 {
                        max_savings = overlap;
                        best_action = Action::Prepend(candidate_id);
                    }
                }
            }

            match best_action {
                Action::None => break,
                Action::Append(id) => {
                    chain.push_back((id, max_savings as u32));
                    root_is_available[id] = false;
                    right_edge_id = id;
                    remaining_count -= 1;
                }
                Action::Prepend(id) => {
                    chain.push_front((id, max_savings as u32));
                    root_is_available[id] = false;
                    left_edge_id = id;
                    remaining_count -= 1;
                }
            }
        }

        // Finalize chain to buffer
        if chain.is_empty() {
            continue;
        }

        let mut current_write_pos = super_buffer.len() as u32;

        // Handle first item in chain
        let first_id = chain[0].0;
        root_final_offsets.insert(first_id, current_write_pos);
        super_buffer.extend_from_slice(unique_strings[first_id].as_bytes());

        // Handle rest
        let mut prev_id = first_id;
        for i in 1..chain.len() {
            let next_id = chain[i].0;
            // For Prepend, we pushed (id, overlap).
            // For Append, we pushed (id, overlap).
            // In both cases, the 'overlap' value in the tuple represented overlap
            // relative to the neighbor in the direction we grew.
            // Since we ordered the deque correctly [Left ... Right],
            // we can just re-calculate linear overlaps to be 100% safe and simple.

            let prev_s = &unique_strings[prev_id];
            let next_s = &unique_strings[next_id];
            let ov = calc_overlap(prev_s, next_s);

            // Write only the non-overlapping suffix
            let next_bytes = next_s.as_bytes();
            if ov < next_bytes.len() {
                let to_write = &next_bytes[ov..];
                // The logical start of this string is 'ov' bytes before the end of buffer
                let start_pos = super_buffer.len() as u32 - ov as u32;
                root_final_offsets.insert(next_id, start_pos);
                super_buffer.extend_from_slice(to_write);
            } else {
                // Fully contained (should be rare given Step 2, but possible)
                let start_pos = super_buffer.len() as u32 - ov as u32;
                root_final_offsets.insert(next_id, start_pos);
            }
            prev_id = next_id;
        }
    }

    // ==================================================================================
    // STEP 4: Finalize Pointers
    // ==================================================================================
    println!("    Updating pointers...");

    for (i, node) in nodes.iter_mut().enumerate() {
        let unique_id = node_to_unique_id[i];

        let (root_id, offset_in_root) = step2_resolution[unique_id];

        // Safety: root_id comes from active_roots.
        // If root was empty string, it's in the map (offset 0).
        // If root was merged, it's in the map.
        let root_base = *root_final_offsets
            .get(&root_id)
            .expect("Root ID missing from offsets");

        node.label_start = root_base + offset_in_root;
    }

    labels.clear();
    labels.append(&mut super_buffer);

    println!(
        "    Total compression complete. Final size: {} bytes.",
        labels.len()
    );
}

// Helper: Calculate overlap length

// Helper to find length of common prefix
// fn common_prefix_len(s1: &[u8], s2: &[u8]) -> usize {
//     s1.iter()
//         .zip(s2)
//         .take_while(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
//         .count()
// }
fn common_prefix_len(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter()
        .zip(s2)
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion_and_search() {
        let mut builder = TrieBuilder::new();
        builder.insert("apple");
        builder.insert("app");
        builder.insert("banana");
        builder.insert("bandana");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        assert!(trie.contains("apple"));
        assert!(trie.contains("app"));
        assert!(trie.contains("banana"));
        assert!(trie.contains("bandana"));

        assert!(!trie.contains("ban"));
        assert!(!trie.contains("apples"));
        assert!(!trie.contains("orange"));
    }

    #[test]
    fn test_split_logic() {
        let mut builder = TrieBuilder::new();
        builder.insert("test");
        builder.insert("team");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        assert!(trie.contains("test"));
        assert!(trie.contains("team"));
    }

    #[test]
    fn test_compact_node_memory_layout() {
        // Verify CompactNode is 8 bytes
        assert_eq!(std::mem::size_of::<CompactNode>(), 8);

        let node = CompactNode::new(100, 200, 50, true, true);
        assert_eq!(node.label_start, 100);
        assert_eq!(node.first_child(), 200);
        assert_eq!(node.label_len(), 50);
        assert_eq!(node.is_terminal(), true);
        assert_eq!(node.has_next_sibling(), true);
    }

    #[test]
    fn test_empty_trie() {
        let builder = TrieBuilder::new();
        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        assert!(!trie.contains(""));
        assert!(!trie.contains("anything"));

        let suggestions = trie.suggest("test", 10);
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_single_word() {
        let mut builder = TrieBuilder::new();
        builder.insert("hello");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        assert!(trie.contains("hello"));
        assert!(!trie.contains("hel"));
        assert!(!trie.contains("hello world"));
        assert!(!trie.contains(""));
    }

    #[test]
    fn test_prefix_words() {
        let mut builder = TrieBuilder::new();
        builder.insert("a");
        builder.insert("ab");
        builder.insert("abc");
        builder.insert("abcd");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        assert!(trie.contains("a"));
        assert!(trie.contains("ab"));
        assert!(trie.contains("abc"));
        assert!(trie.contains("abcd"));
        assert!(!trie.contains("abcde"));
    }

    #[test]
    fn test_suggest_comprehensive() {
        let mut builder = TrieBuilder::new();

        // Insert words with common prefixes
        builder.insert("apple");
        builder.insert("application");
        builder.insert("apply");
        builder.insert("app");
        builder.insert("appreciate");
        builder.insert("banana");
        builder.insert("bandana");
        builder.insert("band");
        builder.insert("test");
        builder.insert("testing");
        builder.insert("tester");
        builder.insert("team");
        builder.insert("tea");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        // Test 1: Suggestions for "app" prefix
        let suggestions = trie.suggest("app", 10);
        assert_eq!(
            suggestions.len(),
            5,
            "Should find 5 words starting with 'app'"
        );
        assert!(suggestions.contains(&"app".to_string()));
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
        assert!(suggestions.contains(&"apply".to_string()));
        assert!(suggestions.contains(&"appreciate".to_string()));

        // Test 2: Suggestions for "ban" prefix
        let suggestions = trie.suggest("ban", 10);
        assert_eq!(
            suggestions.len(),
            3,
            "Should find 3 words starting with 'ban'"
        );
        assert!(suggestions.contains(&"banana".to_string()));
        assert!(suggestions.contains(&"bandana".to_string()));
        assert!(suggestions.contains(&"band".to_string()));

        // Test 3: Suggestions for "te" prefix
        let suggestions = trie.suggest("te", 10);
        assert_eq!(
            suggestions.len(),
            5,
            "Should find 5 words starting with 'te'"
        );
        assert!(suggestions.contains(&"test".to_string()));
        assert!(suggestions.contains(&"testing".to_string()));
        assert!(suggestions.contains(&"tester".to_string()));
        assert!(suggestions.contains(&"team".to_string()));
        assert!(suggestions.contains(&"tea".to_string()));

        // Test 4: Limit suggestions count
        let suggestions = trie.suggest("app", 2);
        assert_eq!(suggestions.len(), 2, "Should limit to 2 suggestions");

        // Test 5: Exact match that also has extensions
        let suggestions = trie.suggest("test", 10);
        assert!(
            suggestions.len() >= 3,
            "Should find 'test' and its extensions"
        );
        assert!(suggestions.contains(&"test".to_string()));
        assert!(suggestions.contains(&"testing".to_string()));
        assert!(suggestions.contains(&"tester".to_string()));

        // Test 6: No matching prefix
        let suggestions = trie.suggest("xyz", 10);
        assert_eq!(
            suggestions.len(),
            0,
            "Should find no suggestions for non-existent prefix"
        );

        // Test 7: Prefix that's longer than any word
        let suggestions = trie.suggest("applicationextended", 10);
        assert_eq!(
            suggestions.len(),
            0,
            "Should find no suggestions for too-long prefix"
        );

        // Test 8: Single character prefix
        let suggestions = trie.suggest("a", 10);
        assert!(
            suggestions.len() >= 5,
            "Should find all words starting with 'a'"
        );
        assert!(suggestions.contains(&"app".to_string()));
        assert!(suggestions.contains(&"apple".to_string()));

        // Test 9: Empty prefix (edge case)
        let suggestions = trie.suggest("", 5);
        // Depending on implementation, this might return all words or none
        // This test documents the behavior
        assert!(
            suggestions.len() <= 5,
            "Should respect limit even for empty prefix"
        );

        // Test 10: Limit of 0
        let suggestions = trie.suggest("app", 0);
        assert_eq!(
            suggestions.len(),
            0,
            "Should return no suggestions when limit is 0"
        );

        // Test 11: Prefix that matches exactly one word with no extensions
        let suggestions = trie.suggest("banana", 10);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0], "banana");

        // Test 12: Case sensitivity
        let suggestions = trie.suggest("APP", 10);
        assert_eq!(
            suggestions.len(),
            0,
            "Should be case sensitive - no matches for uppercase. Got: {:?}", suggestions
        );
    }

    #[test]
    fn test_suggest_with_nested_prefixes() {
        let mut builder = TrieBuilder::new();

        // Create a scenario with nested prefix words
        builder.insert("car");
        builder.insert("card");
        builder.insert("care");
        builder.insert("career");
        builder.insert("careful");
        builder.insert("carefully");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        // Test suggestions at different prefix levels
        let suggestions = trie.suggest("car", 10);
        assert_eq!(suggestions.len(), 6);
        assert!(suggestions.contains(&"car".to_string()));
        assert!(suggestions.contains(&"card".to_string()));
        assert!(suggestions.contains(&"care".to_string()));
        assert!(suggestions.contains(&"career".to_string()));
        assert!(suggestions.contains(&"careful".to_string()));
        assert!(suggestions.contains(&"carefully".to_string()));

        let suggestions = trie.suggest("care", 10);
        assert_eq!(suggestions.len(), 4);
        assert!(suggestions.contains(&"care".to_string()));
        assert!(suggestions.contains(&"career".to_string()));
        assert!(suggestions.contains(&"careful".to_string()));
        assert!(suggestions.contains(&"carefully".to_string()));

        let suggestions = trie.suggest("careful", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"careful".to_string()));
        assert!(suggestions.contains(&"carefully".to_string()));
    }

    #[test]
    fn test_deduplication() {
        let mut builder = TrieBuilder::new();
        // Insert "ax" and "bx". 
        // "x" is terminal for both.
        // Structure:
        // Root children: a, b.
        // a -> x
        // b -> x
        // The 'x' node should be shared.
        // Create "a->x" and "b->x" structure explicitly.
        builder.insert("a");
        builder.insert("ax");
        builder.insert("b");
        builder.insert("bx");
        
        // Disable println in build (if possible) or just ignore noise.
        let (nodes, _labels) = builder.build();
        
        // Root is at 0.
        let root = &nodes[0];
        // Root children: 'a' and 'b'.
        // They should be siblings.
        let child_start = root.first_child();
        assert!(child_start != crate::trie::COMPACT_NONE);
        
        // We expect child_start to be 'a' (sorted 'a' < 'b')
        // And child_start + 1 to be 'b'.
        
        let a_idx = child_start;
        let b_idx = child_start + 1;
        
        let a_node = &nodes[a_idx as usize];
        let b_node = &nodes[b_idx as usize];
        
        // Verify 'a' has next sibling (which is 'b')
        assert!(a_node.has_next_sibling());
        // Verify 'b' does NOT have next sibling
        assert!(!b_node.has_next_sibling());
        
        // Check children of a and b
        let a_child = a_node.first_child();
        let b_child = b_node.first_child();
        
        assert!(a_child != crate::trie::COMPACT_NONE);
        assert!(b_child != crate::trie::COMPACT_NONE);
        
        // The deduplication logic should make them point to the SAME index
        assert_eq!(a_child, b_child, "Children of 'ax' and 'bx' should point to same 'x' node");
    }
    
    #[test]
    fn test_case_sensitivity_isolated() {
        let mut builder = TrieBuilder::new();
        builder.insert("app");
        builder.insert("apple");
        
        // Uncomment if you re-enabled compress_labels
        let (nodes, labels) = builder.build();
        
        println!("Nodes: {:?}", nodes);
        println!("Labels: {:?}", String::from_utf8_lossy(&labels));
        for (i, node) in nodes.iter().enumerate() {
            println!("Node {}: label_start={}, label_len={}, label='{}'", 
                i, node.label_start, node.label_len(), 
                String::from_utf8_lossy(&labels[node.label_start as usize..(node.label_start + node.label_len() as u32) as usize])); 
        }

        let trie = CompactRadixTrie::new(&nodes, &labels);
    
        // Test 12: Case sensitivity
        let suggestions = trie.suggest("APP", 10);
        assert_eq!(
            suggestions.len(),
            0,
            "Should be case sensitive - no matches for uppercase. Got: {:?}", suggestions
        );
    }

    #[test]
    fn test_suggest_with_special_characters() {
        let mut builder = TrieBuilder::new();

        builder.insert("hello-world");
        builder.insert("hello-there");
        builder.insert("hello_world");
        builder.insert("hello.world");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        let suggestions = trie.suggest("hello", 10);
        assert_eq!(suggestions.len(), 4);
        assert!(suggestions.contains(&"hello-world".to_string()));
        assert!(suggestions.contains(&"hello-there".to_string()));
        assert!(suggestions.contains(&"hello_world".to_string()));
        assert!(suggestions.contains(&"hello.world".to_string()));

        let suggestions = trie.suggest("hello-", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"hello-world".to_string()));
        assert!(suggestions.contains(&"hello-there".to_string()));
    }
}

use std::{
    collections::{HashMap, VecDeque},
    convert::TryInto,
    mem,
};

/// Sentinel
const COMPACT_NONE: u32 = 0xFFFFFFFF;

// =========================================================================
// BITMASKS & LAYOUT DEFINITIONS
// =========================================================================
// Both Node Types (Word 0)
const MASK_IS_LEAF: u32    = 1 << 31;
const MASK_HAS_NEXT: u32   = 1 << 30;
const MASK_LABEL_LEN: u32  = 0x3E000000; // 5 bits (25..29) -> max label length 31
const MASK_LABEL_START: u32 = 0x01FFFFFF; // 25 bits (0..24) -> 32MB Label Limit

// Internal Node Only (Word 1)
const MASK_IS_TERMINAL: u32 = 1 << 31;
const MASK_CHILD_IDX: u32   = 0x7FFFFFFF; // 31 bits

/// Helper to decode node data on the fly
#[derive(Debug, Copy, Clone)]
pub struct NodeView {
    pub label_start: u32,
    pub label_len: u16,
    pub has_next: bool,
    pub is_leaf: bool,
    // Fields below only valid if !is_leaf
    pub is_terminal: bool,
    pub first_child: u32,
}

impl NodeView {
    #[inline(always)]
    pub fn from_slice(data: &[u32], idx: usize) -> Self {
        let w0 = data[idx];
        let is_leaf = (w0 & MASK_IS_LEAF) != 0;
        let has_next = (w0 & MASK_HAS_NEXT) != 0;
        let label_len = ((w0 & MASK_LABEL_LEN) >> 25) as u16;  // 5-bit field starts at bit 25
        let label_start = w0 & MASK_LABEL_START;

        if is_leaf {
            NodeView {
                label_start,
                label_len,
                has_next,
                is_leaf: true,
                // Leaves are implicitly terminal in a path-compressed trie
                // (Assuming we don't store "empty" leaves that aren't words)
                is_terminal: true, 
                first_child: COMPACT_NONE,
            }
        } else {
            let w1 = data[idx + 1];
            NodeView {
                label_start,
                label_len,
                has_next,
                is_leaf: false,
                is_terminal: (w1 & MASK_IS_TERMINAL) != 0,
                first_child: w1 & MASK_CHILD_IDX,
            }
        }
    }

    fn set_label_start(nodes: &mut Vec<u32>, idx: usize, new_start: u32) {
        let w0 = nodes[idx];
        let label_len = w0 & MASK_LABEL_LEN;
        let has_next = w0 & MASK_HAS_NEXT;
        let is_leaf = w0 & MASK_IS_LEAF;

        let new_w0 = (new_start & MASK_LABEL_START) | label_len | has_next | is_leaf;
        nodes[idx] = new_w0;
    }
}

// =========================================================================
// BUILDER
// =========================================================================

#[derive(Debug, Default)]
struct Node {
    prefix: String,
    children: HashMap<char, Node>,
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

    /// Inserts a word into the radix trie using path compression.
    /// Uses node splitting when partial matches are found.
    pub fn insert(&mut self, word: &str) {
        let mut current_node = &mut self.root;
        let mut remaining_key = word;

        while !remaining_key.is_empty() {
            let first_char = remaining_key.chars().next().unwrap();

            if current_node.children.contains_key(&first_char) {
                let child_node = current_node.children.get_mut(&first_char).unwrap();
                let common_len = Self::common_prefix_len(&child_node.prefix, remaining_key);

                if common_len == child_node.prefix.len() {
                    // Full prefix match - traverse deeper
                    remaining_key = &remaining_key[common_len..];
                    current_node = child_node;
                    if remaining_key.is_empty() {
                        current_node.is_leaf = true;
                    }
                } else {
                    // Partial match - split the node
                    let child_suffix = child_node.prefix[common_len..].to_string();
                    let input_suffix = remaining_key[common_len..].to_string();
                    child_node.prefix.truncate(common_len);

                    // Move existing data to new child
                    let mut split_node = Node::new(child_suffix, child_node.is_leaf);
                    split_node.children = std::mem::take(&mut child_node.children);

                    // Update current node
                    child_node.is_leaf = false;
                    let split_key = split_node.prefix.chars().next().unwrap();
                    child_node.children.insert(split_key, split_node);

                    // Add new branch or mark as terminal
                    if !input_suffix.is_empty() {
                        let input_key = input_suffix.chars().next().unwrap();
                        child_node
                            .children
                            .insert(input_key, Node::new(input_suffix, true));
                    } else {
                        child_node.is_leaf = true;
                    }
                    return;
                }
            } else {
                // No existing child - create new leaf
                current_node
                    .children
                    .insert(first_char, Node::new(remaining_key.to_string(), true));
                return;
            }
        }
    }

    pub fn build(&self) -> (Vec<u32>, Vec<u8>) {
        println!("Started building compact trie (Mixed Node Size)...");

        // Handle empty trie case - if root has no children and is not terminal, return empty
        if self.root.children.is_empty() && !self.root.is_leaf {
            return (Vec::new(), Vec::new());
        }

        let mut nodes = Vec::new();
        let mut labels = Vec::new();

        // Caches for deduplication
        let mut node_hash_map: HashMap<(String, bool, i32, i32), i32> = HashMap::new();
        let mut dedup_map: HashMap<i32, u32> = HashMap::new();
        let mut next_hash_id = 0;

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
        nodes: &mut Vec<u32>,
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

        // 1. Allocate space (First Pass)
        // We need to know IF they are leaves to allocate 1 or 2 words.
        let mut sibling_offsets = Vec::with_capacity(siblings.len());
        
        for node in siblings {
            sibling_offsets.push(nodes.len()); // Store where this sibling starts
            if node.children.is_empty() {
                // Leaf Node: 1 Word
                nodes.push(0);
            } else {
                // Internal Node: 2 Words
                nodes.push(0);
                nodes.push(0);
            }
        }

        // 2. Recurse on children
        let mut sibling_data = Vec::with_capacity(siblings.len());

        for node in siblings.iter() {
            let mut children: Vec<&Node> = node.children.values().collect();
            children.sort_by(|a, b| a.prefix.cmp(&b.prefix));

            let (child_idx, child_hash) = self.build_recursive(
                &children,
                nodes,
                labels,
                node_hash_map,
                dedup_map,
                next_hash_id,
            );

            let label_len = node.prefix.len();
            if label_len > 31 { panic!("Label too long (max 31 bytes)"); }  // 5-bit field can store 0-31
            let label_start = labels.len() as u32;
            labels.extend_from_slice(node.prefix.as_bytes());

            sibling_data.push((label_start, label_len, child_idx, child_hash));
        }

        // 3. Backward Pass & Encoding
        let mut next_sibling_hash = -1;

        for i in (0..siblings.len()).rev() {
            let node = siblings[i];
            let (label_start, label_len, child_idx, child_hash) = sibling_data[i];
            let is_terminal = node.is_leaf;
            let has_next = i < siblings.len() - 1;
            let node_idx = sibling_offsets[i];

            // --- Compute Hash ---
            let key = (node.prefix.clone(), is_terminal, child_hash, next_sibling_hash);
            let my_hash = if let Some(&h) = node_hash_map.get(&key) {
                h
            } else {
                let h = *next_hash_id;
                *next_hash_id += 1;
                node_hash_map.insert(key, h);
                h
            };

            // --- Encode Node ---
            assert!(label_start <= MASK_LABEL_START,
                    "Label start {} exceeds 25-bit limit {}. The labels array has grown beyond 32MB! \
                     Consider implementing better label compression.",
                    label_start, MASK_LABEL_START);

            let mut w0 = label_start
                | ((label_len as u32) << 25)  // 5-bit field starts at bit 25
                | if has_next { MASK_HAS_NEXT } else { 0 };

            if node.children.is_empty() {
                // Leaf Node Encoding
                w0 |= MASK_IS_LEAF; // Bit 31 = 1
                nodes[node_idx] = w0;
            } else {
                // Internal Node Encoding
                w0 &= !MASK_IS_LEAF; // Bit 31 = 0
                nodes[node_idx] = w0;
                
                let mut w1 = child_idx & MASK_CHILD_IDX;
                if is_terminal { w1 |= MASK_IS_TERMINAL; }
                nodes[node_idx + 1] = w1;
            }

            // --- Dedup Check (Only on first sibling) ---
            if i == 0 {
                // IMPORTANT: Don't deduplicate the root node (at index 0)
                // The root must always be at index 0 for the trie to work correctly
                if start_idx == 0 {
                    // This is the root node, don't deduplicate it
                    dedup_map.insert(my_hash, start_idx);
                    return (start_idx, my_hash);
                } else if let Some(&existing_idx) = dedup_map.get(&my_hash) {
                    nodes.truncate(start_idx as usize);
                    labels.truncate(labels_start_len);
                    return (existing_idx, my_hash);
                } else {
                    dedup_map.insert(my_hash, start_idx);
                    return (start_idx, my_hash);
                }
            }
            next_sibling_hash = my_hash;
        }

        (COMPACT_NONE, -1)
    }

    fn common_prefix_len(s1: &str, s2: &str) -> usize {
        s1.bytes().zip(s2.bytes()).take_while(|(a, b)| a == b).count()
    }
}

// =========================================================================
// RUNTIME TRIE
// =========================================================================

pub struct CompactRadixTrie<'a> {
    pub nodes: &'a [u32],
    pub labels: &'a [u8],
}

impl<'a> CompactRadixTrie<'a> {
    pub fn new(nodes: &'a [u32], labels: &'a [u8]) -> Self {
        Self { nodes, labels }
    }

    pub fn from_bytes(data: &'a [u8]) -> Self {
        let node_count = u32::from_le_bytes(data[0..4].try_into().unwrap());
        
        let nodes_start = 4;
        // Nodes are u32 (4 bytes)
        let nodes_end = nodes_start + (node_count as usize * 4); 
        let nodes_bytes = &data[nodes_start..nodes_end];

        let labels_count = u32::from_le_bytes(data[nodes_end..nodes_end + 4].try_into().unwrap());
        let labels_start = nodes_end + 4;
        let labels_end = labels_start + (labels_count as usize);
        let labels_bytes = &data[labels_start..labels_end];

        let nodes: &[u32] = unsafe {
            std::slice::from_raw_parts(
                nodes_bytes.as_ptr() as *const u32,
                nodes_bytes.len() / 4,
            )
        };

        Self {
            nodes,
            labels: labels_bytes,
        }
    }

    pub fn get_label(&self, start: u32, len: u16) -> &[u8] {
        let s = start as usize;
        let e = s + len as usize;
        &self.labels[s..e]
    }

    pub fn contains(&self, key: &str) -> bool {
        let key_bytes = key.as_bytes();
        let mut key_cursor = 0;

        // Root handling: Root is usually internal, but check safety
        if self.nodes.is_empty() { return false; }

        // Start processing from root        
        let root_view = NodeView::from_slice(self.nodes, 0);
        let root_label = self.get_label(root_view.label_start, root_view.label_len);
        
        if !key_bytes.starts_with(root_label) {
            return false;
        }
        key_cursor += root_label.len();
        
        if key_cursor == key_bytes.len() {
            return root_view.is_terminal;
        }
        
        let mut child_ptr = root_view.first_child;
        
        while child_ptr != COMPACT_NONE {
            // child_ptr points to the start of a list of siblings
            let mut matched_child = false;
            let mut curr = child_ptr as usize;

            loop {
                // Decode current sibling
                let node = NodeView::from_slice(self.nodes, curr);
                
                let label = self.get_label(node.label_start, node.label_len);
                let remaining = &key_bytes[key_cursor..];
                
                // Case-insensitive match
                if remaining.len() >= label.len() 
                   && remaining[..label.len()].eq_ignore_ascii_case(label) 
                {
                    key_cursor += label.len();
                    
                    if key_cursor == key_bytes.len() {
                        return node.is_terminal;
                    }
                    
                    // Traverse down
                    child_ptr = node.first_child;
                    matched_child = true;
                    break;
                }

                if node.has_next {
                    // Jump: 1 word if Leaf, 2 words if Internal
                    curr += if node.is_leaf { 1 } else { 2 };
                } else {
                    break;
                }
            }

            if !matched_child {
                return false;
            }
        }

        false
    }

    pub fn suggest(&self, prefix: &str, num_suggestions: usize) -> Vec<String> {
        let mut all_results = Vec::new();
        if self.nodes.is_empty() { return all_results; }

        self.suggest_helper(prefix.as_bytes(), 0, 0, String::new(), &mut all_results);

        all_results.sort_by(|a, b| {
            a.len().cmp(&b.len()).then_with(|| a.cmp(b))
        });
        all_results.truncate(num_suggestions);
        all_results
    }

    fn suggest_helper(
        &self,
        prefix_bytes: &[u8],
        node_idx: usize,
        prefix_pos: usize,
        current_path: String,
        all_results: &mut Vec<String>,
    ) {
        let node = NodeView::from_slice(self.nodes, node_idx);

        // Check if we fully matched prefix
        if prefix_pos >= prefix_bytes.len() {
            let mut buffer = current_path;
            if node.is_terminal {
                all_results.push(buffer.clone());
            }

            let child = node.first_child;
            if child != COMPACT_NONE {
                self.collect_suggestions(child, 0, &mut buffer, all_results, usize::MAX);
            }
            return;
        }

        // Try match children
        let mut child_idx = node.first_child;
        if child_idx == COMPACT_NONE { return; }

        loop {
            let child = NodeView::from_slice(self.nodes, child_idx as usize);
            let child_label = self.get_label(child.label_start, child.label_len);
            let remaining_prefix = &prefix_bytes[prefix_pos..];
            
            let common_len = common_prefix_len_case_insensitive(child_label, remaining_prefix);

            if common_len > 0 {
                let label_str = unsafe { std::str::from_utf8_unchecked(&child_label[..common_len]) };
                let mut new_path = current_path.clone();
                new_path.push_str(label_str);

                if common_len == child_label.len() {
                    // Full label match, go deeper
                    self.suggest_helper(
                        prefix_bytes,
                        child_idx as usize,
                        prefix_pos + common_len,
                        new_path,
                        all_results
                    );
                } else if common_len == remaining_prefix.len() {
                    // Full prefix match, partial label -> Collect everything under here
                    let mut buffer = new_path;
                    self.collect_suggestions(child_idx, common_len, &mut buffer, all_results, usize::MAX);
                }
            }

            if child.has_next {
                child_idx += if child.is_leaf { 1 } else { 2 };
            } else {
                break;
            }
        }
    }

    /// Collects all words under a node using DFS traversal.
    fn collect_suggestions(
        &self,
        node_idx: u32, // Start of a sibling list
        offset: usize, // Offset into the label of the specific node we are entering
        buffer: &mut String,
        results: &mut Vec<String>,
        limit: usize,
    ) {
        if results.len() >= limit { return; }
        
        let mut curr_idx = node_idx;
        let processing_specific_node = offset > 0;

        loop {
            let node = NodeView::from_slice(self.nodes, curr_idx as usize);
                        
            // Add label portion to current path
            let full_label = self.get_label(node.label_start, node.label_len);
            let part_label = &full_label[if processing_specific_node { offset } else { 0 }..];
            let part_str = unsafe { std::str::from_utf8_unchecked(part_label) };
            
            let added_len = part_str.len();
            buffer.push_str(part_str);

            // Check if current path forms a complete word
            if node.is_terminal {
                results.push(buffer.clone());
                if results.len() >= limit {
                    buffer.truncate(buffer.len() - added_len);
                    return;
                }
            }

            // Recursively collect from children
            let child = node.first_child;
            if child != COMPACT_NONE {
                self.collect_suggestions(child, 0, buffer, results, limit);
                if results.len() >= limit {
                     buffer.truncate(buffer.len() - added_len);
                     return;
                }
            }

            buffer.truncate(buffer.len() - added_len);

            // Process siblings unless we're in a specific node due to partial match
            if processing_specific_node {
                break; 
            }

            if node.has_next {
                curr_idx += if node.is_leaf { 1 } else { 2 };
            } else {
                break;
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        let node_count = self.nodes.len() as u32;
        data.extend_from_slice(&node_count.to_le_bytes());

        let nodes_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.nodes.as_ptr() as *const u8,
                self.nodes.len() * 4,
            )
        };
        data.extend_from_slice(nodes_bytes);

        let label_count = self.labels.len() as u32;
        data.extend_from_slice(&label_count.to_le_bytes());
        data.extend_from_slice(self.labels);

        data
    }

    pub fn size_in_bytes(&self) -> usize {
        (self.nodes.len() * mem::size_of::<u32>()) + self.labels.len()
    }

    pub fn analyze_stats(&self) {
        let mut total_nodes = 0;
        let mut leaf_structs = 0;
        let mut internal_structs = 0;
        
        let mut curr = 0;
        while curr < self.nodes.len() {
            total_nodes += 1;
            let node = NodeView::from_slice(self.nodes, curr);
            if node.is_leaf {
                leaf_structs += 1;
                curr += 1;
            } else {
                internal_structs += 1;
                curr += 2;
            }
        }
        
        println!("=== Trie Statistics ===");
        println!("Total Nodes: {}", total_nodes);
        println!("Leaf Structs (4 bytes): {}", leaf_structs);
        println!("Internal Structs (8 bytes): {}", internal_structs);
        println!("Total Node Bytes: {}", (leaf_structs * 4) + (internal_structs * 8));
    }

}

pub fn compress_labels(labels: &mut Vec<u8>, nodes: &mut Vec<u32>) {

    fn calc_overlap(a: &str, b: &str) -> usize {
        let a_bytes = a.as_bytes();
        let b_bytes = b.as_bytes();
        let max_ov = std::cmp::min(a_bytes.len(), b_bytes.len());
        
        for k in (1..=max_ov).rev() {
            if a_bytes[a_bytes.len()-k..] == b_bytes[..k] {
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
    println!("Starting multi-stage compression on {} nodes...", total_nodes);

    // ==================================================================================
    // STEP 1: Basic Deduplication
    // ==================================================================================
    let mut string_to_id = HashMap::new();
    let mut unique_strings = Vec::new();
    let mut node_to_unique_id = HashMap::new();

    let mut i = 0;
    while i < nodes.len() {
        let node = NodeView::from_slice(nodes, i);
        let start = node.label_start as usize;
        let end = start + node.label_len as usize;
        let slice = if end <= labels.len() { &labels[start..end] } else { &[] };
        let s = String::from_utf8_lossy(slice).to_string();

        if let Some(&id) = string_to_id.get(&s) {
            node_to_unique_id.insert(i, id);
        } else {
            let id = unique_strings.len();
            string_to_id.insert(s.clone(), id);
            unique_strings.push(s);
            node_to_unique_id.insert(i, id);
        }

        // Skip properly: 1 word for leaf, 2 words for internal
        i += if node.is_leaf { 1 } else { 2 };
    }

    let num_uniques = unique_strings.len();
    println!("    Reduced to {} unique strings. Analyzing substrings...", num_uniques);

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
        if len > 0 { length_groups.entry(len).or_default().push(id); }
    }

    let mut distinct_lengths: Vec<_> = length_groups.keys().cloned().collect();
    distinct_lengths.sort_unstable();

    // Rabin-Karp setup
    const P: u64 = 131;
    let max_len = if num_uniques > 0 { unique_strings[sorted_by_len[num_uniques - 1]].len() } else { 0 };
    let mut pow_p = vec![1u64; max_len + 1];
    for i in 1..=max_len { pow_p[i] = pow_p[i - 1].wrapping_mul(P); }

    let mut substring_hashes: HashMap<u64, (usize, u32)> = HashMap::with_capacity(50_000);
    let mut targets_start_idx = 0;

    for &len in &distinct_lengths {
        let candidates = &length_groups[&len];

        // Identify targets (strings strictly longer than current length)
        while targets_start_idx < num_uniques {
            let id = sorted_by_len[targets_start_idx];
            if unique_strings[id].len() > len { break; }
            targets_start_idx += 1;
        }
        if targets_start_idx >= num_uniques { break; }

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
                current_hash = current_hash.wrapping_mul(P).wrapping_add(target_bytes[k] as u64); 
            }
            substring_hashes.entry(current_hash).or_insert((target_id, 0));

            // Rolling window
            for i in 1..=(target_bytes.len() - len) {
                let prev = target_bytes[i - 1] as u64;
                let new = target_bytes[i + len - 1] as u64;
                current_hash = current_hash.wrapping_sub(prev.wrapping_mul(lead_power));
                current_hash = current_hash.wrapping_mul(P).wrapping_add(new);
                substring_hashes.entry(current_hash).or_insert((target_id, i as u32));
            }
        }

        // Match candidates
        for &short_id in candidates {
            let short_bytes = unique_strings[short_id].as_bytes();
            let mut h: u64 = 0;
            for &b in short_bytes { h = h.wrapping_mul(P).wrapping_add(b as u64); }

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
            if next == curr { break; } // safety
            total_offset += off;
            curr = next;
            depth += 1;
            if depth > 1000 { break; } // cycle breaker
        }
        step2_resolution[i] = (curr, total_offset);
    }

    for i in 0..num_uniques {
        if is_active[i] {
            active_roots.push(i);
        }
    }

    println!("    Step 2 complete. Merging {} root strings...", active_roots.len());

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
        by_end_byte[bytes[bytes.len()-1] as usize].push(root_id);
        
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
                    if !root_is_available[candidate_id] { continue; }
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
                    if !root_is_available[candidate_id] { continue; }
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
                },
                Action::Prepend(id) => {
                    chain.push_front((id, max_savings as u32));
                    root_is_available[id] = false;
                    left_edge_id = id;
                    remaining_count -= 1;
                }
            }
        }
        
        // Finalize chain to buffer
        if chain.is_empty() { continue; }

        let current_write_pos = super_buffer.len() as u32;
        
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

    let mut i = 0;
    while i < nodes.len() {
        let node = NodeView::from_slice(nodes, i);
        let unique_id = *node_to_unique_id.get(&i).expect("Node index missing from map");

        let (root_id, offset_in_root) = step2_resolution[unique_id];

        // Safety: root_id comes from active_roots.
        // If root was empty string, it's in the map (offset 0).
        // If root was merged, it's in the map.
        let root_base = *root_final_offsets.get(&root_id).expect("Root ID missing from offsets");

        NodeView::set_label_start(nodes, i, root_base + offset_in_root);

        // Skip properly: 1 word for leaf, 2 words for internal
        i += if node.is_leaf { 1 } else { 2 };
    }

    labels.clear();
    labels.append(&mut super_buffer);

    println!( "    Total compression complete. Final size: {} bytes.", labels.len());
}


fn common_prefix_len_case_insensitive(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter().zip(s2).take_while(|(a, b)| a.eq_ignore_ascii_case(b)).count()
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

        // Test 12: Case insensitivity
        let suggestions = trie.suggest("APP", 10);
        assert_eq!(
            suggestions.len(),
            5,
            "Contains should be case insensitive. Got: {:?}", suggestions
        );
        // Verify we got the expected words (in their original case from the trie)
        assert!(suggestions.contains(&"app".to_string()));
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
        assert!(suggestions.contains(&"apply".to_string()));
        assert!(suggestions.contains(&"appreciate".to_string()));
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
        let root = NodeView::from_slice(&nodes, 0);
        // Root children: 'a' and 'b'.
        // They should be siblings.
        let child_start = root.first_child;
        assert!(child_start != crate::trie::COMPACT_NONE);
        
        // We expect child_start to be 'a' (sorted 'a' < 'b')
        // b's position depends on whether 'a' is a leaf (1 word) or internal (2 words)

        let a_idx = child_start as usize;
        let a_node = NodeView::from_slice(&nodes, a_idx);

        // Calculate b_idx based on a's size (1 word for leaf, 2 words for internal)
        let b_idx = if a_node.is_leaf { a_idx + 1 } else { a_idx + 2 };
        let b_node = NodeView::from_slice(&nodes, b_idx);
        
        // Verify 'a' has next sibling (which is 'b')
        assert!(a_node.has_next);
        // Verify 'b' does NOT have next sibling
        assert!(!b_node.has_next);
        
        // Check children of a and b
        let a_child = a_node.first_child;
        let b_child = b_node.first_child;
        
        assert!(a_child != crate::trie::COMPACT_NONE);
        assert!(b_child != crate::trie::COMPACT_NONE);
        
        // The deduplication logic should make them point to the SAME index
        assert_eq!(a_child, b_child, "Children of 'ax' and 'bx' should point to same 'x' node");
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

    #[test]
    fn test_suggest_case_insensitive() {
        let mut builder = TrieBuilder::new();

        // Insert words in lowercase
        builder.insert("apple");
        builder.insert("application");
        builder.insert("apply");
        builder.insert("ApplePie");
        builder.insert("BANANA");
        builder.insert("BaNaNa-Split");
        builder.insert("cherry");
        builder.insert("Cherry-Pie");

        let (nodes, labels) = builder.build();
        let trie = CompactRadixTrie::new(&nodes, &labels);

        // Test 1: Lowercase prefix matching mixed case words
        let suggestions = trie.suggest("app", 10);
        assert_eq!(
            suggestions.len(),
            4,
            "Should find all words starting with 'app' regardless of case"
        );
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
        assert!(suggestions.contains(&"apply".to_string()));
        assert!(suggestions.contains(&"ApplePie".to_string()));

        // Test 2: Uppercase prefix matching lowercase words
        let suggestions = trie.suggest("APP", 10);
        assert_eq!(
            suggestions.len(),
            4,
            "Uppercase prefix should match lowercase words"
        );
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
        assert!(suggestions.contains(&"apply".to_string()));
        assert!(suggestions.contains(&"ApplePie".to_string()));

        // Test 3: Mixed case prefix
        let suggestions = trie.suggest("ApP", 10);
        assert_eq!(
            suggestions.len(),
            4,
            "Mixed case prefix should work"
        );

        // Test 4: Lowercase prefix matching uppercase stored word
        let suggestions = trie.suggest("ban", 10);
        assert_eq!(
            suggestions.len(),
            2,
            "Should find words stored in uppercase"
        );
        assert!(suggestions.contains(&"BANANA".to_string()));
        assert!(suggestions.contains(&"BaNaNa-Split".to_string()));

        // Test 5: Uppercase prefix matching mixed case
        let suggestions = trie.suggest("BANAN", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"BANANA".to_string()));
        assert!(suggestions.contains(&"BaNaNa-Split".to_string()));

        // Test 6: Case insensitive matching with special characters
        let suggestions = trie.suggest("cherry", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"cherry".to_string()));
        assert!(suggestions.contains(&"Cherry-Pie".to_string()));

        // Test 7: Uppercase variant
        let suggestions = trie.suggest("CHERRY", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"cherry".to_string()));
        assert!(suggestions.contains(&"Cherry-Pie".to_string()));

        // Test 8: Partial match with case variation
        let suggestions = trie.suggest("aPpLe", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"ApplePie".to_string()));
    }
}

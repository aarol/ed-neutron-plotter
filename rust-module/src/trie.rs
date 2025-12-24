use std::collections::{HashMap, VecDeque};
use std::mem::size_of;

use succinct::{
    BinSearchSelect, BitRankSupport, BitVec, BitVecPush, BitVector, JacobsonRank, Select1Support,
    select::Select0Support,
};

/// A LOUDS (Level-Order Unary Degree Sequence) Trie implementation with radix compression
/// The child-parent hierarchy is encoded in a bit-vector for minimal space usage.
///
/// The rank/select structures are computed at load-time and are not stored on disk.
///
/// The labels on the edges are stored separately, with 1-byte labels in a simple array
/// and longer labels stored as (u32) offsets/lengths into a label store.
///
/// Currently, the largest space bottleneck is the
pub struct LoudsTrie {
    bits: succinct::BitVector<u64>,
    bits_select: succinct::select::BinSearchSelect<JacobsonRank<BitVector<u64>>>,
    terminals: succinct::BitVector<u64>,
    terminals_rank: JacobsonRank<BitVector<u64>>,

    label_type: succinct::BitVector<u64>,
    label_type_rank: JacobsonRank<BitVector<u64>>,

    simple_labels: Vec<u8>,
    complex_labels: Vec<u32>,
    label_store: Vec<u8>,
}

impl LoudsTrie {
    /// Builds a LOUDS trie from a sorted list of strings.
    /// Returns the trie and a mapping from input indices to trie IDs.
    pub fn new(keys: &[&str]) -> (Self, Vec<usize>) {
        assert!(!keys.is_empty());
        assert!(
            keys.windows(2).all(|w| w[0] <= w[1]),
            "Keys must be sorted lexicographically"
        );

        // 1. Build standard Trie in memory first
        #[derive(Default)]
        struct TempNode {
            id: Option<usize>,
            children: Vec<(u8, TempNode)>, // Ordered children
            is_terminal: bool,
        }

        let mut root = TempNode::default();
        for (id, key) in keys.iter().enumerate() {
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
            node.id = Some(id);
            node.is_terminal = true;
        }

        // 2. Compress into Radix Trie
        struct CompressedNode {
            id: Option<usize>,
            children: Vec<(Vec<u8>, CompressedNode)>,
            is_terminal: bool,
        }

        fn compress(node: TempNode) -> CompressedNode {
            let mut new_children = Vec::new();
            for (byte, child) in node.children {
                let mut c_child = compress(child);
                let mut label = vec![byte];

                // Try to merge child into this edge
                while c_child.children.len() == 1 && !c_child.is_terminal {
                    let next_len = c_child.children[0].0.len();
                    if label.len() + next_len > 127 {
                        break;
                    }
                    let (next_label, next_node) = c_child.children.pop().unwrap();
                    label.extend(next_label);
                    c_child = next_node;
                }
                new_children.push((label, c_child));
            }
            CompressedNode {
                id: node.id,
                children: new_children,
                is_terminal: node.is_terminal,
            }
        }

        let root_compressed = compress(root);

        // 3. BFS to generate LOUDS bits and Labels
        let mut bits = BitVector::new();
        let mut terminals = BitVector::new();
        let mut temp_node_labels = Vec::new();
        let mut temp_label_store = Vec::new();
        let mut queue = VecDeque::new();

        let mut coords_indices = vec![0; keys.len()];

        // Push fake super-root to start the sequence "10" for the actual root
        bits.push_bit(true);
        bits.push_bit(false);

        queue.push_back(&root_compressed);
        terminals.push_bit(root_compressed.is_terminal);
        let mut curr_terminal_index = 0;

        while let Some(node) = queue.pop_front() {
            for (label, child) in &node.children {
                bits.push_bit(true); // '1' for every child

                let len = label.len();
                let offset = temp_label_store.len();
                temp_label_store.extend_from_slice(label);

                // Pack (offset, len) into u32
                // offset: 25 bits (32MB max), len: 7 bits (127 max)
                assert!(len <= 127, "Label too long");
                assert!(offset < (1 << 25), "Label store too large");
                let packed = ((offset as u32) << 7) | (len as u32);
                temp_node_labels.push(packed);

                terminals.push_bit(child.is_terminal);
                if let Some(id) = child.id {
                    coords_indices[id] = curr_terminal_index;
                }
                if child.is_terminal {
                    curr_terminal_index += 1;
                }

                queue.push_back(child);
            }
            bits.push_bit(false); // '0' to end this node's list of children
        }

        // 4. Split and Compress Labels
        let mut simple_labels = Vec::new();
        let mut complex_labels = Vec::new();
        let mut complex_labels_vec = Vec::new();
        let mut label_type = BitVector::new();

        for packed in temp_node_labels {
            let len = (packed & 0x7F) as usize;
            let offset = (packed >> 7) as usize;
            let s = &temp_label_store[offset..offset + len];

            if len == 1 {
                label_type.push_bit(false);
                simple_labels.push(s[0]);
            } else {
                label_type.push_bit(true);
                complex_labels_vec.push(s.to_vec());
            }
        }

        let (new_store, new_mappings) = compress_labels(&complex_labels_vec);

        for (offset, len) in new_mappings {
            assert!(len < 128, "Label too long");
            assert!(offset < (1 << 25), "Label store too large");
            complex_labels.push((offset << 7) | len);
        }

        let rank = JacobsonRank::new(bits.clone());
        let select = BinSearchSelect::new(rank);
        let label_type_rank = JacobsonRank::new(label_type.clone());
        let terminals_rank = JacobsonRank::new(terminals.clone());

        (
            Self {
                bits,
                terminals,
                bits_select: select,
                label_type,
                label_type_rank,
                simple_labels,
                complex_labels,
                label_store: new_store,
                terminals_rank,
            },
            coords_indices,
        )
    }

    // Returns the bit-index of the first child of the node at `index`.
    // Formula: select0(rank1(index) - 1) + 1
    fn first_child(&self, index: u64) -> Option<u64> {
        if index >= self.bits.bit_len() || !self.bits.get_bit(index) {
            return None;
        }
        let r1 = self.bits_select.rank1(index);
        if r1 == 0 {
            return None;
        }
        let s0 = self.bits_select.select0(r1 - 1)?;

        // Check if the child position is valid (should be a '1' bit representing a node)
        let child_pos = s0 + 1;
        if child_pos >= self.bits.bit_len() || !self.bits.get_bit(child_pos) {
            return None; // No children
        }

        Some(child_pos)
    }

    // Returns the bit-index of the parent of the node at `index`.
    // Formula: select1(rank0(index) - 1)
    fn parent(&self, index: u64) -> Option<u64> {
        if index <= 1 {
            return None;
        } // Super-root (0) and root edge (1) have no parent

        let r0 = self.bits_select.rank0(index);
        if r0 == 0 {
            return None;
        } // No '0' bits before this position

        // select1 is 0-indexed in succinct, so we need r0 - 1 for parent
        let s1 = self.bits_select.select1(r0 - 1)?;

        Some(s1)
    }

    // Returns the Label (slice of bytes) leading to this node.
    fn get_label(&self, index: u64) -> &[u8] {
        let r1 = self.bits_select.rank1(index);
        if r1 < 2 {
            return &[];
        } // Root (Node 1) has no label

        // Node ID (rank1) is 1-based.
        // We map to 0-based index for label arrays: idx = r1 - 2.
        let idx = r1 - 2;

        if self.label_type.get_bit(idx) {
            // Complex label
            let rank = self.label_type_rank.rank1(idx);
            // rank is 1-based count of 1s up to idx.
            // Vector is 0-based, so use rank - 1.
            if rank == 0 {
                return &[];
            } // Should not happen if bit is 1
            let packed = self.complex_labels[(rank - 1) as usize];
            let len = (packed & 0x7F) as usize;
            let offset = (packed >> 7) as usize;
            &self.label_store[offset..offset + len]
        } else {
            // Simple label
            // rank0 gives count of 0s.
            // rank0(idx) gives count of 0s in [0..idx] (inclusive? succinct docs say inclusive)
            // Let's verify rank0 behavior. Usually rank0(i) is count of 0s in 0..=i.
            // So if idx is the k-th 0, rank0(idx) = k.
            // We want the index in simple_labels.
            let rank0 = idx + 1 - self.label_type_rank.rank1(idx);
            // rank0 is 1-based count.
            if rank0 == 0 {
                return &[];
            }
            let simple_idx = (rank0 - 1) as usize;
            &self.simple_labels[simple_idx..simple_idx + 1]
        }
    }

    /// 3. BOTTOM-UP TRAVERSAL
    /// Given a Node ID (bit index), reconstruct the full string.
    pub fn reconstruct_key(&self, mut node_index: u64) -> String {
        let mut result = Vec::new();

        // Walk up until we hit the root (index 0)
        while let Some(p_index) = self.parent(node_index) {
            let label = self.get_label(node_index);
            // We are walking up, so we prepend the label
            // But extending and reversing later is more efficient for Vec
            result.extend(label.iter().rev());

            node_index = p_index;
            if node_index == 0 {
                break;
            }
        }

        result.reverse();
        String::from_utf8(result).unwrap_or_default()
    }

    /// Checks if the exact key exists in the Trie. Returns the ID of the terminal node if found.
    pub fn find(&self, key: &str) -> Option<u64> {
        // Start at the root node (position 0 in LOUDS represents super-root → root edge)
        let mut curr_node_idx = 0;
        let mut key_bytes = key.as_bytes();

        while !key_bytes.is_empty() {
            // Get the starting bit-index where the children of `curr_node_idx` are stored.
            let mut scan_idx = self.first_child(curr_node_idx)?; // No children → key not found

            let mut found = false;

            // Iterate through the children (consecutive '1's in bits)
            while scan_idx < self.bits.bit_len() && self.bits.get_bit(scan_idx) {
                let label = self.get_label(scan_idx);

                if key_bytes.starts_with(label) {
                    curr_node_idx = scan_idx;
                    key_bytes = &key_bytes[label.len()..];
                    found = true;
                    break;
                }
                scan_idx += 1; // Move to next sibling
            }

            if !found {
                return None;
            }
        }

        // Final Check: Is this node a valid end-of-word?
        // If so, return its terminal rank1 (uniquely identifies the key)
        self.is_terminal(curr_node_idx).then(|| {
            let node_id = self.bits_select.rank1(curr_node_idx);
            self.terminals_rank.rank1(node_id - 1) - 1 // -1 to make it 0-based
        })
    }

    /// Helper: Checks if a node_idx is marked as terminal.
    fn is_terminal(&self, node_idx: u64) -> bool {
        // Map bit-index to Node ID (rank1).
        // Node ID 0 is usually super-root, Node ID 1 is Root.
        let node_id = self.bits_select.rank1(node_idx);

        // Check bounds and look up in the terminals bit-vector
        if node_id > 0 && node_id <= self.terminals.bit_len() {
            // -1 because rank is 1-based count, vector is 0-based
            self.terminals.get_bit(node_id - 1)
        } else {
            false
        }
    }

    #[allow(dead_code)]
    fn children(&self, node_idx: u64) -> Vec<String> {
        let mut result = Vec::new();
        if let Some(mut child_idx) = self.first_child(node_idx) {
            while child_idx < self.bits.bit_len() && self.bits.get_bit(child_idx) {
                let label = self.get_label(child_idx);
                result.push(String::from_utf8_lossy(label).to_string());
                child_idx += 1;
            }
        }
        result
    }

    /// Returns autocomplete suggestions for a given prefix (case-insensitive).
    /// Returns up to `limit` suggestions that start with the given prefix.
    pub fn suggest(&self, prefix: &str, limit: usize) -> Vec<String> {
        if limit == 0 {
            return Vec::new();
        }

        // Find all nodes matching the prefix (case-insensitive)
        let start_nodes = self.find_all_prefix_nodes_case_insensitive(prefix);

        if start_nodes.is_empty() {
            return Vec::new();
        }

        // Collect all terminal nodes under these subtrees using BFS
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = std::collections::HashSet::new();

        for &node in &start_nodes {
            if visited.insert(node) {
                queue.push_back(node);
            }
        }

        while let Some(node_idx) = queue.pop_front() {
            if results.len() >= limit {
                break;
            }

            // Check if this node is terminal
            if self.is_terminal(node_idx) {
                results.push(self.reconstruct_key(node_idx));
            }

            // Add all children to the queue
            if let Some(mut child_idx) = self.first_child(node_idx) {
                while child_idx < self.bits.bit_len() && self.bits.get_bit(child_idx) {
                    if visited.insert(child_idx) {
                        queue.push_back(child_idx);
                    }
                    child_idx += 1;
                }
            }
        }

        results
    }

    /// Finds all nodes matching the given prefix (case-insensitive).
    /// Returns the bit-indices of all matching nodes.
    fn find_all_prefix_nodes_case_insensitive(&self, prefix: &str) -> Vec<u64> {
        let prefix_bytes = prefix.as_bytes();
        let mut results = Vec::new();
        let mut queue = VecDeque::new();

        // Queue stores (node_idx, remaining_prefix_slice)
        queue.push_back((0u64, prefix_bytes));

        while let Some((node_idx, curr_prefix)) = queue.pop_front() {
            if curr_prefix.is_empty() {
                // We matched the full prefix. This node is a valid result root.
                results.push(node_idx);
                continue;
            }

            if let Some(mut child_idx) = self.first_child(node_idx) {
                while child_idx < self.bits.bit_len() && self.bits.get_bit(child_idx) {
                    let label = self.get_label(child_idx);

                    let common_len = common_prefix_len_case_insensitive(label, curr_prefix);

                    if common_len == curr_prefix.len() {
                        // Case 1: Prefix is fully matched by (a prefix of) the label.
                        // e.g. Label="apple", Prefix="app". Common=3.
                        // This node matches.
                        results.push(child_idx);
                    } else if common_len == label.len() {
                        // Case 2: Label is fully matched by (a prefix of) the prefix.
                        // e.g. Label="ap", Prefix="apple". Common=2.
                        // Continue down.
                        queue.push_back((child_idx, &curr_prefix[common_len..]));
                    }

                    child_idx += 1;
                }
            }
        }

        results
    }

    pub fn node_count(&self) -> u64 {
        self.bits_select.rank1(self.bits.bit_len() - 1)
    }

    pub fn analyze_structure(&self) {
        println!("LoudsTrie Analysis:");
        println!("  - Total nodes: {}", self.node_count());
        println!("  - Total bits: {:<.2} MB", bits_to_mb(self.bits.bit_len()));
        println!(
            "  - Total terminals: {:<.2} MB",
            bits_to_mb(self.terminals.bit_len())
        );
        println!(
            "  - Simple Labels: {:<.2} MB",
            self.simple_labels.len() as f64 / 1024.0 / 1024.0
        );
        println!(
            "  - Complex Labels: {:<.2} MB",
            self.complex_labels.len() as f64 * 4.0 / 1024.0 / 1024.0
        );
        println!(
            "  - Label store (u8): {:<.2} MB",
            bits_to_mb(self.label_store.len() as u64 * 8)
        );
    }

    pub fn size_on_disk(&self) -> usize {
        let bits_bytes = self.bits.block_len() * size_of::<usize>();
        let terminals_bytes = self.terminals.block_len() * size_of::<usize>();
        let simple_bytes = self.simple_labels.len();
        let complex_bytes = self.complex_labels.len() * size_of::<u32>();
        let label_store_bytes = self.label_store.len();
        let label_type_bytes = self.label_type.block_len() * size_of::<usize>();

        bits_bytes
            + terminals_bytes
            + simple_bytes
            + complex_bytes
            + label_store_bytes
            + label_type_bytes
    }
}

fn bits_to_mb(bits: u64) -> f64 {
    bits as f64 / 8.0 / 1024.0 / 1024.0
}

impl Into<Vec<u8>> for LoudsTrie {
    fn into(self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Serialize bits
        let mut bits_u64s = vec![];
        for i in 0..self.bits.block_len() {
            let block = self.bits.get_block(i);
            bits_u64s.extend_from_slice(&block.to_le_bytes());
        }
        let bits_len = bits_u64s.len() as u32;
        bytes.extend_from_slice(&bits_len.to_le_bytes());
        bytes.extend_from_slice(&bits_u64s);

        // Serialize terminals
        let mut terminals_bytes = vec![];
        for i in 0..self.terminals.block_len() {
            let block = self.terminals.get_block(i);
            terminals_bytes.extend_from_slice(&block.to_le_bytes());
        }
        let terminals_len = terminals_bytes.len() as u32;
        bytes.extend_from_slice(&terminals_len.to_le_bytes());
        bytes.extend_from_slice(&terminals_bytes);

        // Serialize label_type
        let mut lt_bytes = vec![];
        for i in 0..self.label_type.block_len() {
            let block = self.label_type.get_block(i);
            lt_bytes.extend_from_slice(&block.to_le_bytes());
        }
        let lt_len = lt_bytes.len() as u32;
        bytes.extend_from_slice(&lt_len.to_le_bytes());
        bytes.extend_from_slice(&lt_bytes);

        // Serialize simple_labels
        let simple_len = self.simple_labels.len() as u32;
        bytes.extend_from_slice(&simple_len.to_le_bytes());
        bytes.extend_from_slice(&self.simple_labels);

        // Serialize complex_labels (u32)
        let complex_len = self.complex_labels.len() as u32;
        bytes.extend_from_slice(&complex_len.to_le_bytes());
        for &val in &self.complex_labels {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Serialize label_store (u8)
        let label_store_len = self.label_store.len() as u32;
        bytes.extend_from_slice(&label_store_len.to_le_bytes());
        bytes.extend_from_slice(&self.label_store);

        bytes
    }
}

impl From<&[u8]> for LoudsTrie {
    fn from(value: &[u8]) -> Self {
        let mut offset = 0;

        // Deserialize bits
        let bits_len = u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut bits_u64s = Vec::with_capacity(bits_len / 8);
        for _ in 0..(bits_len / 8) {
            let block = u64::from_le_bytes(value[offset..offset + 8].try_into().unwrap());
            bits_u64s.push(block);
            offset += 8;
        }
        let mut bits = BitVector::block_with_capacity(bits_len);
        for &block in bits_u64s.iter() {
            bits.push_block(block);
        }

        // Deserialize terminals
        let terminals_len =
            u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut terminals_u64s = Vec::with_capacity(terminals_len / 8);
        for _ in 0..(terminals_len / 8) {
            let block = u64::from_le_bytes(value[offset..offset + 8].try_into().unwrap());
            terminals_u64s.push(block);
            offset += 8;
        }
        let mut terminals = BitVector::block_with_capacity(terminals_len);
        for &block in terminals_u64s.iter() {
            terminals.push_block(block);
        }

        // Deserialize label_type
        let lt_len = u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut lt_u64s = Vec::with_capacity(lt_len / 8);
        for _ in 0..(lt_len / 8) {
            let block = u64::from_le_bytes(value[offset..offset + 8].try_into().unwrap());
            lt_u64s.push(block);
            offset += 8;
        }
        let mut label_type = BitVector::block_with_capacity(lt_len);
        for &block in lt_u64s.iter() {
            label_type.push_block(block);
        }

        // Deserialize simple_labels
        let simple_len = u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let simple_labels = value[offset..offset + simple_len].to_vec();
        offset += simple_len;

        // Deserialize complex_labels
        let complex_len =
            u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut complex_labels = Vec::with_capacity(complex_len);
        for _ in 0..complex_len {
            let val = u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap());
            complex_labels.push(val);
            offset += 4;
        }

        // Deserialize label_store
        let label_store_len =
            u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let label_store = value[offset..offset + label_store_len].to_vec();

        let rank = JacobsonRank::new(bits.clone());
        let select = BinSearchSelect::new(rank);
        let label_type_rank = JacobsonRank::new(label_type.clone());
        let terminals_rank = JacobsonRank::new(terminals.clone());

        Self {
            bits,
            terminals,
            bits_select: select,
            label_type,
            label_type_rank,
            simple_labels,
            complex_labels,
            label_store,
            terminals_rank,
        }
    }
}

impl LoudsTrie {
    pub fn compress_store(&mut self) {
        // We only compress the complex labels.
        // Simple labels are already minimal (1 byte).

        // 1. Extract complex labels
        let mut complex_labels_vec: Vec<Vec<u8>> = Vec::new();
        let mut complex_indices: Vec<usize> = Vec::new();

        for i in 0..self.complex_labels.len() {
            let packed = self.complex_labels[i];
            let len = (packed & 0x7F) as usize;
            let offset = (packed >> 7) as usize;
            let s = self.label_store[offset..offset + len].to_vec();
            complex_labels_vec.push(s);
            complex_indices.push(i);
        }

        if complex_labels_vec.is_empty() {
            return;
        }

        // 2. Compress
        let (new_store, new_mappings) = compress_labels(&complex_labels_vec);

        // 3. Update store
        self.label_store = new_store;

        // 4. Update pointers
        for (i, &(offset, len)) in new_mappings.iter().enumerate() {
            let original_idx = complex_indices[i];
            assert!(offset < (1 << 25), "Label store too large");
            assert!(len < 128, "Label too long");
            self.complex_labels[original_idx] = (offset << 7) | len;
        }
    }
}

#[allow(dead_code)]
fn common_prefix_len_case_insensitive(s1: &[u8], s2: &[u8]) -> usize {
    s1.iter()
        .zip(s2)
        .take_while(|(a, b)| a.eq_ignore_ascii_case(b))
        .count()
}

fn compress_labels(labels: &[Vec<u8>]) -> (Vec<u8>, Vec<(u32, u32)>) {
    #[derive(Debug, Clone, Copy)]
    enum Action {
        None,
        Append(usize),
        Prepend(usize),
    }

    fn calc_overlap(a: &[u8], b: &[u8]) -> usize {
        let max_ov = a.len().min(b.len());
        for len in (1..=max_ov).rev() {
            if a[a.len() - len..] == b[..len] {
                return len;
            }
        }
        0
    }
    // 1. Extract and Deduplicate
    let mut string_to_id: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut unique_strings: Vec<Vec<u8>> = Vec::new();
    let mut label_to_unique_id: Vec<usize> = Vec::with_capacity(labels.len());

    for s in labels {
        if let Some(&id) = string_to_id.get(s) {
            label_to_unique_id.push(id);
        } else {
            let id = unique_strings.len();
            string_to_id.insert(s.clone(), id);
            unique_strings.push(s.clone());
            label_to_unique_id.push(id);
        }
    }

    let num_uniques = unique_strings.len();
    println!(
        "Compressing {} labels into {} unique strings",
        labels.len(),
        num_uniques
    );
    if num_uniques == 0 {
        return (Vec::new(), vec![(0, 0); labels.len()]);
    }

    // 2. Substring Compression (Parent/Child)
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
    let max_len = unique_strings[sorted_by_len[num_uniques - 1]].len();
    let mut pow_p = vec![1u64; max_len + 1];
    for i in 1..=max_len {
        pow_p[i] = pow_p[i - 1].wrapping_mul(P);
    }

    let mut substring_hashes: HashMap<u64, (usize, u32)> = HashMap::with_capacity(num_uniques);
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

        // If no targets longer than this length, we can't be a substring of anything longer
        if targets_start_idx < num_uniques {
            let target_indices = &sorted_by_len[targets_start_idx..];
            substring_hashes.clear();

            let lead_power = pow_p[len - 1];

            // Hash targets
            for &target_id in target_indices {
                let target_s = &unique_strings[target_id];
                let mut current_hash: u64 = 0;

                // Initial window
                for k in 0..len {
                    current_hash = current_hash
                        .wrapping_mul(P)
                        .wrapping_add(target_s[k] as u64);
                }
                substring_hashes
                    .entry(current_hash)
                    .or_insert((target_id, 0));

                // Rolling window
                for i in 1..=(target_s.len() - len) {
                    let prev = target_s[i - 1] as u64;
                    let new = target_s[i + len - 1] as u64;
                    current_hash = current_hash.wrapping_sub(prev.wrapping_mul(lead_power));
                    current_hash = current_hash.wrapping_mul(P).wrapping_add(new);
                    substring_hashes
                        .entry(current_hash)
                        .or_insert((target_id, i as u32));
                }
            }

            // Match candidates
            for &short_id in candidates {
                let short_bytes = &unique_strings[short_id];
                let mut h: u64 = 0;
                for &b in short_bytes {
                    h = h.wrapping_mul(P).wrapping_add(b as u64);
                }

                if let Some(&(target_id, offset)) = substring_hashes.get(&h) {
                    // Verify to avoid collisions
                    let target_bytes = &unique_strings[target_id];
                    if short_bytes.as_slice()
                        == &target_bytes[offset as usize..(offset as usize + len)]
                    {
                        redirects[short_id] = (target_id, offset);
                        is_active[short_id] = false;
                    }
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

    // 3. Greedy Superstring Merge
    let mut by_start_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];
    let mut by_end_byte: Vec<Vec<usize>> = vec![Vec::new(); 256];

    let mut root_is_available = vec![false; num_uniques];
    let mut root_final_offsets: HashMap<usize, u32> = HashMap::with_capacity(active_roots.len());

    let mut remaining_count = 0;

    for &root_id in &active_roots {
        let s = &unique_strings[root_id];
        if s.is_empty() {
            root_final_offsets.insert(root_id, 0);
            continue;
        }

        by_start_byte[s[0] as usize].push(root_id);
        by_end_byte[s[s.len() - 1] as usize].push(root_id);

        root_is_available[root_id] = true;
        remaining_count += 1;
    }

    let mut super_buffer = Vec::new();

    while remaining_count > 0 {
        let mut best_seed = None;
        while let Some(candidate) = active_roots.pop() {
            if root_is_available[candidate] {
                best_seed = Some(candidate);
                break;
            }
        }

        if best_seed.is_none() {
            break;
        }
        let seed_id = best_seed.unwrap();

        root_is_available[seed_id] = false;
        remaining_count -= 1;

        let mut chain: VecDeque<(usize, u32)> = VecDeque::new();
        chain.push_back((seed_id, 0));

        let mut left_edge_id = seed_id;
        let mut right_edge_id = seed_id;

        loop {
            let mut best_action = Action::None;
            let mut max_savings = 0;

            // Try Append
            let r_str = &unique_strings[right_edge_id];
            if !r_str.is_empty() {
                let last_char = r_str[r_str.len() - 1] as usize;
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
                let first_char = l_str[0] as usize;
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

        if chain.is_empty() {
            continue;
        }

        let current_write_pos = super_buffer.len() as u32;
        let first_id = chain[0].0;
        root_final_offsets.insert(first_id, current_write_pos);
        super_buffer.extend_from_slice(&unique_strings[first_id]);

        let mut prev_id = first_id;
        for i in 1..chain.len() {
            let next_id = chain[i].0;
            let prev_s = &unique_strings[prev_id];
            let next_s = &unique_strings[next_id];
            let ov = calc_overlap(prev_s, next_s);

            let next_bytes = next_s.as_slice();
            if ov < next_bytes.len() {
                let to_write = &next_bytes[ov..];
                let start_pos = super_buffer.len() as u32 - ov as u32;
                root_final_offsets.insert(next_id, start_pos);
                super_buffer.extend_from_slice(to_write);
            } else {
                let start_pos = super_buffer.len() as u32 - ov as u32;
                root_final_offsets.insert(next_id, start_pos);
            }
            prev_id = next_id;
        }
    }

    // 4. Finalize Pointers
    let mut result_mappings = Vec::with_capacity(labels.len());

    let mut count_under_16bit = 0;
    for &unique_id in &label_to_unique_id {
        let (root_id, offset_in_root) = step2_resolution[unique_id];
        let root_base = *root_final_offsets.get(&root_id).unwrap_or(&0);
        let final_offset = root_base + offset_in_root;
        let len = unique_strings[unique_id].len();
        if final_offset + (len as u32) < 65536 {
            count_under_16bit += 1;
        }

        result_mappings.push((final_offset, len as u32));
    }

    println!(
        "{} / {} labels fit within 16-bit offsets after compression",
        count_under_16bit,
        result_mappings.len()
    );

    dbg!(super_buffer.len());

    (super_buffer, result_mappings)
}

#[cfg(test)]
mod tests {
    use super::LoudsTrie;

    #[test]
    fn test_louds_trie() {
        let mut words: Vec<&str> = vec!["A", "B"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        assert!(trie.find("A").is_some());
        assert!(trie.find("B").is_some());
        assert!(trie.find("C").is_none());
        assert!(trie.find("AB").is_none());
    }

    #[test]
    fn test_suggest_basic() {
        let mut words: Vec<&str> = vec!["apple", "application", "apply", "banana", "band"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        let suggestions = trie.suggest("app", 10);
        assert_eq!(suggestions.len(), 3);
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
        assert!(suggestions.contains(&"apply".to_string()));

        let suggestions = trie.suggest("ban", 10);
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.contains(&"banana".to_string()));
        assert!(suggestions.contains(&"band".to_string()));
    }

    #[test]
    fn test_suggest_case_insensitive() {
        let mut words: Vec<&str> = vec!["Apple", "APPLICATION", "Apply", "Banana"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        // Search with lowercase should find uppercase entries
        let suggestions = trie.suggest("app", 10);
        assert_eq!(suggestions.len(), 3);
        assert!(suggestions.contains(&"Apple".to_string()));
        assert!(suggestions.contains(&"APPLICATION".to_string()));
        assert!(suggestions.contains(&"Apply".to_string()));

        // Search with uppercase should also work
        let suggestions = trie.suggest("APP", 10);
        assert_eq!(suggestions.len(), 3);

        // Mixed case search
        let suggestions = trie.suggest("ApP", 10);
        assert_eq!(suggestions.len(), 3);
    }

    #[test]
    fn test_suggest_limit() {
        let mut words: Vec<&str> = vec!["a1", "a2", "a3", "a4", "a5"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        let suggestions = trie.suggest("a", 3);
        assert_eq!(suggestions.len(), 3);

        let suggestions = trie.suggest("a", 10);
        assert_eq!(suggestions.len(), 5);
    }

    #[test]
    fn test_suggest_no_match() {
        let mut words: Vec<&str> = vec!["apple", "banana"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        let suggestions = trie.suggest("xyz", 10);
        assert!(suggestions.is_empty());

        let suggestions = trie.suggest("c", 10);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_empty_prefix() {
        let mut words: Vec<&str> = vec!["apple", "banana"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        // Empty prefix should return all words (up to limit)
        // Note: Empty prefix starts from root which might include root itself
        let suggestions = trie.suggest("a", 10);
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions.contains(&"apple".to_string()));

        let suggestions = trie.suggest("b", 10);
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions.contains(&"banana".to_string()));
    }

    #[test]
    fn test_suggest_exact_match() {
        let mut words: Vec<&str> = vec!["app", "apple", "application"];
        words.sort();
        let (trie, _) = LoudsTrie::new(&words);

        // "app" is both a valid word and a prefix
        let suggestions = trie.suggest("app", 10);
        assert_eq!(suggestions.len(), 3);
        assert!(suggestions.contains(&"app".to_string()));
        assert!(suggestions.contains(&"apple".to_string()));
        assert!(suggestions.contains(&"application".to_string()));
    }
}

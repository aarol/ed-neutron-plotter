use std::collections::{HashMap, VecDeque};

use bitvec::vec;

struct Trie {
    root: TrieNode,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u8, TrieNode>,
    is_terminal: bool,
}

impl Trie {
  fn add(&mut self, word: &str) {
      let mut current_node = &mut self.root;
      for c in word.bytes() {
            current_node = current_node.children.entry(c).or_default();
      }
      current_node.is_terminal = true;
  }

  fn suggest(&self, prefix: &str) -> Vec<String> {
      let mut current_node = &self.root;
      // Find the node corresponding to the end of the prefix
      for c in prefix.bytes() {
          match current_node.children.get(&c) {
              Some(node) => current_node = node,
              None => return vec![],
          }
      }
      let mut suggestions = Vec::new();
      self.collect_words(current_node, prefix.to_string(), &mut suggestions);
      suggestions
  }

  fn collect_words(&self, node: &TrieNode, prefix: String, suggestions: &mut Vec<String>) {
      if node.is_terminal {
          suggestions.push(prefix.clone());
      }
      for (&c, child_node) in &node.children {
          let mut new_prefix = prefix.clone();
          new_prefix.push(c as char);
          self.collect_words(child_node, new_prefix, suggestions);
      }
  }

  fn build(&mut self) {

      let mut queue = VecDeque::new();
      queue.push_back(&self.root);

      while let Some(current_node) = queue.pop_front() {
          for (char, child_node) in current_node.children.iter() {
              // 

              queue.push_back(child_node);
          }
      }

      
  }
}

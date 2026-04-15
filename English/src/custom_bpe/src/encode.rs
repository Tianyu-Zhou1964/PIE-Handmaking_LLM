// encode.rs
// Inference side: highly optimized Byte-Level BPE Tokenizer
// Core optimizations:
// 1. FxHashMap: uses an ultra-fast non-cryptographic hash map for O(1) lookup with small integer keys.
// 2. Aho-Corasick automaton: O(N) linear-time scan for all special tokens, avoiding complex regex backtracking.
// 3. GIL release: uses safe String ownership transfer to unlock Python's global interpreter lock at runtime, avoiding main-thread blocking.
// 4. Doubly-linked list merging: for long token sequences, uses an array-simulated linked list to eliminate the O(N^2) memory-move cliff from Vec::remove.
// 5. Zero-allocation buffer: pre-allocates memory outside the loop and only calls clear() on the hot path, achieving zero allocations.

use pyo3::prelude::*;
use rustc_hash::FxHashMap; // Replaces the standard hash map; 30%+ speed improvement
use regex::Regex;
use std::collections::HashMap as StdHashMap;
use aho_corasick::{AhoCorasick, MatchKind};


/// Doubly-linked list node for avoiding O(N) memory moves in long text chunks
#[derive(Clone, Copy)]
struct ListNode {
    id: u32,
    prev: usize,
    next: usize,
}

#[pyclass]
pub struct Tokenizer {
    regex: Regex,
    // FxHashMap replaces the default hash; computing the default hash for (u32, u32) is extremely slow
    merges: FxHashMap<(u32, u32), (usize, u32)>,
    // Aho-Corasick automaton for ultra-fast matching of all special tokens like <|user|>
    special_matcher: Option<AhoCorasick>,
    // Used with the AC automaton: directly maps a match ID to token ID in O(1)
    special_token_ids: Vec<u32>,
}

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(
        merges_list: Vec<((u32, u32), u32)>, 
        special_tokens_dict: StdHashMap<String, u32>
    ) -> PyResult<Self> {
        
        // 1. Rebuild the pre-tokenization regex (GPT-2 / LLaMA standard)
        let pat_str = [
            r"'(?:s|t|ll|ve|re|m|d)",
            r" ?\p{L}+",
            r" ?\p{N}+",
            r" ?[^\s\p{L}\p{N}]+",
            r"\s+",
        ].join("|");
        
        let regex = Regex::new(&pat_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // 2. Build the ultra-fast FxHashMap merge rule table
        let mut merges = FxHashMap::default();
        for (rank, ((p0, p1), new_id)) in merges_list.into_iter().enumerate() {
            merges.insert((p0, p1), (rank, new_id));
        }

        // 3. Build the Aho-Corasick special token matcher
        let mut patterns = Vec::with_capacity(special_tokens_dict.len());
        let mut special_token_ids = Vec::with_capacity(special_tokens_dict.len());

        for (pat, id) in special_tokens_dict {
            patterns.push(pat);
            special_token_ids.push(id);
        }

        let special_matcher = if !patterns.is_empty() {
            // Use builder to specify leftmost-longest match semantics
            let ac = AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Some(ac)
        } else {
            None
        };

        println!("Rust Tokenizer (fast mode) initialized! Rules: {}, Special tokens: {}", merges.len(), patterns.len());

        Ok(Tokenizer {
            regex,
            merges,
            special_matcher,
            special_token_ids,
        })
    }

    /// encode method called from Python
    /// Note: receives `String` rather than `&str`.
    /// Reason: to call `allow_threads` and release the GIL, data must be detached from Python's memory management.
    pub fn encode(&self, py: Python<'_>, text: String) -> Vec<u32> {
        // Release the GIL: while this closure runs, other Python threads are free to run without blocking!
        py.allow_threads(|| {
            self._encode_internal(&text)
        })
    }
}

// -------------------------------------------------------------------------
// Rust internal private implementation
// -------------------------------------------------------------------------
impl Tokenizer {
    fn _encode_internal(&self, text: &str) -> Vec<u32> {
        // Pre-estimate capacity to reduce reallocation overhead
        let mut final_tokens = Vec::with_capacity(text.len() / 4); 
        
        // [Core optimization]: lift buffers outside the loop to avoid thousands of heap allocations per call!
        let mut word_buffer: Vec<u32> = Vec::with_capacity(256);
        let mut list_buffer: Vec<ListNode> = Vec::with_capacity(256);

        let mut last_end = 0;

        if let Some(ref ac) = self.special_matcher {
            // Aho-Corasick finds all special tokens in O(N) linear time
            for mat in ac.find_iter(text) {
                let start = mat.start();
                let end = mat.end();

                // 1. Process the normal text before the special token
                if start > last_end {
                    self.encode_normal_text(
                        &text[last_end..start], 
                        &mut final_tokens, 
                        &mut word_buffer, 
                        &mut list_buffer
                    );
                }

                // 2. Directly get the token ID for this special token in O(1) and push to the final array
                let pattern_idx = mat.pattern().as_usize();
                final_tokens.push(self.special_token_ids[pattern_idx]);

                last_end = end;
            }
        }

        // Process any remaining normal text at the tail
        if last_end < text.len() {
            self.encode_normal_text(
                &text[last_end..], 
                &mut final_tokens, 
                &mut word_buffer,
                &mut list_buffer
            );
        }

        final_tokens
    }

    /// Processes a plain text chunk (no special tokens)
    fn encode_normal_text(
        &self, 
        text: &str, 
        final_tokens: &mut Vec<u32>, 
        word_buffer: &mut Vec<u32>,
        list_buffer: &mut Vec<ListNode>
    ) {
        for mat in self.regex.find_iter(text) {
            // clear() does not free memory; it just resets the length to 0 — zero overhead!
            word_buffer.clear(); 
            
            // [Core logic]: convert the UTF-8 text chunk to raw bytes.
            // Since the initial token ID for bytes 0-255 is exactly 0-255, cast directly to u32!
            // Perfect 1:1 mapping — completely eliminates the garbled-string lookup from the Python side.
            word_buffer.extend(mat.as_str().as_bytes().iter().map(|&b| b as u32));

            // Run the ultra-fast BPE merge
            self.bpe_merge(word_buffer, list_buffer);

            // Append the merged result to the final output
            final_tokens.extend_from_slice(word_buffer);
        }
    }

    /// Core algorithm: BPE merge with no O(N^2) memory-move cliff
    fn bpe_merge(&self, ids: &mut Vec<u32>, list_nodes: &mut Vec<ListNode>) {
        if ids.len() < 2 {
            return;
        }

        // For short tokens (e.g., common ASCII letter combos, 3~8 bytes),
        // contiguous memmove is actually faster than building a complex linked list — fall back to the simple approach
        if ids.len() < 50 {
            loop {
                let mut best_rank = usize::MAX;
                let mut best_idx = usize::MAX;
                let mut target_id = 0;

                for i in 0..ids.len() - 1 {
                    if let Some(&(rank, new_id)) = self.merges.get(&(ids[i], ids[i + 1])) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                            target_id = new_id;
                        }
                    }
                }

                if best_idx == usize::MAX { break; }

                // Merge and remove the right-hand old ID
                ids[best_idx] = target_id;
                ids.remove(best_idx + 1); 
            }
            return;
        }

        // =================================================================
        // [Overflow protection]: for extremely long chunks (e.g., thousands of consecutive spaces
        // or long URLs in malformed text), we must use an array-based doubly-linked list
        // to strictly bound time complexity.
        // =================================================================
        list_nodes.clear();
        for (i, &id) in ids.iter().enumerate() {
            list_nodes.push(ListNode {
                id,
                prev: if i == 0 { usize::MAX } else { i - 1 },
                next: if i == ids.len() - 1 { usize::MAX } else { i + 1 },
            });
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;
            let mut target_id = 0;

            let mut curr = 0; // 0 is the head; it only ever absorbs its right neighbor, so index 0 is never deleted
            while curr != usize::MAX {
                let next = list_nodes[curr].next;
                if next != usize::MAX {
                    let pair = (list_nodes[curr].id, list_nodes[next].id);
                    if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = curr;
                            target_id = new_id;
                        }
                    }
                }
                curr = next;
            }

            // No more mergeable pairs — exit
            if best_idx == usize::MAX { break; }

            // O(1) merge: update linked list pointers
            let right_idx = list_nodes[best_idx].next;
            let next_next = list_nodes[right_idx].next;

            // Left node absorbs the merged result
            list_nodes[best_idx].id = target_id;
            // Skip over the right node
            list_nodes[best_idx].next = next_next;
            
            if next_next != usize::MAX {
                list_nodes[next_next].prev = best_idx;
            }
        }

        // Drain the linked list back into ids in order
        ids.clear();
        let mut curr = 0;
        while curr != usize::MAX {
            ids.push(list_nodes[curr].id);
            curr = list_nodes[curr].next;
        }
    }
}
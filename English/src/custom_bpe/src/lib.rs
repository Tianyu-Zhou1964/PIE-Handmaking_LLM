// Preprocessing: receives text from Python, performs streaming regex tokenization and frequency counting,
// and compresses the large corpus into a high-density "unique word structure" (Word {ids, count}).
// Initialization: iterates over the Word list once, fills pair_freqs (global Pair frequency table),
// builds pair_to_words (inverted index), and pushes all Pairs into a BinaryHeap (max-heap).
// Core loop:
//   1. Heap Pop: pop the top-frequency Pair from the heap.
//   2. Lazy validation: compare the heap frequency against the real-time frequency in pair_freqs;
//      discard stale entries if they don't match.
//   3. Rule recording: once confirmed as the best Pair, record it in the Merges rule table.
//   4. Local Update: use the pair_to_words inverted index to precisely locate affected Words
//      and perform the merge-and-replace inside them.
//   5. Frequency rollback and injection: remove frequencies of old neighbor pairs (e.g., CA, BD),
//      add frequencies of new neighbor pairs (e.g., CC, CD), and push the new frequencies into the heap.
//   6. Termination: stop when the target vocab size is reached or no Pair with frequency > 1 remains.
//
// train_bpe() core logic flow
// │
// ├── 1. Regex pre-tokenization
// │   └── Uses GPT-2 standard regex to split raw long text into pieces and counts
// │       the frequency of each unique word.
// │       Result: word_counts { [72, 105]: 5000 times }  (i.e., "Hi" appears 5000 times in the corpus)
// │
// ├── 2. Build inverted index
// │   └── Core optimization: record which Word(s) contain each Pair.
// │       pair_to_words: { (72, 105) -> [index1, index5, ...] }
// │
// ├── 3. Initialize priority queue (BinaryHeap)
// │   └── Performance key: push all Pairs with their frequencies into a max-heap.
// │       The heap top is always the Pair with the highest frequency in the entire corpus.
// │
// ├── 4. BPE main loop (efficient merge phase) -> repeat until vocab is full:
// │   ├── a. Heap Pop: directly retrieve the highest-frequency Pair (A, B) from the top.
// │   │      (Validation: if the Pair's frequency is stale, skip it — lazy deletion)
// │   ├── b. Record rule: store (A, B) -> new ID (C) in the Merges table.
// │   ├── c. Local Update:
// │   │      ① Use pair_to_words index to locate only the affected Words — no full scan.
// │   │      ② Merge A+B into C; disconnect old neighbors (CA, BD); establish new neighbors (CC, CD).
// │   │      ③ Dynamically increment/decrement frequencies in pair_freqs; push affected new frequencies into the heap.
// │   └── d. Repeat.
// │
// └── 5. PyO3 export
//     └── Compiled as a Python module class for import

use pyo3::prelude::*; // Equivalent to `from pyo3 import prelude`; PyO3 enables writing Python modules in Rust
use hashbrown::HashMap; // High-performance HashMap
use regex::Regex; // Regex library
use pyo3::types::PyModule; // Ensure PyModule type is imported
use std::collections::HashMap as StdHashMap; // Python only accepts the standard library hash map as input and return type
use std::collections::BinaryHeap; // Use heap sort; greatly reduces time complexity — gets faster as it runs

mod encode;
mod decode;
mod engine;

struct Word { // Struct is roughly equivalent to a Python class with only attributes and no methods
    ids: Vec<u32>, // The token IDs this word currently consists of; Vec is a vector (contiguous memory) holding u32 values
    count: u64,    // The total frequency of this word across the entire corpus
}

/// Entry point called from Python: receives a list of raw texts and performs regex tokenization + BPE training entirely in Rust
#[pyfunction] // Marks this function as callable from Python; applies only to the immediately following fn
fn train_bpe(texts: Vec<String>, vocab_size: usize) -> PyResult<(StdHashMap<String, u32>, Vec<((u32, u32), u32)>)> {
    // This function takes two parameters: texts (a vector of strings) and vocab_size (an integer).
    // It returns a string hash map with u32 IDs, and a Vec<((u32, u32), u32)>
    // where (u32, u32) is a token pair and the last u32 is the merged ID.
    // Print a log to the Python console informing the user that pre-tokenization has started
    println!("Rust: received {} texts, starting regex pre-tokenization...", texts.len());
    // ! means the macro accepts a variable number of arguments; each {} corresponds to one argument

    // 1. Regex pre-tokenization
    // This regex is used for BPE pre-tokenization, based on GPT-2's standard tokenization logic.
    // let declares an immutable variable in Rust and binds a name to it              
    let pat_str = [
    // pat_str stands for pattern_string: the regex rule string
        r"'(?:s|t|ll|ve|re|m|d)",
        // 1. Match contraction suffixes such as 's, 't, 'll, 've, 're, 'm, 'd
        // r"..." means: treat everything inside the quotes as literal text — do not interpret escape characters!
        // The leading single quote ' triggers matching when encountered; (?:) treats what follows as a group.

        r" ?\p{L}+",
        // 2. Match letters: optional space + one or more Unicode letters (includes CJK characters, etc.)
        // \p{...} matches Unicode properties; e.g., \p{L} matches letters, \p{N} matches digits.
        // This line means: "an optional space, followed by one or more consecutive letters."

        r" ?\p{N}+",
        // 3. Match digits: optional space + one or more Unicode digits
        
        r" ?[^\s\p{L}\p{N}]+",
        // 4. Match punctuation and symbols: optional space + one or more non-letter/digit/whitespace characters

        r"\s+",
        // 5. Handle pure whitespace

    ].join("|"); // Join the array elements with the "or" operator

    // Compile the regex; return a Python exception if the regex format is invalid
    // 1. Try to turn the assembled long string pat_str into a real regex engine
    let re_result = Regex::new(&pat_str);
    // Creates a Result<Regex, Error> enum, which is either Regex or Error; we'll use match to destructure it.
    // A result enum always exists but may be either the correct variant Ok(Regex) or the error variant Err(error type, message, location).
    // :: is the path separator; it navigates to the new method on the Regex struct.
    // new creates a new rule instance; & is the borrowing (borrowing) operator — takes the address of pat_str without allowing mutation.
    // re_result is now a generic Result<Regex, Error> with two possible variants:
    // Ok if the logic is correct, Err if not — we use that next.

    // 2. Check the compilation result
    let re = match re_result { // match is pattern matching
        Ok(r) => r, // If the tag is Ok, take this branch.
                    // Ok(r) pattern-matches the Ok variant and transfers ownership of the contained Regex to r;
                    // the Error part is discarded and its memory freed.
                    // (condition) => (action); the right side is just r, a single expression.
                    // match statements have return values; here the return value is r, transferring ownership to re.
        
        // Failed: wrap the Rust error message and return it as a Python ValueError
        Err(e) => {
            // Similarly, if the tag is Err, take this branch.
            let error_msg = e.to_string();
            // Create a variable to hold the error message string
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg));
            // return terminates the main function immediately — which is correct here since an error occurred
            // Returns a Python-style raise ValueError error
        }
    };

    // Count word frequencies: mapping -> word byte data (Vec<u8>) -> occurrence count (u64)
    let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    // mut means mutable; this variable can be modified

    // Iterate over each text in the input
    for text in texts.iter() { // .iter() is a borrowing iterator that returns &String immutable references
                                // texts contains many news articles; let's assume there are many instances of "Hi there!"
                                // text is one specific article; I'll use "Hi there!" as the running example
        for mat in re.find_iter(text) {
            // re is now a Regex struct; .find_iter() returns an iterator (also a struct)
            // The iterator contains three parts: 1) an immutable reference to re (the regex machine),
            // 2) an immutable reference to the current text, and 3) the position of the last match (last_end)
            // mat stores the old and new last_end: the former is the start position, the latter is the end position
            // mat holds only two usize integers as start/end indices

            let word_bytes = mat.as_str().as_bytes().to_vec();
            // mat.as_str() converts the positions into a fat pointer recording the first element address
            // and element count within text. .as_bytes() converts to bytes for efficient processing.
            // .to_vec() allocates new memory, copies the content there, and names it word_bytes.
            
            *word_counts.entry(word_bytes).or_insert(0) += 1;
            // .entry() on word_counts looks for word_bytes; it returns an enum wrapped value:
            // either Occupied (exists) or Vacant (empty). .or_insert() has a built-in match to unwrap it.
            // .or_insert(0) means: if not found, set this position's value to 0; otherwise do nothing.
            // Then the counter increments by 1 (counts += 1).
            // The leading * is dereferencing: word_counts.entry(...).or_insert(0) returns a mutable reference;
            // * dereferences it so we operate on the value, not the address.
        }
    }

    // Print the number of unique words identified after pre-tokenization
    println!("Rust: regex tokenization complete, identified {} unique words. Starting BPE loop...", word_counts.len());

    // 2. Initialize BPE data structures
    let mut words: Vec<Word> = Vec::with_capacity(word_counts.len());
    // Create a words vector to store all Word structs in the current corpus
    
    let mut pair_freqs: HashMap<(u32, u32), u64> = HashMap::new();
    // Define a global hash map — our core object to continuously maintain.
    // It tracks the frequency of adjacent token pairs: (ID1, ID2) -> total occurrence count

    let mut pair_to_words: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    // Create a pair-to-word index table: maps a pair to the indices of words in the words vector that contain it.
    // Building this now avoids scanning for them in every loop iteration.
    

    // Convert the counted byte data into initial Word structs and store them in the words vector
    for (bytes, count) in word_counts {
        // Convert bytes (u8) to token IDs (u32)
        let ids: Vec<u32> = bytes.into_iter().map(|b| b as u32).collect();
        // bytes is the UTF-8 encoding of e.g. "Hi there!"; since all are ASCII, each character is one byte.
        // So we get exactly 9 bytes. .into_iter() extracts them one by one.
        // .map(...) applies a transformation to each element.
        // |b| b as u32 is a closure: |b| is the parameter, b as u32 is the function body that casts b to u32.
        // .collect() gathers all iterator outputs into a collection.
        // The final ids vector for "Hi there!" looks like: vec![72, 105, 32, 116, 104, 101, 114, 101, 33]
        
        // If the word has length >= 2, count its adjacent Pair frequencies
        if ids.len() >= 2 {
            for window in ids.windows(2) { 
                // Creates a window of length 2 that slides from left to right.
                // Returns an iterator; each iteration yields a sub-slice of 2 elements.
                let pair = (window[0], window[1]);
                // Create a variable pair to hold these two elements
                // Update the global Pair frequency (Pair frequency = frequency of containing Word * occurrences of Pair in that Word)
                *pair_freqs.entry(pair).or_insert(0) += count;
                // The outer loop shows this text appears count times, so increment the corresponding position by count

                let word_idx = words.len();
                // Save the index at which the current word is about to be inserted

                let word_list = pair_to_words.entry(pair).or_insert_with(Vec::new);
                // Look up pair in the inverted index; returns an Occupied or Vacant enum variant.
                // .or_insert_with handles both: if Occupied, returns a mutable reference to the existing vector;
                // if Vacant, creates a new Vec::new and returns a mutable reference to it (a fat pointer: first address + element count)

                if word_list.last() != Some(&word_idx) {
                // Only push if the last element of the vector is not already this word's index — avoids duplicates.
                // Example: processing the word "banana" which is at index 42 in the corpus.
                // Its pairs are: (b,a), (a,n), (n,a), (a,n), (n,a).
                // The first time we encounter (a,n), we look up pair_to_words[(a,n)].
                // Say it already has [10, 23]; the last is not 42, so we push 42 → [10, 23, 42].
                // The second time we encounter (a,n), the last is already 42, so we don't push again — avoiding duplicates.
                    word_list.push(word_idx);
                }
            }   
        }
        // Add the initialized Word to the words vector
        words.push(Word { ids, count });
    }


    let mut successful_merges = 0;
    // Track the number of successful merges

    let mut merges: Vec<((u32, u32), u32)> = Vec::new();
    // Create a mutable tuple vector to store merge rules: ((old_id1, old_id2), new_id)
    // This is the final object we are working so hard to produce

    let mut next_id: u32 = 256; 
    // New token IDs start from 256 (0-255 are reserved for base bytes)
    
    let target_merges = vocab_size.saturating_sub(256 + 64);
    // Maximum number of merges = target vocab size - (256 base tokens + 3 special tokens + reserved slots)
    // The 3 special tokens are:
    //      1. <PAD> (Padding): used to pad shorter sequences to match longer ones during training.
    //      2. <BOS> (Begin of Sentence): signals the model that a new conversation has started.
    //      3. <EOS> (End of Sentence): signals the model it can stop — the message is complete.
    // .saturating_sub: saturating subtraction — a safe built-in Rust subtraction that returns 0 instead of
    // wrapping around to a huge positive integer (e.g., 100 - 256 would overflow a u32 otherwise).
    // Each merge operation essentially combines two old tokens into one new token.
    // 1 merge: vocabulary total increases by 1 (a new character combination), but also removes 2 old tokens.
    // BPE retains the highest-frequency pairs as tokens and assigns them IDs.

    let mut heap = BinaryHeap::with_capacity(pair_freqs.len());
    // Initialize a standard heap (a monotone complete binary tree)
    // Set initial capacity to the number of pairs

    for (&pair, &freq) in pair_freqs.iter() {
        if freq > 0 {
            heap.push((freq, pair)); // The heap auto-sorts based on the first tuple element (freq)
            // Pushing means inserting at the bottom of the binary tree, then bubbling up automatically
        }
    }

    // 3. BPE loop: core training process (build one new rule per iteration until vocab is full)
    //  a. Find: identify the most frequent pair (A, B)
    //  b. Merge: use the inverted index to find all locations containing pair (A, B) and replace with new token C
    //  c. Update: update pair_freqs — complex step involving removing old pairs and adding new pairs
    //  d. Build merge table (Merges Table): record the (A, B) -> C mapping for future text encoding

    while successful_merges < target_merges {
        // Repeat as long as the number of successful merges hasn't reached the target
        // a. Find the highest-frequency Pair
        let mut best_pair = None;
        // Initialize best_pair

        // --- Core optimization: retrieve the highest-frequency valid Pair from the heap ---
        while let Some((freq, pair)) = heap.pop() {
            // Key: check whether the frequency popped from the heap matches the real-time frequency in pair_freqs.
            // If they differ, this Pair's frequency changed in a previous merge — it's stale garbage data; discard it.
            // .pop() retrieves the root node of the max-heap and removes it; the heap auto-rebalances afterward.
            if let Some(&current_freq) = pair_freqs.get(&pair) {
            // if let is a combined construct equivalent to a match that only handles the success case.
            // If the frequency retrieved for this pair is Some, enter the block; otherwise skip.

                if current_freq == freq && freq > 1 {
                    best_pair = Some((pair, freq));
                    // Set this pair as the current best
                    break;
                }
            }
        }
        let (best_pair, _max_freq) = match best_pair {
        // Destructure the Option<best_pair> via match to extract its content.
        // Re-assign to best_pair.
        // After this operation: best_pair is the tuple (id1, id2); _max_freq is the frequency.
        // Variables starting with _ mean: "I know this exists but I don't need it; it dies immediately."
        // E.g., for "Hi": best_pair {[72, 105], 5} becomes the final (72, 105)

            Some(p) => p,
            None => {
                println!("Rust: stopping early — no mergeable Pair with frequency > 1");
                break;
            }
        };

        // let best_pair = pair_freqs.iter()
        //     // Iterate over the pair-frequency hash map; each iteration returns a reference to a tuple of two references: &(&Pair, &u64)
        //     .max_by(|a, b| {
        //     // .max_by() scans the iterator to find the maximum value.
        //     // It uses a tournament-style comparison (champion vs. challenger), repeatedly comparing two elements.
        //     // The closure |a, b| specifies the comparison rule.
        //         match a.1.cmp(b.1) {
        //         // a.1.cmp(b.1) compares the second element of each tuple (the frequency); .cmp() returns an Ordering.
        //         // match handles the two possible outcomes:
        //             std::cmp::Ordering::Equal => a.0.cmp(b.0),  // If equal, continue comparing pair IDs until a difference is found
        //                                                         // ultimately returning Greater or Less
        //             other => other, // If unequal, return Greater or Less directly
        //         }
        //     })
        //     .map(|(k, v)| (*k, *v));
        //     // Since we got indices, dereference to extract the actual values

        // // Destructure the Option<best_pair> via match to get the concrete tuple
        // let (best_pair, _max_freq) = match best_pair {
        //     Some((pair, freq)) if freq > 1 => (pair, freq),
        //     // Some means best_pair has content; the content triggers the => action and returns (pair, freq)
        //     _ => {
        //     // _ handles the case where there's no content
        //         println!("Rust: stopping early — no mergeable Pair with frequency > 1");
        //         break;
        //     }
        // };


        let unique_indices = match pair_to_words.remove(&best_pair){
            // Record the indices of Word structs in the words vector that will be affected; returns a vector
            Some(indices) => {
                pair_freqs.remove(&best_pair);
                // Remove best_pair from the pair-frequency hash map and mark its slot as DELETED
                indices
            },
            None => {
                pair_freqs.remove(&best_pair); // Clean up this anomalous entry
                continue; // No word contains this pair; skip to the next iteration
                // This shouldn't normally happen, but just in case
            }
        };

        merges.push((best_pair, next_id));
        // Push the merge rule into the result vector — the only place we update our final output
        successful_merges += 1;
        // Increment the next ID


        // b. Update IDs in all affected words: replace the old Pair with the new ID
        for word_idx in unique_indices {
            // Iterate over the affected word indices
            // Each word is a Word struct: Word {ids, count}
            // Recall what a Word is: e.g., a news article "Hi there!" after regex splitting gives "Hi", " there", "!"
            // After as_bytes().to_vec(), their ids are:
            // 3 Word structs with ids: vec![72, 105], [32, 116, 104, 101, 114, 101], [33]  (ASCII codes)
            // So each Word is a struct containing {token IDs from the regex-split corpus chunk, occurrence count}
            // E.g., "Hi" becomes a Word struct: {[72, 105], 5} (assuming "Hi" appears 5 times in the article)
            
            let word_count = words[word_idx].count;
            // Extract count; we'll need it shortly

            let ids = &mut words[word_idx].ids;

            if ids.len() < 2 { continue; }
            // If only one element remains, it's already minimal — skip to the next iteration for efficiency

            let mut j = 0;
            // Create a pointer j representing the current index in the ids vector

            let mut changed = false;
            // Boolean flag: tracks whether this Word's ids vector was actually modified in this merge round

            let mut new_ids = Vec::with_capacity(ids.len());
            // Result vector: stores the new u32 sequence after merge-and-replace

            while j < ids.len() { // Keep advancing the pointer until the end
                // Check whether the two IDs at the current position equal the best_pair to merge
                // For "HiHithere": ids = [72, 105, 72, 105, 116, 104, 101, 114, 101]
                if j < ids.len() - 1 && ids[j] == best_pair.0 && ids[j + 1] == best_pair.1 {
                // Using "Hi" as an example: suppose Hi is best_pair (72, 105).
                // If element at position j is 72 and the next element j+1 is 105,
                // we've found the best combination in our word.
                    
                    // If there's a previous token: after merging 'AB', all combinations containing 'A' (like 'CA')
                    // and 'B' (like 'BD') have their frequencies changed.
                    // We now disconnect the left neighbor and the right neighbor.
                    // First: the left neighbor
                        if new_ids.len() > 0 { // Ensure the result vector has at least one element
                            let prev_id = *new_ids.last().unwrap();
                            // Get the last element of the result vector. .last() returns an Option reference.
                            // .unwrap() unpacks the Option — safe here since we already checked for elements.
                            // * dereferences it: copies the last element's value into a new memory location and assigns it to prev_id.
                            
                            // -------------------------
                            // [Old buggy code commented out]
                            // if let Some(c) = pair_freqs.get_mut(&(prev_id, ids[j])) {
                            // // Some() as a wrapper means: if the right side is None, skip the block;
                            // // if Some(c), assign the content to c and execute the block.
                            // // Look in pair_freqs for (prev_id, current_id); if found,
                            // // assign its mutable reference to c — c is the global occurrence count of this pair.
                            //
                            // // pair_freqs is a global hash map (over hundreds of thousands of articles);
                            // // word is a struct specific to one regex slice of one article.
                            //     *c = c.saturating_sub(word_count); 
                            //     // Can't read this left to right — read the * first:
                            //     // it means we're operating on the value at the address, not the address itself.
                            //     // So the global pair_freqs entry for (prev_id, ids[j]) decreases by the current slice's count.
                            //     // Why subtract word.count: when we merge best_pair in this word,
                            //     // the frequency this word contributed to its neighbors must be taken back.
                            //     // Because this word contributed word.count occurrences of best_pair, subtract word.count globally.
                            // }
                            
                            if let Some(c) = pair_freqs.get_mut(&(prev_id, ids[j])) {
                            // Some() wrapper: if the right side is None, skip the block;
                            // if Some(c), assign content to c and execute the block.
                            // Look in pair_freqs for (prev_id, current_id); if found,
                            // assign its mutable reference to c — c is the global occurrence count of this pair.

                            // pair_freqs is a global hash map; word is a struct for one regex slice of one article.
                                *c = c.saturating_sub(word_count); 
                                // Read the * first: operate on the value at the address, not the address itself.
                                // The global entry for (prev_id, ids[j]) in pair_freqs decreases by this slice's count.
                                // Why subtract word.count: when we merge best_pair in this word,
                                // the frequency this word contributed to its neighbors must be reclaimed.
                                // Since this word contributed word.count occurrences of best_pair, subtract word.count globally.
                                
                                if *c > 0 { // Push the updated frequency of the new neighbor into the heap — critical logic!
                                    heap.push((*c, (prev_id, ids[j])));
                                }
                            }
                            // -------------------------

                            let freq_ref = pair_freqs.entry((prev_id, next_id)).or_insert(0);
                            *freq_ref += word_count; // Update the frequency in the hash map
                            // Core: push the latest increased frequency into the heap for competition!
                            // Even if this Pair was previously in the heap, that's fine — we validate freq when popping.
                            heap.push((*freq_ref, (prev_id, next_id))); 
                            // =========================

                            pair_to_words.entry((prev_id, next_id)).or_insert_with(Vec::new).push(word_idx);
                        }

                        //     // 2. Increase frequency of new Pair (previous token, new TokenID)
                        //     *pair_freqs.entry((prev_id, next_id)).or_insert(0) += word_count;
                        //     // j and j+1 merged into next_id, which now pairs with j-1; this (possibly new) pair increases by word.count
                        //     // next_id auto-increments with each outer loop iteration, starting at 256 (0-255 reserved for ASCII)

                        //     pair_to_words.entry((prev_id, next_id)).or_insert_with(Vec::new).push(word_idx);
                        // }

                        // j and j+1 are merged; we've already handled everything before j.
                        // Now we handle after j+1 — i.e., disconnect j+1 from j+2.
                        
                        // -------------------------
                        // [Old buggy code commented out]
                        // if j + 2 < ids.len() {
                        //     if let Some(c) = pair_freqs.get_mut(&(ids[j+1], ids[j+2])) {
                        //         *c = c.saturating_sub(word_count);
                        //     }
                        //     let freq_ref = pair_freqs.entry((next_id, ids[j+2])).or_insert(0);
                        //     *freq_ref += word_count;
                        //     heap.push((*freq_ref, (next_id, ids[j+2]))); // Push the new frequency into the heap
                        //
                        //     // Same logic: j and j+1 merged into next_id, which now pairs with j+2...
                        //     pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                        //
                        //     // *pair_freqs.entry((next_id, ids[j+2])).or_insert(0) += word_count;
                        //     // pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                        // }
                        
                        // [Correct code]
                        if j + 2 < ids.len() {
                            if let Some(c) = pair_freqs.get_mut(&(ids[j+1], ids[j+2])) {
                                *c = c.saturating_sub(word_count);
                                
                                if *c > 0 { // Also push the right neighbor's updated frequency into the heap!
                                    heap.push((*c, (ids[j+1], ids[j+2])));
                                }
                            }
                            let freq_ref = pair_freqs.entry((next_id, ids[j+2])).or_insert(0);
                            *freq_ref += word_count;
                            heap.push((*freq_ref, (next_id, ids[j+2]))); // Push the new frequency into the heap
                            // =========================

                            // Same: j and j+1 merged into next_id, which now pairs with j+2...
                            pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);

                            // *pair_freqs.entry((next_id, ids[j+2])).or_insert(0) += word_count;
                            // pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                    }
                    // -------------------------

                    new_ids.push(next_id);
                    // Push the merged result into the new_ids vector
                    j += 2;
                    // Since we've processed j and j+1, the next iteration starts at j+2
                    changed = true;
                    // If the code reaches here, a merge definitely occurred — set changed to true
                } else {
                    // No matching merge pair found at this position; keep the current ID unchanged
                    new_ids.push(ids[j]);
                    j += 1;
                }
            }
            // If the Word's ID sequence changed, update the Word struct
            if changed {
                *ids = new_ids;
            // Replace the ids vector of this word's struct with the new merged new_ids
            }
        }

        // Print progress every 100 merges
        if successful_merges != 0 && (successful_merges ) % 100 == 0 {
            println!("Rust: merge progress {} / {} -> new token ID: {}", successful_merges, target_merges, next_id);
        }
        next_id += 1;
        // Increment the count of successful merges
    }

    // Return an empty dictionary (reserved) and all generated merge rules
    Ok((StdHashMap::new(), merges))
    // merges is a vector; each element is ((old_id1, old_id2), new_id) — records our merge rules.
    // We create a standard hash map to return our result because Python only recognizes that type.
    // Looks like no return value? Ok(...) is a result enum — it IS the return value.
    // To extract its contents, the caller unpacks it; PyO3 handles that during conversion to Python code.
    // Python code doesn't know what Ok is — PyO3 strips it away.
    // This thing is a result enum wrapping a tuple of (hash_map, merge_rules).
}

// This is a macro invocation telling the Rust compiler: the following function custom_bpe
// is an entry point for defining a Python extension module.
#[pymodule]
fn custom_bpe<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // <'py> is a lifetime — it names "the duration while the Python interpreter is alive".
    // _py represents "the Python interpreter itself"; the underscore means we don't use it.
    // m is a reference to this Python module (the file treated as an importable entity).
    // -> PyResult<()> declares the return type: a result enum that is either () (unit) or an Err exception.

    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    // Reading from inside out: wrap_pyfunction!(train_bpe, m) — the ! means this is a macro call.
    // Macros can generate code at compile time, doing things ordinary functions cannot.
    // wrap_pyfunction!(train_bpe, m) wraps the Rust function train_bpe into a format Python can recognize.
    // The trailing ? is syntactic sugar: it's shorthand for a match on the result enum —
    // Ok(...) => continue, Err(...) => immediately return with the error, ending the function.
    // .add_function registers the function into the module; it returns a result enum.
    // The final ? works the same way — if registration fails, propagate the error to Python.

    m.add_class::<encode::Tokenizer>()?;
    m.add_class::<decode::Decoder>()?;
    m.add_class::<engine::BpeEngine>()?;
    
    Ok(())
    // Return empty (unit)
}
// encode.rs
// 推理端：高度优化的 Byte-Level BPE Tokenizer
// 核心优化：
// 1. FxHashMap: 使用极速非加密哈希表处理小整数键的 O(1) 查找。
// 2. Aho-Corasick 自动机: O(N) 线性时间瞬间扫描所有特殊词元，避免复杂正则回溯。
// 3. 释放 GIL: 采用安全的 String 所有权转移，运行期解除 Python 全局锁，避免阻塞主进程。
// 4. 双向链表合并: 在长词合并时，采用数组模拟链表，彻底干掉 Vec::remove 带来的 O(N^2) 内存移动悬崖。
// 5. 零分配 Buffer: 在循环外部预分配内存，热路径中只做 clear()，实现零 Allocation。

use pyo3::prelude::*;
use rustc_hash::FxHashMap; // 替代标准哈希，速度提升 30%+
use regex::Regex;
use std::collections::HashMap as StdHashMap;
use aho_corasick::{AhoCorasick, MatchKind};
use std::collections::BinaryHeap;
use std::cmp::Reverse;


/// 用于在长文本块中避免 O(N) 内存移动的双向链表节点
#[derive(Clone, Copy)]
struct ListNode {
    id: u32,
    prev: usize,
    next: usize,
    gen: u32,
}

#[pyclass]
pub struct Tokenizer {
    regex: Regex,
    // 使用 FxHashMap 替代默认哈希，(u32, u32) 计算默认哈希极其耗时
    merges: FxHashMap<(u32, u32), (usize, u32)>,
    // AhoCorasick 自动机，用于极速匹配 <|user|> 等所有特殊词元
    special_matcher: Option<AhoCorasick>,
    // 配合 AC 自动机使用：通过 match 的 ID 直接 O(1) 映射到 Token ID
    special_token_ids: Vec<u32>,
}

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(
        merges_list: Vec<((u32, u32), u32)>, 
        special_tokens_dict: StdHashMap<String, u32>
    ) -> PyResult<Self> {
        
        // 1. 重建预分词正则 (GPT-2 / LLaMA 标准)
        let pat_str = [
            r"'(?:s|t|ll|ve|re|m|d)",
            r" ?\p{L}+",
            r" ?\p{N}+",
            r" ?[^\s\p{L}\p{N}]+",
            r"\s+",
        ].join("|");
        
        let regex = Regex::new(&pat_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // 2. 构建极速 FxHashMap 合并规则表
        let mut merges = FxHashMap::default();
        for (rank, ((p0, p1), new_id)) in merges_list.into_iter().enumerate() {
            merges.insert((p0, p1), (rank, new_id));
        }

        // 3. 构建 Aho-Corasick 特殊词元匹配器
        let mut patterns = Vec::with_capacity(special_tokens_dict.len());
        let mut special_token_ids = Vec::with_capacity(special_tokens_dict.len());

        for (pat, id) in special_tokens_dict {
            patterns.push(pat);
            special_token_ids.push(id);
        }

        let special_matcher = if !patterns.is_empty() {
            // 使用 builder 指定最长匹配规则
            let ac = AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Some(ac)
        } else {
            None
        };

        println!("Rust Tokenizer 极速版初始化完成！规则数: {}, 特殊词元数: {}", merges.len(), patterns.len());

        Ok(Tokenizer {
            regex,
            merges,
            special_matcher,
            special_token_ids,
        })
    }

    /// Python 端调用的 encode
    /// 注意：这里接收的是 `String` 而非 `&str`。
    /// 原因是如果要调用 `allow_threads` 释放 GIL，数据必须脱离 Python 的内存管理。
    pub fn encode(&self, py: Python<'_>, text: String) -> Vec<u32> {
        // 释放 GIL 锁：在此闭包运行期间，Python 的其他线程可以自由运行，不会被阻塞！
        py.allow_threads(|| {
            self._encode_internal(&text)
        })
    }
}

// -------------------------------------------------------------------------
// Rust 内部私有实现
// -------------------------------------------------------------------------
impl Tokenizer {
    fn _encode_internal(&self, text: &str) -> Vec<u32> {
        // 预估容量，减少扩容开销
        let mut final_tokens = Vec::with_capacity(text.len() / 4); 
        
        // 【核心优化】：将 Buffer 提至外层，避免在循环中产生数千次的堆内存分配！
        let mut word_buffer: Vec<u32> = Vec::with_capacity(256);
        let mut list_buffer: Vec<ListNode> = Vec::with_capacity(256);

        let mut last_end = 0;

        if let Some(ref ac) = self.special_matcher {
            // Aho-Corasick 将以极快的线性时间找出所有的特殊词元
            for mat in ac.find_iter(text) {
                let start = mat.start();
                let end = mat.end();

                // 1. 处理特殊词元前面的普通文本
                if start > last_end {
                    self.encode_normal_text(
                        &text[last_end..start], 
                        &mut final_tokens, 
                        &mut word_buffer, 
                        &mut list_buffer
                    );
                }

                // 2. 直接 O(1) 拿到这个特殊词元对应的 ID，推入最终数组
                let pattern_idx = mat.pattern().as_usize();
                final_tokens.push(self.special_token_ids[pattern_idx]);

                last_end = end;
            }
        }

        // 处理尾部剩余的普通文本
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

    /// 处理纯普通文本块（无特殊词元）
    fn encode_normal_text(
        &self, 
        text: &str, 
        final_tokens: &mut Vec<u32>, 
        word_buffer: &mut Vec<u32>,
        list_buffer: &mut Vec<ListNode>
    ) {
        for mat in self.regex.find_iter(text) {
            // clear 不会释放内存，仅仅重置长度为 0，零开销！
            word_buffer.clear(); 
            
            // 【核心逻辑】：将 UTF-8 文本块转为底层字节 (Byte)。
            // 因为 0-255 字节的初始 Token ID 就是 0-255，所以直接强转为 u32 即可！
            // 完美的 1:1 映射，彻底摆脱 Python 端的乱码字符串查找
            word_buffer.extend(mat.as_str().as_bytes().iter().map(|&b| b as u32));

            // 执行极速 BPE 融合
            self.bpe_merge(word_buffer, list_buffer);

            // 融合完成后，整体追加到最终结果中
            final_tokens.extend_from_slice(word_buffer);
        }
    }

    /// 核心算法：无 O(N^2) 内存移动悬崖的极限 BPE 融合
    fn bpe_merge(&self, ids: &mut Vec<u32>, list_nodes: &mut Vec<ListNode>) {
        if ids.len() < 2 {
            return;
        }

        // 如果这个词特别短 (比如常见的英文字母组合 3~8 个字节)
        // 底层的连续内存移动 (memmove) 速度其实快于构建复杂链表，退化为传统方式
        if ids.len() < 50 {
            // 预扫一遍，为每个位置缓存 rank
            let mut ranks: Vec<usize> = Vec::with_capacity(ids.len());
            for i in 0..ids.len() - 1 {
                ranks.push(
                    self.merges.get(&(ids[i], ids[i + 1]))
                        .map_or(usize::MAX, |&(r, _)| r)
                );
            }
            ranks.push(usize::MAX); // 末位哨兵

            loop {
                // 找全局最优（只读 ranks 数组，无哈希查找）
                let mut best_rank = usize::MAX;
                let mut best_idx = usize::MAX;
                for i in 0..ranks.len() - 1 {
                    if ranks[i] < best_rank {
                        best_rank = ranks[i];
                        best_idx = i;
                    }
                }
                if best_idx == usize::MAX { break; }

                // 拿合并结果（这次必命中）
                let (_, target_id) = self.merges[&(ids[best_idx], ids[best_idx + 1])];
                ids[best_idx] = target_id;
                ids.remove(best_idx + 1);
                ranks.remove(best_idx + 1);

                // 只刷新受影响的 1~2 个位置
                if best_idx < ranks.len() - 1 {
                    ranks[best_idx] = self.merges.get(&(ids[best_idx], ids[best_idx + 1]))
                        .map_or(usize::MAX, |&(r, _)| r);
                }
                if best_idx > 0 {
                    ranks[best_idx - 1] = self.merges.get(&(ids[best_idx - 1], ids[best_idx]))
                        .map_or(usize::MAX, |&(r, _)| r);
                }
            }
            return;
        }

        // =================================================================
        // 【防爆机制】：对于极长块（如连续几千个空格/长链接等畸形文本），
        // 必须采用数组双向链表 (Array-based Linked List)，时间复杂度严格压制。
        // =================================================================
        list_nodes.clear();
        for (i, &id) in ids.iter().enumerate() {
            list_nodes.push(ListNode {
                id,
                prev: if i == 0 { usize::MAX } else { i - 1 },
                next: if i == ids.len() - 1 { usize::MAX } else { i + 1 },
                gen: 0,
            });
        }

        // 堆条目：(rank, left_idx, 合并时left节点的id快照, 合并时right节点的id快照, new_id)
        // 用 id 快照做 lazy deletion：pop 出来时验证快照是否仍匹配当前节点 id
        // 若不匹配说明该节点已被合并过（id 已变），直接丢弃
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        struct HeapEntry(Reverse<usize>, usize, u32, u32, u32);
        // rank(越小越优) left_idx  gen_l  gen_r  new_id
        

        // 初始化堆
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut curr = 0;
        while curr != usize::MAX {
            let right = list_nodes[curr].next;
            if right != usize::MAX {
                let pair = (list_nodes[curr].id, list_nodes[right].id);
                if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                    heap.push(HeapEntry(
                        Reverse(rank), curr,
                        list_nodes[curr].gen, list_nodes[right].gen,
                        new_id,
                    ));
                }
            }
            curr = list_nodes[curr].next;
        }

        // 主循环
        loop {
            let HeapEntry(_, left, snap_l, snap_r, new_id) = match heap.pop() {
                Some(e) => e,
                None => break,
            };

            let right = list_nodes[left].next;
            if right == usize::MAX
                || list_nodes[left].gen != snap_l
                || list_nodes[right].gen != snap_r
            {
                continue;
            }

            let next_next = list_nodes[right].next;
            list_nodes[left].id = new_id;
            list_nodes[left].gen += 1;
            list_nodes[left].next = next_next;
            if next_next != usize::MAX {
                list_nodes[next_next].prev = left;
            }

            // 内联：检查 left → next_next 这对
            {
                let right2 = list_nodes[left].next;
                if right2 != usize::MAX {
                    let pair = (list_nodes[left].id, list_nodes[right2].id);
                    if let Some(&(rank, nid)) = self.merges.get(&pair) {
                        heap.push(HeapEntry(
                            Reverse(rank), left,
                            list_nodes[left].gen, list_nodes[right2].gen,
                            nid,
                        ));
                    }
                }
            }

            // 内联：检查 prev → left 这对
            {
                let prev = list_nodes[left].prev;
                if prev != usize::MAX {
                    let pair = (list_nodes[prev].id, list_nodes[left].id);
                    if let Some(&(rank, nid)) = self.merges.get(&pair) {
                        heap.push(HeapEntry(
                            Reverse(rank), prev,
                            list_nodes[prev].gen, list_nodes[left].gen,
                            nid,
                        ));
                    }
                }
            }
        }

        // 把链表里的结果按照顺序倒回 ids 里
        ids.clear();
        let mut curr = 0;
        while curr != usize::MAX {
            ids.push(list_nodes[curr].id);
            curr = list_nodes[curr].next;
        }
    }
}
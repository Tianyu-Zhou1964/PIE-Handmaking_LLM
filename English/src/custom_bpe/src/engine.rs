// engine.rs
// ==============================================================================
// Unified BPE Engine: Encoder + Decoder Combined, Single File I/O
// ==============================================================================
// Motivation for refactoring:
//   In the old architecture, Tokenizer (encode.rs) and Decoder (decode.rs) were
//   initialized independently. The Python side had to manually read vocab.json /
//   merges.txt, parse nested formats, strip special tokens, split merges line by
//   line, and look up dictionaries to assemble tuples — all the slowest possible
//   Python string operations.
//   Now everything is internalized into Rust:
//     1. serde_json reads vocab.json directly (nested format handled automatically)
//     2. Rust parses merges.txt line by line and looks up vocab for IDs internally
//     3. A single pass over vocab builds both encoder-side and decoder-side structures
//     4. Special token detection and routing handled entirely within Rust
//
// Python usage (reduced from 50+ lines to 2):
//   engine = custom_bpe.BpeEngine("path/to/vocab.json", "path/to/merges.txt")
//   ids = engine.encode("Hello, world!") 
//   {ids = engine.encode("床前明月光")  // Chinese text example}
//   text = engine.decode(ids)
//   chunk = engine.decode_stream(next_id)
//   engine.reset_stream()
//
// All performance optimizations preserved:
//   - FxHashMap: ultra-fast non-cryptographic hash, O(1) lookup for small int keys
//   - Aho-Corasick automaton: O(N) linear scan over all special tokens
//   - Dual-path BPE merging: short tokens use ranks array scan, long tokens use
//     doubly-linked array + BinaryHeap
//   - Zero-allocation buffer: pre-allocated outside the hot loop, hot path only calls clear()
//   - GIL release: encoding/decoding does not block other Python threads
//   - Streaming decode: automatic UTF-8 byte buffering for incomplete CJK characters
// ==============================================================================

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use regex::Regex;
use std::collections::HashMap as StdHashMap;
use aho_corasick::{AhoCorasick, MatchKind};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::fs;


// ==============================================================================
// 辅助结构体
// ==============================================================================

/// BPE 合并时用于长文本块的双向链表节点（避免 O(N²) 内存移动）
#[derive(Clone, Copy)]
struct ListNode {
    id: u32,
    prev: usize,
    next: usize,
    generation: u32,
}

/// 长文本 BPE 合并堆条目：(rank越小越优, 左节点索引, 左节点代数快照, 右节点代数快照, 合并后新ID)
/// 用 generation 快照做 lazy deletion：pop 时验证快照是否匹配当前节点，不匹配则丢弃
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct HeapEntry(Reverse<usize>, usize, u32, u32, u32);


// ==============================================================================
// GPT-2 Byte-to-Unicode 反向映射（解码侧需要）
// ==============================================================================

/// 构建 GPT-2 标准的 Unicode字符 → 原始字节 反向映射表
///
/// GPT-2 BPE 把每个字节（0-255）映射成可见 Unicode 字符，让词表 JSON 不炸。
/// 解码时需要反向走一遍：vocab 中的 Unicode 子词字符串 → 原始字节序列。
///
/// 映射规则：
///   - 可见字节（33-126, 161-172, 174-255）→ 直接映射到自己
///   - 不可见字节（0-32, 127-160, 173）→ 映射到 Unicode 256+ 区域
fn build_unicode_to_byte_map() -> FxHashMap<char, u8> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(33u32..=126);   // 可见 ASCII
    bs.extend(161u32..=172);  // 可见拉丁补充
    bs.extend(174u32..=255);  // 可见拉丁补充

    let mut mapping = FxHashMap::default();

    // 安全字节：直接映射到自己
    for &b in &bs {
        mapping.insert(char::from_u32(b).unwrap(), b as u8);
    }

    // 不安全字节：从 Unicode 256 开始依次映射
    let mut is_safe = [false; 256];
    for &b in &bs {
        is_safe[b as usize] = true;
    }

    let mut offset = 0u32;
    for b in 0u32..256 {
        if !is_safe[b as usize] {
            let unicode_char = char::from_u32(256 + offset).unwrap();
            mapping.insert(unicode_char, b as u8);
            offset += 1;
        }
    }

    mapping
}


// ==============================================================================
// BpeEngine：编码 + 解码统一引擎
// ==============================================================================

#[pyclass]
pub struct BpeEngine {
    // ---- 编码侧 ----
    /// GPT-2 标准预分词正则
    regex: Regex,
    /// 合并规则表：(旧ID1, 旧ID2) → (优先级rank, 合并后新ID)
    merges: FxHashMap<(u32, u32), (usize, u32)>,
    /// Aho-Corasick 自动机，O(N) 线性扫描所有特殊词元
    special_matcher: Option<AhoCorasick>,
    /// AC 自动机 match ID → Token ID 的映射数组
    special_token_ids: Vec<u32>,

    // ---- 解码侧 ----
    /// Token ID → 原始字节序列（核心解码查找表）
    id_to_bytes: FxHashMap<u32, Vec<u8>>,
    /// 流式解码的 UTF-8 字节缓冲区（处理"半个汉字"）
    stream_buffer: Vec<u8>,
    str_to_id: FxHashMap<String, u32>,
}

#[pymethods]
impl BpeEngine {
    // ==================================================================
    // 初始化：直接接收文件路径，Rust 内部完成所有解析
    // ==================================================================

    /// 创建 BPE 引擎
    ///
    /// Python 端用法：
    ///   engine = custom_bpe.BpeEngine("/path/to/vocab.json", "/path/to/merges.txt")
    ///
    /// 内部流程：
    ///   1. serde_json 读 vocab.json（自动处理 {"model":{"vocab":{...}}} 嵌套格式）
    ///   2. 一次遍历 vocab：
    ///      - 特殊 token（<eos> 等）→ 收集到 AC 自动机 patterns
    ///      - 所有 token → 构建 id_to_bytes 解码表
    ///   3. Rust 内部逐行解析 merges.txt + 查 vocab 拿 ID → 构建 merges 编码表
    ///   4. 构建 Aho-Corasick 匹配器 + 预分词正则
    #[new]
    pub fn new(vocab_path: String, merges_path: String) -> PyResult<Self> {
        // ----------------------------------------------------------------
        // Step 1: 读取并解析 vocab.json
        // ----------------------------------------------------------------
        let vocab_raw = fs::read_to_string(&vocab_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法读取 vocab 文件 '{}': {}", vocab_path, e)
            ))?;

        // serde_json 解析，自动处理两种格式：
        //   格式1: {"token": id, ...}             （平坦）
        //   格式2: {"model": {"vocab": {...}}}      （嵌套，如 HuggingFace 格式）
        let json_value: serde_json::Value = serde_json::from_str(&vocab_raw)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("vocab JSON 解析失败: {}", e)
            ))?;

        let vocab_obj = if let Some(model) = json_value.get("model") {
            if let Some(v) = model.get("vocab") {
                v
            } else {
                &json_value
            }
        } else {
            &json_value
        };

        // 转成 HashMap<String, u32>
        let vocab_map: StdHashMap<String, u32> = match vocab_obj.as_object() {
            Some(obj) => {
                let mut map = StdHashMap::with_capacity(obj.len());
                for (k, v) in obj {
                    let id = v.as_u64()
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("vocab 中 '{}' 的值不是整数", k)
                        ))? as u32;
                    map.insert(k.clone(), id);
                }
                map
            }
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "vocab JSON 顶层不是对象"
                ));
            }
        };

        println!("Rust BpeEngine: vocab 加载完成，共 {} 个词条", vocab_map.len());

        // ----------------------------------------------------------------
        // Step 2: 一次遍历 vocab，同时构建编码侧和解码侧
        // ----------------------------------------------------------------
        let unicode_to_byte = build_unicode_to_byte_map();
        let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();
        let mut special_patterns: Vec<String> = Vec::new();
        let mut special_token_ids: Vec<u32> = Vec::new();

        for (token_str, &token_id) in &vocab_map {
            // ---- 判断是否为特殊 token ----
            if token_str.starts_with('<') && token_str.ends_with('>') {
                // 解码侧：特殊 token 按 UTF-8 原样存储
                id_to_bytes.insert(token_id, token_str.as_bytes().to_vec());
                // 编码侧：收集到 AC 自动机的 patterns
                special_patterns.push(token_str.clone());
                special_token_ids.push(token_id);
                continue;
            }

            // ---- 普通子词：逐字符查反向映射表，还原为原始字节 ----
            let mut bytes = Vec::with_capacity(token_str.len());
            let mut valid = true;

            for ch in token_str.chars() {
                match unicode_to_byte.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        // 兜底：不在映射表中的字符，原样塞 UTF-8 字节
                        bytes.extend_from_slice(ch.to_string().as_bytes());
                        valid = false;
                    }
                }
            }

            if !valid {
                eprintln!(
                    "⚠️ 子词 {:?}(ID={}) 包含不在 GPT-2 映射表中的字符，已做兜底处理",
                    token_str, token_id
                );
            }

            id_to_bytes.insert(token_id, bytes);
        }

        // ----------------------------------------------------------------
        // Step 3: Rust 内部解析 merges.txt（不再让 Python 干这活）
        // ----------------------------------------------------------------
        let merges_raw = fs::read_to_string(&merges_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
                format!("无法读取 merges 文件 '{}': {}", merges_path, e)
            ))?;

        let mut merges: FxHashMap<(u32, u32), (usize, u32)> = FxHashMap::default();
        let mut rank: usize = 0;

        let mut combined_buf = String::with_capacity(64);
        for line in merges_raw.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue; // 跳过空行和注释行
            }

            // 每行格式："子词A 子词B"，split 后查 vocab 拿 ID
            let mut parts = line.splitn(2, ' ');
            let p0_str = match parts.next() {
                Some(s) => s,
                None => continue,
            };
            let p1_str = match parts.next() {
                Some(s) => s,
                None => continue,
            };

            // 查 vocab 拿三个 ID
            let p0_id = match vocab_map.get(p0_str) {
                Some(&id) => id,
                None => continue, // 词表里找不到就跳过
            };
            let p1_id = match vocab_map.get(p1_str) {
                Some(&id) => id,
                None => continue,
            };
            combined_buf.clear();
            combined_buf.push_str(p0_str);
            combined_buf.push_str(p1_str);
            let new_id = match vocab_map.get(&combined_buf) {
                Some(&id) => id,
                None => continue,
            };

            merges.insert((p0_id, p1_id), (rank, new_id));
            rank += 1;
        }

        println!(
            "Rust BpeEngine: merges 加载完成，共 {} 条规则，{} 个特殊词元",
            merges.len(),
            special_patterns.len()
        );

        // ----------------------------------------------------------------
        // Step 4: 构建 Aho-Corasick 自动机 + 预分词正则
        // ----------------------------------------------------------------
        let special_matcher = if !special_patterns.is_empty() {
            let ac = AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&special_patterns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Aho-Corasick 构建失败: {}", e)
                ))?;
            Some(ac)
        } else {
            None
        };

        let pat_str = [
            r"'(?:s|t|ll|ve|re|m|d)",
            r" ?\p{L}+",
            r" ?\p{N}+",
            r" ?[^\s\p{L}\p{N}]+",
            r"\s+",
        ].join("|");

        let regex = Regex::new(&pat_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("正则编译失败: {}", e)
            ))?;

        println!("Rust BpeEngine 初始化完成！");

        let mut str_to_id: FxHashMap<String, u32> = FxHashMap::default();
        for (token_str, &token_id) in &vocab_map {
            str_to_id.insert(token_str.clone(), token_id);
        }

        Ok(BpeEngine {
            regex,
            merges,
            special_matcher,
            special_token_ids,
            id_to_bytes,
            stream_buffer: Vec::with_capacity(64),
            str_to_id,
        })
    }

    // ==================================================================
    // 编码接口
    // ==================================================================

    /// 编码：文本 → Token ID 序列
    /// 释放 GIL，不阻塞 Python 其他线程
    pub fn encode(&self, py: Python<'_>, text: String) -> Vec<u32> {
        py.allow_threads(|| {
            self._encode_internal(&text)
        })
    }

    // ==================================================================
    // 解码接口
    // ==================================================================

    /// 一次性解码：Token ID 序列 → 完整 UTF-8 字符串
    /// 释放 GIL
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        py.allow_threads(|| {
            self._decode_internal(&ids)
        })
    }

    /// 流式解码：逐个 Token 输入，尽可能输出文字
    ///
    /// 返回值：
    ///   - 字节凑齐 → 返回解码出的文字
    ///   - 字节没凑齐（半个汉字）→ 返回空字符串 ""
    ///   - Token ID 不在词表中 → 返回空字符串 ""
    ///
    /// 每次新对话前要调用 reset_stream()
    pub fn decode_stream(&mut self, token_id: u32) -> String {
        let raw_bytes = match self.id_to_bytes.get(&token_id) {
            Some(b) => b,
            None => return String::new(),
        };

        self.stream_buffer.extend_from_slice(raw_bytes);

        match std::str::from_utf8(&self.stream_buffer) {
            Ok(s) => {
                let result = s.to_string();
                self.stream_buffer.clear();
                result
            }
            Err(e) => {
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    let result = std::str::from_utf8(&self.stream_buffer[..valid_len])
                        .unwrap()
                        .to_string();
                    self.stream_buffer.drain(..valid_len);
                    result
                } else {
                    String::new()
                }
            }
        }
    }

    /// 重置流式解码器的内部状态
    pub fn reset_stream(&mut self) {
        self.stream_buffer.clear();
    }

    // ==================================================================
    // 工具方法：暴露特殊 token ID 给 Python（比如拿 <eos> 的 ID）
    // ==================================================================

    /// 查询某个 token 字符串对应的 ID（找不到返回 None）
    /// 用法：eos_id = engine.token_to_id("<eos>")
    pub fn token_to_id(&self, token_str: String) -> Option<u32> {
        self.str_to_id.get(&token_str).copied()
    }
}


// ==============================================================================
// Rust 内部私有实现（不暴露给 Python）
// ==============================================================================

impl BpeEngine {
    // ------------------------------------------------------------------
    // 编码内部实现（保留全部优化）
    // ------------------------------------------------------------------

    fn _encode_internal(&self, text: &str) -> Vec<u32> {
        let mut final_tokens = Vec::with_capacity(text.len() / 4);

        // 【零分配 Buffer】循环外预分配，热路径只做 clear()
        let mut word_buffer: Vec<u32> = Vec::with_capacity(256);
        let mut list_buffer: Vec<ListNode> = Vec::with_capacity(256);

        let mut last_end = 0;

        // 【Aho-Corasick】O(N) 线性扫描特殊词元
        if let Some(ref ac) = self.special_matcher {
            for mat in ac.find_iter(text) {
                let start = mat.start();
                let end = mat.end();

                if start > last_end {
                    self._encode_normal_text(
                        &text[last_end..start],
                        &mut final_tokens,
                        &mut word_buffer,
                        &mut list_buffer,
                    );
                }

                let pattern_idx = mat.pattern().as_usize();
                final_tokens.push(self.special_token_ids[pattern_idx]);
                last_end = end;
            }
        }

        if last_end < text.len() {
            self._encode_normal_text(
                &text[last_end..],
                &mut final_tokens,
                &mut word_buffer,
                &mut list_buffer,
            );
        }

        final_tokens
    }

    /// 处理纯普通文本块（无特殊词元）
    fn _encode_normal_text(
        &self,
        text: &str,
        final_tokens: &mut Vec<u32>,
        word_buffer: &mut Vec<u32>,
        list_buffer: &mut Vec<ListNode>,
    ) {
        for mat in self.regex.find_iter(text) {
            word_buffer.clear(); // 零开销：不释放内存，仅重置长度

            // 直接用原始字节值作为初始 Token ID（0-255 = 字节值）
            word_buffer.extend(mat.as_str().as_bytes().iter().map(|&b| b as u32));

            // 执行 BPE 合并（双路径优化）
            self._bpe_merge(word_buffer, list_buffer);

            final_tokens.extend_from_slice(word_buffer);
        }
    }

    /// 核心算法：双路径 BPE 合并
    ///   - 短词（< 50 字节）：ranks 数组线性扫描，cache-friendly
    ///   - 长词（≥ 50 字节）：数组双向链表 + BinaryHeap + lazy deletion
    fn _bpe_merge(&self, ids: &mut Vec<u32>, list_nodes: &mut Vec<ListNode>) {
        if ids.len() < 2 {
            return;
        }

        // ============================================================
        // 短路径：ranks 数组扫描（短词的 memmove 快于链表开销）
        // ============================================================
        if ids.len() < 50 {
            let mut ranks: Vec<usize> = Vec::with_capacity(ids.len());
            for i in 0..ids.len() - 1 {
                ranks.push(
                    self.merges.get(&(ids[i], ids[i + 1]))
                        .map_or(usize::MAX, |&(r, _)| r)
                );
            }
            ranks.push(usize::MAX); // 末位哨兵

            loop {
                // 全局最优（只读 ranks 数组，无哈希查找）
                let mut best_rank = usize::MAX;
                let mut best_idx = usize::MAX;
                for i in 0..ranks.len() - 1 {
                    if ranks[i] < best_rank {
                        best_rank = ranks[i];
                        best_idx = i;
                    }
                }
                if best_idx == usize::MAX { break; }

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

        // ============================================================
        // 长路径：数组双向链表 + BinaryHeap（防爆机制）
        // ============================================================
        list_nodes.clear();
        for (i, &id) in ids.iter().enumerate() {
            list_nodes.push(ListNode {
                id,
                prev: if i == 0 { usize::MAX } else { i - 1 },
                next: if i == ids.len() - 1 { usize::MAX } else { i + 1 },
                generation: 0,
            });
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(ids.len());
        let mut curr = 0;
        while curr != usize::MAX {
            let right = list_nodes[curr].next;
            if right != usize::MAX {
                let pair = (list_nodes[curr].id, list_nodes[right].id);
                if let Some(&(rank, new_id)) = self.merges.get(&pair) {
                    heap.push(HeapEntry(
                        Reverse(rank), curr,
                        list_nodes[curr].generation, list_nodes[right].generation,
                        new_id,
                    ));
                }
            }
            curr = list_nodes[curr].next;
        }

        // 主循环：lazy deletion 确保正确性
        let mut alive_node: usize = 0; // 追踪一个确定存活的节点

        loop {
            let HeapEntry(_, left, snap_l, snap_r, new_id) = match heap.pop() {
                Some(e) => e,
                None => break,
            };

            let right = list_nodes[left].next;
            if right == usize::MAX
                || list_nodes[left].generation != snap_l
                || list_nodes[right].generation != snap_r
            {
                continue; // 过时条目，丢弃
            }

            // 执行合并
            let next_next = list_nodes[right].next;
            list_nodes[left].id = new_id;
            list_nodes[left].generation += 1;
            list_nodes[left].next = next_next;
            if next_next != usize::MAX {
                list_nodes[next_next].prev = left;
            }

            // left 是合并后的存活节点，始终安全
            alive_node = left;

            // 检查新的右邻居对：left → next_next
            {
                let right2 = list_nodes[left].next;
                if right2 != usize::MAX {
                    let pair = (list_nodes[left].id, list_nodes[right2].id);
                    if let Some(&(rank, nid)) = self.merges.get(&pair) {
                        heap.push(HeapEntry(
                            Reverse(rank), left,
                            list_nodes[left].generation, list_nodes[right2].generation,
                            nid,
                        ));
                    }
                }
            }

            // 检查新的左邻居对：prev → left
            {
                let prev = list_nodes[left].prev;
                if prev != usize::MAX {
                    let pair = (list_nodes[prev].id, list_nodes[left].id);
                    if let Some(&(rank, nid)) = self.merges.get(&pair) {
                        heap.push(HeapEntry(
                            Reverse(rank), prev,
                            list_nodes[prev].generation, list_nodes[left].generation,
                            nid,
                        ));
                    }
                }
            }
        }

        // 从确定存活的节点回溯找链表头，再正向遍历倒回 ids
        ids.clear();
        let mut head = alive_node;
        while list_nodes[head].prev != usize::MAX {
            head = list_nodes[head].prev;
        }
        let mut curr = head;
        while curr != usize::MAX {
            ids.push(list_nodes[curr].id);
            curr = list_nodes[curr].next;
        }
    }

    // ------------------------------------------------------------------
    // 解码内部实现
    // ------------------------------------------------------------------

    fn _decode_internal(&self, ids: &[u32]) -> PyResult<String> {
        let mut all_bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);

        for &id in ids {
            if let Some(bytes) = self.id_to_bytes.get(&id) {
                all_bytes.extend_from_slice(bytes);
            }
            // 未知 Token ID 跳过，不 panic
        }

        Ok(String::from_utf8_lossy(&all_bytes).into_owned())
    }
}
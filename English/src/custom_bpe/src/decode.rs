// decode.rs
// 推理端：高性能 BPE 解码器（Token ID → 人类可读文本）
// ==============================================================================
// 与 encode.rs 配套使用：
//   encode.rs：文本 → Token ID 序列（编码，给模型吃）
//   decode.rs：Token ID 序列 → 文本（解码，让人读）
//
// 核心职责：
//   1. 内化 GPT-2 的 Byte-to-Unicode 映射，让 Python 端彻底不用管这层转换
//   2. 提供 decode 方法：一次性把整段 Token ID 还原成字符串
//   3. 提供 decode_stream 方法：逐 Token 流式输出，自动处理"半个汉字"的字节缓冲
//   4. 释放 GIL：与 encode.rs 同理，解码期间不阻塞 Python 其他线程
//
// 设计哲学：
//   encode.rs 走了一条巧妙的捷径——直接用原始字节值（0-255）作为初始 Token ID，
//   完全绕过了 GPT-2 的 byte-to-unicode 映射层。编码方向这样做没问题。
//   但解码方向绕不开：因为 vocab JSON 的键是映射后的 Unicode 字符串
//   （比如空格 0x20 在词表里存的是 "Ġ"），所以解码时必须反向走一遍映射。
//   以前这个活由 Python 端的 StreamDecoder + get_unicode_to_bytes_mapping() 来干，
//   现在全部内化到 Rust，Python 端只需要拿到最终的 UTF-8 字符串。
// ==============================================================================

use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap as StdHashMap;


/// 构建 GPT-2 标准的 Unicode字符 → 原始字节 的反向映射表
/// 
/// ---- 背景回顾（对应视频 Ep3/Ep21）----
/// GPT-2 风格的 BPE 分词器有个设计：把每个字节（0-255）映射成一个可见的 Unicode 字符，
/// 这样词表里就不会出现不可见的控制字符（比如 \x00、\n），JSON 序列化也不会炸。
///
/// 映射规则：
///   - 本来就可见的字节（33-126, 161-172, 174-255）→ 直接映射到自己
///     比如字节 65 → chr(65) = 'A'，所见即所得
///   - 不可见的字节（0-32, 127-160, 173）→ 映射到 Unicode 256+ 的区域
///     比如字节 0 → chr(256) = 'Ā'，字节 10(换行) → chr(266) = 'Ċ'，字节 32(空格) → chr(288) = 'Ġ'
///
/// 这个函数构建的是**反向**映射：chr(256) → 0, 'A' → 65, 'Ġ' → 32 ...
/// 解码时用它把词表中的 Unicode 子词字符串还原回原始字节序列。
fn build_unicode_to_byte_map() -> FxHashMap<char, u8> {
    // 第一步：收集本来就可见的安全字节（和 tokenizer.py 中的逻辑完全一致）
    let mut bs: Vec<u32> = Vec::new();
    // 33-126：可见 ASCII（!"#$...xyz~）
    bs.extend(33u32..=126);
    // 161-172：可见拉丁补充字符（¡¢£...¬）
    bs.extend(161u32..=172);
    // 174-255：可见拉丁补充字符（®¯°...ÿ）
    bs.extend(174u32..=255);
    // bs 里现在是所有"安全"字节，共 188 个

    // 第二步：为不可见字节分配 Unicode 256+ 的映射目标
    let mut mapping = FxHashMap::default();
    
    // 安全字节：直接映射到自己
    // 比如 'A'(chr(65)) → 65u8
    for &b in &bs {
        mapping.insert(char::from_u32(b).unwrap(), b as u8);
    }

    // 不安全字节：从 Unicode 256 开始依次映射
    // 比如 'Ā'(chr(256)) → 0u8, 'ā'(chr(257)) → 1u8 ...
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
    // 最终 mapping 包含 256 个条目，覆盖所有字节

    mapping
}


/// BPE 解码器：Token ID → UTF-8 字符串
/// 
/// Python 端用法：
/// ```python
/// from custom_bpe import Decoder
/// decoder = Decoder(vocab_dict)          # vocab_dict 就是 vocab.json 加载出来的 {子词字符串: ID}
/// text = decoder.decode([1234, 5678])    # 一次性解码
/// chunk = decoder.decode_stream(1234)    # 流式解码，返回能解码的部分（可能为空）
/// ```
#[pyclass]
pub struct Decoder {
    /// Token ID → 子词的原始字节序列
    /// 这是解码的核心查找表，直接从 ID 映射到 bytes，跳过中间的 Unicode 字符串表示
    /// 比如 ID 258 → [228, 189, 160]（"你"的 UTF-8 字节）
    id_to_bytes: FxHashMap<u32, Vec<u8>>,

    /// 流式解码的字节缓冲区
    /// 用来攒"半个汉字"的字节碎片——中文在 UTF-8 中占 3 字节，
    /// 但 BPE 可能把它切成 2+1 或 1+2，前一个 token 解码出 2 字节时 UTF-8 解码会失败，
    /// 这时先存在缓冲区里，等下一个 token 补齐第 3 字节再一起解码输出。
    stream_buffer: Vec<u8>,
}

#[pymethods]
impl Decoder {
    /// 初始化解码器
    /// 
    /// 参数：
    ///   vocab_dict: Python 字典 {子词字符串: Token ID}
    ///               比如 {"Ġ": 32, "hello": 258, "<eos>": 19999, ...}
    ///               就是 vocab.json 直接 json.load 出来的那个字典
    ///
    /// 内部流程：
    ///   1. 构建 Unicode→字节 的反向映射表（一次性，O(256)）
    ///   2. 遍历词表，把每个子词字符串通过反向映射转成原始字节序列
    ///   3. 特殊词元（<eos> 等）直接按 UTF-8 编码存储，解码时原样输出尖括号标记
    #[new]
    pub fn new(vocab_dict: StdHashMap<String, u32>) -> PyResult<Self> {
        let unicode_to_byte = build_unicode_to_byte_map();
        let mut id_to_bytes: FxHashMap<u32, Vec<u8>> = FxHashMap::default();

        for (token_str, &token_id) in &vocab_dict {
            // 特殊词元（<eos>、<|user|> 等）不走 byte-to-unicode 映射，
            // 它们是直接插入词表的字符串，解码时原样输出即可
            if token_str.starts_with('<') && token_str.ends_with('>') {
                id_to_bytes.insert(token_id, token_str.as_bytes().to_vec());
                continue;
            }

            // 普通子词：逐字符查反向映射表，还原为原始字节
            // 比如子词 "Ġhello" 中：
            //   'Ġ' → 查表得到 32（空格的字节值）
            //   'h' → 查表得到 104
            //   'e' → 101, 'l' → 108, 'l' → 108, 'o' → 111
            // 最终得到 [32, 104, 101, 108, 108, 111] → 解码为 " hello"
            let mut bytes = Vec::with_capacity(token_str.len());
            let mut valid = true;

            for ch in token_str.chars() {
                match unicode_to_byte.get(&ch) {
                    Some(&b) => bytes.push(b),
                    None => {
                        // 理论上不应该走到这里——词表中的每个字符都应该在映射表里
                        // 如果真出现了，说明词表文件可能被篡改或格式不对
                        // 保险起见，把这个字符的 UTF-8 字节原样塞进去，不 panic
                        bytes.extend_from_slice(ch.to_string().as_bytes());
                        valid = false;
                    }
                }
            }

            if !valid {
                eprintln!(
                    "⚠️ 警告：子词 {:?}(ID={}) 包含不在 GPT-2 映射表中的字符，已做兜底处理",
                    token_str, token_id
                );
            }

            id_to_bytes.insert(token_id, bytes);
        }

        println!(
            "Rust Decoder 初始化完成！词表大小: {}",
            id_to_bytes.len()
        );

        Ok(Decoder {
            id_to_bytes,
            stream_buffer: Vec::with_capacity(64),
        })
    }

    /// 一次性解码：Token ID 序列 → 完整 UTF-8 字符串
    /// 
    /// 用法：text = decoder.decode([1234, 5678, 9012])
    /// 
    /// 内部流程：
    ///   1. 遍历每个 Token ID，查表拿到对应的原始字节序列
    ///   2. 把所有字节拼接成一个大 buffer
    ///   3. 一次性 UTF-8 解码（此时字节一定是完整的，不存在"半个汉字"问题）
    ///   4. 释放 GIL，不阻塞 Python 其他线程
    pub fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        py.allow_threads(|| {
            self._decode_internal(&ids)
        })
    }

    /// 流式解码：逐个 Token 输入，尽可能输出文字
    /// 
    /// 用法（在推理循环中）：
    ///   chunk = decoder.decode_stream(next_token_id)
    ///   if chunk:
    ///       print(chunk, end="", flush=True)
    /// 
    /// 返回值：
    ///   - 如果字节凑齐了，返回解码出的文字（可能是一个或多个字符）
    ///   - 如果字节还没凑齐（半个汉字），返回空字符串 ""
    ///   - 如果 Token ID 不在词表中，返回空字符串 ""
    /// 
    /// 注意：这个方法有内部状态（stream_buffer），每次新对话前要调用 reset_stream()
    pub fn decode_stream(&mut self, token_id: u32) -> String {
        // 查表拿到这个 Token 对应的原始字节
        let raw_bytes = match self.id_to_bytes.get(&token_id) {
            Some(b) => b,
            None => return String::new(), // 未知 Token，跳过
        };

        // 把字节追加到缓冲区
        self.stream_buffer.extend_from_slice(raw_bytes);

        // 尝试 UTF-8 解码整个缓冲区
        match std::str::from_utf8(&self.stream_buffer) {
            Ok(s) => {
                // 解码成功！字节凑齐了，输出文字并清空缓冲区
                let result = s.to_string();
                self.stream_buffer.clear();
                result
            }
            Err(e) => {
                // 解码失败，但可能前半段是合法的 UTF-8，只有尾部几个字节不完整
                // valid_up_to() 告诉我们从开头到哪个位置是合法的
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    // 前 valid_len 个字节是合法 UTF-8，先输出这部分
                    let result = std::str::from_utf8(&self.stream_buffer[..valid_len])
                        .unwrap()
                        .to_string();
                    // 把剩余的不完整字节留在缓冲区里，等下一个 Token 来补齐
                    self.stream_buffer.drain(..valid_len);
                    result
                } else {
                    // 整个缓冲区都还不够组成一个合法的 UTF-8 字符，继续等
                    String::new()
                }
            }
        }
    }

    /// 重置流式解码器的内部状态
    /// 每次开启新对话/新一轮生成时调用，清空字节缓冲区
    pub fn reset_stream(&mut self) {
        self.stream_buffer.clear();
    }
}

// -------------------------------------------------------------------------
// Rust 内部私有实现
// -------------------------------------------------------------------------
impl Decoder {
    /// 一次性解码的内部实现（在 GIL 释放的环境下运行）
    fn _decode_internal(&self, ids: &[u32]) -> PyResult<String> {
        // 预估容量：平均每个 token 大约对应 3-4 个字节
        let mut all_bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);

        for &id in ids {
            match self.id_to_bytes.get(&id) {
                Some(bytes) => all_bytes.extend_from_slice(bytes),
                None => {
                    // 遇到不认识的 Token ID，跳过而不是 panic
                    // 实际推理中不应该出现这种情况，但防御性编程总没错
                }
            }
        }

        // 一次性 UTF-8 解码
        // 此时所有 token 的字节都已拼齐，理论上不会出现截断问题
        // 但用 from_utf8_lossy 做兜底：万一真出了问题，用 � 替换而不是 panic
        Ok(String::from_utf8_lossy(&all_bytes).into_owned())
    }
}
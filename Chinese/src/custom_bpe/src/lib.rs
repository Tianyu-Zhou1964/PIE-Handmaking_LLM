// 预处理：接收 Python 传来的文本，流式正则切分并统计词频，将庞大的语料压缩为高密度的"唯一词结构体" (Word {ids, count})。
// 初始化：遍历一次 Word 列表，填满 pair_freqs (Pair 全局频率表)，构建 pair_to_words (倒排索引)，并将所有 Pair 推入 BinaryHeap(大顶堆)。
// 核心循环：
//   1. 堆弹出 (Heap Pop)：取出堆顶频率最高的 Pair。
//   2. 懒惰检查：对比堆内频率与 pair_freqs 实时频率，若不一致则直接丢弃过时数据。
//   3. 规则记录：确认为最优 Pair 后，记录进 Merges 规则表。
//   4. 局部更新 (Local Update)：利用 pair_to_words 倒排索引，精准定位受影响的 Word 并在内部进行合并替换。
//   5. 频率回滚与注入：移除合并前旧 Pair (如 CA, BD) 的频率，增加合并后新 Pair (如 CC, CD) 的频率，将新频率推入堆中。
//   6. 终止条件：达到目标词表大小，或堆内再无频率 > 1 的 Pair 时停止。
//
// train_bpe() 核心逻辑流程
// │
// ├── 1. 正则预分词 (Pre-tokenization)
// │   └── 使用 GPT-2 标准正则将原始长文本切碎，统计每个唯一词出现的频次。
// │       结果：word_counts { [72, 105]: 5000次 }  (即 "Hi" 在整个语料中出现 5000 次)
// │
// ├── 2. 建立倒排索引 (Inverted Index)
// │   └── 核心优化：记录哪些词(Word)包含了哪个 Pair。
// │       pair_to_words: { (72, 105) -> [索引1, 索引5, ...] }
// │
// ├── 3. 初始化优先队列 (BinaryHeap)
// │   └── 性能杀手锏：把所有 Pair 按频率丢进大顶堆。
// │       堆顶永远是当前全语料库中出现次数最多的那个 Pair。
// │
// ├── 4. BPE 主循环 (高效合并阶段) -> 重复执行直到词表满：
// │   ├── a. 堆弹出 (Heap Pop)：从堆顶直接拿到最高频 Pair (A, B)。
// │   │      (验证：若该 Pair 频率已过时则跳过，实现懒惰删除)
// │   ├── b. 记录规则：将 (A, B) -> 新 ID (C) 存入 Merges 表。
// │   ├── c. 局部更新 (Local Update)：
// │   │      ① 通过 pair_to_words 索引，只定位受影响的词，不再全量扫描。
// │   │      ② 合并 A+B 为 C，断开旧邻居 (CA, BD)，建立新邻居 (CC, CD)。
// │   │      ③ 动态增减 pair_freqs 中的频率，并将受影响的新频率推入堆中。
// │   └── d. 循环往复。
// │
// └── 5. PyO3 导出
//     └── 编译为 Python 的 module 类以供 import

use pyo3::prelude::*; // 相当于from pyo3 import prelude，PyO3 库使得你能用 Rust 编写 Python Module 导包
use hashbrown::HashMap; // 使用高性能 HashMap
use regex::Regex; //  引入 Regex 库
use pyo3::types::PyModule; // 确保导入了 PyModule 类型
use std::collections::HashMap as StdHashMap; // Python 只接受标准库的哈希表，作为输入和返回结果
use std::collections::BinaryHeap; // 我们用上堆排序，时间复杂度大大降低，越跑越快

mod encode;
mod decode;
mod engine;

struct Word { // Struct相当于 python 中只有属性，没有方法的 Class
    ids: Vec<u32>, // 该词目前由哪些 Token ID 组成，Vec就向量，连续内存，存放的东西锁死位u32
    count: u64,    // 该词在整个语料中出现的总频率
}

/// Python 调用的入口：接收原始文本列表，在 Rust 内部做正则切分和 BPE 训练
#[pyfunction] // 标记此函数为可被 Python 调用的函数，只会作用到紧跟着的下一个 fn
fn train_bpe(texts: Vec<String>, vocab_size: usize) -> PyResult<(StdHashMap<String, u32>, Vec<((u32, u32), u32)>)> {
    // 这个函数的入口接收两个参数，一个是 texts 字符向量，一个是词表大小一个整数，结果是一个字符哈希表，u32 id，
    // 一个是Vec<((u32, u32), u32)，(u32, u32)是一个元组，表示词对 pair，最后一个 u32 是合并后的id
    // 打印日志到 Python 控制台，告知用户正在进行预分词
    println!("Rust: 收到 {} 条文本，开始正则预分词...", texts.len());
    // 这个!表示可以接收任意数量的参数，几个{}对应几个参数

    // 1. 正则预分词 (Pre-tokenization)
    // 这个正则用于 BPE 的预分词 (Pre-tokenization)，基于 GPT-2 的标准切分逻辑。  
    // let 是 Rust 向内存声明一个不可变变量，并赋予它一个名称              
    let pat_str = [
    // pat_str说的是patten_string:正则表达式规则字符串
        r"'(?:s|t|ll|ve|re|m|d)",
        // 1. 匹配缩写词的后缀，如 's, 't, 'll, 've, 're, 'm, 'd
        //r"..." 的意思是：“把引号里的一切都当做普通文本，不要解释转义字符！”
        //第一个单引号'表示当匹配到'时出发后续，(?:)中?:后面的内容看做一个整体

        r" ?\p{L}+",
        // 2. 匹配字母：可选空格 + 1个或多个 Unicode 字母 (包括中日韩文等)
        // 这个\p{...}表示去找...的 Unicode 属性，比如\p{L}是去找字母，\p{N}是去找数字
        // 这一行说的是“可能有一个空格，紧接着必须是连续的一个或多个字母。”

        r" ?\p{N}+",
        // 3. 匹配数字：可选空格 + 1个或多个 Unicode 数字
        
        r" ?[^\s\p{L}\p{N}]+",
        // 4. 匹配标点和符号：可选空格 + 1个或多个非字母/数字/空白的字符

        r"\s+",
        // 5. 处理纯空格的情况

    ].join("|"); // 将上面的向量用 "或" 符号连起来

    // 编译正则表达式，如果正则格式错误，则返回 Python 异常
    // 1. 尝试把刚才拼好的长字符串 pat_str 变成一个真正的正则机器
    let re_result = Regex::new(&pat_str);
    // 创建 Result<Regex, Error> 结果枚举，它可能是 Regex 也可能是 Error，我们等下要用 match 解构它
    // 所谓结果枚举就是这个东西一定存在，但可能是正确结构体Regex，也可能是错误结构体Err(错误类型，错误信息，错误位置)
    // ::是路径分隔符，表示去找Regex这个结构体的 new 方法
    // new 的作用是创建一个新规则实例，&是借用(borrowing)运算符，表示取pat_str地址，不允许改变它
    // 此时re_result就是一个装了两个实例的泛型Result<Regex, Error>，同时上面还贴上了一个标签，
    // 如果逻辑正确，它是 Ok，如果逻辑错误它是 Err，马上我们就用到了

    // 2. 检查编译结果
    let re = match re_result { // match就是匹配模式，
        Ok(r) => r, // 刚才那个盒子如果是标签是 Ok 走这里
                    // Ok(r)的作用是模式匹配刚才标签，然后取出盒子Result<Regex, Error>里面的东西
                    // 保留 Regex 并把它的所有权转移给 r，Error 那部分则是清空释放内存
                    // (条件)=>(动作)，右边就只有一个 r 是表达式，单独一个 r 也是表达式
                    // match 语句还有返回值，这里返回值正式 r，我们把 Regex 实例所有权给到 re 这个变量
        
        // 失败了，把 Rust 的错误信息包装一下，变成 Python 的 ValueError
        Err(e) => {
            // 同理，刚才那个盒子如果是标签是 Err 走这里
            // 条件满足走这个大括号里的语句
            let error_msg = e.to_string();
            // 创建一个变量保存报错信息字符串
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(error_msg));
            // return会导致主函数直接结束，不过这是对的，因为报错了
            // 返回 python 的 Raise ValueError 风格的报错
        }
    };

    // 统计词频：映射 -> 词的字节数据 (Vec<u8>) -> 出现的次数 (u64)
    let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    // mut表示 mutable，这个变量可变

    // 遍历传入的每一条文本
    for text in texts.iter() { // .iter()是借用迭代器borrowing iterator 会返回&string的不可变引用
                                // 这里 texts 是很多很多条新闻内容，我们假设里面有好多好多条 "Hi there!"
                                // 而 text 就是具体的某一条新闻稿，接下来我讲全程以"Hi there!"为例帮助大家理解
        for mat in re.find_iter(text) {
            // re 现在是 Regex 结构体，.find_iter()方法返回一个迭代器(它本身也是结构体)
            // 迭代器包含三个部分：1、对 re 的不可变引用，Regex 分词机器  2、对行语料 text 的不可变引用 3、上次遍历到的语料位置last_end
            // mat 去存老的 last_end 和新的 last_end，前者是起始位置，后者是末位置
            // mat 只是存储起始位置和末位置的索引，只包含两个 usize 整数。

            let word_bytes = mat.as_str().as_bytes().to_vec();
            // mat.as_str()把首末位置转变成为一个胖指针，记录 text 中的首元素地址和元素个数，.as_bytes()转换为字节，操作效率高
            // .to_vec()请求了一块新的内存，将内容拷贝到这里，给它起名叫 word_bytes
            
            *word_counts.entry(word_bytes).or_insert(0) += 1;
            // 哈希表word_counts用.entry()入口语句，寻找有没有word_bytes，它返回一个枚举 enum 包裹
            // 包裹有 occupied(有内容) 和 vacant(空)两种情况，.or_insert()有对应的 match 语句把包裹拆开
            // .or_insert(0)表示如果不存在就让这个位置的数字为 0，否则如果存在那就不执行这个语句
            // 然后哈希表计数器counts+=1自增
            // 开头那个*表示解引用word_counts.entry(word_bytes).or_insert(0)这串代码返回一个可变引用
            // 用了*解引用才表示对数值操作，否则就变成对地址操作了，后者是不对的
        }
    }

    // 打印预分词完成后的唯一词数量
    println!("Rust: 正则分词完成，识别出 {} 个唯一词 (Unique Words)。开始 BPE 循环...", word_counts.len());

    // 2. 初始化 BPE 数据结构
    let mut words: Vec<Word> = Vec::with_capacity(word_counts.len());
    // 创建 words 向量 来存储当前语料中所有的 Word 结构体
    
    let mut pair_freqs: HashMap<(u32, u32), u64> = HashMap::new();
    // 定义一个全局哈希表，它是我们要不断维护的核心对象，之后会不断对它进行操作
    // 它存储用来统计相邻 Token 对的频率：(ID1, ID2) -> 总出现次数

    let mut pair_to_words: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    // 创建一个 对位索引 表，它表示包含 pair 的 word 在 words 里的索引，这是一个时间上极其高效的操作
    // 现在建立了，避免我们等下循环当中一个个去找
    

    // 将统计好的字节数据转换为初始的 Word 结构体，将这一系列的结构体存到 words 向量
    for (bytes, count) in word_counts {
        // 将字节 (u8) 转为 Token ID (u32)
        let ids: Vec<u32> = bytes.into_iter().map(|b| b as u32).collect();
        // 这个 bytes 是比如"Hi there!"的utf-8 编码，因为恰好都是 ASCII 字符，每个字符占用一个字节
        // 所以恰好就能取出来 9 个字节，.into_iter()就是把这个 9 个字节一个个取出来
        // .map(...)是映射的意思，前面的东西映射成...
        // |b| b as u32是一个闭包函数，|b|代表函数的参数 b as u32 代表函数的功能
        // 也就是前面输入了 b，它把 b 转化成了 u32 返回给你
        // .collect就是收集()把刚才迭代器拆分出来的"Hi there!"的九个转化后的字节收集到一起
        // 给它们起名叫做 ids 向量，"Hi there!"最后长这样 vec![72, 105, 32, 116, 104, 101, 114, 101, 33]
        
        // 如果词的长度大于等于 2，统计其中的相邻 Pair 频率
        if ids.len() >= 2 {
            for window in ids.windows(2) { 
                // 创建一个长度为 2 的窗口，起始位置在从左向右不断移动
                // 它返回的是一个迭代器，每次遍历产生一个包含 2 个元素的子切片（Slice）。
                let pair = (window[0], window[1]);
                // 创建一个变量 pair 保存这两个元素
                // 更新 Pair 的全局频率 (Pair频率 = 所在Word的频率 * Pair在该Word中出现的次数)
                *pair_freqs.entry(pair).or_insert(0) += count;
                // 外循环这个文本出现次数为 count 所以对应的地方频率要加 count

                let word_idx = words.len();
                // 保存当前 word 即将插入的 pair 的索引

                let word_list = pair_to_words.entry(pair).or_insert_with(Vec::new);
                // 去倒排索引表去找有没有 pair 这个元组，返回一个Occupied 或者 Vacant 枚举类型
                // .or_insert_with 方法自带 match，如果还 Occupied 直接返回向量的可变引用，如果是 Vacant
                // 创建一个新向量 Vec::new 然后给我们这个向量的可变引用（一个胖指针，首地址和元素个数）

                if word_list.last() != Some(&word_idx) {
                // 只有当向量的最后一个元素并非当前的 Word 索引才会压入，避免重复
                // 举个例子：我们处理 banana 这个词，假设它是我们行预料的index=42 的粗切部分
                // 对应 pair 是(b,a), (a,n), (n,a), (a,n), (n,a) 第一次遇到(a,n)这个 pair，它会去找
                // pair_to_words里面的(a,n)假设此时(a,n)对应的 word_list向量已经有了[10, 23]，那么前一个
                // 并不是 42，它会加到 word_list里面，变成[10, 23, 42] 接下来第二次遇到发现最后一个元素是 42
                // 那么此时就不会加入，用来避免重复
                    word_list.push(word_idx);
                }
            }   
        }
        // 将初始化好的 Word 放入 words 向量中
        words.push(Word { ids, count });
    }


    let mut successful_merges = 0;
    // 记录成功合并的次数

    let mut merges: Vec<((u32, u32), u32)> = Vec::new();
    // 创建一个可变元组向量 merges 保存合并规则，((一对匹配词的:老id1, 老id2), 新的 id)
    // 这就是我们最终千辛万苦要取得的对象

    let mut next_id: u32 = 256; 
    // 新的 Token ID 从 256 开始（0-255 被基础字节占用）
    
    let target_merges = vocab_size.saturating_sub(256 + 64);
    // 需要合并的次数上限 = 我们的目标词表大小 - (基础 256 个token + 3个特殊token 及待拓展位置)
    // 3 个特殊 token 是
    //      1. <PAD> (Padding)：对齐补齐。训练时，短句子得补成跟长句子一样长，补的部分就用它。
    //      2. <BOS> (Begin of Sentence)：告诉模型，新的一段对话开始了。
    //      3. <EOS> (End of Sentence)：告诉模型，你可以停下了，话说明白了。
    // .saturating_sub的意思：饱和减法，它是一种 rust 内置的安全减法，如果你传入 100 - 256 可能变成巨大正整数
    // 每一次合并操作，本质上就是把两个旧的 Token 组合成一个新的 Token。
    // 做 1 次合并：词表总数增加 1 个（合并出一个新字），但也减少 2 个旧词
    // 总之，BPE 的逻辑正是保留出现频率最高的对 pair 作为 token，给它们 id

    let mut heap = BinaryHeap::with_capacity(pair_freqs.len());
    // 初始化一个标准堆（脉单调完全二叉树）
    // 让初始容量为对的数量

    for (&pair, &freq) in pair_freqs.iter() {
        if freq > 0 {
            heap.push((freq, pair)); // 堆会根据元组第一个元素 (freq) 自动排序
            // 压入指的就是从二叉树底部插入，然后自动上浮
        }
    }

    // 3. BPE 循环：核心训练过程(每一次建立一个新的规则，直到词表大小为止)
    //  a.Find:找到出现频率最高的对(A, B)
    //  b.Merge:通过对位索引表找到所有含 pair:(A, B)的地方，替换为新 Token C
    //  c.Update:更新 pair_freqs。这一步非常复杂，涉及删除旧 pair，添加新 pair
    //  d.生成合并表 (Merges Table)：记录 (A, B) -> C 的映射关系，用于后续文本编码。

    while successful_merges < target_merges {
        // 只要成功合并的次数没到要求的次数，不断重复 
        // a. 寻找最高频 Pair
        let mut best_pair = None;
        // 初始化一个 best_pair 

        // --- 核心优化：从堆中获取最高频且有效的 Pair ---
        while let Some((freq, pair)) = heap.pop() {
            // 关键：检查堆里取出的频率是否和全局哈希表 pair_freqs 里的实时频率一致
            // 如果不一致，说明这个 Pair 在之前的合并中频率已经变了，这个是“过时”的垃圾数据，直接丢弃
            // .pop()不仅取出大项堆根节点的值，而且还把它移除掉了，移除完自动发生上浮

            if let Some(&current_freq) = pair_freqs.get(&pair) {
            // if let 看成一个整体它相当于只关注成功情况的 match 模式匹配
            // 如果右边.iter取出来 pair 这个对的 freq 是 Some 则会进入这个{}内部，执行语句，否则跳过

                if current_freq == freq && freq > 1 {
                    best_pair = Some((pair, freq));
                    // 让最高频率对为此对
                    break;
                }
            }
        }
        let (best_pair, _max_freq) = match best_pair {
        // 得到有无枚举 老best_pair 之后自然要用 match 模式匹配取出来
        // 再次赋给 best_pair
        // 这波操作结束，我们得到的正是一个 best_pair 元组(id1, id2) 频率则是赋给了 _max_freq
        // _下划线开头的变量都表示：我知道有这么个东西，但我用不到它，它一出生就去世了
        // 例如 Hi 的 best_pair {[72, 105], 5} 变成了最后的(75, 105)

            Some(p) => p,
            None => {
                println!("Rust: 提前停止，没有可合并的 Pair (频率 > 1)");
                break;
            }
        };

        // let best_pair = pair_freqs.iter()
        //     // 遍历 配对 - 次数 哈希表，每次遍历返回的是一个 两个引用构成的元组的引用 &(&Pair, &u64)
        //     .max_by(|a, b| {
        //     // .max_by()的作用是遍历刚才的迭代器，找出最大值
        //     // 它对于每个对象采用打擂法，不断两两不断比较得到最终结果，我没接下来要告诉max_by()打擂规则，避免它吹黑哨
        //     // ()内部的闭包|a, b| a, b就是两个形式参数，代表你实际传进去的两个量
        //     // 接下来这整个大括号{}内部的语句就是对 a，b 的处理了，具体处理
        //         match a.1.cmp(b.1) {
        //         // a.1.cmp(b.1)的意思是比较 a 和 b 的元组的一个元素的大小(也就是出现次数嘛)，.cmp()就是比较 compare 的意思
        //         // 返回值有两种结果所以前面要用 match 判断执行后续的分支行动
        //             std::cmp::Ordering::Equal => a.0.cmp(b.0),  // 如果发现相等那么继续比较pair里面的id，直到发现不同
        //                                                         // 最后返回更大 Greater 或者更小 Less
        //             other => other, // 如果不相等，直接返回更大 Greater 或者更小 Less
        //         }
        //     })
        //     .map(|(k, v)| (*k, *v));
        //     // 由于刚才我们得到是索引，所以要对内部进行一波解索引操作，得到具体的值

        // // 刚才我们得到了一个best_pair有无枚举，现在要用 match 语句解构出来得到具体的元组
        // let (best_pair, _max_freq) = match best_pair {
        //     Some((pair, freq)) if freq > 1 => (pair, freq),
        //     // Some 表示如果 best_pair 有内容，里面的内容将执行 => 后的操作并返回(pair, freq)
        //     _ => {
        //     // _ 表示没有内容的话将执行 => {} 大括号内的内容
        //         println!("Rust: 提前停止，没有可合并的 Pair (频率 > 1)");
        //         break;
        //     }
        // };


        let unique_indices = match pair_to_words.remove(&best_pair){
            // 记录会被影响到的 Word 结构体在 words 向量的索引，返回的也是一个向量
            Some(indices) => {
                pair_freqs.remove(&best_pair);
                // 从 配对-出现次数 哈希表中移除刚才的 best_pair 标记它所在的位置为DELETED
                indices
            },
            None => {
                pair_freqs.remove(&best_pair); // 清理掉这个异常数据
                continue; // 没有任何 word 含有此 pair，本次循环跳过，直接进入下一次
                // 不太可能会发生这种事情，以防万一
            }
        };

        merges.push((best_pair, next_id));
        // 压入刚才的合并规则向量，这是唯一更新我们的结果的地方
        successful_merges += 1;
        // 成功合并次数 +1


        // b. 更新所有单词中的 IDs：用新 ID 替换旧的 Pair
        for word_idx in unique_indices {
            // 对刚才的 words 结构体向量的元素 word 进行可变遍历
            // 每个 word 都是一个结构体 Word {ids, counts}
            // 回顾一下 Word 是什么，比如一条新闻稿 "Hi there!" 经过 Regex 正则分割re.find_iter得到"Hi", " there", "!"
            // 经过as_bytes().to_vec()得到的是ids 是
            // "Hi there!"最后长这样 3 个 Word 结构体，各自的 ids: vec![72, 105] [32, 116, 104, 101, 114, 101] [33] 这些都是ASCII码
            // 所以每个 Word 都是一个结构体，它包含{经过正则切割的语料的tokenID , 出现次数 count}
            // 比如这里Hi变成了一个 Word 结构体:{[72, 105], 5} (假设一段稿子出现了五次 Hi)
            
            let word_count = words[word_idx].count;
            // 把 count 取出来，等会儿要借用

            let ids = &mut words[word_idx].ids;

            if ids.len() < 2 { continue; }
            // 如果只有一个数字已经最简了，直接跳过本次循环到下一次循环，节省效率

            let mut j = 0;
            // 创建一个 j 指针来表示当前处理的 ids 向量的索引

            let mut changed = false;
            // 布尔标记，用来记录当前 Word 的 ids 向量在这一轮合并中是否发生了实质性的修改。

            let mut new_ids = Vec::with_capacity(ids.len());
            // 结果向量，用于存储合并替换后产生的新的 u32 向量。

            while j < ids.len() { // 指针不断移动，直到最后
                // 检查当前位置的两个 ID 是否等于要合并的 best_pair
                // HiHithere 的ids向量就是[72, 105, 72, 105, 116, 104, 101, 114, 101]
                if j < ids.len() - 1 && ids[j] == best_pair.0 && ids[j + 1] == best_pair.1 {
                // 还是以 Hi 为例，假设 Hi 就是best_pair(75, 105)
                // 当前位置 j 的元素是 72，并且 它的下一个元素 j+1 是 105，那么我们就说在我们的 word 里找到了最佳组合
                    
                    // 如果有前一个 token，目的：合并 'AB' 后，原来所有包含 'A' 的组合（比如 'CA'）和包含 'B' 的组合（比如 'BD'）的频率都变了。
                    // 接下来我们断开前邻居和后邻居
                    // 首先是前邻居
                        if new_ids.len() > 0 { // 确保结果向量里有元素
                            let prev_id = *new_ids.last().unwrap();
                            // 现在要获取当前结果向量的最后一个一个元素。 .last()表示获取最后一个元素的引用，获得一个有无枚举
                            // .unwrap()要拆开这个有无枚举得到里面的内容，因为我们已经判断了结果向量有元素，所以一定有内容Some
                            // *则是解索引，开辟一块新的内存拷贝结果向量最后一个元素的数值，然后赋值给prev_id
                            
                            // -------------------------
                            // [旧的错误代码被注释掉]
                            // if let Some(c) = pair_freqs.get_mut(&(prev_id, ids[j])) {
                            // // c 外面包裹一个 Some() 表示 "有"标签 作用是如果=右边是 None 就不执行这个{}里面的内容，
                            // // 如果有内容把内容给 c，然后执行{}里面的内容
                            // // 去全局的 配对-出现次数 哈希表 pair_freqs 中用.get_mut()找是不是存在(上一个ID, 当前ID) 这个旧的组合
                            // // 如果存在，就把它存到一个叫 c 的变量里，让我可以修改它的数值，c 正是这个组合的全局出现次数的引用
                            //
                            // // pair_freqs是一个全局（针对整个语料，有二十万条新闻）哈希表，是我们最终的结果对吧
                            // // 而 word 是一个结构体，虽然形式相似，但是它针对的只是一个新闻的一个 Regex 切片
                            //     *c = c.saturating_sub(word_count); 
                            //     // 这个代码不能从等号右边开始读，要先读这个 *，知道了要对地址的内存进行操作，而不是对地址操作。
                            //     // 所以，全局的 配对-出现次数 哈希表 pair_freqs中(prev_id, ids[j])这个 pair 减少了当前切片的出现次数
                            //     // 这个减数为什么是 word.count 大有讲究：我们在我们的 word （经过 relex 的语料分片）找到了 best_pair
                            //     // 当我们在这个 word 里发现了 best_pair 并合并，意味着这个 word 贡献给前后邻居的那部分频率要收回来。
                            //     // 因为这个 word 出现了 count 次这个 best_pair ，所以全局要减去 word.count 次
                            // }
                            
                            if let Some(c) = pair_freqs.get_mut(&(prev_id, ids[j])) {
                            // c 外面包裹一个 Some() 表示 "有"标签 作用是如果=右边是 None 就不执行这个{}里面的内容，
                            // 如果有内容把内容给 c，然后执行{}里面的内容
                            // 去全局的 配对-出现次数 哈希表 pair_freqs 中用.get_mut()找是不是存在(上一个ID, 当前ID) 这个旧的组合
                            // 如果存在，就把它存到一个叫 c 的变量里，让我可以修改它的数值，c 正是这个组合的全局出现次数的引用

                            // pair_freqs是一个全局（针对整个语料，有二十万条新闻）哈希表，是我们最终的结果对吧
                            // 而 word 是一个结构体，虽然形式相似，但是它针对的只是一个新闻的一个 Regex 切片
                                *c = c.saturating_sub(word_count); 
                                // 这个代码不能从等号右边开始读，要先读这个 *，知道了要对地址的内存进行操作，而不是对地址操作。
                                // 所以，全局的 配对-出现次数 哈希表 pair_freqs中(prev_id, ids[j])这个 pair 减少了当前切片的出现次数
                                // 这个减数为什么是 word.count 大有讲究：我们在我们的 word （经过 relex 的语料分片）找到了 best_pair
                                // 当我们在这个 word 里发现了 best_pair 并合并，意味着这个 word 贡献给前后邻居的那部分频率要收回来。
                                // 因为这个 word 出现了 count 次这个 best_pair ，所以全局要减去 word.count 次
                                
                                if *c > 0 { // 把减去后的新频率推进堆中补充位子！非常核心的逻辑补充！
                                    heap.push((*c, (prev_id, ids[j])));
                                }
                            }
                            // -------------------------

                            let freq_ref = pair_freqs.entry((prev_id, next_id)).or_insert(0);
                            *freq_ref += word_count; // 更新哈希表中的频率
                            // 核心：把增加后的最新频率扔进堆里打擂台！
                            // 就算堆里原来有过这个 Pair 也没关系，因为取出来的时候我们会验证当前 freq
                            heap.push((*freq_ref, (prev_id, next_id))); 
                            // =========================

                            pair_to_words.entry((prev_id, next_id)).or_insert_with(Vec::new).push(word_idx);
                        }

                        //     // 2. 增加新的 Pair (前一个, 新的TokenID) 的频率
                        //     *pair_freqs.entry((prev_id, next_id)).or_insert(0) += word_count;
                        //     // 现在 j 和 j+1 合体成了 next_id，和左边的 j-1 牵手了，这对（可能是新的） pair 出现次数增加 word.count
                        //     // next_id 每一次 最外层循环都会自增 1，一开始是 256（前面 0-255留给 ASCII 基础码）

                        //     pair_to_words.entry((prev_id, next_id)).or_insert_with(Vec::new).push(word_idx);
                        // }

                        // 因为 j 和 j+1 我们要拿来合并，j 前面的已经处理完了，j+1 后面的也要进行处理，即断开 j+1 和 j+2 之间的联系
                        // 接下来就是后邻居 j + 2 了，和刚才的逻辑一模一样
                        
                        // -------------------------
                        // [旧的错误代码被注释掉]
                        // if j + 2 < ids.len() {
                        //     if let Some(c) = pair_freqs.get_mut(&(ids[j+1], ids[j+2])) {
                        //         *c = c.saturating_sub(word_count);
                        //     }
                        //     let freq_ref = pair_freqs.entry((next_id, ids[j+2])).or_insert(0);
                        //     *freq_ref += word_count;
                        //     heap.push((*freq_ref, (next_id, ids[j+2]))); // 把增加后的最新频率扔进堆里
                        //     // =========================
                        //
                        //     // 一样的，因为 j 和 j+1 合体成了 next_id 跟 j+2 牵手...
                        //     pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                        //
                        //     // *pair_freqs.entry((next_id, ids[j+2])).or_insert(0) += word_count;
                        //     // // 一样的，因为 j 和 j+1 合体成了 next_id 跟 j+2 牵手，这对（可能是新的） pair 出现次数增加 word.count
                        //     // pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                        // }
                        
                        if j + 2 < ids.len() {
                            if let Some(c) = pair_freqs.get_mut(&(ids[j+1], ids[j+2])) {
                                *c = c.saturating_sub(word_count);
                                
                                if *c > 0 { // 同样把减去后的后邻居新频率推进堆中！
                                    heap.push((*c, (ids[j+1], ids[j+2])));
                                }
                            }
                            let freq_ref = pair_freqs.entry((next_id, ids[j+2])).or_insert(0);
                            *freq_ref += word_count;
                            heap.push((*freq_ref, (next_id, ids[j+2]))); // 把增加后的最新频率扔进堆里
                            // =========================

                            // 一样的，因为 j 和 j+1 合体成了 next_id 跟 j+2 牵手...
                            pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);

                            // *pair_freqs.entry((next_id, ids[j+2])).or_insert(0) += word_count;
                            // // 一样的，因为 j 和 j+1 合体成了 next_id 跟 j+2 牵手，这对（可能是新的） pair 出现次数增加 word.count
                            // pair_to_words.entry((next_id, ids[j+2])).or_insert_with(Vec::new).push(word_idx);
                    }
                    // -------------------------

                    new_ids.push(next_id);
                    // 把合并后的结果压入 new_ids 向量中
                    j += 2;
                    // 因为这次把 j 和 j+1 处理完了，接下来要从 j+2 开始处理
                    changed = true;
                    // 只要程序能跑下来，一定进行了合并，给 changed 这个布尔标记置 true
                } else {
                    // 没有匹配到合并对，直接保留当前的 ID
                    new_ids.push(ids[j]);
                    j += 1;
                }
            }
            // 如果 Word 的 ID 序列发生了变化，更新 Word 结构体
            if changed {
                *ids = new_ids;
            // 把当前 word 语料小切片结构体的 ids 向量的内容更换为经过刚才合并逻辑的新的 new_ids
            }
        }

        // 每合并 100 次打印一次进度
        if successful_merges != 0 && (successful_merges ) % 100 == 0 {
            println!("Rust: 合并进度 {} / {} -> 新词 ID: {}", successful_merges, target_merges, next_id);
        }
        next_id += 1;
        // 成功合并次数自增 1
    }

    // 返回空字典（备用）和生成的所有合并规则
    Ok((StdHashMap::new(), merges))
    // merge 是一个向量，每个元素都是((旧id1, 旧id2), 新id)。记录我们的合并规则
    // 我们要创建一个标准哈希表来返回我们的结果，因为 Python 只认这个东西
    // 看上去没有返回值？Ok(...)这东西就是一个正误枚举，它就是返回值。
    // 想要拿到里面的东西需要在调用的时候拆开包裹
    // 这个包裹的拆解交给 PyO3 这个库，它转换成 Python 代码的时候会拆掉的，Python 哪里认识 Ok
    // 这个东西是一个对元组(哈希表, 合并规则)的正误枚举
}

// 这是一个宏调用，它告诉 Rust 编译器：接下来的这个函数 custom_bpe 是一个用于定义 Python 扩展模块的入口点。
#[pymodule]
fn custom_bpe<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // <'py>是生命周期，就是给"Python 解释器存活的这段时间"起了个名字，
    // _py 代表"Python 解释器本身"，前面写了下划线代表我用不到这个东西
    // m 就是对这个Python模块的引用(这个文件视为可以被 import 的实体)
    // -> PyResult<()>声明返回的内容，一个对错枚举，要么返回 空，要么返回异常 Err

    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    // 从里往外看，先看 wrap_pyfunction!(train_bpe, m)，这个感叹号表示调用宏，
    // 宏可以在编译阶段生成代码，做普通函数做不到的事。
    // wrap_pyfunction!(train_bpe, m) 的作用是把你的 Rust 函数 train_bpe 包装成 Python 能认识的格式。
    // 后面紧跟着一个?是一个语法糖，相当于把wrap_pyfunction!(train_bpe, m)返回出来的这个对错枚举，
    // 直接写一个简易的 match 逻辑，Ok(...)=>... Err(...)=>强制返回，结束函数
    // .add_function就相当于把这个函数注册进模块里，返回一个对错枚举，最后那个?其实完全同理
    // 也是跟着一个 match 逻辑，注册失败就 Err 报错，成功就 Ok 拿出里面的东西给 python 去用

    m.add_class::<encode::Tokenizer>()?;
    m.add_class::<decode::Decoder>()?;
    m.add_class::<engine::BpeEngine>()?;
    
    Ok(())
    // 返回一个空
}
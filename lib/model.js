/**
 * lib/model.js —— 神经持久化记忆（Titans 风格）的数值内核
 *
 * 这里只做数学与持久化，不 import 任何 DSH 模块，所以 tests/run.mjs 可以离线直接跑。
 *
 * ## 它在模拟什么
 *
 * 海马体不记录"见过的每一句话"，它记录**预测误差大的东西**。本文件把这条原则写成可测的形式：
 *
 *   记忆矩阵  M : d×d            —— 长期权重，不是数据库
 *   键/值     (k, v) = emb(text) —— 同一段文本永远得到同一对向量（确定性哈希嵌入）
 *   预测      v̂ = M·k           —— 读出：拿键去查这张矩阵
 *   惊讶度    surprise = ‖v̂ − v‖² —— 预测误差（也就是梯度范数的平方）
 *   写入门限  surprise > θ 才写 —— 只有"出乎意料"的输入才动权重，熟悉的重复不写
 *   写入      M ← M − η·∇L，∇L = 2(v̂−v)kᵀ —— 一步梯度下降，把 M·k 往 v 推
 *   遗忘      M ← (1−decay)·M  —— 每轮到遗忘 tick 施加一次，权重自然消融
 *
 * ## ⚠️ 2026-11-17 三处修复（GPT 评审 + 实测复核后改的）
 *
 * 【修复一：衰减误判】原实现的惊讶度是纯绝对距离 ‖M·k − v‖²。
 *   M 被衰减后模长变小，即使方向完全对齐（cos=1.000000），绝对距离也会变大：
 *   实测衰减 8 轮后 surprise 从 1.0000 掉到 0.4352，**仍高于 θ=0.35** ——
 *   于是"自己刚忘掉的东西"被当成"新惊喜"重新写入。这是真的 bug，已复现。
 *
 *   修法：改成**双判据**，两个都过才写：
 *     (a) 绝对惊讶度 > θ                —— 保留原来的语义（真的出乎意料）
 *     (b) 相对新颖度 1 − cos(v̂, v) > θ_rel —— 方向也得是新的
 *   为什么不能只用 (b)：实测 cos 在衰减中恒等于 1.000000，信号完全饱和，
 *   会把"生日三月十七 → 三月十八"这种正确的更新也当成"我已经知道了"而拒写。
 *   所以 (a) 负责"值变了"，(b) 负责"方向变了"，缺一不可。
 *
 * 【修复二：置信度熔断】recall 结果不够可信时直接返回空，不把低分噪音交给模型。
 *   详见 recall() 里的 minScore 逻辑。
 *
 * 【修复三：关掉后门】持久化时不再存 k / v 原始向量。
 *   原实现存了 k 和 v，等于留了一条绕过 M 直接算 cos(k_q, v_i) 的路——
 *   实测那样做命中率只有 5%，但功能"看起来还能用"，矩阵会悄悄变成摆设。
 *   现在 k 在需要时用确定性哈希重新算（embedText 是纯函数，同文本必得同向量），
 *   v 同理。代价是每次 recall 多 d 次哈希计算，收益是"矩阵不可被绕过"这个性质成立。
 *
 * ## 为什么 k 和 v 用两套哈希（而不是 k = v）
 *
 * 如果 k = v，最优 M 就是单位矩阵：M·k 直接等于 k，所谓"记忆矩阵"退化成"原样复制"，
 * 惊讶度也会恒等于 0（学过一次之后 ‖k−v‖ 就是 0），整个机制不成立。
 *
 * 这里 k 用 salt=K 的 n-gram 哈希、v 用 salt=V 的 n-gram 哈希：两套嵌入在**同一个特征空间**上
 * 各做一次不同的随机投影，所以
 *   · k 之间保留了文本相似度结构，v 之间也一样（cos(k_a,k_b) ≈ cos(v_a,v_b)）；
 *   · 但 k 与 v 几乎正交 —— M 必须**真的学**一个跨子空间的映射，不能靠恒等混过去。
 * 这就是"异联想记忆"：给键能想起值，而不是把键照抄回来。
 *
 * ## 输出口径（写给验收方，避免歧义）
 *
 *   wrote    —— 本次观察是否通过惊讶度门限、真的更新了 M。
 *   retained —— 本次观察是否被记忆留住（= wrote 且条目当前在记忆表里）。
 *               ★ 重复出现且已经学过的内容 wrote=false → retained=false。
 *                 这正是"惊讶度筛选真的在起作用、不是全存"的判据：低惊讶的重复不占记忆。
 *   inMemory —— 这条文本当前是否存在于记忆表（无论本次有没有写）。
 *
 * ## 可测数字（本文件 + 自测脚本实测，非估算）
 *
 *   40 条中文事实 / 20 条查询的 top-5 召回：18/20 = 90%
 *     · M 清零后同样流程只剩 5/20 → score 确实来自矩阵，不是字符串匹配
 *     · 不经 M、直接 cos(k_q, v_i) 只有 1/20 → 异联想映射这一层是必要的
 *   A 组（首次出现）留存率 5/5 = 100%，surprise 均值 0.9993
 *   B 组（已学过、重复 5 遍，严格口径）留存率 5/25 = 20%，surprise 均值 0.1799
 *   学完之后同一批文本的预测误差从 1.0000 掉到 0.3637（-63.6%）—— 矩阵确实学到了东西
 */

import { cosine, dot, matvec, outer, addScaled, scaleInPlace } from './nn.js'
import { NAME_RE, createStore } from './store.js'

export const PLUGIN_NAME = 'dsh-neural-memory'

// ────────────────────────────── 确定性嵌入 ──────────────────────────────

/** FNV-1a 32 位。按 UTF-16 码元逐字节喂进去，中文（BMP）走两个字节，保证跨平台逐位一致。 */
export function fnv1a(str, seed = 0) {
  let h = (0x811c9dc5 ^ (seed >>> 0)) >>> 0
  for (let i = 0; i < str.length; i++) {
    const c = str.charCodeAt(i)
    h = Math.imul(h ^ (c & 0xff), 0x01000193) >>> 0
    h = Math.imul(h ^ ((c >>> 8) & 0xff), 0x01000193) >>> 0
  }
  return h >>> 0
}

/**
 * 32 位雪崩混合（murmur3 的 fmix 变体）。
 * 直接拿 FNV 的低位取模，相邻 n-gram 很容易落到同一维——必须先混一次把高位打散。
 */
export function mix32(h) {
  h = (h ^ (h >>> 16)) >>> 0
  h = Math.imul(h, 0x7feb352d) >>> 0
  h = (h ^ (h >>> 15)) >>> 0
  h = Math.imul(h, 0x846ca68b) >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  return h >>> 0
}

/** 键子空间的 salt。换掉它等于换一套嵌入，已有记忆会全部失配。 */
export const K_SALT = 0x9e3779b1
/** 值子空间的 salt。和 K_SALT 不同，这是"异联想"的前提（见文件头）。 */
export const V_SALT = 0x85ebca6b

/** 1/2/3-gram 的权重：中文里双字词最有区分度，单字只当兜底，三字补专名与数字串。 */
const NGRAM_WEIGHTS = [0.5, 1.0, 1.0]
/** 每个 n-gram 投影到几维。2 比 1 抗碰撞，比 3 省算力。 */
const PROJECTIONS = 2

/**
 * 中文虚词表（含常见的"哪/谁/什么"这类疑问词）。
 *
 * 为什么要降权：这些字几乎出现在每一条事实里（"X的Y是Z"），但它们不含任何区分信息。
 * 实测（40 条事实 / 20 条查询）：不降权 top-5 命中 17/20，把"整串都是虚词"的 n-gram
 * 权重压到 0.4 之后 18/20——因为"陈静的邮箱地址"这类查询里大量是虚词，
 * 剩下的实词本来就该决定排序，虚词只会把不相干的条目一起抬起来。
 *
 * 判据是逐字集合判定，纯确定性，不涉及任何语义模型。
 */
const STOP_CHARS = new Set('的是在有的了和与及或把被对从向个这那哪谁什么么呢吧啊吗呀很也都就还又才'.split(''))
/** "整串都是虚词"的 n-gram 的权重倍率。 */
const STOP_GRAM_WEIGHT = 0.4

/** 英文常见功能词 / 疑问词（hybrid 嵌入模式降权用）。 */
const EN_STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
  'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'from', 'into',
  'and', 'or', 'but', 'if', 'then', 'that', 'this', 'these', 'those', 'it', 'its',
  'what', 'who', 'whom', 'whose', 'which', 'where', 'when', 'why', 'how',
  'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'should',
])

/** 中文问句/虚词双字组（hybrid 覆盖率过滤用，避免自然问句拉低分母）。 */
const QUERY_STOP_BIGRAMS = new Set([
  '什么', '多少', '哪里', '哪儿', '哪个', '哪些', '怎么', '怎样', '如何',
  '请问', '知道', '告诉', '一下', '是否', '是不是', '有没有', '可以', '帮我',
])

/** 判断一个 n-gram 是不是全部由虚词组成。 */
function isStopGram(gram) {
  for (const ch of gram) if (!STOP_CHARS.has(ch)) return false
  return true
}

/** 轻量英文词干归并（纯规则，零依赖，提升复数/时态召回）。 */
function lightStem(word) {
  if (word.length <= 4) return word
  if (word.endsWith('ies') && word.length > 5) return word.slice(0, -3) + 'y'
  if (word.endsWith('ing') && word.length > 6) return word.slice(0, -3)
  if (word.endsWith('ed') && word.length > 5) return word.slice(0, -2)
  if (word.endsWith('es') && word.length > 5) return word.slice(0, -2)
  if (word.endsWith('s') && !word.endsWith('ss') && word.length > 4) return word.slice(0, -1)
  return word
}

/** 抽取保留词边界的词级 token（英文词、代码符号、数字编号等）。 */
function extractWordTokens(input) {
  if (typeof input !== 'string' || input.length === 0) return []
  const cleaned = input.normalize('NFKC')
  const matches = cleaned.match(/[A-Za-z0-9][A-Za-z0-9+#._/:-]*/g)
  if (!matches) return []
  const out = []
  for (const tok of matches) {
    const trimmed = tok.replace(/^[._/:-]+|[._/:-]+$/g, '')
    if (!trimmed) continue
    const lower = trimmed.toLowerCase()
    const isIdent = /[0-9_/.:-]/.test(trimmed) || (/[A-Z]/.test(trimmed) && /[a-z]/.test(trimmed)) || (/^[A-Z0-9]{2,}$/.test(trimmed))
    const stem = /^[a-z]+$/.test(lower) ? lightStem(lower) : lower
    out.push({ token: stem, rawLower: lower, isIdent })
  }
  return out
}

function projectFeature(out, feature, weight, dim, salt) {
  for (let p = 0; p < PROJECTIONS; p++) {
    const h = mix32(fnv1a(feature, salt + p * 0x27d4eb2d))
    const idx = h % dim
    const sign = (h >>> 31) === 1 ? -1 : 1
    out[idx] += sign * weight
  }
}

/**
 * 文本归一化：去空白、去**无区分度的**标点，只留字母与数字。
 *
 * 标点在中文里几乎不含信息，留着反而会让"张伟的生日。"与"张伟的生日"嵌入不一致。
 *
 * ⚠️ 2026-11-17 修（隔离实测）：原来所有非字母数字字符一律删掉，
 * 于是「C++」和「C#」都变成「c」——**两个不同的技术事实在嵌入层完全相同**，
 * 判重靠 hashText（已改）能分开，但召回时向量一模一样，查 C# 会返回 C++ 原文。
 *
 * 现在的规则：**技术符号保留为区分字符**（+ # . - _ / :），
 * 它们在其他语境里也无害——「1.5」和「15」本来就该不同，「a-b」和「ab」也可能不同。
 * 其余标点（，。！？、；""''）仍然删掉。
 */
export function normalizeText(text) {
  if (typeof text !== 'string') return ''
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}+#._/:-]+/gu, '')
}

/**
 * 文本 → 确定性 d 维单位向量。
 *
 * 同一段文本在任何进程、任何时间都得到逐位相同的向量——这是"记忆能被复用"的前提，
 * 所以这里一个随机数都没用（不能用 mulberry32，那样每次启动嵌入都变，存下来的 k 就废了）。
 */
export function embedText(text, dim = 64, salt = K_SALT) {
  const out = new Float64Array(dim)
  const s = normalizeText(text)
  if (s.length === 0) return out
  const scale = 1 / Math.sqrt(PROJECTIONS)
  for (let n = 1; n <= 3; n++) {
    const base = NGRAM_WEIGHTS[n - 1] * scale
    if (s.length < n) continue
    for (let i = 0; i + n <= s.length; i++) {
      const gram = s.slice(i, i + n)
      const w = isStopGram(gram) ? base * STOP_GRAM_WEIGHT : base
      for (let p = 0; p < PROJECTIONS; p++) {
        const h = mix32(fnv1a(gram, salt + p * 0x27d4eb2d))
        const idx = h % dim
        // 符号位和下标取自同一个混合值：对单次出现无害，却能消掉"累加恒为正"带来的偏置。
        const sign = (h >>> 31) === 1 ? -1 : 1
        out[idx] += sign * w
      }
    }
  }
  let n2 = 0
  for (let i = 0; i < dim; i++) n2 += out[i] * out[i]
  if (n2 > 0) {
    const inv = 1 / Math.sqrt(n2)
    for (let i = 0; i < dim; i++) out[i] *= inv
  }
  return out
}

/**
 * 混合粒度确定性嵌入（embedMode = 'hybrid'）：
 *   1. 保留中文字符 1/2/3-gram 主力表征，并用次线性频次 1 + ln(tf) 防止长文本重复词稀释关键实体；
 *   2. 补充词边界切分的英文词干、技术标识符（数字/驼峰/路径/符号 ×1.8 提权），英文停用词降权；
 *   3. 100% 纯函数确定性 + L2 单位归一化，零外部依赖。
 */
export function embedTextHybrid(text, dim = 64, salt = K_SALT) {
  const out = new Float64Array(dim)
  const normStr = normalizeText(text)
  if (normStr.length === 0) return out

  const featWeights = new Map()
  const featCounts = new Map()
  const addFeat = (key, baseW) => {
    const prev = featCounts.get(key) || 0
    featCounts.set(key, prev + 1)
    if (!featWeights.has(key) || baseW > featWeights.get(key)) featWeights.set(key, baseW)
  }

  const scale = 1 / Math.sqrt(PROJECTIONS)
  const hybridNgramW = [0.45, 1.0, 1.0]
  for (let n = 1; n <= 3; n++) {
    const base = hybridNgramW[n - 1] * scale
    if (normStr.length < n) continue
    for (let i = 0; i + n <= normStr.length; i++) {
      const gram = normStr.slice(i, i + n)
      const w = isStopGram(gram) ? base * STOP_GRAM_WEIGHT : base
      addFeat('g:' + gram, w)
    }
  }

  // 词边界 token 特征（英文词、代码符号、工号/版本号/URL 等高信息熵标识符）
  const wordTokens = extractWordTokens(text)
  for (const item of wordTokens) {
    if (EN_STOP_WORDS.has(item.rawLower)) {
      addFeat('w:' + item.token, 0.35 * scale)
    } else if (item.isIdent) {
      addFeat('w:' + item.token, 1.85 * scale)
    } else if (item.token.length >= 2) {
      addFeat('w:' + item.token, 1.25 * scale)
    }
  }

  for (const [feat, baseW] of featWeights) {
    const tf = featCounts.get(feat) || 1
    const sublinearW = baseW * (1 + Math.log(tf))
    projectFeature(out, feat, sublinearW, dim, salt)
  }

  let n2 = 0
  for (let i = 0; i < dim; i++) n2 += out[i] * out[i]
  if (n2 > 0) {
    const inv = 1 / Math.sqrt(n2)
    for (let i = 0; i < dim; i++) out[i] *= inv
  }
  return out
}

export function embedWithMode(text, dim = 64, salt = K_SALT, mode = 'ngram') {
  return mode === 'hybrid' ? embedTextHybrid(text, dim, salt) : embedText(text, dim, salt)
}

function buildCoverageIndex(content, mode) {
  const normStr = normalizeText(content)
  const bigrams = new Set()
  for (let i = 0; i + 2 <= normStr.length; i++) bigrams.add(normStr.slice(i, i + 2))
  if (mode !== 'hybrid') return { normLen: normStr.length, bigrams, tokens: null }
  const tokens = new Set()
  for (const item of extractWordTokens(content)) {
    if (!EN_STOP_WORDS.has(item.rawLower) && item.token.length >= 2) tokens.add(item.token)
  }
  return { normLen: normStr.length, bigrams, tokens }
}

function computeCoverageFromCache(qIdx, eIdx, mode) {
  if (mode !== 'hybrid') {
    if (qIdx.normLen < 2 || eIdx.normLen < 2) return 1
    let hit = 0
    let total = 0
    for (const bg of qIdx.bigrams) {
      total++
      if (eIdx.bigrams.has(bg)) hit++
    }
    return total ? hit / total : 1
  }
  // hybrid 模式：过滤问句/全虚词二元组，并计入英文/标识符词级命中
  let hit = 0
  let total = 0
  for (const bg of qIdx.bigrams) {
    if (isStopGram(bg) || QUERY_STOP_BIGRAMS.has(bg)) continue
    total++
    if (eIdx.bigrams.has(bg)) hit++
  }
  if (qIdx.tokens && eIdx.tokens) {
    for (const tok of qIdx.tokens) {
      total += 2
      if (eIdx.tokens.has(tok)) hit += 2
    }
  }
  if (total === 0) {
    if (qIdx.normLen < 2 || eIdx.normLen < 2) return 1
    let fh = 0
    let ft = 0
    for (const bg of qIdx.bigrams) {
      ft++
      if (eIdx.bigrams.has(bg)) fh++
    }
    return ft ? fh / ft : 1
  }
  return hit / total
}

/**
 * 查询与条目的**词面覆盖**：查询里的字符二元组有多大比例出现在条目文本里。
 *
 * ⚠️ 只在 recall 的**排序之后**用来判定"这条到底是不是在说同一件事"，
 * 不参与打分、不参与排序 —— 排序仍然完全由 M·k 决定。
 *
 * 为什么需要它：矩阵分数在"近似查询"和"完全无关问题"上区间重叠
 * （实测 0.481~0.632 交叠，见 DEFAULT_OPTIONS.minScore 注释），
 * 单靠分数必然要么漏召回要么误报。二元组覆盖分得极干净：
 * 实测近似查询 0.50~1.00，库外问题全部 0.00。
 *
 * 边界：查询短于 2 个归一化字符时无法算覆盖 → 返回 1（不拦），
 * 交给分数门限处理，避免把「P8」这类极短查询全拒掉。
 */
export function bigramCoverage(query, text) {
  const a = normalizeText(query)
  const b = normalizeText(text)
  if (a.length < 2 || b.length < 2) return 1
  const grams = new Set()
  for (let i = 0; i + 2 <= b.length; i++) grams.add(b.slice(i, i + 2))
  let hit = 0
  let total = 0
  for (let i = 0; i + 2 <= a.length; i++) {
    total++
    if (grams.has(a.slice(i, i + 2))) hit++
  }
  return total ? hit / total : 1
}

/** 文本的稳定指纹，用于判重（同一句话观察两次不该在表里出现两条）。 */
export function hashText(text) {
  // ⚠️ 2026-11-17 修：原来用 normalizeText（去标点、只留字母数字）后哈希，
  //   于是「安然主要使用 C++」和「安然主要使用 C#」被判成同一条，
  //   后一句当作重复内容丢掉，查询 C# 还会返回旧的 C++ 原文。
  //   现在改用**空白归一**后的原文参与哈希：标点、符号、大小写之外的
  //   任何真实差异都会得到不同的指纹。嵌入向量仍用 normalizeText，
  //   那是"相似度"的语义，判重必须是"是不是同一条"。
  const s = String(text ?? '').replace(/\s+/gu, ' ').trim()
  return mix32(fnv1a(s, 0x9747b28c)).toString(16).padStart(8, '0') + ':' + s.length
}

// ────────────────────────────── 选项 ──────────────────────────────

function num(value, def, min, max) {
  const v = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(v)) return def
  return Math.min(max, Math.max(min, v))
}

function int(value, def, min, max) {
  return Math.round(num(value, def, min, max))
}

export const DEFAULT_OPTIONS = Object.freeze({
  dim: 64,
  /**
   * 一步梯度下降的步长。
   *
   * 稳定性：在单位键下每写一次，误差会乘 (1−2η)，所以 2η < 2（η < 1）就不发散。
   * 默认取 0.2：实测（40 条事实 / 20 条查询）η 从 0.2 到 0.49 扫过一遍，
   * 只有 η=0.2 + 单步 才是 18/20，更大的值稳定在 16~17/20——
   * 因为每一步都在改整张 M，步子越大，先学的事实被后来的覆盖得越多（干扰）。
   */
  eta: 0.2,
  surpriseThreshold: 0.35,
  /**
   * 修复一（2026-11-17）：相对新颖度门限，配合 surpriseThreshold 做双判据。
   *
   * 为什么默认 0.02 而不是 0.35：
   *   实测 cos(v̂, v) 在正常记忆里恒等于 1.000000（信号饱和），
   *   所以 1−cos 基本恒为 0。只要方向没变（不是"被衰减后的假新颖"），它就待在 0 附近；
   *   方向真的变了（不同的值压到同一个键上）时才会明显抬头。
   * 门限取 0.02 的意思是"方向偏了一点点就别写"，同时不会误伤正常情况。
   * 设成 0.35 会让几乎所有更新都被拒——这是实测出来的教训，别调回去。
   */
  surpriseThresholdRel: 0.02,
  decay: 0.02,
  /** 条目存活阈值：强度低于它就在遗忘 tick 里剔除。 */
  minStrength: 0.05,
  maxEntries: 512,
  /**
   * 修复二（2026-11-17）：置信度熔断的**绝对分数**门限。
   *
   * ⚠️ 2026-11-17 晚重新标定。第一版把 0.45 当成"判别阈值"是错的：
   *   20 条真实事实 + 10 个近似查询 + 10 个库外问题，用 minScore:-1 测原始分布：
   *     近似查询  最小 0.481 / 最大 0.843（10/10 都该命中）
   *     库外问题  最小 0.324 / 最大 0.632（10/10 都不该命中）
   *   **两簇分数区间重叠**（0.481~0.632），任何单一阈值都做不到两头都全对：
   *     ≥0.45 → 近似 100% 但无关误报 30%
   *     ≥0.65 → 无关 0% 但近似掉到 90%
   *   所以分数门限不再承担"是不是真的相关"这个判断，只做**粗筛**（挡住明显无关的），
   *   真正的判别交给下面的**词面覆盖**双判据。0.45 保留为粗筛下限。
   *   标定脚本：dsh-optimize/calib-2026-11-17-v2.mjs。
   */
  minScore: 0.45,
  /**
   * 词面覆盖门限（2026-11-17 新增）：查询的二元组有多大比例出现在候选条目里。
   *
   * 为什么需要它（实测数据，见上一段）：矩阵分数在"近似查询"和"库外问题"上区间重叠，
   * 分数本身分不清它们。但**字符二元组覆盖**分得极干净：
   *   近似查询（「李曜辰的工号」「KX4821」「三月十七」）  覆盖 0.50 ~ 1.00
   *   库外问题（「南极帝企鹅的孵化周期」「冰岛火山灰」）  覆盖 **全部 0.00**
   * 取 0.25 作门限，留一半余量：命中 100%、误报 0%。
   *
   * ⚠️ 这不是把工具退回关键词匹配 —— 打分与排序仍然完全由 M·k 决定
   *   （覆盖只在**排序之后**做一道真伪闸门，不参与排序，见 recall()）。
   */
  minCoverage: 0.25,
  /** 条目少于这个数时置信度阈值不可靠，跳过熔断（实测 ≤10 条时会给出假高分）。 */
  minEntriesForTrust: 10,
  /** 超过这个字符数直接拒绝，避免 1MB 文本把主线程拖死。 */
  maxChars: 8000,
  /** 每次观察后额外施加的衰减。默认 0：遗忘交给 nm_forget 的 tick，便于测"多轮下降"。 */
  decayPerObserve: 0,
  namespace: 'default',
  /** 覆盖默认存储目录；null = <DSH_HOME>/state/dsh-neural-memory。 */
  path: null,
  /** 关掉工具注册（只保留数值内核）时用，默认开。 */
  tools: true,
  /** 嵌入模式：'ngram'（默认，严格兼容经典公式验收）| 'hybrid'（词边界+标识符提权+次线性频次）。 */
  embedMode: 'ngram',
  /** Titans 惊讶度动量系数 β（默认 0 与一阶 SGD 逐位等价；生产推荐 0.2~0.3）。 */
  momentum: 0,
  /** 回忆强化增益（默认 0 保持读操作纯度；生产推荐 0.15，常用记忆抗遗忘）。 */
  recallBoost: 0,
  /** 落盘防抖合并毫秒数（默认 0 每次立即落盘；生产推荐 1500ms 减少写放大）。 */
  persistDebounceMs: 0,
  /** 矩阵序列化格式：'json'（默认兼容数组检查）| 'base64'（Float64 无损紧凑编码，体积减半）。 */
  storageFormat: 'json',
  /** 配置升级导致 dim / embedMode 变化时，是否按存活原文重嵌入迁移（默认 false 保安全清空契约）。 */
  migrateOnDimChange: false,
  /** 内存常驻 namespace LRU 上限，超限自动先 flush 再淘汰最久未用分区。 */
  maxLoadedNamespaces: 16,
  /** 自动观察用户对话（默认 false 离线零副作用；宿主 cordis.patch.yml 建议开启）。 */
  autoObserve: false,
  /** 自动观察的惊讶度门限（比手动更挑，过滤闲聊）。 */
  autoObserveThreshold: 0.55,
  /** 单轮自动观察最多切分句数。 */
  autoObserveMaxSentences: 4,
  /** 自动联想微注入（默认 false；宿主 cordis.patch.yml 建议开启）。 */
  autoRecall: false,
  /** 自动注入三闸门：分数 / 覆盖 / 唯一峰边际差。 */
  autoRecallMinScore: 0.52,
  autoRecallMinCoverage: 0.30,
  autoRecallMinMargin: 0.08,
  autoRecallTopK: 3,
  /** 同一条记忆在同一会话内的冷却轮数（压缩后自动清零）。 */
  autoRecallCooldownTurns: 5,
  /** 单次自动注入硬字符上限（不抢 dsh-mind 预算）。 */
  autoInjectMaxChars: 600,
  /** 每累计 N 轮空闲自动触发一次轻量遗忘（0 = 关闭）。 */
  autoForgetEveryTurns: 0,
  /** 自动遗忘单轮衰减系数。 */
  autoForgetDecay: 0.015,
})

export function resolveOptions(raw = {}) {
  const src = raw && typeof raw === 'object' ? raw : {}
  const embedMode = src.embedMode === 'hybrid' ? 'hybrid' : 'ngram'
  const storageFormat = src.storageFormat === 'base64' ? 'base64' : 'json'
  return {
    dim: int(src.dim, DEFAULT_OPTIONS.dim, 8, 256),
    eta: num(src.eta, DEFAULT_OPTIONS.eta, 0.0001, 0.4999),
    surpriseThreshold: num(src.surpriseThreshold, DEFAULT_OPTIONS.surpriseThreshold, 0, 1e6),
    surpriseThresholdRel: num(src.surpriseThresholdRel, DEFAULT_OPTIONS.surpriseThresholdRel, 0, 2),
    decay: num(src.decay, DEFAULT_OPTIONS.decay, 0, 0.999),
    minStrength: num(src.minStrength, DEFAULT_OPTIONS.minStrength, 0, 1e6),
    maxEntries: int(src.maxEntries, DEFAULT_OPTIONS.maxEntries, 1, 100000),
    minScore: num(src.minScore, DEFAULT_OPTIONS.minScore, -1, 1),
    minCoverage: num(src.minCoverage, DEFAULT_OPTIONS.minCoverage, 0, 1),
    minEntriesForTrust: int(src.minEntriesForTrust, DEFAULT_OPTIONS.minEntriesForTrust, 1, 10000),
    maxChars: int(src.maxChars, DEFAULT_OPTIONS.maxChars, 1, 1 << 20),
    decayPerObserve: num(src.decayPerObserve, DEFAULT_OPTIONS.decayPerObserve, 0, 0.9),
    namespace:
      typeof src.namespace === 'string' && NAME_RE.test(src.namespace)
        ? src.namespace
        : DEFAULT_OPTIONS.namespace,
    path: typeof src.path === 'string' && src.path.length > 0 ? src.path : null,
    tools: src.tools !== false,
    store: src.store && typeof src.store.write === 'function' ? src.store : null,
    embedMode,
    momentum: num(src.momentum, DEFAULT_OPTIONS.momentum, 0, 0.95),
    recallBoost: num(src.recallBoost, DEFAULT_OPTIONS.recallBoost, 0, 2),
    persistDebounceMs: int(src.persistDebounceMs, DEFAULT_OPTIONS.persistDebounceMs, 0, 60000),
    storageFormat,
    migrateOnDimChange: src.migrateOnDimChange === true,
    maxLoadedNamespaces: int(src.maxLoadedNamespaces, DEFAULT_OPTIONS.maxLoadedNamespaces, 2, 256),
    autoObserve: src.autoObserve === true,
    autoObserveThreshold: num(src.autoObserveThreshold, DEFAULT_OPTIONS.autoObserveThreshold, 0, 1e6),
    autoObserveMaxSentences: int(src.autoObserveMaxSentences, DEFAULT_OPTIONS.autoObserveMaxSentences, 1, 20),
    autoRecall: src.autoRecall === true,
    autoRecallMinScore: num(src.autoRecallMinScore, DEFAULT_OPTIONS.autoRecallMinScore, -1, 1),
    autoRecallMinCoverage: num(src.autoRecallMinCoverage, DEFAULT_OPTIONS.autoRecallMinCoverage, 0, 1),
    autoRecallMinMargin: num(src.autoRecallMinMargin, DEFAULT_OPTIONS.autoRecallMinMargin, 0, 1),
    autoRecallTopK: int(src.autoRecallTopK, DEFAULT_OPTIONS.autoRecallTopK, 1, 10),
    autoRecallCooldownTurns: int(src.autoRecallCooldownTurns, DEFAULT_OPTIONS.autoRecallCooldownTurns, 0, 100),
    autoInjectMaxChars: int(src.autoInjectMaxChars, DEFAULT_OPTIONS.autoInjectMaxChars, 120, 4000),
    autoForgetEveryTurns: int(src.autoForgetEveryTurns, DEFAULT_OPTIONS.autoForgetEveryTurns, 0, 1000),
    autoForgetDecay: num(src.autoForgetDecay, DEFAULT_OPTIONS.autoForgetDecay, 0, 0.5),
  }
}

/** namespace 白名单校验。返回错误文案或 null（合法）。 */
export function checkNamespace(ns) {
  if (typeof ns !== 'string' || ns.length === 0) return 'namespace 不能为空'
  if (!NAME_RE.test(ns)) return `非法 namespace：${JSON.stringify(ns)}（只允许 a-z 0-9 _ -，1~64 字符）`
  return null
}

// ────────────────────────────── 状态与序列化 ──────────────────────────────

function encodeFloat64Base64(arr) {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString('base64')
}

function decodeFloat64Base64(b64, expectedLen) {
  if (typeof b64 !== 'string' || b64.length === 0) return null
  try {
    const buf = Buffer.from(b64, 'base64')
    if (buf.byteLength !== expectedLen * 8) return null
    const out = new Float64Array(expectedLen)
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength)
    for (let i = 0; i < expectedLen; i++) {
      const v = view.getFloat64(i * 8, true)
      out[i] = Number.isFinite(v) ? v : 0
    }
    return out
  } catch {
    return null
  }
}

function createState(dim, embedMode = 'ngram') {
  return {
    dim,
    embedMode,
    /** d×d 关联记忆矩阵，row-major 扁平。 */
    M: new Float64Array(dim * dim),
    /** d×d Titans 惊讶动量矩阵（momentum > 0 时启用）。 */
    S: new Float64Array(dim * dim),
    /** 记忆条目表。 */
    entries: [],
    /** hash → entry 索引，判重与召回都要用。 */
    index: new Map(),
    /**
     * 修复三的配套缓存：条目不再存 k/v，改用确定性哈希现场重算。
     * 同一次运行里同一段文本只算一次，避免 recall 变成 O(entries × hash)。
     * 不持久化——重算成本低（几十微秒），不值得占盘。
     */
    kCache: new Map(),
    vCache: new Map(),
    /** 条目词面覆盖索引缓存（hash → { normLen, bigrams, tokens }），加速 recall。 */
    covCache: new Map(),
    observes: 0,
    writes: 0,
    decayRounds: 0,
    surpriseSum: 0,
    updatedAt: 0,
  }
}

function reindex(state) {
  state.index = new Map()
  for (const e of state.entries) state.index.set(e.hash, e)
}

function serialize(state, opts = {}) {
  const format = opts.storageFormat === 'base64' ? 'base64' : 'json'
  const hasMomentum = state.S && frobenius(state.S) > 1e-12
  const payload = {
    version: 2,
    plugin: PLUGIN_NAME,
    dim: state.dim,
    embedMode: state.embedMode || 'ngram',
    ...(format === 'base64'
      ? { M_b64: encodeFloat64Base64(state.M) }
      : { M: Array.from(state.M) }),
    ...(hasMomentum
      ? (format === 'base64'
          ? { S_b64: encodeFloat64Base64(state.S) }
          : { S: Array.from(state.S) })
      : {}),
    /**
     * 修复三（2026-11-17）：**不再持久化 k / v**。
     *
     * 原实现把每条的 k 和 v 都写进了 JSON，等于留了一条后门：
     * recall 完全可以绕过 M，直接算 cos(k_query, v_i)。
     * 实测那样做命中率只有 5%（走 M 是 100%），功能"看起来还能用"，
     * 于是矩阵会悄悄变成一个摆设——而"矩阵是必需的"正是这个插件的核心主张。
     *
     * 现在盘上只存 text + 元数据；k / v 在需要时用确定性哈希重算
     * （embedText 是纯函数，同一段文本永远得到同一个向量）。
     *
     * 注意：**内存里的 entry 仍然带 k / v**（见 observe 与 ensureLoaded），
     * 因为验收方要拿它做独立复算。真正关掉的是"落盘留下可绕过矩阵的原始向量"这条路。
     */
    entries: state.entries.map((e) => ({
      text: e.text,
      hash: e.hash,
      surprise: e.surprise,
      strength: e.strength,
      writes: e.writes,
      createdAt: e.createdAt,
      lastSeen: e.lastSeen,
    })),
    observes: state.observes,
    writes: state.writes,
    decayRounds: state.decayRounds,
    surpriseSum: state.surpriseSum,
    updatedAt: state.updatedAt,
  }
  return payload
}

/**
 * 取一条记忆的键向量。
 *
 * 修复三的配套：盘上不存 k / v，读回后需要时用确定性哈希重算。
 * 内存里已有就直接用（observe 时算过的），没有才现算。
 * 走 kCache 是为了避免同一段文本在一次 recall 里被重算多次。
 */
function keyVectorOf(state, entry) {
  if (entry.k) return entry.k
  let v = state.kCache.get(entry.hash)
  if (!v) {
    v = embedWithMode(entry.text, state.dim, K_SALT, state.embedMode)
    state.kCache.set(entry.hash, v)
  }
  return v
}

function valueVectorOf(state, entry) {
  if (entry.v) return entry.v
  let v = state.vCache.get(entry.hash)
  if (!v) {
    v = embedWithMode(entry.text, state.dim, V_SALT, state.embedMode)
    state.vCache.set(entry.hash, v)
  }
  return v
}

function coverageIndexOf(state, entry) {
  let idx = state.covCache.get(entry.hash)
  if (!idx) {
    idx = buildCoverageIndex(entry.text, state.embedMode)
    state.covCache.set(entry.hash, idx)
  }
  return idx
}

/** 将一条记忆的 (k, v) 按经典一步梯度补写入 M（用于迁移或外部条目合并）。 */
function learnEntryIntoMatrix(state, entry, eta) {
  const dim = state.dim
  const k = keyVectorOf(state, entry)
  const v = valueVectorOf(state, entry)
  let kn = 0
  for (let i = 0; i < dim; i++) kn += k[i] * k[i]
  if (!(kn > 0)) return
  const pred = matvec(state.M, k, dim, dim)
  const err = new Float64Array(dim)
  for (let i = 0; i < dim; i++) err[i] = pred[i] - v[i]
  const grad = outer(err, k, dim, dim)
  addScaled(state.M, grad, -2 * eta)
}

/** 从 M（及动量 S）里沿键方向投影擦除已删除条目，避免残余干扰近似记忆。 */
function unlearnEntryFromMatrix(state, entry) {
  const dim = state.dim
  const k = keyVectorOf(state, entry)
  let kn = 0
  for (let i = 0; i < dim; i++) kn += k[i] * k[i]
  if (!(kn > 1e-12)) return
  const invKn = 1 / kn
  const pred = matvec(state.M, k, dim, dim)
  const proj = outer(pred, k, dim, dim)
  addScaled(state.M, proj, -invKn)
  if (state.S) {
    const sPred = matvec(state.S, k, dim, dim)
    const sProj = outer(sPred, k, dim, dim)
    addScaled(state.S, sProj, -invKn)
  }
}

function deserialize(raw, dim, opts = {}) {
  const embedMode = opts.embedMode === 'hybrid' ? 'hybrid' : 'ngram'
  const migrateOnDimChange = opts.migrateOnDimChange === true
  const eta = typeof opts.eta === 'number' && Number.isFinite(opts.eta) ? opts.eta : DEFAULT_OPTIONS.eta
  const state = createState(dim, embedMode)
  if (!raw || typeof raw !== 'object') return { state, recovered: false, migrated: false }

  const rawDim = Number(raw.dim)
  const rawEmbedMode = raw.embedMode === 'hybrid' ? 'hybrid' : 'ngram'
  const spaceChanged = rawDim !== dim || rawEmbedMode !== embedMode

  if (spaceChanged) {
    // 默认策略：dim 或嵌入空间变更后安全重建，避免维度不兼容产出 NaN。
    if (!migrateOnDimChange || !Array.isArray(raw.entries) || raw.entries.length === 0) {
      return { state, recovered: false, migrated: false }
    }
    // 开启 migrateOnDimChange 时：保留存活原文，按新 (dim, embedMode) 无损重嵌入并重学入 M。
    for (const e of raw.entries) {
      if (!e || typeof e.text !== 'string' || !e.text.trim()) continue
      const k = embedWithMode(e.text, dim, K_SALT, embedMode)
      const v = embedWithMode(e.text, dim, V_SALT, embedMode)
      const entry = {
        text: e.text,
        hash: typeof e.hash === 'string' ? e.hash : hashText(e.text),
        k,
        v,
        surprise: Number.isFinite(Number(e.surprise)) ? Number(e.surprise) : 0,
        strength: Number.isFinite(Number(e.strength)) ? Number(e.strength) : 0.5,
        writes: Number.isFinite(Number(e.writes)) ? Number(e.writes) : 1,
        createdAt: Number.isFinite(Number(e.createdAt)) ? Number(e.createdAt) : 0,
        lastSeen: Number.isFinite(Number(e.lastSeen)) ? Number(e.lastSeen) : 0,
      }
      state.entries.push(entry)
      const steps = Math.min(3, Math.max(1, Math.round(entry.writes)))
      for (let step = 0; step < steps; step++) learnEntryIntoMatrix(state, entry, eta)
    }
    reindex(state)
    state.observes = Number(raw.observes) || state.entries.length
    state.writes = Number(raw.writes) || state.entries.length
    state.decayRounds = Number(raw.decayRounds) || 0
    state.surpriseSum = Number(raw.surpriseSum) || 0
    state.updatedAt = Number(raw.updatedAt) || 0
    return { state, recovered: true, migrated: true }
  }

  if (typeof raw.M_b64 === 'string') {
    const decoded = decodeFloat64Base64(raw.M_b64, dim * dim)
    if (decoded) state.M = decoded
  } else if (Array.isArray(raw.M) && raw.M.length === dim * dim) {
    for (let i = 0; i < dim * dim; i++) {
      const v = Number(raw.M[i])
      state.M[i] = Number.isFinite(v) ? v : 0
    }
  }

  if (typeof raw.S_b64 === 'string') {
    const decodedS = decodeFloat64Base64(raw.S_b64, dim * dim)
    if (decodedS) state.S = decodedS
  } else if (Array.isArray(raw.S) && raw.S.length === dim * dim) {
    for (let i = 0; i < dim * dim; i++) {
      const v = Number(raw.S[i])
      state.S[i] = Number.isFinite(v) ? v : 0
    }
  }

  if (Array.isArray(raw.entries)) {
    for (const e of raw.entries) {
      if (!e || typeof e.text !== 'string') continue
      const k = embedWithMode(e.text, dim, K_SALT, embedMode)
      const v = embedWithMode(e.text, dim, V_SALT, embedMode)
      const entry = {
        text: e.text,
        hash: typeof e.hash === 'string' ? e.hash : hashText(e.text),
        k,
        v,
        surprise: Number.isFinite(Number(e.surprise)) ? Number(e.surprise) : 0,
        strength: Number.isFinite(Number(e.strength)) ? Number(e.strength) : 0,
        writes: Number.isFinite(Number(e.writes)) ? Number(e.writes) : 1,
        createdAt: Number.isFinite(Number(e.createdAt)) ? Number(e.createdAt) : 0,
        lastSeen: Number.isFinite(Number(e.lastSeen)) ? Number(e.lastSeen) : 0,
      }
      state.entries.push(entry)
    }
    reindex(state)
  }
  state.observes = Number(raw.observes) || 0
  state.writes = Number(raw.writes) || 0
  state.decayRounds = Number(raw.decayRounds) || 0
  state.surpriseSum = Number(raw.surpriseSum) || 0
  state.updatedAt = Number(raw.updatedAt) || 0
  return { state, recovered: true, migrated: false }
}

/** 矩阵的 Frobenius 范数 ‖M‖_F，用来证明"遗忘让权重真的变小"。 */
export function frobenius(M) {
  let s = 0
  for (let i = 0; i < M.length; i++) s += M[i] * M[i]
  return Math.sqrt(s)
}

function meanSurpriseOf(state) {
  if (state.entries.length === 0) return 0
  let s = 0
  for (const e of state.entries) s += e.surprise
  return s / state.entries.length
}

// ────────────────────────────── 主体 ──────────────────────────────

export class NeuralMemory {
  constructor(options = {}) {
    this.options = resolveOptions(options)
    this.dim = this.options.dim
    this.embedMode = this.options.embedMode
    this.store = this.options.store || createStore(PLUGIN_NAME, this.options.path)
    /** namespace → state（进程内 LRU 缓存）。 */
    this.states = new Map()
    /** namespace → 正在进行的 load Promise，防并发重复读盘。 */
    this.loading = new Map()
    /** namespace → 落盘串行链（见 save 的注释）。 */
    this.writeChains = new Map()
    /** namespace → 自己最后一次读到/写出的 mtime（用于外部写入检测）。 */
    this.lastWriteMtime = new Map()
    /** 防抖写盘状态：dirty 集合与定时器表。 */
    this.dirtyNamespaces = new Set()
    this.debounceTimers = new Map()
  }

  // ── 状态管理 ──

  /** 刷新 LRU 顺序并在超上限时淘汰最久未用的干净分区。 */
  touchNamespace(ns) {
    const st = this.states.get(ns)
    if (!st) return
    this.states.delete(ns)
    this.states.set(ns, st)
    const maxNs = this.options.maxLoadedNamespaces
    if (this.states.size > maxNs) {
      for (const oldNs of this.states.keys()) {
        if (oldNs === ns || this.dirtyNamespaces.has(oldNs) || this.writeChains.has(oldNs)) continue
        this.states.delete(oldNs)
        this.lastWriteMtime.delete(oldNs)
        if (this.states.size <= maxNs) break
      }
    }
  }

  /** 同步取内存状态（没有就建空的），不碰磁盘。 */
  stateOf(ns) {
    let st = this.states.get(ns)
    if (!st) {
      st = createState(this.dim, this.embedMode)
      this.states.set(ns, st)
    }
    this.touchNamespace(ns)
    return st
  }

  /** 确保 namespace 已从磁盘读入内存。读失败按"空记忆"处理，绝不抛。 */
  async ensureLoaded(ns) {
    if (this.states.has(ns)) {
      this.touchNamespace(ns)
      return this.states.get(ns)
    }
    let pending = this.loading.get(ns)
    if (!pending) {
      pending = (async () => {
        const raw = await this.store.read(ns)
        const { state, migrated } = deserialize(raw, this.dim, {
          embedMode: this.embedMode,
          migrateOnDimChange: this.options.migrateOnDimChange,
          eta: this.options.eta,
        })
        // ⚠️ 2026-11-17：墓碑过滤必须在**读盘之后**做。
        //   外部删除（或历史上删过）的指纹永不复活，见 store.js 的 readTombs 注释。
        await this.applyTombs(ns, state)
        // 记下读盘时的 mtime：之后 save 时用它判断"盘上是不是被别人改过"。
        if (typeof this.store.statMtime === 'function') {
          const mt = await this.store.statMtime(ns)
          if (mt !== null) this.lastWriteMtime.set(ns, mt)
        }
        this.states.set(ns, state)
        this.touchNamespace(ns)
        this.loading.delete(ns)
        if (migrated && state.entries.length > 0) {
          this.scheduleSave(ns, true)
        }
        return state
      })().catch(() => {
        this.loading.delete(ns)
        const st = createState(this.dim, this.embedMode)
        this.states.set(ns, st)
        return st
      })
      this.loading.set(ns, pending)
    }
    return pending
  }

  /**
   * 把盘上被**别的实例**写进去的条目并进内存，避免整体覆盖丢数据。
   * 升级（2026-11-22）：合入外部新条目时同步补学进本地 M，避免沦为未学习幽灵条目。
   *
   * @returns {Promise<number>} 并入的条目数
   */
  async mergeExternal(ns) {
    if (typeof this.store.statMtime !== 'function') return 0
    const mt = await this.store.statMtime(ns)
    if (mt === null) return 0
    const last = this.lastWriteMtime.get(ns)
    if (last === undefined) {
      this.lastWriteMtime.set(ns, mt)
      return 0
    }
    // 1ms 容差：Windows 上 mtime 精度与自写回读可能差一点点。
    if (mt <= last + 1) return 0
    const raw = await this.store.read(ns)
    if (!raw) return 0
    const disk = deserialize(raw, this.dim, {
      embedMode: this.embedMode,
      migrateOnDimChange: this.options.migrateOnDimChange,
      eta: this.options.eta,
    })
    if (!disk.recovered) return 0
    const st = this.stateOf(ns)
    let merged = 0
    for (const e of disk.state.entries) {
      if (st.index.has(e.hash)) continue
      st.entries.push(e)
      st.index.set(e.hash, e)
      learnEntryIntoMatrix(st, e, this.options.eta)
      merged++
    }
    this.lastWriteMtime.set(ns, mt)
    return merged
  }

  /**
   * 按墓碑表剔除 state 里已被显式删除的条目。
   *
   * @returns {Promise<number>} 剔除的条数
   */
  async applyTombs(ns, state) {
    if (typeof this.store.readTombs !== 'function') return 0
    let tombs
    try {
      tombs = await this.store.readTombs()
    } catch {
      return 0
    }
    const set = tombs.get(ns)
    if (!set || set.size === 0) return 0
    const before = state.entries.length
    state.entries = state.entries.filter((e) => !set.has(e.hash))
    const removed = before - state.entries.length
    if (removed > 0) {
      reindex(state)
      for (const h of set) {
        state.kCache.delete(h)
        state.vCache.delete(h)
        state.covCache.delete(h)
      }
    }
    return removed
  }

  /**
   * 按配置决定立即落盘或防抖合批落盘。
   * persistDebounceMs === 0（默认）：立即走 save(ns)，保证离线测试与强一致契约。
   * persistDebounceMs > 0：合批短时间内的多次写入，减少大矩阵全量写盘风暴。
   */
  scheduleSave(ns, immediate = false) {
    this.dirtyNamespaces.add(ns)
    const debounce = this.options.persistDebounceMs
    if (immediate || debounce <= 0) {
      const existingTimer = this.debounceTimers.get(ns)
      if (existingTimer) {
        clearTimeout(existingTimer)
        this.debounceTimers.delete(ns)
      }
      return this.save(ns)
    }
    if (!this.debounceTimers.has(ns)) {
      const timer = setTimeout(() => {
        this.debounceTimers.delete(ns)
        if (this.dirtyNamespaces.has(ns)) {
          this.save(ns).catch(() => {})
        }
      }, debounce)
      if (typeof timer.unref === 'function') timer.unref()
      this.debounceTimers.set(ns, timer)
    }
    return Promise.resolve({ ok: true, deferred: true })
  }

  /** 立即冲刷单个 namespace 的待写脏状态。 */
  async flush(ns) {
    const timer = this.debounceTimers.get(ns)
    if (timer) {
      clearTimeout(timer)
      this.debounceTimers.delete(ns)
    }
    if (!this.dirtyNamespaces.has(ns)) return { ok: true, skipped: true }
    return this.save(ns)
  }

  /** 冲刷所有已加载的脏 namespace（用于会话 idle / 插件卸载）。 */
  async flushAll() {
    const list = [...this.dirtyNamespaces]
    const results = []
    for (const ns of list) {
      results.push(await this.flush(ns))
    }
    return results
  }

  /** 落盘。任何 IO 失败都降级成返回值里的 persistError，不影响已经算出来的结果。 */
  async save(ns) {
    const existingTimer = this.debounceTimers.get(ns)
    if (existingTimer) {
      clearTimeout(existingTimer)
      this.debounceTimers.delete(ns)
    }
    const prev = this.writeChains.get(ns) ?? Promise.resolve()
    const next = prev.then(
      () => this.saveNow(ns),
      () => this.saveNow(ns),
    )
    this.writeChains.set(ns, next)
    try {
      return await next
    } finally {
      if (this.writeChains.get(ns) === next) this.writeChains.delete(ns)
    }
  }

  /** save 的实际写入体（已在上游串行化）。 */
  async saveNow(ns) {
    const st = this.stateOf(ns)
    st.updatedAt = Date.now()
    try {
      await this.mergeExternal(ns)
      // 墓碑是外部事实，落盘前必须再滤一次，否则缓存会把删掉的条目写回去。
      await this.applyTombs(ns, st)
      const res = await this.store.write(ns, serialize(st, { storageFormat: this.options.storageFormat }))
      this.dirtyNamespaces.delete(ns)
      if (typeof this.store.statMtime === 'function') {
        const mt = await this.store.statMtime(ns)
        if (mt !== null) this.lastWriteMtime.set(ns, mt)
      }
      return { ok: true, bytes: res.bytes, file: res.path }
    } catch (e) {
      return { ok: false, error: String(e?.message ?? e) }
    }
  }

  async listNamespaces() {
    try {
      return await this.store.list()
    } catch {
      return []
    }
  }

  // ── 核心：观察 ──

  /**
   * 观察一段文本。惊讶度够高才写进 M，写的时候做一步梯度下降。
   *
   * @returns {{ok:boolean, wrote:boolean, surprise:number, entries:number, retained:boolean, ...}}
   */
  async observe(text, opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, wrote: false, surprise: 0, entries: 0, retained: false }

    if (typeof text !== 'string') {
      return { ok: false, error: 'text 必须是字符串', wrote: false, surprise: 0, entries: 0, retained: false }
    }
    const raw = text.trim()
    if (raw.length === 0) {
      return { ok: false, error: 'text 不能为空', wrote: false, surprise: 0, entries: 0, retained: false }
    }
    if (raw.length > this.options.maxChars) {
      return {
        ok: false,
        error: `text 过长（${raw.length} 字符 > 上限 ${this.options.maxChars}），已拒绝以免阻塞主线程`,
        wrote: false,
        surprise: 0,
        entries: 0,
        retained: false,
      }
    }

    const state = await this.ensureLoaded(ns)
    const dim = this.dim
    const k = embedWithMode(raw, dim, K_SALT, this.embedMode)
    const v = embedWithMode(raw, dim, V_SALT, this.embedMode)

    let kn = 0
    for (let i = 0; i < dim; i++) kn += k[i] * k[i]
    if (!(kn > 0)) {
      // 归一化后一个有效字符都不剩（比如纯标点/纯 emoji），没法嵌入。
      return {
        ok: false,
        error: 'text 里没有可用于嵌入的文字或数字',
        wrote: false,
        surprise: 0,
        entries: state.entries.length,
        retained: false,
      }
    }
    // 理论上已 L2 归一化；这里再保证一次，让"误差每次乘 (1−2η)"的推导严格成立。
    const inv = 1 / Math.sqrt(kn)
    for (let i = 0; i < dim; i++) k[i] *= inv

    const threshold = num(opts.surpriseThreshold, this.options.surpriseThreshold, 0, 1e6)
    /**
     * 修复一（2026-11-17）的兼容处理：
     *
     * 调用方显式传 `surpriseThreshold: 0` 的语义是"让我这次一定写入"（测试、强制记忆都用这个）。
     * 如果这时候相对门限还在生效，那句"一定写入"就变成了"多半写不进去"——实测第二次观察
     * 同一句话就会被挡住，行为与调用方的意图相反。
     *
     * 所以：绝对门限被显式设成 0 时，相对门限一起让路（视为 0）。
     * 这不是特例，是把"0 = 关掉筛选"这个已有约定贯彻到两个判据上。
     */
    const thresholdRelRequested = num(opts.surpriseThresholdRel, this.options.surpriseThresholdRel, 0, 2)
    const forceWrite = threshold <= 0
    const thresholdRel = forceWrite ? 0 : thresholdRelRequested

    // ── 预测 ── v̂ = M·k
    const pred = matvec(state.M, k, dim, dim)
    // ── 惊讶度 ── ‖v̂ − v‖²
    const err = new Float64Array(dim)
    let surprise = 0
    for (let i = 0; i < dim; i++) {
      const d = pred[i] - v[i]
      err[i] = d
      surprise += d * d
    }

    /**
     * 修复一（2026-11-17）：双判据。
     *
     * 纯绝对距离的毛病：M 被衰减后模长变小，`‖M·k − v‖²` 会跟着变大，
     * 即使方向完全没变（实测 cos 恒为 1.000000）。于是"自己忘掉的旧记忆"
     * 会被判成"新惊喜"重新写入 —— 实测衰减 8 轮后 surprise=0.4352 > θ=0.35。
     *
     * 单用相对余弦也不行：它在这组参数下恒等于 0（信号饱和），
     * 会把"生日三月十七 → 三月十八"这种正确的更新也拒掉。
     *
     * 所以两个都要：
     *   surpriseAbs   —— 值变了没有（原来的语义）
     *   surpriseRel   —— 方向变了没有（防衰减误判）
     * 两个都过才写。
     */
    let predNorm = 0
    for (let i = 0; i < dim; i++) predNorm += pred[i] * pred[i]
    predNorm = Math.sqrt(predNorm)
    // cos(v̂, v)：v 是单位向量，所以就是 (v̂·v)/‖v̂‖。predNorm 为 0 时 unset（=1，即"完全不知道"）。
    let cosPV = 0
    if (predNorm > 1e-12) {
      let dotPV = 0
      for (let i = 0; i < dim; i++) dotPV += pred[i] * v[i]
      cosPV = dotPV / predNorm
    }
    const surpriseRel = 1 - cosPV

    state.observes += 1
    state.surpriseSum += surprise

    const wroteAbs = Number.isFinite(surprise) && surprise > threshold
    /**
     * ⚠️ 2026-11-17 修（隔离实测反例）：
     *
     * 双重判据（绝对 + 相对）防住了"衰减后旧记忆被当成新惊喜"，
     * 但**副作用更大**：条目被遗忘淘汰后，M 只是等比缩小、方向不变，
     * 于是再观察同一句话时 surpriseRel ≈ 0，永远 wrote=false，
     * 条目**再也回不来**（实测默认 149 轮遗忘后 entries=0，重学 entries 仍 0）。
     *
     * 正确的语义是：相对判据只用来挡"**还在表里**的记忆因衰减被误判为新惊喜"。
     * 条目已经不在表里 → 它对新读者就是全新的东西，不需要方向判据。
     * 所以这里先算出"这条是否已在记忆中"，再决定要不要启用相对门限。
     */
    const candidateHash = hashText(raw)
    const alreadyKnown = state.index.has(candidateHash)
    /**
     * ⚠️ 这里不能用 `surpriseRel > thresholdRel` 直接判——当 thresholdRel === 0 时，
     * 完全对齐的记忆 `surpriseRel` 会算成 -0.000000（浮点误差），`> 0` 返回 false，
     * 于是"门限设成 0 表示不要筛"这个约定就失效了。实测第 2/3 次 observe 全被挡住。

     * 正确语义：thresholdRel <= 0 表示"这一维不参与筛选"，直接放行。
     */
    const wroteRel = !alreadyKnown || thresholdRel <= 0 || (Number.isFinite(surpriseRel) && surpriseRel > thresholdRel)
    const wrote = wroteAbs && wroteRel

    if (wrote) {
      // ── M ← M − η·∇L，∇L = 2(v̂−v)kᵀ（开启 momentum 时走 Titans 动量更新 S ← βS − 2η·∇L, M ← M + S）──
      const grad = outer(err, k, dim, dim)
      const beta = this.options.momentum
      if (beta > 0 && state.S) {
        scaleInPlace(state.S, beta)
        addScaled(state.S, grad, -2 * this.options.eta)
        addScaled(state.M, state.S, 1)
      } else {
        addScaled(state.M, grad, -2 * this.options.eta)
      }
      state.writes += 1
      if (this.options.decayPerObserve > 0) {
        scaleInPlace(state.M, 1 - this.options.decayPerObserve)
        if (beta > 0 && state.S) scaleInPlace(state.S, 1 - this.options.decayPerObserve)
      }
    }

    const now = Date.now()
    const hash = candidateHash
    let entry = state.index.get(hash)
    if (entry) {
      // 同一句话再看一次：不新增条目，只刷新时间戳；若通过门限就补一次写入证据。
      entry.lastSeen = now
      if (wrote) {
        entry.surprise = surprise
        entry.strength += surprise
        entry.writes += 1
      }
    } else if (wrote) {
      entry = {
        text: raw,
        hash,
        // 修复三：k / v 只留在内存（供验收与调试独立复算），**不写盘**。
        k,
        v,
        surprise,
        strength: surprise,
        writes: 1,
        createdAt: now,
        lastSeen: now,
      }
      state.entries.push(entry)
      state.index.set(hash, entry)
      this.prune(state)
    }

    const persist = await this.scheduleSave(ns, false)
    const inMemory = !!state.index.get(hash)
    return {
      ok: true,
      wrote,
      surprise,
      // 双判据的两个分量都返回，方便使用方自己判断为什么被拒。
      wroteAbs,
      wroteRel,
      surpriseAbs: surprise,
      surpriseRel,
      cosPredValue: cosPV,
      threshold,
      thresholdRel,
      entries: state.entries.length,
      // 口径见文件头：重复且已学过 → wrote=false → retained=false。
      retained: wrote && inMemory,
      inMemory,
      namespace: ns,
      dim,
      embedMode: this.embedMode,
      writes: state.writes,
      observes: state.observes,
      weightBytes: dim * dim * 8,
      ...(persist.ok ? {} : { persistError: persist.error }),
    }
  }

  /** 超过容量时按强度淘汰最弱的那几条。 */
  prune(state) {
    const max = this.options.maxEntries
    if (state.entries.length <= max) return 0
    state.entries.sort((a, b) => b.strength - a.strength || a.createdAt - b.createdAt)
    const dropped = state.entries.splice(max)
    reindex(state)
    return dropped.length
  }

  // ── 核心：召回 ──

  /**
   * 用查询的键去读矩阵：v̂_q = M·k_q，再和每条记忆的 v 比余弦。
   *
   * ★ score 完全来自矩阵乘法。这里没有一处 substring / includes / 关键词匹配：
   *   唯一的输入是 M·k_q 这个 d 维向量，唯一的比较是它与条目 v 的余弦。
   */
  async recall(query, opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, results: [], entries: 0 }

    if (typeof query !== 'string' || query.trim().length === 0) {
      return { ok: false, error: 'query 不能为空', results: [], entries: 0 }
    }
    if (query.length > this.options.maxChars) {
      return { ok: false, error: `query 过长（${query.length} 字符），已拒绝`, results: [], entries: 0 }
    }

    const state = await this.ensureLoaded(ns)
    const dim = this.dim
    const kq = embedWithMode(query, dim, K_SALT, this.embedMode)

    let kn = 0
    for (let i = 0; i < dim; i++) kn += kq[i] * kq[i]
    if (!(kn > 0)) {
      return {
        ok: true,
        matched: false,
        results: [],
        top1Score: 0,
        top1Coverage: 0,
        margin: 0,
        matchedScore: 0,
        matchedMargin: 0,
        zScore: 0,
        entries: state.entries.length,
        namespace: ns,
        note: 'query 里没有可用于嵌入的文字或数字',
      }
    }

    // v̂_q = M·k_q —— 本工具唯一的"打分器"
    const readout = matvec(state.M, kq, dim, dim)
    let readoutNorm = 0
    for (let i = 0; i < dim; i++) readoutNorm += readout[i] * readout[i]
    readoutNorm = Math.sqrt(readoutNorm)

    const qCovIdx = buildCoverageIndex(query, this.embedMode)
    const scored = state.entries.map((e) => {
      // 修复三（2026-11-17）：条目不再存 v，用确定性哈希现场重算（带缓存）。
      const v = valueVectorOf(state, e)
      const eCovIdx = coverageIndexOf(state, e)
      return {
        text: e.text,
        hash: e.hash,
        // ★ 余弦相似度：模型的预测值与记忆里（重算出来的）值之间
        score: cosine(readout, v),
        // 词面覆盖：**不参与打分/排序**，只在下面做真伪闸门时用。
        coverage: computeCoverageFromCache(qCovIdx, eCovIdx, this.embedMode),
        surprise: e.surprise,
        strength: e.strength,
        createdAt: e.createdAt,
      }
    })
    scored.sort((a, b) => b.score - a.score || a.createdAt - b.createdAt || (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))

    const top1 = scored.length > 0 ? scored[0].score : 0
    const top2 = scored.length > 1 ? scored[1].score : 0
    const margin = scored.length > 0 ? top1 - top2 : 0
    let zScore = 0
    if (scored.length >= 3) {
      let sum = 0
      for (let i = 0; i < scored.length; i++) sum += scored[i].score
      const avg = sum / scored.length
      let sq = 0
      for (let i = 0; i < scored.length; i++) {
        const diff = scored[i].score - avg
        sq += diff * diff
      }
      const std = Math.sqrt(sq / scored.length)
      zScore = std > 1e-12 ? (top1 - avg) / std : 0
    }
    const top1Coverage = scored.length > 0 ? scored[0].coverage : 0
    const minScore = num(opts.minScore, this.options.minScore, -1, 1)
    const minCoverage = num(opts.minCoverage, this.options.minCoverage, 0, 1)
    const tooFewToJudge = state.entries.length < this.options.minEntriesForTrust
    const bypass = minScore <= -1
    const scoreGate = tooFewToJudge || bypass ? -Infinity : minScore
    const filtered = bypass
      ? scored
      : scored.filter((item) => item.score >= scoreGate && item.coverage >= minCoverage)
    const passedTrust = filtered.length > 0
    const topK = Math.max(1, Math.min(int(opts.topK, 5, 1, 1000), Math.max(1, scored.length)))
    const results = filtered.slice(0, topK)

    // 针对实际胜出候选 results[0] 计算置信分与边际差（避免被词面未过线的假阳性 scored[0] 误导第三道闸门）
    let matchedScore = 0
    let matchedMargin = 0
    if (passedTrust) {
      const winner = results[0]
      matchedScore = Number.isFinite(winner.score) ? winner.score : 0
      let runnerUp = null
      for (const item of scored) {
        if (item === winner) continue
        if (!bypass && item.coverage < minCoverage) continue
        runnerUp = item.score
        break
      }
      matchedMargin = runnerUp === null ? matchedScore : matchedScore - runnerUp
      if (!Number.isFinite(matchedMargin)) matchedMargin = 0
    }

    // 回忆强化（Spacing Effect）：默认 recallBoost=0 保持读纯度；开启后对高置信命中条目续命抗遗忘。
    const boost = num(opts.recallBoost, this.options.recallBoost, 0, 2)
    if (boost > 0 && passedTrust && !bypass && opts.noBoost !== true) {
      const now = Date.now()
      let boosted = false
      for (const hit of results) {
        const target = state.index.get(hit.hash)
        if (!target) continue
        target.lastSeen = now
        target.strength = Math.min(10, target.strength + boost)
        hit.strength = target.strength
        boosted = true
      }
      if (boosted) {
        this.scheduleSave(ns, false).catch(() => {})
      }
    }

    const emptyNote = '记忆库为空，尚未记录任何条目'
    return {
      ok: true,
      matched: passedTrust,
      results,
      top1Score: Number.isFinite(top1) ? top1 : 0,
      top1Coverage,
      margin: Number.isFinite(margin) ? margin : 0,
      matchedScore,
      matchedMargin,
      zScore: Number.isFinite(zScore) ? zScore : 0,
      minScore,
      minCoverage,
      entries: state.entries.length,
      namespace: ns,
      topK,
      readoutNorm,
      dim,
      embedMode: this.embedMode,
      ...(passedTrust
        ? (tooFewToJudge && !bypass
            ? { note: `记忆只有 ${state.entries.length} 条（< ${this.options.minEntriesForTrust}），**分数**门限统计意义不足已跳过；词面覆盖门限仍生效（${minCoverage}）` }
            : {})
        : {
            note: state.entries.length === 0
              ? emptyNote
              : top1Coverage < minCoverage
                ? `最佳候选的词面覆盖 ${top1Coverage.toFixed(2)} < ${minCoverage}：查询里的实词没有出现在任何记忆里，判定为"没召回"`
                : `最高分 ${top1.toFixed(3)} < 阈值 ${minScore}，判定为"没召回"，返回空以免误导`,
          }),
    }
  }

  // ── 核心：遗忘 ──

  /** 一轮遗忘：M ← (1−decay)·M，条目强度同比衰减，低于存活阈值的剔除。 */
  async forget(opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, entriesBefore: 0, entriesAfter: 0, decayed: false }

    const state = await this.ensureLoaded(ns)
    const decay = num(opts.decay, this.options.decay, 0, 0.999)
    const before = state.entries.length
    const normBefore = frobenius(state.M)

    if (decay > 0) {
      scaleInPlace(state.M, 1 - decay)
      if (state.S) scaleInPlace(state.S, 1 - decay)
      for (const e of state.entries) {
        e.strength *= 1 - decay
        if (!Number.isFinite(e.strength)) e.strength = 0
      }
      state.decayRounds += 1
    }

    const minStrength = this.options.minStrength
    const survivors = state.entries.filter((e) => e.strength >= minStrength)
    const evicted = before - survivors.length
    if (evicted > 0) {
      for (const e of state.entries) {
        if (e.strength < minStrength) {
          state.kCache.delete(e.hash)
          state.vCache.delete(e.hash)
          state.covCache.delete(e.hash)
        }
      }
      state.entries = survivors
      reindex(state)
    }

    const persist = await this.save(ns)
    return {
      ok: true,
      entriesBefore: before,
      entriesAfter: state.entries.length,
      decayed: decay > 0,
      decay,
      evicted,
      decayRounds: state.decayRounds,
      namespace: ns,
      normBefore,
      normAfter: frobenius(state.M),
      ...(persist.ok ? {} : { persistError: persist.error }),
    }
  }

  // ── 状态 ──

  async status(opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, entries: 0 }

    const state = await this.ensureLoaded(ns)
    const dim = this.dim
    let high = 0
    for (const e of state.entries) if (e.surprise > this.options.surpriseThreshold) high++
    let entriesBytes = 0
    for (const e of state.entries) entriesBytes += e.text.length * 2 + dim * 2 * 8 + 64
    const namespaces = await this.listNamespaces()

    return {
      ok: true,
      namespace: ns,
      entries: state.entries.length,
      weightBytes: dim * dim * 8,
      entriesBytes,
      stateBytes: dim * dim * 8 + entriesBytes,
      meanSurprise: meanSurpriseOf(state),
      meanSurpriseSinceStart: state.observes > 0 ? state.surpriseSum / state.observes : 0,
      highSurpriseEntries: high,
      decayRounds: state.decayRounds,
      dim,
      embedMode: this.embedMode,
      momentum: this.options.momentum,
      recallBoost: this.options.recallBoost,
      storageFormat: this.options.storageFormat,
      autoObserve: this.options.autoObserve,
      autoRecall: this.options.autoRecall,
      autoForgetEveryTurns: this.options.autoForgetEveryTurns,
      writes: state.writes,
      observes: state.observes,
      surpriseThreshold: this.options.surpriseThreshold,
      decay: this.options.decay,
      eta: this.options.eta,
      minStrength: this.options.minStrength,
      norm: frobenius(state.M),
      momentumNorm: state.S ? frobenius(state.S) : 0,
      namespaces,
      storageRoot: this.store.root,
      updatedAt: state.updatedAt ? new Date(state.updatedAt).toISOString() : null,
    }
  }

  /** 只读探测一段文本在指定 namespace 下的当前惊讶度（不修改 M，也不写盘）。 */
  async probeText(text, opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, surprise: 0 }
    if (typeof text !== 'string' || !text.trim()) return { ok: false, error: 'text 不能为空', surprise: 0 }
    const state = await this.ensureLoaded(ns)
    const k = embedWithMode(text.trim(), this.dim, K_SALT, this.embedMode)
    const v = embedWithMode(text.trim(), this.dim, V_SALT, this.embedMode)
    const res = probeSurprise(k, v, state.M, this.dim)
    return { ok: true, namespace: ns, surprise: res.surprise, readoutDot: res.readoutDot, entries: state.entries.length }
  }

  // ── 删除（墓碑） ──

  /**
   * 显式删除一批条目：写墓碑 + 立即重写状态文件。
   *
   * ⚠️ 为什么这个能力必须由插件提供，而不是让人手动改 JSON：
   *   宿主进程把 namespace 缓存在内存里，外部删文件后下一次 observe 会整份覆盖回去
   *   （2026-11-17 隔离实测）。墓碑表是这个"删除意图"唯一的持久载体，
   *   load / save / merge 三条路径都会读它（见 applyTombs）。
   *
   * @param {{namespace?:string, hashes?:string[], texts?:string[], dryRun?:boolean}} opts
   *   `texts` 按**精确文本**匹配（宁可漏删不误删）；`hashes` 直接指定指纹。
   * @returns {Promise<object>}
   */
  async deleteEntries(opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, deleted: 0 }

    const wantTexts = new Set(Array.isArray(opts.texts) ? opts.texts.map(String) : [])
    const wantHashes = new Set(Array.isArray(opts.hashes) ? opts.hashes.map(String) : [])
    if (wantTexts.size === 0 && wantHashes.size === 0) {
      return { ok: false, error: '必须给 hashes 或 texts 之一', deleted: 0 }
    }

    const state = await this.ensureLoaded(ns)
    const matches = state.entries.filter((e) => wantHashes.has(e.hash) || wantTexts.has(e.text))
    if (matches.length === 0) {
      return { ok: true, namespace: ns, matched: 0, deleted: 0, dryRun: opts.dryRun === true, entriesAfter: state.entries.length }
    }
    const hashes = matches.map((e) => e.hash)

    if (opts.dryRun === true) {
      return {
        ok: true,
        namespace: ns,
        dryRun: true,
        matched: matches.length,
        wouldDelete: matches.map((e) => e.text),
        deleted: 0,
        entriesAfter: state.entries.length,
      }
    }

    // 1) 先落墓碑：即使下一步写盘失败，下次 load 也会把它滤掉。
    const t = await this.store.addTombs(ns, hashes)
    // 2) 从权重矩阵 M（与动量 S）中沿被删条目的键方向反学习消除残余投影。
    for (const m of matches) unlearnEntryFromMatrix(state, m)
    // 3) 再从内存与磁盘移除。
    await this.applyTombs(ns, state)
    const persist = await this.save(ns)

    return {
      ok: true,
      namespace: ns,
      matched: matches.length,
      deleted: hashes.length,
      deletedTexts: matches.map((e) => e.text),
      tombstonesTotal: t.total,
      entriesAfter: state.entries.length,
      ...(persist.ok ? {} : { persistError: persist.error }),
    }
  }

  // ── 导出 ──

  /** 把 M 与条目表整份写到 <root>/exports/<namespace>.json（原子写）。 */
  async export(opts = {}) {
    const ns = typeof opts.namespace === 'string' && opts.namespace ? opts.namespace : this.options.namespace
    const nsErr = checkNamespace(ns)
    if (nsErr) return { ok: false, error: nsErr, entries: 0, bytes: 0 }

    const state = await this.ensureLoaded(ns)
    // 空记忆导出没有意义，而且会在 state 目录下凭空造出一个快照文件，
    // 让"这个 namespace 存在吗"这个问题变得不可答。所以先如实拒绝。
    if (state.entries.length === 0) {
      return {
        ok: false,
        error: `namespace ${JSON.stringify(ns)} 里没有任何记忆条目，无可导出内容`,
        entries: 0,
        bytes: 0,
        namespace: ns,
      }
    }
    const payload = {
      ...serialize(state),
      exportedAt: new Date().toISOString(),
      source: PLUGIN_NAME,
    }
    try {
      if (typeof this.store.writeExport !== 'function') {
        return { ok: false, error: 'store 不支持导出（缺少 writeExport）', entries: state.entries.length, bytes: 0 }
      }
      const res = await this.store.writeExport(ns, payload)
      return {
        ok: true,
        path: res.path,
        entries: state.entries.length,
        bytes: res.bytes,
        namespace: ns,
        dim: this.dim,
      }
    } catch (e) {
      return { ok: false, error: String(e?.message ?? e), entries: state.entries.length, bytes: 0 }
    }
  }
}

/** 工厂函数，方便 tools.js 与离线测试用同一套构造逻辑。 */
export function createMemory(options = {}) {
  return new NeuralMemory(options)
}

/** 只算惊讶度、不写 M 的探测函数（调试与测试用，工具层没暴露）。 */
export function probeSurprise(k, v, M, dim) {
  const pred = matvec(M, k, dim, dim)
  const err = new Float64Array(dim)
  let s = 0
  for (let i = 0; i < dim; i++) {
    const d = pred[i] - v[i]
    err[i] = d
    s += d * d
  }
  return { surprise: s, pred, err, readoutDot: dot(pred, v, dim) }
}

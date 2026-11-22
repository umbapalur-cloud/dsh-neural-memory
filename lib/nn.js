/**
 * core/nn.js —— 三个插件共用的数值内核
 *
 * 为什么要自己写而不引依赖：
 *   1. 这三个插件的价值就在"真的在算"，引一个 autograd 库反而看不清梯度到底怎么走的；
 *   2. 插件装进 profile 后零依赖最稳，不会因为依赖树变化被禁用；
 *   3. 矩阵操作用 Float64Array 扁平存储，比 number[][] 快 3~5 倍且省一半内存。
 *
 * 所有矩阵一律 row-major 扁平 Float64Array：A[i][j] === A[i*n + j]，m 行 n 列。
 * 所有随机一律走 mulberry32(seed)，同 seed 必须逐位复现（测试要靠这个比 baseline）。
 *
 * 实现者请把本文件复制到各自插件的 lib/nn.js，之后可以往上加，
 * 但**不许删或改已有导出**——别人的测试可能在用。
 */

// ────────────────────────────── 随机 ──────────────────────────────

/**
 * 32 位确定性伪随机数发生器。不用 Math.random，因为测试要能复现同一个 baseline。
 * 返回 [0,1) 的函数。
 */
export function mulberry32(seed) {
  let a = seed >>> 0
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 高斯分布（Box–Muller），用于初始化权重。 */
export function gaussian(rand) {
  let u = 0
  let v = 0
  while (u === 0) u = rand()
  while (v === 0) v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// ────────────────────────────── 分配 ──────────────────────────────

export function zeros(n) {
  return new Float64Array(n)
}

/**
 * 按「fan-in 缩放」的高斯初始化。标准差取 1/sqrt(k)，这是线性层最常见的做法，
 * 能让前向输出的方差大致保持稳定，不至于一上来就爆。
 */
export function randMatrix(m, n, rand, scale) {
  const out = new Float64Array(m * n)
  const std = scale ?? 1 / Math.sqrt(Math.max(1, n))
  for (let i = 0; i < out.length; i++) out[i] = gaussian(rand) * std
  return out
}

export function randVector(n, rand, scale) {
  return randMatrix(1, n, rand, scale)
}

// ────────────────────────────── 基本运算 ──────────────────────────────

export function dot(a, b, n) {
  let s = 0
  for (let i = 0; i < n; i++) s += a[i] * b[i]
  return s
}

export function norm(a) {
  return Math.sqrt(dot(a, a, a.length))
}

/** out = A(m×k) · B(k×n)，返回新矩阵 m×n。 */
export function matmul(A, B, m, k, n) {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    const aRow = i * k
    const oRow = i * n
    for (let p = 0; p < k; p++) {
      const a = A[aRow + p]
      if (a === 0) continue
      const bRow = p * n
      for (let j = 0; j < n; j++) out[oRow + j] += a * B[bRow + j]
    }
  }
  return out
}

/** y = A(m×n) · x(n)，返回新向量 m。 */
export function matvec(A, x, m, n) {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    const row = i * n
    let s = 0
    for (let j = 0; j < n; j++) s += A[row + j] * x[j]
    out[i] = s
  }
  return out
}

/** y += A(m×n) · x(n)，原地累加，省一次分配。 */
export function matvecAdd(y, A, x, m, n) {
  for (let i = 0; i < m; i++) {
    const row = i * n
    let s = 0
    for (let j = 0; j < n; j++) s += A[row + j] * x[j]
    y[i] += s
  }
  return y
}

/** out = A ⊗ bᵀ，即外积：A(m×1) · b(1×n) → m×n。用于构造梯度。 */
export function outer(a, b, m, n) {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    const a0 = a[i]
    const row = i * n
    for (let j = 0; j < n; j++) out[row + j] = a0 * b[j]
  }
  return out
}

/** 原地：A += scale · B。用于 SGD 更新时避免每次 new。 */
export function addScaled(A, B, scale) {
  for (let i = 0; i < A.length; i++) A[i] += scale * B[i]
  return A
}

/** 原地：A *= scale。权重衰减用它。 */
export function scaleInPlace(A, scale) {
  for (let i = 0; i < A.length; i++) A[i] *= scale
  return A
}

export function elu(x) {
  return x > 0 ? x : Math.exp(x) - 1
}

/** 线性注意力的特征映射 φ(x)=elu(x)+1，保证非负，从而状态矩阵单调累加。 */
export function phi(x) {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = elu(x[i]) + 1
  return out
}

export function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x))
  const e = Math.exp(x)
  return e / (1 + e)
}

// ────────────────────────────── 概率 ──────────────────────────────

/**
 * softmax（数值稳定版）：先减最大值。
 * 不稳定的实现会在 logits 到 50 左右时就溢出成 NaN，训练到一半突然全 NaN 很难查。
 */
export function softmax(logits) {
  const n = logits.length
  let max = -Infinity
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i]
  const out = new Float64Array(n)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max)
    out[i] = e
    sum += e
  }
  if (sum === 0 || !Number.isFinite(sum)) {
    // 极端情况：全 -Infinity。退化成均匀分布，保证下游不出 NaN。
    for (let i = 0; i < n; i++) out[i] = 1 / n
    return out
  }
  for (let i = 0; i < n; i++) out[i] /= sum
  return out
}

export function logSoftmax(logits) {
  const n = logits.length
  let max = -Infinity
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i]
  let sum = 0
  for (let i = 0; i < n; i++) sum += Math.exp(logits[i] - max)
  const logSum = Math.log(sum) + max
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = logits[i] - logSum
  return out
}

/** 交叉熵（自然对数）。target 是类别下标。 */
export function crossEntropy(logits, target) {
  const ls = logSoftmax(logits)
  const v = -ls[target]
  return Number.isFinite(v) ? v : 60 // 截断，避免单个坏样本把平均值拉到 Infinity
}

/** 按概率采样（temperature 越低越贪心）。 */
export function sample(probs, rand) {
  const r = rand()
  let acc = 0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]
    if (r <= acc) return i
  }
  return probs.length - 1
}

/** argmax，平局取下标最小的，保证同 seed 同结果。 */
export function argmax(arr) {
  let best = 0
  let bestV = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > bestV) {
      bestV = arr[i]
      best = i
    }
  }
  return best
}

// ────────────────────────────── 优化器 ──────────────────────────────

/**
 * Adam 状态。测试时训练（TTT）里用它比纯 SGD 稳得多——
 * 因为同一个 W 要在极短的窗口内反复吃梯度，纯 SGD 很快会学到一个只会输出高频字符的退化解。
 */
export function createAdam(shape, opts = {}) {
  const { lr = 0.02, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = opts
  const n = shape.reduce((a, b) => a * b, 1)
  return {
    lr,
    beta1,
    beta2,
    eps,
    t: 0,
    m: new Float64Array(n),
    v: new Float64Array(n),
  }
}

/** 用梯度 g 更新参数 p（原地）。 */
export function adamStep(adam, p, g) {
  adam.t += 1
  const { lr, beta1, beta2, eps } = adam
  const bc1 = 1 - Math.pow(beta1, adam.t)
  const bc2 = 1 - Math.pow(beta2, adam.t)
  for (let i = 0; i < p.length; i++) {
    const gi = g[i]
    adam.m[i] = beta1 * adam.m[i] + (1 - beta1) * gi
    adam.v[i] = beta2 * adam.v[i] + (1 - beta2) * gi * gi
    const mHat = adam.m[i] / bc1
    const vHat = adam.v[i] / bc2
    p[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps)
  }
  return p
}

// ────────────────────────────── 序列化 ──────────────────────────────

/** Float64Array → 普通数组。JSON 不认 TypedArray，直接 stringify 会变成 {"0":..} 对象。 */
export function toPlain(arr) {
  return Array.from(arr)
}

/** 普通数组 → Float64Array。读盘时用。 */
export function fromPlain(obj) {
  if (obj instanceof Float64Array) return obj
  if (Array.isArray(obj)) return Float64Array.from(obj)
  throw new TypeError('期望数组，收到 ' + typeof obj)
}

/** 取浮点均值，用于报"平均惊讶度"这类统计。 */
export function mean(arr) {
  if (arr.length === 0) return 0
  let s = 0
  for (let i = 0; i < arr.length; i++) s += arr[i]
  return s / arr.length
}

/** 余弦相似度，长度 0 或零向量返回 0 而不是 NaN。 */
export function cosine(a, b) {
  const na = norm(a)
  const nb = norm(b)
  if (na === 0 || nb === 0) return 0
  return dot(a, b, a.length) / (na * nb)
}

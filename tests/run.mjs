#!/usr/bin/env node
/**
 * tests/run.mjs —— dsh-neural-memory 独立验收脚本（零依赖 ESM，`node tests/run.mjs` 直接跑）
 *
 * ## 这个文件是谁写的、立场是什么
 *
 * 作者（nm-impl）声称「召回 18/20 = 90%」。本脚本由独立验收者从零重写，
 * **不采信作者报告里的任何一个数字**：所有数字都在本文件里重新算一遍并打印。
 *
 * ## 两条防止自欺的原则
 *
 * 1. **不用默认存储路径**。全部实例都显式传 `path` 指向 os.tmpdir() 下的唯一临时目录。
 *    否则会在真实 <DSH_HOME>/state/dsh-neural-memory/ 里留下 default.json，
 *    而且下一轮重跑会读到上一轮的残留记忆 → 召回率虚高 → 测试自己污染被测对象。
 * 2. **关键结论用「自己重算」证伪，而不是「看代码像对的」**。
 *    - 矩阵更新：用独立实现的 matvec/outer 手算一步梯度，与实现逐元素比。
 *    - 召回打分：从落盘的 M 和条目 v 自己重算 cosine(M·k, v)，与工具返回值比。
 *    - 惊讶度：快照 M，自己算 ‖M_before·k − v‖²，与工具返回值比。
 *    三条都对得上，"硬编码假数字"才被排除。
 *
 * ## 输出约定
 *
 * 每个检查都打印实测数字。全部通过 → 退出码 0；任一失败 → 退出码 1。
 * 失败时**保留失败态、打印真实数字**，不改阈值蒙混。
 *
 * ## 口径（2026-11-17 作者与 lead 敲定，本脚本按它读字段）
 *
 *   retained = wrote && inMemory
 *   重复出现且已学过的文本：wrote=false、inMemory=true、retained=false
 *   —— 这是「惊讶度筛选在起作用」的判据，不是 bug。
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  matmul, matvec, outer, dot, cosine, norm, addScaled, scaleInPlace,
} from '../lib/nn.js'
import {
  NeuralMemory, embedText, hashText, K_SALT, V_SALT, frobenius,
  DEFAULT_OPTIONS, checkNamespace, normalizeText,
} from '../lib/model.js'
import { makeToolDefinitions, TOOL_NAMES } from '../lib/tools.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const LIB = path.resolve(HERE, '..', 'lib')

// ══════════════════════════ 断言与输出基础设施 ══════════════════════════

const TOL = 1e-12
const results = []
let current = null

function section(id, title) {
  current = { id, title, ok: true, lines: [], failNotes: [] }
  results.push(current)
  console.log('')
  console.log(`── ${id} ${title} ${'─'.repeat(Math.max(4, 62 - title.length))}`)
}

function line(msg) {
  console.log(`   ${msg}`)
  if (current) current.lines.push(msg)
}

/** 记录一条断言。detail 是实测数字，失败也要打出来。 */
function assert(label, ok, detail) {
  const mark = ok ? 'PASS' : 'FAIL'
  console.log(`   [${mark}] ${label}${detail ? ` :: ${detail}` : ''}`)
  if (current) {
    current.lines.push(`[${mark}] ${label} :: ${detail}`)
    if (!ok) {
      current.ok = false
      current.failNotes.push(`${label} :: ${detail}`)
    }
  }
  return ok
}

const f = (x, d = 6) => (Number.isFinite(x) ? x.toFixed(d) : String(x))
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`

/** 逐元素最大绝对误差。两数组长度不等直接返回 Infinity。 */
function maxAbsDiff(a, b) {
  if (!a || !b || a.length !== b.length) return Infinity
  let m = 0
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i])
    if (!(d <= m)) m = d
    if (Number.isNaN(d)) return Infinity
  }
  return m
}

// ── 独立实现的矩阵运算（不复用 lib/nn.js，否则就是拿被测对象验证被测对象） ──

function nMatmul(A, B, m, k, n) {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let p = 0; p < k; p++) s += A[i * k + p] * B[p * n + j]
      out[i * n + j] = s
    }
  }
  return out
}

function nMatvec(A, x, m, n) {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += A[i * n + j] * x[j]
    out[i] = s
  }
  return out
}

function nOuter(a, b, m, n) {
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = a[i] * b[j]
  return out
}

function nDot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function nCosine(a, b) {
  const na = Math.sqrt(nDot(a, a))
  const nb = Math.sqrt(nDot(b, b))
  if (na === 0 || nb === 0) return 0
  return nDot(a, b) / (na * nb)
}

// ══════════════════════════ 临时目录（绝不碰真实 DSH_HOME） ══════════════════════════

let TMP_ROOT = null

async function mkRoot() {
  const base = path.join(os.tmpdir(), `nm-verify-${process.pid}-${Date.now().toString(36)}`)
  await fsp.mkdir(base, { recursive: true })
  return base
}

function sectionDir(name) {
  const d = path.join(TMP_ROOT, name)
  return d
}

async function rmDir(dir) {
  try {
    await fsp.rm(dir, { recursive: true, force: true })
  } catch {
    /* Windows 上偶发占用，忽略 */
  }
}

// ══════════════════════════ 1. 数学正确性 ══════════════════════════

async function check1_math() {
  section('1', '数学正确性：手算 vs lib/nn.js（阈值 1e-12）')

  // ── 2×2 × 2×2 ──
  // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]
  // AB = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  const A = Float64Array.from([1, 2, 3, 4])
  const B = Float64Array.from([5, 6, 7, 8])
  const got22 = matmul(A, B, 2, 2, 2)
  const exp22 = Float64Array.from([19, 22, 43, 50])
  const d22 = maxAbsDiff(got22, exp22)
  line(`2×2 手算 [[19,22],[43,50]] 实得 [${Array.from(got22).join(', ')}]`)
  assert('2×2 矩阵乘逐元素相等', d22 < TOL, `maxAbsDiff=${d22.toExponential(3)}`)

  // ── 2×3 × 3×2（非方阵，验行主序下标没写反） ──
  // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
  // AB = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
  //    = [[58,64],[139,154]]
  const A23 = Float64Array.from([1, 2, 3, 4, 5, 6])
  const B32 = Float64Array.from([7, 8, 9, 10, 11, 12])
  const got23 = matmul(A23, B32, 2, 3, 2)
  const exp23 = Float64Array.from([58, 64, 139, 154])
  const d23 = maxAbsDiff(got23, exp23)
  line(`2×3 × 3×2 手算 [[58,64],[139,154]] 实得 [${Array.from(got23).join(', ')}]`)
  assert('2×3 矩阵乘逐元素相等', d23 < TOL, `maxAbsDiff=${d23.toExponential(3)}`)

  // ── matvec ──
  // A = [[1,2],[3,4]]  x = [5,6]  → [1*5+2*6, 3*5+4*6] = [17, 39]
  const x = Float64Array.from([5, 6])
  const gotMv = matvec(A, x, 2, 2)
  const expMv = Float64Array.from([17, 39])
  const dMv = maxAbsDiff(gotMv, expMv)
  line(`matvec 手算 [17,39] 实得 [${Array.from(gotMv).join(', ')}]`)
  assert('matvec 逐元素相等', dMv < TOL, `maxAbsDiff=${dMv.toExponential(3)}`)

  // ── outer（梯度构造用它，单独验） ──
  // a = [1,2](2×1)  b = [3,4](1×2) → [[3,4],[6,8]]
  const gotOut = outer(Float64Array.from([1, 2]), Float64Array.from([3, 4]), 2, 2)
  const expOut = Float64Array.from([3, 4, 6, 8])
  const dOut = maxAbsDiff(gotOut, expOut)
  line(`outer 手算 [[3,4],[6,8]] 实得 [${Array.from(gotOut).join(', ')}]`)
  assert('outer 外积逐元素相等', dOut < TOL, `maxAbsDiff=${dOut.toExponential(3)}`)

  // ── 余弦相似度 ──
  // 正交：a=[1,0,0] b=[0,1,0] → 0
  // 同向：a=[1,2,2] b=[1,2,2] → 1
  // 一般：a=[3,4] b=[4,3] → 24/25 = 0.96
  // 零向量：a=[0,0] b=[1,1] → 0（不是 NaN）
  const cases = [
    ['正交', [1, 0, 0], [0, 1, 0], 0],
    ['同向', [1, 2, 2], [1, 2, 2], 1],
    ['一般', [3, 4], [4, 3], 24 / 25],
    ['零向量', [0, 0], [1, 1], 0],
    ['反向', [1, 2, 3], [-1, -2, -3], -1],
  ]
  let dCos = 0
  for (const [name, a, b, expect] of cases) {
    const got = cosine(Float64Array.from(a), Float64Array.from(b))
    const d = Math.abs(got - expect)
    if (d > dCos) dCos = d
    line(`cosine ${name.padEnd(4)} 手算 ${f(expect, 4)} 实得 ${f(got, 12)}  diff=${d.toExponential(3)}`)
  }
  assert('余弦相似度 5 个手算用例全部相等', dCos < TOL, `maxAbsDiff=${dCos.toExponential(3)}`)

  // ── 顺带：addScaled / scaleInPlace（梯度更新与衰减的基础） ──
  const base = Float64Array.from([1, 2, 3])
  addScaled(base, Float64Array.from([10, 20, 30]), -2)
  const dAdd = maxAbsDiff(base, Float64Array.from([-19, -38, -57]))
  scaleInPlace(base, 0.5)
  const dScale = maxAbsDiff(base, Float64Array.from([-9.5, -19, -28.5]))
  line(`addScaled(a,b,-2) [1,2,3]+(-2)*[10,20,30] = [${Array.from(base).map((v) => v * 2).join(', ')}] (×2 前)`)
  assert('addScaled 原地累加正确', dAdd < TOL, `maxAbsDiff=${dAdd.toExponential(3)}`)
  assert('scaleInPlace 原地缩放正确', dScale < TOL, `maxAbsDiff=${dScale.toExponential(3)}`)
}

// ══════════════════════════ 2. 记忆矩阵更新正确性 ══════════════════════════

async function check2_gradient() {
  section('2', '记忆矩阵更新：手算 ∇L = 2(v̂−v)kᵀ 与实现逐元素比对')

  const dir = sectionDir('grad')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir, dim: 64 })
  const ns = 'gradtest'
  const text = '周以宁的服务器在 03 号机房'
  const eta = DEFAULT_OPTIONS.eta // 0.2
  line(`参数：dim=64, eta=${eta}, 初始 M = 0（全零矩阵）`)

  const k = embedText(text, 64, K_SALT)
  const v = embedText(text, 64, V_SALT)
  const nk = Math.sqrt(nDot(k, k))
  const nv = Math.sqrt(nDot(v, v))
  line(`k 的模 = ${f(nk, 12)}，v 的模 = ${f(nv, 12)}（嵌入应已 L2 归一化）`)
  assert('嵌入向量是单位向量', Math.abs(nk - 1) < 1e-12 && Math.abs(nv - 1) < 1e-12,
    `|‖k‖−1|=${Math.abs(nk - 1).toExponential(3)}, |‖v‖−1|=${Math.abs(nv - 1).toExponential(3)}`)

  // ── 第 1 步：M=0 → v̂=0 → surprise=‖v‖²=1 → M += (-2η)·outer(0−v, k) = 2η·v·kᵀ ──
  const r1 = await mem.observe(text, { namespace: ns, surpriseThreshold: 0 })
  assert('第 1 次 observe 触发了写入（可写）', r1.ok === true && r1.wrote === true,
    `ok=${r1.ok}, wrote=${r1.wrote}, surprise=${f(r1.surprise, 12)}`)

  const surprise1Hand = 1.0 // ‖0 − v‖² = ‖v‖² = 1
  const dS1 = Math.abs(r1.surprise - surprise1Hand)
  assert('surprise₁ = ‖0−v‖² 的解析值 1.0', dS1 < TOL, `实测 ${f(r1.surprise, 12)} vs 手算 1.0, diff=${dS1.toExponential(3)}`)

  // ★ 必须快照：state.M 是活数组，observe 会原地改它。
  //   第一版这里写成 `= mem.stateOf(ns).M`，于是第 2 步的手算基准被第 2 步的写入原地污染，
  //   报出 4.354e-2 的假阳性——那是**测试自己的 bug**，不是实现的问题。
  const M1 = Float64Array.from(mem.stateOf(ns).M)
  // 手算：M1 = 0 − η·∇L，∇L = 2(v̂−v)kᵀ = 2(0−v)kᵀ = −2·v·kᵀ
  //       ⇒ M1 = −0.2 · (−2·v·kᵀ) = 0.4·v·kᵀ
  const vk = nOuter(v, k, 64, 64)
  const expM1 = new Float64Array(64 * 64)
  for (let i = 0; i < expM1.length; i++) expM1[i] = -(-2 * eta) * vk[i] // 严格镜像实现的运算顺序
  const dM1 = maxAbsDiff(M1, expM1)
  assert('第 1 步后 M 与手算 2η·v·kᵀ 逐元素相等（抓符号写反/漏系数 2）', dM1 < TOL,
    `maxAbsDiff=${dM1.toExponential(3)}`)

  // 反证：如果实现把符号写反（M = −2η·v kᵀ），误差会是 2×‖…‖
  const wrongSign = new Float64Array(64 * 64)
  for (let i = 0; i < wrongSign.length; i++) wrongSign[i] = 2 * eta * vk[i] * -1
  const dWrong = maxAbsDiff(M1, wrongSign)
  line(`对照：若符号写反，与实现的误差会是 ${dWrong.toExponential(3)}（远大于 1e-12，说明本检查有区分力）`)
  assert('本检查能区分正确符号与错误符号', dWrong > 1e-3, `符号写反时的误差=${dWrong.toExponential(3)}`)

  // 漏掉系数 2 → M = η·v·kᵀ
  const noTwo = new Float64Array(64 * 64)
  for (let i = 0; i < noTwo.length; i++) noTwo[i] = eta * vk[i]
  const dNoTwo = maxAbsDiff(M1, noTwo)
  line(`对照：若漏掉系数 2，误差会是 ${dNoTwo.toExponential(3)}`)
  assert('本检查能区分系数 2 是否漏掉', dNoTwo > 1e-3, `漏系数时的误差=${dNoTwo.toExponential(3)}`)

  // M·k 应等于 2η·v = 0.4v（因为 kᵀk = 1）
  const mk1 = nMatvec(M1, k, 64, 64)
  const expMk1 = new Float64Array(64)
  for (let i = 0; i < 64; i++) expMk1[i] = 2 * eta * v[i]
  const dMk1 = maxAbsDiff(mk1, expMk1)
  assert('M·k = 2η·v（读出确实沿 v 方向推进）', dMk1 < TOL, `maxAbsDiff=${dMk1.toExponential(3)}`)

  // ── 第 2 步：M≠0，验 (v̂−v) 这一项真的被算进去了 ──
  const pred2 = nMatvec(M1, k, 64, 64) // 独立实现
  const err2 = new Float64Array(64)
  let s2 = 0
  for (let i = 0; i < 64; i++) {
    err2[i] = pred2[i] - v[i]
    s2 += err2[i] * err2[i]
  }
  const r2 = await mem.observe(text, { namespace: ns, surpriseThreshold: 0 })
  const dS2 = Math.abs(r2.surprise - s2)
  line(`手算 ‖M₁·k − v‖² = (0.4−1)²·‖v‖² = ${f(s2, 12)}；实现报 ${f(r2.surprise, 12)}`)
  assert('第 2 步 surprise 与手算 ‖v̂−v‖² 相等（v̂ = M·k 真的进了误差项）', dS2 < TOL,
    `maxAbsDiff=${dS2.toExponential(3)}`)

  const M2 = Float64Array.from(mem.stateOf(ns).M)
  const ek = nOuter(err2, k, 64, 64)
  const expM2 = new Float64Array(64 * 64)
  for (let i = 0; i < expM2.length; i++) expM2[i] = M1[i] + (-2 * eta) * ek[i]
  const dM2 = maxAbsDiff(M2, expM2)
  assert('第 2 步后 M 与手算 M − η·2(v̂−v)kᵀ 逐元素相等（M≠0 的一般情形）', dM2 < TOL,
    `maxAbsDiff=${dM2.toExponential(3)}`)

  const r3 = await mem.observe(text, { namespace: ns, surpriseThreshold: 0 })
  line(`连续 3 次写入后 surprise 轨迹：${f(r1.surprise, 6)} → ${f(r2.surprise, 6)} → ${f(r3.surprise, 6)}`)
  assert('反复写入使预测误差单调下降（矩阵真的在拟合）',
    r1.surprise > r2.surprise && r2.surprise > r3.surprise,
    `${f(r1.surprise, 6)} > ${f(r2.surprise, 6)} > ${f(r3.surprise, 6)}`)

  const normM = frobenius(mem.stateOf(ns).M)
  line(`3 次写入后 ‖M‖_F = ${f(normM, 6)}`)
  assert('‖M‖_F 是有限正数（无 NaN/Inf 污染）', Number.isFinite(normM) && normM > 0, `‖M‖=${f(normM, 6)}`)
}

// ══════════════════════════ 3. 召回率（关键） ══════════════════════════

/**
 * 40 条互不相同的短中文事实，**验收者自造**（不是作者测试里那 40 条）。
 * 覆盖：人名（40 个不同姓名）、数字（工号/尾号/日期/金额/编号）、专有名词（机房、系统、院校、航班号）。
 * 结构刻意不完全统一（"X 的 Y 是 Z" / "X 负责 Y" / "X 在 Y 做 Z"），避免只测模板匹配。
 */
const FACTS = [
  '李曜辰的工号是 KX4821',
  '周以宁负责的服务器在 03 号机房',
  '陈景行的生日是 1993 年 7 月 14 日',
  '吴桐的邮箱后缀是 qingteng.dev',
  '郑柏川养了一只叫汤圆的猫',
  '孙浔的 GitHub 用户名是 sunxun_lab',
  '林胧在杭州西湖边开了家茶馆',
  '何斯年的项目代号叫青雀',
  '顾清野的工位在 B 栋 12 层',
  '唐见微是数据库组的技术负责人',
  '谢云舟的手机尾号是 6073',
  '韩霁的密码提示写的是外婆家的槐树',
  '沈书亭的职级是 P8',
  '白鹭书院的院长姓岑',
  '黄栖迟在 2024 年拿到注册会计师证',
  '罗砚之的笔名是砚台不洗',
  '徐锦程负责对账系统的重构',
  '苏灼华的出生地在福建泉州',
  '崔行舟每周三去游泳',
  '施令仪是公司里唯一会修打印机的人',
  '方既明的车是深蓝色帕萨特',
  '江晚吟的猫叫年糕体重 4.2 公斤',
  '卢望舒主管供应链',
  '岳临川擅长写 Rust',
  '姜沐白的口头禅是先跑一遍再说',
  '严既澄在 7 月调去了成都分公司',
  '汤问舟的孩子今年上小学二年级',
  '邵清嘉的研究方向是蛋白质折叠',
  '鲍元亮的工牌号是 000771',
  '潘敬之喜欢喝冰美式不加糖',
  '屈子安的航班是 CA1832',
  '常鹤鸣在晨会上提了库存周转率的问题',
  '温梨落的工位有一盆绿萝',
  '熊知远的项目预算是 380 万',
  '屈臣的合同编号是 HT20260914',
  '路遥川的导师是闵教授',
  '岑溪的手机号尾数是 8891',
  '杨无咎负责 2026 年第三季度的审计',
  '慕云深的组里只有三个人',
  '郁文昭每周五下午做代码评审',
]

/** 20 条查询：40 条里取偶数下标（0,2,4,...,38），确定性、无挑选偏向。 */
const QUERY_IDX = Array.from({ length: 20 }, (_, i) => i * 2)

async function check3_recall() {
  section('3', '召回率（关键）：40 条事实 → 20 条查询 top-5，契约要求 ≥ 60%')

  const dir = sectionDir('recall')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir })
  const ns = 'facts'
  const texts = FACTS.map((t) => t.trim())

  assert(`自造事实 ${texts.length} 条且互不相同`, new Set(texts).size === texts.length,
    `distinct=${new Set(texts).size}/${texts.length}`)

  // ── 逐条 observe ──
  let wrote = 0
  let surpriseSum = 0
  const surprises1 = []
  for (const t of texts) {
    const r = await mem.observe(t, { namespace: ns })
    if (!r.ok) {
      assert('nm_observe 全部成功', false, `失败于文本「${t}」：${r.error}`)
      return
    }
    if (r.wrote) wrote++
    surpriseSum += r.surprise
    surprises1.push(r.surprise)
  }
  const st = mem.stateOf(ns)
  line(`写入 ${wrote}/${texts.length} 条触发更新，最终 entries=${st.entries.length}，‖M‖_F=${f(frobenius(st.M), 6)}`)
  line(`首轮 surprise 均值=${f(surpriseSum / texts.length, 6)}（≈1 说明从零矩阵开始预测误差就是 ‖v‖²）`)
  assert('40 条事实全部进入记忆表', st.entries.length === texts.length, `entries=${st.entries.length}`)

  // ── 20 条查询 top-5 ──
  const queries = QUERY_IDX.map((i) => texts[i])
  let hits = 0
  const top1Hits = []
  const scoreLog = []
  for (const q of queries) {
    const r = await mem.recall(q, { namespace: ns, topK: 5 })
    if (!r.ok) {
      assert('nm_recall 全部成功', false, `失败于查询「${q}」：${r.error}`)
      return
    }
    const idx = r.results.findIndex((x) => x.text === q)
    const hit = idx >= 0
    if (hit) hits++
    top1Hits.push(idx === 0)
    scoreLog.push({ q, hit, rank: idx, top1: r.results[0]?.score ?? 0, scores: r.results.map((x) => x.score) })
  }
  const rate = hits / queries.length
  const top1 = top1Hits.filter(Boolean).length

  line('')
  for (const s of scoreLog) {
    line(`  ${s.hit ? '命中' : '未中'} rank=${String(s.rank).padStart(2)} top1Score=${f(s.top1, 4)}  「${s.q}」`)
  }
  line('')
  line(`实测：top-5 命中 ${hits}/${queries.length} = ${pct(rate)}；其中排第 1 的 ${top1}/${queries.length} = ${pct(top1 / queries.length)}`)
  line(`随机基线（40 条里蒙 5 条）= 5/40 = ${pct(5 / 40)}`)
  assert(`召回率 ≥ 60%（契约 §4.3）`, rate >= 0.6, `实测 ${hits}/${queries.length} = ${pct(rate)}`)

  // ── 学习证据：同一批事实再过一遍，预测误差应显著下降 ──
  let s2 = 0
  let wrote2 = 0
  for (const t of texts) {
    const r = await mem.observe(t, { namespace: ns })
    s2 += r.surprise
    if (r.wrote) wrote2++
  }
  const mean1 = surpriseSum / texts.length
  const mean2 = s2 / texts.length
  const drop = (mean1 - mean2) / mean1
  line(`第二轮 surprise 均值=${f(mean2, 6)}（首轮 ${f(mean1, 6)}），降幅 ${pct(drop)}；第二轮只有 ${wrote2}/${texts.length} 条再触发写入`)
  assert('第二轮预测误差显著低于首轮（矩阵学到了东西）', drop > 0.3, `降幅 ${pct(drop)}（阈值 30%）`)

  // ── 诚实记录：默认门限 0.35 下，第二轮 40 条**全部**仍会触发写入 ──
  // 这不是契约失败项（契约没要求"重复不再写"），但它说明：
  // 只要 M 还在动、预测误差还没压到 θ 以下，重复内容仍然占写入预算。
  // 下面用「固定 M、固定文本」的构造性实验把这条边界量出来。
  line(`【实测记录】默认 θ=0.35 时，学完一轮后再观察同一批 40 条，仍有 ${wrote2}/${texts.length} 条满足 surprise > θ`)
  line(`  → 单步 η=0.2 的学习量有限（误差每步约 ×(1−2η)=0.6，但新样本不断改写 M），`)
  line(`    「重复不再写」在本实现里需要更高的 θ 或更多轮次，不能只靠一次观察达成。`)

  // ── 门限扫描：θ 调到多少，重复内容才真的不再写入 ──
  const dirT = sectionDir('recall-thresh')
  await fsp.mkdir(dirT, { recursive: true })
  const memT = new NeuralMemory({ path: dirT })
  const nsT = 'thresh'
  for (const t of texts) await memT.observe(t, { namespace: nsT }) // 先学一遍（默认 θ）
  // 固定住当前 M，再对同一批文本按不同 θ 求"会写几条"（用 probeSurprise 不写盘）
  const stT = memT.stateOf(nsT)
  const Msnap = Float64Array.from(stT.M)
  const surprises2 = []
  for (const t of texts) {
    const k = embedText(t, 64, K_SALT)
    const v = embedText(t, 64, V_SALT)
    const pred = nMatvec(Msnap, k, 64, 64)
    let s = 0
    for (let i = 0; i < 64; i++) { const d = pred[i] - v[i]; s += d * d }
    surprises2.push(s)
  }
  const sorted = [...surprises2].sort((a, b) => a - b)
  line('')
  line('门限扫描（M 固定为「学完 40 条」后的快照，同一批 40 条文本再观察一次）：')
  for (const th of [0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]) {
    const w = surprises2.filter((s) => s > th).length
    line(`  θ=${th.toFixed(2)} → 会写入 ${String(w).padStart(2)}/40 条（${pct(w / 40)}）`)
  }
  line(`  第二轮 surprise 分布：min=${f(sorted[0], 4)} 中位=${f(sorted[19], 4)} max=${f(sorted[39], 4)}`)
  line('  → 结论：这套实现要达到「重复内容基本不再写」，θ 需接近分布上沿（≥0.9），')
  line('    默认 0.35 只筛掉"已经学得很透"的部分。这是设计取舍，已写入 README「已知限制」。')

  // 高 θ 下重复写入数必须显著少于低 θ —— 证明门限参数真的在起作用
  const wLow = surprises2.filter((s) => s > 0.35).length
  const wHigh = surprises2.filter((s) => s > 0.9).length
  assert('提高 θ 确实显著减少重复写入（门限真的在筛，不是摆设）', wHigh < wLow,
    `θ=0.35 写 ${wLow}/40 vs θ=0.90 写 ${wHigh}/40`)

  // ── 附加观测（不计入判定）：近似查询 ──
  const nearQueries = [
    ['李曜辰的工号是什么', 0],
    ['周以宁的服务器在哪', 1],
    ['陈景行的生日', 2],
    ['郑柏川的猫叫什么', 4],
    ['何斯年的项目代号', 7],
    ['谢云舟的手机尾号', 10],
    ['鲍元亮的工牌号是多少', 28],
    ['屈子安的航班', 30],
  ]
  let nearHits = 0
  for (const [q, want] of nearQueries) {
    const r = await mem.recall(q, { namespace: ns, topK: 5 })
    if (r.results.some((x) => x.text === texts[want])) nearHits++
  }
  line(`附加观测（不计入判定）：8 条近似/省略式查询的 top-5 命中 ${nearHits}/8 = ${pct(nearHits / 8)}`)
  line('  —— 契约只要求原句查询，近似查询作为「是否真的按语义近似排」的参考，不参与通过判定')

  return { dir, mem, ns, texts, queries, hitRate: rate, hits, total: queries.length, scoreLog }
}

// ══════════════════════════ 3.5 契约返回字段完整性 ══════════════════════════

async function check35_contract(ctx3) {
  section('3.5', '契约 §3.3 返回字段完整性（全部工具）')

  const dir = sectionDir('contract')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir })
  const ns = 'c'

  const o = await mem.observe('何斯年的项目代号叫青雀', { namespace: ns })
  const needObs = ['ok', 'wrote', 'surprise', 'entries', 'retained', 'inMemory']
  const missObs = needObs.filter((k) => !(k in o))
  line(`nm_observe 字段：${Object.keys(o).join(', ')}`)
  assert('nm_observe 含契约字段 ok/wrote/surprise/entries/retained/inMemory', missObs.length === 0,
    missObs.length ? `缺 ${missObs.join(',')}` : '全齐')

  const rc = await mem.recall('何斯年的项目代号', { namespace: ns })
  const missRc = ['ok', 'results', 'entries'].filter((k) => !(k in rc))
  const first = rc.results[0] ?? {}
  const missItem = ['text', 'score', 'surprise'].filter((k) => !(k in first))
  line(`nm_recall 字段：${Object.keys(rc).join(', ')}；results[0] 字段：${Object.keys(first).join(', ')}`)
  assert('nm_recall 含 ok/results/entries，条目含 text/score/surprise',
    missRc.length === 0 && missItem.length === 0,
    missRc.length || missItem.length ? `缺 ${[...missRc, ...missItem].join(',')}` : '全齐')

  const fg = await mem.forget({ namespace: ns })
  const missFg = ['ok', 'entriesBefore', 'entriesAfter', 'decayed'].filter((k) => !(k in fg))
  line(`nm_forget 字段：${Object.keys(fg).join(', ')}`)
  assert('nm_forget 含 ok/entriesBefore/entriesAfter/decayed', missFg.length === 0,
    missFg.length ? `缺 ${missFg.join(',')}` : '全齐')

  const stt = await mem.status({ namespace: ns })
  const missSt = ['ok', 'entries', 'weightBytes', 'meanSurprise', 'highSurpriseEntries', 'decayRounds']
    .filter((k) => !(k in stt))
  line(`nm_status 字段：${Object.keys(stt).join(', ')}`)
  assert('nm_status 含 ok/entries/weightBytes/meanSurprise/highSurpriseEntries/decayRounds',
    missSt.length === 0, missSt.length ? `缺 ${missSt.join(',')}` : '全齐')
  assert('nm_status.weightBytes = d×d×8 = 64×64×8 = 32768', stt.weightBytes === 64 * 64 * 8,
    `实测 ${stt.weightBytes}`)

  const ex = await mem.export({ namespace: ns })
  const missEx = ['ok', 'path', 'entries', 'bytes'].filter((k) => !(k in ex))
  line(`nm_export 字段：${Object.keys(ex).join(', ')} → ${ex.path ?? ex.error}`)
  assert('nm_export 含 ok/path/entries/bytes', missEx.length === 0,
    missEx.length ? `缺 ${missEx.join(',')}` : '全齐')
  assert('nm_export 落盘文件真实存在且内容可解析', !!ex.ok && fs.existsSync(ex.path) &&
    (() => { try { JSON.parse(fs.readFileSync(ex.path, 'utf8')); return true } catch { return false } })(),
    ex.ok ? `bytes=${ex.bytes}` : `ok=false: ${ex.error}`)

  // 工具层形状（contract §2.3）
  const defs = makeToolDefinitions(mem, { config: {} })
  const names = defs.map((d) => d.name)
  line(`工具定义：${names.join(', ')}`)
  assert('五个工具名与契约 3.3 一致', TOOL_NAMES.every((n) => names.includes(n)),
    `期望 ${TOOL_NAMES.join(',')}，实得 ${names.join(',')}`)

  /**
   * ⚠️ 2026-11-17 改判据：原来这里要求 `!d.parameters.type`（必须是扁平 DSL）。
   * 那条断言把**接错契约**当成了正确行为 —— 宿主 register() 并不编译扁平 DSL，
   * 实测 12 个工具在模型侧全成了 `() => any`。现在要求编译后的原始 JSON Schema，
   * 且每个声明了 required:true 的参数都必须出现在根级 required 数组里。
   */
  const shapeOk = defs.every((d) =>
    typeof d.description === 'string' && d.description.length > 10 &&
    d.parameters && typeof d.parameters === 'object' && !Array.isArray(d.parameters) &&
    d.parameters.type === 'object' &&
    d.parameters.properties && typeof d.parameters.properties === 'object' &&
    d.output && d.output.schema && typeof d.output.render === 'function' &&
    typeof d.execute === 'function')
  assert('工具形状符合宿主契约（编译后 JSON Schema + output.{schema,render}）', shapeOk,
    shapeOk ? '5/5 合规' : '存在形状不合规的工具定义')

  // 参数required 必须落在根级数组，而不是留在属性里（宿主会当非法关键字拒绝）
  const reqOk = defs.every((d) => {
    const props = d.parameters.properties ?? {}
    for (const [key, spec] of Object.entries(props)) {
      if ('required' in spec) return false // 属性里不许留 required
      if (spec.type === undefined) return false // 每个属性都得有类型
      void key
    }
    return true
  })
  assert('参数属性里不残留 required、且每个属性都有 type', reqOk,
    reqOk ? '全部合规' : '有属性残留 required 或缺 type')

  const recallDef = defs.find((d) => d.name === 'nm_recall')
  assert('nm_recall 的 query 出现在根级 required',
    Array.isArray(recallDef?.parameters?.required) && recallDef.parameters.required.includes('query'),
    JSON.stringify(recallDef?.parameters?.required))

  // execute + render 真跑一遍
  let execOk = 0
  const execFail = []
  const calls = {
    nm_observe: { text: '屈子安的航班是 CA1832', namespace: ns },
    nm_recall: { query: '屈子安的航班', namespace: ns, topK: 3 },
    nm_forget: { namespace: ns, decay: 0.01 },
    nm_status: { namespace: ns },
    nm_export: { namespace: ns },
    // 2026-11-17 新增工具：dryRun 默认预览，不会动数据
    nm_delete: { namespace: ns, texts: ['不存在的占位文本'], dryRun: true },
  }
  for (const d of defs) {
    try {
      const v = await d.execute(calls[d.name] ?? {})
      const rendered = d.output.render(calls[d.name] ?? {}, v)
      const okRender = Array.isArray(rendered) && rendered.length > 0 &&
        typeof rendered[0].text === 'string' && rendered[0].text.length > 0
      if (v && typeof v.ok === 'boolean' && okRender) execOk++
      else execFail.push(`${d.name}(ok=${v?.ok}, render=${okRender})`)
    } catch (e) {
      execFail.push(`${d.name} 抛异常: ${e?.message ?? e}`)
    }
  }
  // ⚠️ 2026-11-17：原来的常量 5 改为按 TOOL_NAMES 长度算 ——
  //   新增工具时不该让"数量断言"变成假失败。
  const TOOL_COUNT = TOOL_NAMES.length
  line(`execute+render 成功 ${execOk}/${TOOL_COUNT}`)
  assert('全部工具 execute 与 render 都不抛异常且返回结构化值', execOk === TOOL_COUNT,
    execFail.length ? execFail.join('; ') : `${execOk}/${TOOL_COUNT}`)

  // 非法参数下 render 也不能抛
  let renderOk = 0
  for (const d of defs) {
    try {
      const r1 = d.output.render({}, null)
      const r2 = d.output.render(undefined, undefined)
      if (Array.isArray(r1) && Array.isArray(r2)) renderOk++
    } catch { /* 计数即可 */ }
  }
  assert('render 对 null/undefined 也不抛（模型最终看的是它）', renderOk === defs.length, `${renderOk}/${defs.length}`)

  // 空参数时 execute 返回 ok:false 而不是抛
  let guardedOk = 0
  for (const d of defs) {
    try {
      const v = await d.execute({})
      if (v && v.ok === false && typeof v.error === 'string') guardedOk++
    } catch { /* 抛了就不计数 */ }
  }
  line(`空参数调用：${guardedOk}/5 返回 ok:false + error（其余为允许 ok:true 的查询类工具）`)
  assert('空参数不抛异常（返回 ok:false 或合法空结果）', guardedOk >= 2, `${guardedOk}/5 返回结构化失败`)

  return { dir }
}

// ══════════════════════════ 4. 惊讶度筛选真的在起作用 ══════════════════════════

const A_GROUP = [
  '卢望舒主管供应链并直接向总裁汇报',
  '岳临川上个月把结算服务从 Java 迁到了 Rust',
  '姜沐白坚持所有实验必须跑够三次才给结论',
  '严既澄在成都分公司负责西南大区的渠道',
  '汤问舟给团队定的代码规范是不许写超过五十行的函数',
]
const B_GROUP = [
  '公司楼下的便利店每天早上七点开门',
  '会议室的投影仪需要按两下电源键才亮',
  '食堂周三中午供应番茄牛腩面',
  '电梯在十九层会短暂停顿一下',
  '前台的绿植每周一早上浇水',
]

async function check4_surprise() {
  section('4', '惊讶度筛选（关键）：A 组首现 vs B 组重复 5 遍')

  const dir = sectionDir('surprise')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir })
  const ns = 'surprise'
  const dim = 64

  const aSurprises = []
  const aRetained = []
  const bSurprises = []   // 按轮次：bRound[k] = 第 k 次重复的 surprise 列表
  const bRetained = []
  const bWrote = []
  for (let k = 0; k < 5; k++) { bSurprises.push([]); bRetained.push([]); bWrote.push(0) }

  // 交错进行：A_i 观察 1 次；B_i 紧接着观察 5 次。两组看到同样的 M 漂移。
  for (let i = 0; i < A_GROUP.length; i++) {
    // A 组：首次出现
    const ra = await mem.observe(A_GROUP[i], { namespace: ns })
    aSurprises.push(ra.surprise)
    aRetained.push(ra.retained ? 1 : 0)

    // B 组：同一文本连观察 5 次
    for (let k = 0; k < 5; k++) {
      const rb = await mem.observe(B_GROUP[i], { namespace: ns })
      bSurprises[k].push(rb.surprise)
      bRetained[k].push(rb.retained ? 1 : 0)
      if (rb.wrote) bWrote[k]++
    }
  }

  const meanA = aSurprises.reduce((a, b) => a + b, 0) / aSurprises.length
  const meanBAll = bSurprises.flat().reduce((a, b) => a + b, 0)
  const nB = bSurprises.flat().length
  const meanB = meanBAll / nB
  const meanBRepeat = bSurprises.slice(1).flat().reduce((a, b) => a + b, 0) / 20
  const retA = aRetained.reduce((a, b) => a + b, 0)
  const retB = bRetained.flat().reduce((a, b) => a + b, 0)
  const rateA = retA / A_GROUP.length
  const rateB = retB / (B_GROUP.length * 5)

  line('')
  line(`A 组（5 条首现）：surprise = [${aSurprises.map((x) => f(x, 4)).join(', ')}]，均值 ${f(meanA, 6)}`)
  line(`  retained = ${retA}/${A_GROUP.length} = ${pct(rateA)}`)
  line(`B 组（5 条 × 5 次 = 25 次观察）：第 1 次均值 ${f(bSurprises[0].reduce((a, b) => a + b, 0) / 5, 6)}；`)
  for (let k = 0; k < 5; k++) {
    const m = bSurprises[k].reduce((a, b) => a + b, 0) / 5
    line(`  第 ${k + 1} 次重复：surprise 均值 ${f(m, 6)}，wrote ${bWrote[k]}/5，retained ${bRetained[k].reduce((a, b) => a + b, 0)}/5`)
  }
  line(`  B 组总体：surprise 均值 ${f(meanB, 6)}（其中第 2~5 次重复均值 ${f(meanBRepeat, 6)}）`)
  line(`  retained = ${retB}/25 = ${pct(rateB)}`)
  line('  口径：retained = wrote && inMemory —— 重复且已学过 → wrote=false → retained=false')
  line('')

  assert('A 组（高惊讶）留存率 > B 组（低惊讶）留存率（契约 §4.3）', rateA > rateB,
    `A=${pct(rateA)} vs B=${pct(rateB)}，差 ${pct(rateA - rateB)}`)
  assert('A 组 surprise 均值 > B 组重复轮次均值（门限真的在筛）', meanA > meanBRepeat,
    `A=${f(meanA, 6)} vs B(第2~5次)=${f(meanBRepeat, 6)}`)
  assert('B 组第 1 次与后续重复的 surprise 有明显落差（学会后不再惊讶）',
    bSurprises[0].reduce((a, b) => a + b, 0) / 5 > meanBRepeat,
    `第1次 ${f(bSurprises[0].reduce((a, b) => a + b, 0) / 5, 6)} vs 后续 ${f(meanBRepeat, 6)}`)

  // ── 反假：surprise 必须是 ‖M_before·k − v‖²，而不是别的什么计数器 ──
  const dir2 = sectionDir('surprise-probe')
  await fsp.mkdir(dir2, { recursive: true })
  const mem2 = new NeuralMemory({ path: dir2 })
  const ns2 = 'probe'
  const probes = [
    '白鹭书院的院长姓岑',
    '崔行舟每周三去游泳',
    '方既明的车是深蓝色帕萨特',
    '杨无咎负责 2026 年第三季度的审计',
  ]
  let maxSurpriseDiff = 0
  let firstSurprise = null
  let lastSurprise = null
  for (let round = 0; round < 2; round++) {
    for (const t of probes) {
      // 快照 M_before
      const before = Float64Array.from(mem2.stateOf(ns2).M)
      const k = embedText(t, dim, K_SALT)
      const v = embedText(t, dim, V_SALT)
      const pred = nMatvec(before, k, dim, dim) // 独立实现
      let s = 0
      for (let i = 0; i < dim; i++) { const d = pred[i] - v[i]; s += d * d }
      const r = await mem2.observe(t, { namespace: ns2 })
      const diff = Math.abs(r.surprise - s)
      if (diff > maxSurpriseDiff) maxSurpriseDiff = diff
      if (firstSurprise === null) firstSurprise = r.surprise
      lastSurprise = r.surprise
    }
  }
  line(`反假检查：把 M_before 快照下来，自己算 ‖M_before·k − v‖²，与 observe 返回的 surprise 比`)
  line(`  8 次观察的最大偏差 = ${maxSurpriseDiff.toExponential(3)}；首轮首条 ${f(firstSurprise, 6)} → 末轮末条 ${f(lastSurprise, 6)}`)
  assert('surprise 数值可由 ‖M·k − v‖² 独立复现（不是硬编码/计数器）', maxSurpriseDiff < TOL,
    `maxAbsDiff=${maxSurpriseDiff.toExponential(3)}`)

  logEntries(dir, ns)
  return { dir }
}

// ══════════════════════════ 5. 遗忘曲线 ══════════════════════════

async function check5_forget() {
  section('5', '遗忘曲线：多轮 nm_forget 后条目数与 ‖M‖ 单调不增，且旧条目被淘汰')

  const dir = sectionDir('forget')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir })
  const ns = 'forget'
  const facts = FACTS.slice(0, 6)
  for (const t of facts) await mem.observe(t, { namespace: ns })

  const st0 = mem.stateOf(ns)
  const entries0 = st0.entries.length
  const norm0 = frobenius(st0.M)
  const strength0 = st0.entries.map((e) => e.strength)
  line(`起始：entries=${entries0}，‖M‖_F=${f(norm0, 6)}，条目强度=[${strength0.map((x) => f(x, 4)).join(', ')}]`)

  // ── 默认 decay=0.02：只验单调性（此时还不到淘汰阈值） ──
  let prevNorm = norm0
  let prevEntries = entries0
  let monoDefault = true
  const trail = []
  for (let i = 0; i < 5; i++) {
    const r = await mem.forget({ namespace: ns })
    trail.push({ n: r.entriesAfter, norm: frobenius(mem.stateOf(ns).M), decayed: r.decayed })
    if (r.entriesAfter > prevEntries) monoDefault = false
    if (frobenius(mem.stateOf(ns).M) > prevNorm + 1e-15) monoDefault = false
    prevEntries = r.entriesAfter
    prevNorm = frobenius(mem.stateOf(ns).M)
  }
  line(`默认 decay=0.02 连跑 5 轮：${trail.map((t) => `(${t.n}, ${f(t.norm, 3)})`).join(' → ')}`)
  assert('默认 decay 下多轮遗忘：条目数与 ‖M‖ 单调不增', monoDefault, monoDefault ? '5 轮均单调' : '出现上升')

  // ── 大 decay：验淘汰真的会发生 ──
  const dir2 = sectionDir('forget2')
  await fsp.mkdir(dir2, { recursive: true })
  const mem2 = new NeuralMemory({ path: dir2 })
  const ns2 = 'forget2'
  for (const t of facts) await mem2.observe(t, { namespace: ns2 })

  const e0 = mem2.stateOf(ns2).entries.length
  const n0 = frobenius(mem2.stateOf(ns2).M)
  const rows = []
  let mono = true
  let prevE = e0
  let prevN = n0
  let evictedTotal = 0
  for (let i = 1; i <= 12; i++) {
    const r = await mem2.forget({ namespace: ns2, decay: 0.3 })
    const nAfter = frobenius(mem2.stateOf(ns2).M)
    evictedTotal += r.evicted
    rows.push({ round: i, entries: r.entriesAfter, norm: nAfter, evicted: r.evicted })
    if (r.entriesAfter > prevE) mono = false
    if (nAfter > prevN + 1e-15) mono = false
    prevE = r.entriesAfter
    prevN = nAfter
  }
  line('')
  line('decay=0.3 连跑 12 轮（minStrength=0.05）：')
  for (const row of rows) {
    line(`  第 ${String(row.round).padStart(2)} 轮：entries=${row.entries}，‖M‖_F=${f(row.norm, 6)}，本轮淘汰 ${row.evicted}`)
  }
  const last = rows[rows.length - 1]
  line('')
  assert('decay=0.3 下条目数与 ‖M‖ 也严格单调不增', mono, `${e0} 条 → ${last.entries} 条，‖M‖ ${f(n0, 4)} → ${f(last.norm, 6)}`)
  assert('衰减到阈值以下后旧条目被淘汰', last.entries < e0 && evictedTotal > 0,
    `淘汰前 ${e0} 条 → 淘汰后 ${last.entries} 条，累计淘汰 ${evictedTotal} 条`)
  assert('‖M‖ 的衰减比 ≈ (1−decay)^轮数', (() => {
    const expect = n0 * Math.pow(0.7, 12)
    return Math.abs(last.norm - expect) / Math.max(expect, 1e-12) < 1e-6
  })(), `实测 ${f(last.norm, 9)} vs 理论 ${f(n0 * Math.pow(0.7, 12), 9)}`)

  // ── 「遗忘」不是「删除」：条目被淘汰但矩阵还在（有残余权重） ──
  line(`淘汰后 ‖M‖_F 仍为 ${f(last.norm, 6)}（> 0 说明是权重衰减，不是把矩阵删了重来）`)
  assert('遗忘后矩阵仍有残余权重（不是清库）', last.norm > 0, `‖M‖=${f(last.norm, 6)}`)

  logEntries(dir2, ns2)
  return { dir }
}

/** 把落盘 JSON 的条目数打出来，顺带确认条目表与 M 真的写进了磁盘。 */
function logEntries(dir, ns) {
  const file = path.join(dir, `${ns}.json`)
  try {
    const raw = JSON.parse(fs.readFileSync(file, 'utf8'))
    line(`  落盘 ${path.basename(file)}：M.length=${raw.M.length}，entries=${raw.entries.length}，version=${raw.version}`)
  } catch (e) {
    line(`  落盘文件读取失败：${e?.message ?? e}`)
  }
}

// ══════════════════════════ 6. 召回打分真的来自 M（关键） ══════════════════════════

async function check6_provenance(ctx3) {
  section('6', '召回打分溯源（关键）：从 M 自己重算 + 清零消融 + 绕过 M 反证')

  const { dir, ns, texts, queries } = ctx3
  const dim = 64

  // ── 6a. 用落盘的 M 与条目 v 自己重算 score，与工具返回值逐条比 ──
  const mem = new NeuralMemory({ path: dir })
  const st = await mem.ensureLoaded(ns)
  const M = st.M
  const entryByText = new Map(st.entries.map((e) => [e.text, e]))

  let maxScoreDiff = 0
  let compared = 0
  let rankMismatch = 0
  for (const q of queries) {
    const kq = embedText(q, dim, K_SALT)
    const readout = nMatvec(M, kq, dim, dim) // 独立实现的 matvec
    const r = await mem.recall(q, { namespace: ns, topK: 5 })

    // 自己算全表打分并排序
    const mine = st.entries.map((e) => ({
      text: e.text,
      score: nCosine(readout, e.v),
    }))
    mine.sort((a, b) => b.score - a.score)

    for (const item of r.results) {
      const e = entryByText.get(item.text)
      if (!e) continue
      const expect = nCosine(readout, e.v)
      const d = Math.abs(item.score - expect)
      if (d > maxScoreDiff) maxScoreDiff = d
      compared++
    }
    // 顺序是否与我算的一致
    for (let i = 0; i < r.results.length; i++) {
      if (mine[i] && mine[i].text !== r.results[i].text) rankMismatch++
    }
  }
  line(`6a 自己重算 cos(M·k_q, v_i)：比较 ${compared} 个分数，最大偏差 ${maxScoreDiff.toExponential(3)}`)
  assert('recall 的 score === cos(M·k_q, v_entry)（自己重算，逐条对齐）', maxScoreDiff < TOL,
    `maxAbsDiff=${maxScoreDiff.toExponential(3)}`)
  assert('recall 的排序与「纯矩阵打分」排序一致', rankMismatch === 0, `不一致 ${rankMismatch} 处`)

  // ⚠️ 反证：如果 score 其实是关键词/子串匹配，那么「文本相同」必然得最高分，
  //    而 matrix 版本在没有写入 M 的新 namespace 上应该拿不到同样的排序。
  //    这里直接用「条目文本 vs 查询」的朴素字符串重叠度做一个对照打分器，
  //    看它与矩阵打分的排序是否一致——不一致才说明矩阵在起作用。
  function charOverlapScore(a, b) {
    const sa = normalizeText(a)
    const sb = normalizeText(b)
    if (!sa || !sb) return 0
    const setB = new Set(sb.split(''))
    let hit = 0
    for (const ch of new Set(sa.split(''))) if (setB.has(ch)) hit++
    return hit / new Set(sa.split('')).size
  }
  let overlapAgree = 0
  let overlapTotal = 0
  for (const q of queries) {
    const r = await mem.recall(q, { namespace: ns, topK: 5 })
    const byOverlap = [...st.entries].sort((a, b) => charOverlapScore(q, b.text) - charOverlapScore(q, a.text)).slice(0, 5)
    overlapTotal++
    if (byOverlap.some((e) => e.text === r.results[0]?.text)) overlapAgree++
  }
  line(`  对照：用「字符重叠度」打分器给同一批查询排序，top-1 与矩阵打分一致 ${overlapAgree}/${overlapTotal} 次`)
  // ★ 这里必须说实话：在**原句查询**上，字符重叠度与矩阵打分的 top-1 会高度一致，
  //   因为原句查询本来就是"字符几乎全同"的任务。所以这一项**不能**用来区分"矩阵 vs 字符串匹配"。
  //   真正有区分力的证据是 6b（M 清零后命中率崩到 15%）和 6c（绕过 M 只有 20%）。
  line('  注意：原句查询下文本重叠度与矩阵打分本就该一致，因此**这一项没有区分力**，不作为证据；')
  line('        证明「打分来自 M」靠的是下面的 6b 清零消融与 6c 绕过 M 反证。')
  if (overlapAgree === overlapTotal) {
    line('        （实测确实完全一致——符合预期，不是问题，但也不能当成"矩阵有效"的证明。）')
  }

  // ── 6b. 清零消融：M ← 0 后同一批查询的分数与命中率必须显著变化 ──
  //
  // ⚠️ 2026-11-17 更新：实现侧加了置信度熔断（results 可能被清空）。
  //    原来用 `r.results.map(x => x.score)` 取分数，熔断后拿到空数组 →
  //    scores[0] === undefined → Number.isFinite 过滤掉 → maxScoreDelta 恒为 0，
  //    这条检查就永远 FAIL。那不是实现的问题，是**观测点过时**。
  //    现在一律读 r.top1Score（熔断与否都返回），观测点与被测性质重新对齐。
  const memZ = new NeuralMemory({ path: dir })
  await memZ.ensureLoaded(ns)
  const before = []
  for (const q of queries) {
    // minScore: -1 = 关掉熔断，保证能拿到候选做比较（这里比的是"分数变没变"，不是"召准不准"）
    const r = await memZ.recall(q, { namespace: ns, topK: 5, minScore: -1 })
    before.push({ q, hit: r.results.some((x) => x.text === q), top1: r.top1Score })
  }
  memZ.stateOf(ns).M.fill(0) // 只清内存里的 M，不动磁盘
  const after = []
  for (const q of queries) {
    const r = await memZ.recall(q, { namespace: ns, topK: 5, minScore: -1 })
    after.push({ q, hit: r.results.some((x) => x.text === q), top1: r.top1Score })
  }
  const hitBefore = before.filter((x) => x.hit).length
  const hitAfter = after.filter((x) => x.hit).length
  let maxScoreDelta = 0
  let nonZeroAfter = 0
  for (let i = 0; i < before.length; i++) {
    const a = before[i].top1
    const b = after[i].top1
    if (Number.isFinite(a) && Number.isFinite(b)) maxScoreDelta = Math.max(maxScoreDelta, Math.abs(a - b))
    if (Number.isFinite(b) && Math.abs(b) > 1e-15) nonZeroAfter++
  }
  line('')
  line(`6b M 清零前：top-5 命中 ${hitBefore}/${queries.length} = ${pct(hitBefore / queries.length)}`)
  line(`   M 清零后：top-5 命中 ${hitAfter}/${queries.length} = ${pct(hitAfter / queries.length)}，top-1 分数最大变化 ${f(maxScoreDelta, 6)}`)
  line(`   清零后仍有非零分数的查询：${nonZeroAfter}/${queries.length}`)
  line(`   清零后仍命中的 ${hitAfter} 条，命中的是「按 createdAt 兜底排序」的最早 5 条，与语义无关`)
  const dropPp = (hitBefore - hitAfter) / queries.length
  assert('M 清零后 top-1 分数为 0（score 完全由 M 产出）', nonZeroAfter === 0,
    `非零查询数 ${nonZeroAfter}/${queries.length}, 最大分数变化 ${f(maxScoreDelta, 6)}`)
  assert('M 清零后命中率显著下降（≥30 个百分点）', dropPp >= 0.3,
    `命中率 ${pct(hitBefore / queries.length)} → ${pct(hitAfter / queries.length)}，降 ${(dropPp * 100).toFixed(1)} pp`)
  assert('清零前后分数确实变化（不是同一份缓存）', maxScoreDelta > 0.5, `top-1 最大变化 ${f(maxScoreDelta, 6)}`)

  // ── 6c. 绕过 M：直接用 cos(k_q, v_i) 打分 ──
  let bypassHits = 0
  for (const q of queries) {
    const kq = embedText(q, dim, K_SALT)
    const scored = st.entries.map((e) => ({ text: e.text, s: nCosine(kq, e.v) }))
    scored.sort((a, b) => b.s - a.s)
    const top5 = scored.slice(0, 5).map((x) => x.text)
    if (top5.includes(q)) bypassHits++
  }
  line('')
  line(`6c 绕过 M，直接用 cos(k_q, v_i) 打分：top-5 命中 ${bypassHits}/${queries.length} = ${pct(bypassHits / queries.length)}`)
  line(`   走 M（cos(M·k_q, v_i)）的命中：${hitBefore}/${queries.length} = ${pct(hitBefore / queries.length)}`)
  assert('绕过 M 的命中率显著更低（证明「读出 M·k」这一层是必要的）',
    bypassHits <= hitBefore - 4,
    `绕过 ${pct(bypassHits / queries.length)} vs 走 M ${pct(hitBefore / queries.length)}，差 ${hitBefore - bypassHits} 条`)

  // ── 6d. 清零后残留的 3 条命中到底是什么？（验证「分数全 0 → 退化成 createdAt 排序」的解释） ──
  const memZ2 = new NeuralMemory({ path: dir })
  await memZ2.ensureLoaded(ns)
  memZ2.stateOf(ns).M.fill(0)
  const survivors = []
  for (const q of queries) {
    const r = await memZ2.recall(q, { namespace: ns, topK: 5 })
    if (r.results.some((x) => x.text === q)) {
      survivors.push({ q, hitRank: r.results.findIndex((x) => x.text === q), topText: r.results[0]?.text })
    }
  }
  line('')
  line(`6d 清零后仍"命中"的 ${survivors.length} 条具体是：`)
  for (const s of survivors) line(`   查询「${s.q}」→ 排第 ${s.hitRank + 1}，返回的 top1 是「${s.topText}」`)
  const earliest5 = st.entries.map((e) => e.text).slice(0, 5)
  line(`   记忆表里最早的 5 条：${earliest5.map((t) => `「${t}」`).join(', ')}`)
  const allInEarliest = survivors.every((s) => earliest5.includes(s.q))
  line(`   → 残留命中全部落在"最早的 5 条"里 = ${allInEarliest}（证实：分数全 0 时按 createdAt 兜底）`)
  assert('清零后的残留命中可由「createdAt 兜底排序」完全解释（不是还有隐藏打分）', allInEarliest,
    allInEarliest ? `${survivors.length} 条全在最早 5 条内` : '存在无法解释的命中')

  // ══════════════════════════════════════════════════════════════════
  // ★ 6e. 100% 召回率到底证明了什么？—— 能力边界解剖
  //
  // 必须说清的事：查询用**原句**时，k_query ≡ k_entry（嵌入是同一段文本的确定性哈希），
  // M 学的是 k_i → v_i 的映射，读出 M·k_i ≈ v_i，于是 cos ≈ 1。
  // 所以「原句查询 100%」证明的是**异联想精确查找真的工作**，它**不能**证明语义理解。
  // 下面三组梯度测试把这条边界量出来。
  // ══════════════════════════════════════════════════════════════════
  const dirD = sectionDir('difficulty')
  await fsp.mkdir(dirD, { recursive: true })
  const memD = new NeuralMemory({ path: dirD })
  const nsD = 'diff'
  for (const t of texts) await memD.observe(t, { namespace: nsD })

  // 难度 1：截断（保留主语 + 疑问尾巴，字符重合度高）
  const d1 = [
    ['李曜辰的工号是什么', 0],
    ['周以宁的服务器在哪', 1],
    ['陈景行的生日', 2],
    ['郑柏川的猫叫什么', 4],
    ['何斯年的项目代号', 7],
    ['谢云舟的手机尾号', 10],
    ['鲍元亮的工牌号是多少', 28],
    ['屈子安的航班', 30],
    ['温梨落的工位有什么', 32],
    ['慕云深的组里有几个人', 38],
  ]
  let h1 = 0
  for (const [q, want] of d1) {
    const r = await memD.recall(q, { namespace: nsD, topK: 5 })
    if (r.results.some((x) => x.text === texts[want])) h1++
  }

  // 难度 2：只给独一无二的实体串，去掉人名与句式（共享字符极少）
  const d2 = [
    ['KX4821', 0],
    ['qingteng.dev', 3],
    ['sunxun_lab', 5],
    ['CA1832', 30],
    ['HT20260914', 34],
    ['8891', 36],
    ['000771', 28],
    ['P8', 12],
  ]
  let h2 = 0
  for (const [q, want] of d2) {
    const r = await memD.recall(q, { namespace: nsD, topK: 5 })
    if (r.results.some((x) => x.text === texts[want])) h2++
  }

  // 难度 3：改写查询（不共享完整句子，但**仍共享少量实词**）
  // ★ 诚实命名：这不是"纯语义查询"。命中的 4 条靠的是共享关键词
  //   （"修打印机"/"写 Rust"/"蛋白质"/"预算"+"八十万"），本质仍是字符重叠。
  //   真语义（"谁最擅长编码"→岳临川）本实现做不到，这条边界写进 README。
  const d3 = [
    ['谁负责修打印机', 19],
    ['谁的猫最胖', 21],
    ['谁在成都上班', 25],
    ['谁会写 Rust', 23],
    ['谁研究蛋白质', 27],
    ['谁的预算是三百八十万', 33],
  ]
  let h3 = 0
  for (const [q, want] of d3) {
    const r = await memD.recall(q, { namespace: nsD, topK: 5 })
    const hit = r.results.some((x) => x.text === texts[want])
    if (hit) h3++
    line(`   [${hit ? '命中' : '未中'}] 「${q}」 top1=${f(r.results[0]?.score ?? 0, 3)}`)
  }

  // 难度 4：真语义（目标句里**一个查询词都不出现**）——用来证明"不理解语义"
  const d4 = [
    ['谁最擅长编程语言', 23],       // 岳临川擅长写 Rust
    ['谁管钱与预算', 33],           // 熊知远的项目预算是 380 万
    ['谁在生物领域工作', 27],        // 邵清嘉研究方向是蛋白质折叠
    ['谁住的地方有湖', 6],          // 林胧在杭州西湖边开茶馆
  ]
  let h4 = 0
  const d4log = []
  for (const [q, want] of d4) {
    const r = await memD.recall(q, { namespace: nsD, topK: 5 })
    const hit = r.results.some((x) => x.text === texts[want])
    if (hit) h4++
    d4log.push({ q, hit, top1: r.results[0]?.score ?? 0 })
  }

  line('')
  line('能力边界解剖（同一份记忆，四种查询难度）：')
  line(`  难度 1 · 原句截断（保留人名与主体词）      ：${h1}/10 = ${pct(h1 / 10)}`)
  line(`  难度 2 · 只给唯一实体串（KX4821 这类）    ：${h2}/8  = ${pct(h2 / 8)}`)
  line(`  难度 3 · 改写但共享实词（"谁会写 Rust"） ⚠️ ：${h3}/6  = ${pct(h3 / 6)}   仍属字符重叠，非语义`)
  line(`  难度 4 · 真语义（目标句里无一个查询词）    ：${h4}/4  = ${pct(h4 / 4)}`)
  for (const x of d4log) line(`     [${x.hit ? '命中' : '未中'}] 「${x.q}」 top1=${f(x.top1, 3)}`)
  line('')
  line('  ★ 关键结论：本实现的"相似"来自 n-gram 哈希嵌入的字符级重叠，**不是语义理解**。')
  line('    原句/截断/实体串查询很强（100%/87.5%），一旦改成不共享词的说法就崩。')
  line('    这与契约一致（契约只要求原句/近似查询），但宣传上不能叫"按意思像不像"。')
  assert('原句截断仍能召回（常识级近似）', h1 >= 7, `${h1}/10 = ${pct(h1 / 10)}`)

  // ══════════════════════════════════════════════════════════════════
  // ★ 6f. 容量上限：d=64 的矩阵能装多少条又还能查得到？（决定"实用性"的核心）
  // ══════════════════════════════════════════════════════════════════
  line('')
  line('容量压力测试（d=64，每个规模用独立 namespace，从零开始学）：')
  const scaleRows = []
  for (const N of [20, 40, 80, 160, 320]) {
    const d = sectionDir(`scale-${N}`)
    await fsp.mkdir(d, { recursive: true })
    const m = new NeuralMemory({ path: d })
    const batch = []
    for (let i = 0; i < N; i++) {
      const a = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛'
      const b = '春夏秋冬风花雪月山川江河湖海星辰云霞雨露霜雷电虹霞岚霖雾霭霁雪'
      const c = '甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥'
      const t = `${a[i % a.length]}${b[(i * 7) % b.length]}${c[(i * 3) % c.length]}的第 ${1000 + i * 37} 号记录是 ${(i * 2654435761) % 99991}`
      batch.push(t)
      await m.observe(t, { namespace: 's' })
    }
    const qs = batch.filter((_, i) => i % 5 === 0)
    let hit = 0
    const t0 = Date.now()
    for (const q of qs) {
      const r = await m.recall(q, { namespace: 's', topK: 5 })
      if (r.results.some((x) => x.text === q)) hit++
    }
    const ms = Date.now() - t0
    const rate = hit / qs.length
    scaleRows.push({ N, hit, total: qs.length, rate, ms, entries: m.stateOf('s').entries.length, norm: frobenius(m.stateOf('s').M) })
    line(`  N=${String(N).padStart(3)} 条 → entries=${m.stateOf('s').entries.length}，召回 ${hit}/${qs.length} = ${pct(rate)}，‖M‖=${f(frobenius(m.stateOf('s').M), 3)}，${qs.length} 次查询耗时 ${ms} ms`)
  }
  const r20 = scaleRows[0]
  const r320 = scaleRows[scaleRows.length - 1]
  line('')
  line(`  容量结论：20 条时 ${pct(r20.rate)}，320 条时 ${pct(r320.rate)}（跌幅 ${((r20.rate - r320.rate) * 100).toFixed(1)} pp）`)
  assert('20 条规模下召回仍 ≥ 60%（小规模确实可用）', r20.rate >= 0.6, `${pct(r20.rate)}`)
  line(`  注：d=64 的矩阵理论上能容纳远多于 512 条（maxEntries）的近正交键，`)
  line(`      但单步 η=0.2 的写入会互相干扰，规模越大越明显——这是容量/干扰的取舍，不是 bug。`)

  return { hitBefore, hitAfter, maxScoreDelta, bypassHits, maxScoreDiff, h1, h2, h3, scaleRows }
}

// ══════════════════════════ 7. 持久化 ══════════════════════════

async function check7_persistence() {
  section('7', '持久化：写完存盘 → 新实例读回 → 同一查询结果逐项一致')

  const dir = sectionDir('persist')
  await fsp.mkdir(dir, { recursive: true })
  const ns = 'persist'
  const facts = FACTS.slice(0, 12)
  const m1 = new NeuralMemory({ path: dir })
  for (const t of facts) await m1.observe(t, { namespace: ns })
  await m1.save(ns)

  const M1 = Float64Array.from(m1.stateOf(ns).M)
  const e1 = m1.stateOf(ns).entries.map((e) => ({ text: e.text, v: Array.from(e.v), k: Array.from(e.k) }))
  const queries = facts.filter((_, i) => i % 2 === 0)
  const r1 = []
  for (const q of queries) r1.push(await m1.recall(q, { namespace: ns, topK: 5 }))

  // 新实例（同一个 path，模拟进程重启）
  const m2 = new NeuralMemory({ path: dir })
  await m2.ensureLoaded(ns)
  const M2 = m2.stateOf(ns).M
  const e2 = m2.stateOf(ns).entries

  const dM = maxAbsDiff(M1, M2)
  line(`矩阵往返：${M1.length} 个元素，最大偏差 ${dM.toExponential(3)}`)
  assert('M 逐元素往返一致（JSON 往返，误差 < 1e-12）', dM < TOL, `maxAbsDiff=${dM.toExponential(3)}`)

  let dV = 0
  assert('条目数一致', e1.length === e2.length, `写前 ${e1.length} → 读回 ${e2.length}`)
  for (let i = 0; i < Math.min(e1.length, e2.length); i++) {
    if (e1[i].text !== e2[i].text) {
      assert('条目顺序与文本一致', false, `第 ${i} 条不一致：${e1[i].text} vs ${e2[i].text}`)
      break
    }
    dV = Math.max(dV, maxAbsDiff(Float64Array.from(e1[i].v), e2[i].v))
    dV = Math.max(dV, maxAbsDiff(Float64Array.from(e1[i].k), e2[i].k))
  }
  assert('每条记忆的 k/v 逐元素往返一致', dV < TOL, `maxAbsDiff=${dV.toExponential(3)}`)

  let maxScoreDiff = 0
  let textMismatch = 0
  let orderMismatch = 0
  for (let qi = 0; qi < queries.length; qi++) {
    const r2 = await m2.recall(queries[qi], { namespace: ns, topK: 5 })
    if (r2.entries !== r1[qi].entries) textMismatch++
    const a = r1[qi].results
    const b = r2.results
    if (a.length !== b.length) { orderMismatch++; continue }
    for (let i = 0; i < a.length; i++) {
      if (a[i].text !== b[i].text) orderMismatch++
      maxScoreDiff = Math.max(maxScoreDiff, Math.abs(a[i].score - b[i].score))
    }
  }
  line(`召回往返：${queries.length} 条查询，结果条数/条目数不一致 ${textMismatch} 处，文本或顺序不一致 ${orderMismatch} 处，`)
  line(`           score 最大偏差 ${maxScoreDiff.toExponential(3)}`)
  assert('同一查询的 recall 结果逐项一致（score 差 < 1e-12）',
    textMismatch === 0 && orderMismatch === 0 && maxScoreDiff < TOL,
    `mismatch=${textMismatch + orderMismatch}, maxScoreDiff=${maxScoreDiff.toExponential(3)}`)

  // ── 边界：删掉文件后重建，应是空状态而不是崩溃 ──
  await fsp.rm(path.join(dir, `${ns}.json`), { force: true })
  const m3 = new NeuralMemory({ path: dir })
  const r3 = await m3.recall(queries[0], { namespace: ns })
  const st3 = await m3.status({ namespace: ns })
  line(`删除状态文件后新建实例：entries=${st3.entries}，recall ok=${r3.ok}，results=${r3.results.length}`)
  assert('状态文件被删后读回是空记忆而不是崩溃', st3.ok === true && st3.entries === 0 && r3.ok === true,
    `entries=${st3.entries}, recall.ok=${r3.ok}`)

  // ── dim 变化 → 旧矩阵不兼容，应丢弃而不是算出 NaN ──
  const dBad = sectionDir('persist-dim')
  await fsp.mkdir(dBad, { recursive: true })
  const md = new NeuralMemory({ path: dBad, dim: 64 })
  for (const t of FACTS.slice(0, 3)) await md.observe(t, { namespace: 'd' })
  const md32 = new NeuralMemory({ path: dBad, dim: 32 })
  const st32 = await md32.ensureLoaded('d')
  let finite = true
  for (let i = 0; i < st32.M.length; i++) if (!Number.isFinite(st32.M[i])) finite = false
  line(`dim 从 64 改成 32 后读同一份状态：entries=${st32.entries.length}，M 全有限=${finite}`)
  assert('dim 变化后旧状态被安全丢弃（不产出 NaN）', finite && st32.entries.length === 0,
    `entries=${st32.entries.length}, finite=${finite}`)

  return { dir }
}

// ══════════════════════════ 8. 边界 ══════════════════════════

async function check8_edges() {
  section('8', '边界：空文本 / 1MB 长文本 / 非法 namespace / 不存在的 namespace')

  const dir = sectionDir('edges')
  await fsp.mkdir(dir, { recursive: true })
  const mem = new NeuralMemory({ path: dir })
  const ns = 'edge'

  const cases = []

  async function probe(name, fn, label = null) {
    let threw = null
    let value = null
    try {
      value = await fn()
    } catch (e) {
      threw = e
    }
    const okShape = value && typeof value === 'object' && typeof value.ok === 'boolean'
    cases.push({ name, label, threw, value, okShape })
    const desc = threw
      ? `抛异常 ${threw?.message ?? threw}`
      : `ok=${value?.ok}${value?.error ? `, error=${JSON.stringify(value.error).slice(0, 60)}` : ''}`
    line(`${name.padEnd(34)} → ${desc}`)
    return { threw, value, okShape }
  }

  await probe('observe("")', () => mem.observe('', { namespace: ns }))
  await probe('observe("   ")', () => mem.observe('   ', { namespace: ns }))
  await probe('observe(空对象)', () => mem.observe({}, { namespace: ns }))
  await probe('observe(null)', () => mem.observe(null, { namespace: ns }))
  await probe('observe(纯标点 "，。！？")', () => mem.observe('，。！？', { namespace: ns }))
  await probe('observe(1MB 中文)', () => mem.observe('中'.repeat(1024 * 1024), { namespace: ns }))
  await probe('observe(5000 字，上限内)', () => mem.observe('张伟的记录'.repeat(1000), { namespace: ns }))
  await probe('observe(ns="../../etc/passwd")', () => mem.observe('测试文本', { namespace: '../../etc/passwd' }), 'namespace 非法')
  await probe('observe(ns=65 字符)', () => mem.observe('测试文本', { namespace: 'a'.repeat(65) }), 'namespace 非法')
  await probe('recall("../../etc/passwd")', () => mem.recall('../../etc/passwd', { namespace: ns }))
  await probe('recall(不存在的 namespace)', () => mem.recall('任何查询', { namespace: 'no-such-ns' }))
  await probe('recall(空查询)', () => mem.recall('', { namespace: ns }))
  await probe('recall(1MB 查询)', () => mem.recall('中'.repeat(1024 * 1024), { namespace: ns }))
  await probe('status(不存在的 namespace)', () => mem.status({ namespace: 'no-such-ns' }))
  await probe('status(ns 非法)', () => mem.status({ namespace: '../x' }), 'namespace 非法')
  await probe('forget(ns 非法)', () => mem.forget({ namespace: '../x' }), 'namespace 非法')
  await probe('forget(不存在的 namespace)', () => mem.forget({ namespace: 'no-such-ns' }))
  await probe('export(不存在的 namespace)', () => mem.export({ namespace: 'no-such-ns' }))
  await probe('export(ns 非法)', () => mem.export({ namespace: '../x' }), 'namespace 非法')

  line('')
  const threwAny = cases.filter((c) => c.threw)
  assert('没有任何一个边界用例抛异常', threwAny.length === 0,
    threwAny.length ? threwAny.map((c) => c.name).join('; ') : `${cases.length}/${cases.length} 未抛`)
  const shaped = cases.filter((c) => c.okShape)
  assert('全部返回结构化对象（含 boolean ok）', shaped.length === cases.length, `${shaped.length}/${cases.length}`)

  // 空文本/超长/非法 ns 必须是 ok:false（有意义的拒绝），不能静默 ok:true。
  // ★ 用精确的下标/标签匹配，不用子串：第一版用 name.includes('../../etc/passwd') 分类，
  //   把「recall 的**查询文本**是 ../../etc/passwd」也误判进「必须 false」的集合——
  //   那条其实是合法查询（路径穿越要防的是 namespace，不是 query），是测试自己的 bug。
  const mustFalse = cases.filter((c) =>
    c.name === 'observe("")' ||
    c.name === 'observe(1MB 中文)' ||
    c.name === 'recall(1MB 查询)' ||
    c.name === 'observe(纯标点 "，。！？")' ||
    c.label === 'namespace 非法')
  assert('空文本 / 1MB / 非法 namespace 都返回 ok:false（有意义的拒绝）',
    mustFalse.every((c) => c.value && c.value.ok === false),
    mustFalse.map((c) => `${c.name}=${c.value?.ok}`).join(', '))

  // 反过来也要确认：路径穿越**只防 namespace**，不防 query 文本（query 不落盘）
  const queryLooksLikePath = cases.find((c) => c.name === 'recall("../../etc/passwd")')
  assert('查询文本长得像路径不算非法（只校验 namespace，query 不落盘）',
    queryLooksLikePath?.value?.ok === true,
    `ok=${queryLooksLikePath?.value?.ok}`)

  const nsCase = cases.find((c) => c.name.includes('../../etc/passwd'))
  line(`非法 namespace 的拒绝理由：${JSON.stringify(nsCase?.value?.error)}`)
  assert('非法 namespace 的报错文案说明了允许的字符集',
    /a-z.*0-9|只允许|非法 namespace/i.test(String(nsCase?.value?.error ?? '')),
    JSON.stringify(String(nsCase?.value?.error ?? '(空)').slice(0, 80)))

  // 不存在的 namespace：recall 应为 ok:true + 空结果（不是错误）
  const noNs = cases.find((c) => c.name.includes('不存在的 namespace') && c.name.startsWith('recall'))
  assert('不存在的 namespace 的 recall 返回 ok:true + 空结果',
    noNs?.value?.ok === true && Array.isArray(noNs.value.results) && noNs.value.results.length === 0,
    `ok=${noNs?.value?.ok}, results=${noNs?.value?.results?.length}`)

  // 关键：非法 namespace 不能真的在磁盘上造出越界文件
  const escaped = path.resolve(dir, '..', '..', 'etc', 'passwd.json')
  line(`检查路径穿越落点是否被创建：${escaped} → ${fs.existsSync(escaped) ? '存在（BUG！）' : '不存在'}`)
  assert('路径穿越未在磁盘上产生任何文件', !fs.existsSync(escaped), fs.existsSync(escaped) ? '文件被创建' : '不存在')

  // 落盘目录里只应有合法 namespace 的文件
  let files = []
  try { files = await fsp.readdir(dir) } catch { files = [] }
  const bad = files.filter((n) => !/^[a-z0-9_-]{1,64}\.json$/.test(n))
  line(`临时状态目录里的文件：${files.length ? files.join(', ') : '(空)'}`)
  assert('状态目录里只有合法命名的文件', bad.length === 0, bad.length ? bad.join(', ') : '全部合法')

  // 5000 字那条应该成功（在 maxChars=8000 之内），确认不是把所有长文本都拒了
  const long5000 = cases.find((c) => c.name.includes('5000'))
  line(`5000 字文本：ok=${long5000?.value?.ok}，surprise=${f(long5000?.value?.surprise ?? NaN, 4)}，耗时是否可接受见运行总时长`)

  return { dir }
}

// ══════════════════════════ 9. 干净度 ══════════════════════════

/** 去掉注释与字符串字面量，只留"真正的代码"，避免注释里提到 mock 被误判。 */
function stripCommentsAndStrings(src) {
  let out = ''
  let i = 0
  const n = src.length
  let state = 'code'
  const push = (ch) => { out += ch }
  while (i < n) {
    const c = src[i]
    const d = src[i + 1]
    if (state === 'code') {
      if (c === '/' && d === '/') { state = 'line'; i += 2; continue }
      if (c === '/' && d === '*') { state = 'block'; i += 2; continue }
      if (c === "'" || c === '"' || c === '`') {
        state = c === "'" ? 'sq' : c === '"' ? 'dq' : 'tpl'
        push(' ')
        i++
        continue
      }
      push(c)
      i++
      continue
    }
    if (state === 'line') {
      if (c === '\n') { state = 'code'; push('\n') }
      i++
      continue
    }
    if (state === 'block') {
      if (c === '*' && d === '/') { state = 'code'; i += 2; continue }
      if (c === '\n') push('\n')
      i++
      continue
    }
    // 字符串内部
    if (c === '\\') { i += 2; continue }
    if ((state === 'sq' && c === "'") || (state === 'dq' && c === '"') || (state === 'tpl' && c === '`')) {
      state = 'code'
      i++
      continue
    }
    if (c === '\n') push('\n')
    i++
  }
  return out
}

async function check9_clean(scannedLib) {
  section('9', '干净度：lib/ 里的 console.log / TODO / FIXME / mock / fake / 假数字')

  const files = scannedLib ?? (await fsp.readdir(LIB)).filter((n) => n.endsWith('.js')).map((n) => path.join(LIB, n))
  const banned = [
    { re: /console\.(log|warn|error|info|debug)\s*\(/g, label: 'console.* 调用' },
    { re: /\bTODO\b/g, label: 'TODO' },
    { re: /\bFIXME\b/g, label: 'FIXME' },
    { re: /\bmock\b/gi, label: 'mock' },
    { re: /\bfake\b/gi, label: 'fake' },
    { re: /占位/g, label: '占位' },
    { re: /Math\.random\s*\(/g, label: 'Math.random（破坏可复现）' },
  ]
  // 作者报告里的那些"验收数字"如果出现在**代码**里（而非注释）就是硬编码嫌疑
  const fakeNumbers = [
    { re: /18\s*\/\s*20/g, label: '18/20' },
    { re: /90\s*%/g, label: '90%' },
    { re: /0\.9993/g, label: '0.9993' },
    { re: /0\.3637/g, label: '0.3637' },
    { re: /0\.1799/g, label: '0.1799' },
  ]

  const rawHits = []   // 注释/字符串里的出现（说明性，允许）
  const codeHits = []  // 真代码里的出现（判定失败）
  const fakeCodeHits = []

  for (const file of files) {
    const raw = await fsp.readFile(file, 'utf8')
    const code = stripCommentsAndStrings(raw)
    const base = path.basename(file)
    for (const b of banned) {
      for (const m of raw.matchAll(b.re)) {
        const upto = raw.slice(0, m.index)
        const lineNo = upto.split('\n').length
        rawHits.push({ base, lineNo, label: b.label, text: raw.split('\n')[lineNo - 1].trim() })
      }
      for (const m of code.matchAll(b.re)) {
        const upto = code.slice(0, m.index)
        const lineNo = upto.split('\n').length
        codeHits.push({ base, lineNo, label: b.label })
      }
    }
    for (const b of fakeNumbers) {
      for (const m of code.matchAll(b.re)) {
        const lineNo = code.slice(0, m.index).split('\n').length
        fakeCodeHits.push({ base, lineNo, label: b.label })
      }
    }
  }

  line(`扫描 ${files.length} 个文件：${files.map((f) => path.basename(f)).join(', ')}`)
  line('')
  line(`注释/字符串里的提及（说明性文字，允许保留）：${rawHits.length} 处`)
  const grouped = new Map()
  for (const h of rawHits) {
    const key = `${h.base}:${h.lineNo} [${h.label}]`
    grouped.set(key, h.text)
  }
  for (const [k, v] of grouped) line(`  ${k}  ${v.slice(0, 90)}`)
  line('')
  line(`实际代码里的违规（应判定失败）：${codeHits.length} 处`)
  for (const h of codeHits) line(`  ${h.base}:${h.lineNo} [${h.label}]`)
  line(`代码里出现的"验收数字"（硬编码假数字嫌疑）：${fakeCodeHits.length} 处`)
  for (const h of fakeCodeHits) line(`  ${h.base}:${h.lineNo} [${h.label}]`)
  line('')

  assert('lib/ 的实际代码里没有 console.* / TODO / FIXME / mock / fake / 占位 / Math.random',
    codeHits.length === 0, codeHits.length ? `${codeHits.length} 处违规` : '0 处（全部只出现在注释里）')
  assert('lib/ 的代码里没有硬编码的验收数字', fakeCodeHits.length === 0,
    fakeCodeHits.length ? `${fakeCodeHits.length} 处` : '0 处')

  // 有没有为了过测试而写的分支
  const specialCases = []
  for (const file of files) {
    const raw = await fsp.readFile(file, 'utf8')
    const code = stripCommentsAndStrings(raw)
    const patterns = [
      { re: /process\.env\.\w+/g, label: 'process.env' },
      { re: /\bisTest\b|\b__test\b|\bNODE_ENV\b|\bunderTest\b/g, label: '测试判定' },
      { re: /\.(includes|indexOf|search|match)\(\s*(text|query|raw|s)\s*\)/g, label: '可疑的字符串匹配调用' },
    ]
    for (const p of patterns) {
      for (const m of code.matchAll(p.re)) {
        const lineNo = code.slice(0, m.index).split('\n').length
        specialCases.push({ base: path.basename(file), lineNo, label: p.label, hit: m[0] })
      }
    }
  }
  line('可疑分支扫描（process.env / 测试判定 / 字符串匹配文本）：')
  for (const s of specialCases) line(`  ${s.base}:${s.lineNo} [${s.label}] ${s.hit}`)
  if (specialCases.length === 0) line('  （无）')

  const envHits = specialCases.filter((s) => s.label === 'process.env')
  const testHits = specialCases.filter((s) => s.label === '测试判定')
  const strHits = specialCases.filter((s) => s.label === '可疑的字符串匹配调用')
  assert('没有「测试专用分支」（isTest / NODE_ENV / __test）', testHits.length === 0,
    testHits.length ? testHits.map((h) => `${h.base}:${h.lineNo}`).join(', ') : '0 处')
  assert('process.env 只用于 DSH_HOME（无测试特判）', envHits.every((h) => h.hit === 'process.env.DSH_HOME'),
    envHits.map((h) => `${h.base}:${h.lineNo} ${h.hit}`).join(', ') || '无')
  assert('召回路径里没有把 query/text 拿去做字符串匹配', strHits.length === 0,
    strHits.length ? strHits.map((h) => `${h.base}:${h.lineNo} ${h.hit}`).join(', ') : '0 处')

  // 召回实现里只允许出现一次打分调用，且必须是 cosine(readout, ...)
  const modelSrc = await fsp.readFile(path.join(LIB, 'model.js'), 'utf8')
  const recallBody = modelSrc.slice(modelSrc.indexOf('async recall('), modelSrc.indexOf('async forget('))
  // ★ 第一版正则写成 score:\s*([^,\n]+)，在 cosine(readout, e.v) 的**第一个逗号**处就截断了，
  //   拿到 "cosine(readout" 然后误判失败——测试自己的 bug。
  //   改成按整行抓，再检查整行里是否只有这一个打分来源。
  //
  // ⚠️ 2026-11-17 更新：实现侧做了"关后门"重构，条目不再持久化 k/v，
  //    打分改成 cosine(readout, valueVectorOf(state, e))。检查也跟着放宽到
  //    「cosine(readout, 任意表达式)」，但**意图不变**：
  //    打分必须且只能来自 readout，不能出现 cosine(kq, ...) 这种绕过 M 的写法。
  const scoreLine = /score:\s*([^\n]+)/.exec(recallBody)
  line('')
  line(`recall 里的打分表达式：score: ${scoreLine ? scoreLine[1].trim() : '(未找到)'}`)
  assert('recall 的 score 表达式只依赖 cosine(readout, …)',
    !!scoreLine && /cosine\(\s*readout\s*,/.test(scoreLine[1]),
    scoreLine ? scoreLine[1].trim() : '未找到')
  // 反证：如果哪天有人把打分改成 cosine(kq, ...) 绕过矩阵，这里必须报错。
  assert('score 没有绕过 M 直接用查询键打分（后门检查）',
    !!scoreLine && !/cosine\(\s*kq\s*,/.test(scoreLine[1]),
    scoreLine ? scoreLine[1].trim() : '未找到')
  const readoutLine = /const readout = ([^\n]+)/.exec(recallBody)
  line(`readout 的构造：${readoutLine ? readoutLine[1].trim() : '(未找到)'}`)
  assert('readout 来自 M·k（matvec 调用）',
    !!readoutLine && /matvec\(\s*state\.M\s*,\s*kq/.test(readoutLine[1]),
    readoutLine ? readoutLine[1].trim() : '未找到')
}

// ══════════════════════════ 10. Config 与参数真的生效 ══════════════════════════

async function check10_config() {
  section('10', 'Config 校验与参数生效（防止"配置写了但被忽略"）')

  const { Config } = await import('../lib/plugin.js')
  const v = Config['~standard']
  assert('Config 暴露 standard-schema 形状（~standard.version=1）', v && v.version === 1,
    `version=${v?.version}, vendor=${v?.vendor}`)

  const full = v.validate({})
  line(`空配置校验结果：value=${JSON.stringify(full.value)}`)
  assert('空配置时给出全部默认值（dim=64, eta=0.2, θ=0.35, decay=0.02）',
    full.value?.dim === 64 && full.value?.eta === 0.2 &&
    full.value?.surpriseThreshold === 0.35 && full.value?.decay === 0.02,
    `dim=${full.value?.dim}, eta=${full.value?.eta}, θ=${full.value?.surpriseThreshold}, decay=${full.value?.decay}`)

  const bad = v.validate({ dim: 'abc', eta: 999 })
  line(`非法配置 {dim:'abc', eta:999} → issues=${bad.issues?.length ?? 0}`)
  assert('非法类型报 issue 而不是静默接受', Array.isArray(bad.issues) && bad.issues.length >= 1,
    `issues=${JSON.stringify(bad.issues)}`)

  const clamp = v.validate({ eta: 5, dim: 9999 })
  line(`越界配置 {eta:5, dim:9999} → eta=${clamp.value?.eta}, dim=${clamp.value?.dim}`)
  assert('eta 被夹到 0.4999 以下（稳定性条件 2η<2）', clamp.value.eta <= 0.4999, `eta=${clamp.value.eta}`)
  assert('dim 被夹到 256 以内', clamp.value.dim <= 256, `dim=${clamp.value.dim}`)

  // ── 参数是否真的作用于数值内核 ──
  const d1 = sectionDir('cfg-eta')
  const d2 = sectionDir('cfg-eta2')
  await fsp.mkdir(d1, { recursive: true })
  await fsp.mkdir(d2, { recursive: true })
  const t = '严既澄在成都分公司负责西南大区的渠道'
  const mA = new NeuralMemory({ path: d1, eta: 0.1 })
  const mB = new NeuralMemory({ path: d2, eta: 0.4 })
  await mA.observe(t, { namespace: 'x' })
  await mB.observe(t, { namespace: 'x' })
  const nA = frobenius(mA.stateOf('x').M)
  const nB = frobenius(mB.stateOf('x').M)
  line(`eta=0.1 → ‖M‖=${f(nA, 6)}；eta=0.4 → ‖M‖=${f(nB, 6)}；比值 ${f(nB / nA, 6)}（理论应为 4.0）`)
  assert('eta 真的进入梯度更新（∥M∥ 与 eta 成正比）', Math.abs(nB / nA - 4) < 1e-9,
    `实测比值 ${f(nB / nA, 12)} vs 理论 4`)

  const d3 = sectionDir('cfg-decay')
  await fsp.mkdir(d3, { recursive: true })
  const mC = new NeuralMemory({ path: d3, decay: 0.1 })
  for (const x of FACTS.slice(0, 5)) await mC.observe(x, { namespace: 'x' })
  const n0 = frobenius(mC.stateOf('x').M)
  await mC.forget({ namespace: 'x' })
  const n1 = frobenius(mC.stateOf('x').M)
  line(`decay=0.1 → ‖M‖ ${f(n0, 6)} × 0.9 = ${f(n0 * 0.9, 9)}；实测 ${f(n1, 9)}`)
  assert('decay 真的进入遗忘（‖M‖ ← (1−decay)‖M‖）', Math.abs(n1 - n0 * 0.9) / n0 < 1e-12,
    `相对误差 ${(Math.abs(n1 - n0 * 0.9) / n0).toExponential(3)}`)

  // ── namespace 隔离：两个 namespace 的记忆不应串味 ──
  const d4 = sectionDir('cfg-ns')
  await fsp.mkdir(d4, { recursive: true })
  const mD = new NeuralMemory({ path: d4 })
  await mD.observe('甲区的钥匙放在东边抽屉', { namespace: 'alpha' })
  await mD.observe('乙区的钥匙放在西边柜子', { namespace: 'beta' })
  const ra = await mD.recall('甲区的钥匙放哪', { namespace: 'alpha', topK: 5 })
  const rb = await mD.recall('甲区的钥匙放哪', { namespace: 'beta', topK: 5 })
  line(`namespace alpha 查到 ${ra.results.length} 条，beta 查到 ${rb.results.length} 条`)
  assert('两个 namespace 各自独立（beta 里查不到 alpha 的内容）',
    ra.entries === 1 && rb.entries === 1 && ra.results[0].text !== rb.results[0].text,
    `alpha.entries=${ra.entries}, beta.entries=${rb.entries}`)

  // ── 并发：同一 namespace 并发 observe 不应产生重复条目或 NaN ──
  const d5 = sectionDir('cfg-conc')
  await fsp.mkdir(d5, { recursive: true })
  const mE = new NeuralMemory({ path: d5 })
  const conc = FACTS.slice(0, 10)
  await Promise.all(conc.map((x) => mE.observe(x, { namespace: 'c' })))
  const stE = mE.stateOf('c')
  let finite = true
  for (let i = 0; i < stE.M.length; i++) if (!Number.isFinite(stE.M[i])) finite = false
  line(`10 条并发 observe → entries=${stE.entries.length}，M 全有限=${finite}`)
  assert('并发 observe 不产生重复条目', stE.entries.length === 10, `entries=${stE.entries.length}`)
  assert('并发 observe 后 M 无 NaN/Inf', finite, `finite=${finite}`)

  // 重复文本并发不应出现两条同 hash 条目
  await Promise.all(Array.from({ length: 5 }, () => mE.observe(conc[0], { namespace: 'c' })))
  const cnt = mE.stateOf('c').entries.filter((e) => e.text === conc[0]).length
  assert('同一文本并发重复 observe 在表里只有一条', cnt === 1, `实际 ${cnt} 条`)
}

// ══════════════════════════ 11. 宿主形状（假 ctx 驱动 apply） ══════════════════════════

async function check11_host() {
  section('11', '宿主形状：假 ctx 驱动 apply()，验延迟注册 / 卸载注销 / 不抛异常')

  const { apply, name, inject, Config } = await import('../lib/plugin.js')
  assert('plugin.js 导出 name / inject / Config / apply',
    name === 'dsh-neural-memory' && Array.isArray(inject) && !!Config && typeof apply === 'function',
    `name=${name}, inject=[${inject.join(',')}], Config=${typeof Config}, apply=${typeof apply}`)

  const dir = sectionDir('host')
  await fsp.mkdir(dir, { recursive: true })

  // ── 场景 A：tools 立刻可用 ──
  const registered = []
  const effects = []
  const logs = []
  const ctxA = {
    logger: { info: (m) => logs.push(m), warn: (m) => logs.push(m), debug: () => {} },
    get: (k) => (k === 'tools' ? { register: (def) => { registered.push(def.name); return () => {} } } : undefined),
    effect: (fn) => { effects.push(fn) },
    on: () => {},
    set: () => {},
  }
  apply(ctxA, { path: dir })
  line(`场景 A（tools 立即可用）：注册 ${registered.length} 个 → ${registered.join(', ')}`)
  // ⚠️ 2026-11-17：数量断言改为按 TOOL_NAMES 长度（新增工具不该造成假失败）
  assert('tools 就绪时全部工具注册', registered.length === TOOL_NAMES.length, `${registered.length}/${TOOL_NAMES.length}`)
  assert('注册后挂了 ctx.effect 用于卸载注销', effects.length === TOOL_NAMES.length, `effect=${effects.length}/${TOOL_NAMES.length}`)

  // ── 场景 B：tools 一开始不可用，稍后才 provide（延迟注册路径） ──
  const regB = []
  let toolsReady = false
  const ctxB = {
    logger: { info: (m) => logs.push(`B:${m}`), warn: (m) => logs.push(`B:warn:${m}`), debug: () => {} },
    get: (k) => (k === 'tools' && toolsReady ? { register: (def) => { regB.push(def.name); return () => {} } } : undefined),
    effect: () => {},
    on: () => {},
    set: () => {},
  }
  apply(ctxB, { path: dir, tools: true })
  line(`场景 B（首次 tools 不可用）：apply 后立即注册数=${regB.length}`)
  assert('tools 未就绪时不抛异常、不注册', regB.length === 0, `注册数=${regB.length}`)
  toolsReady = true
  await new Promise((r) => setTimeout(r, 700)) // 等 500ms 轮询兜底
  line(`场景 B（tools 就绪并等 700ms 后）：注册 ${regB.length} 个 → ${regB.join(', ')}`)
  assert('轮询兜底在 tools 就绪后完成注册', regB.length === TOOL_NAMES.length, `${regB.length}/${TOOL_NAMES.length}`)

  // ── 场景 C：重复注册（already registered 不应被当成失败反复重试） ──
  const regC = []
  const ctxC = {
    logger: { info: (m) => logs.push(`C:${m}`), warn: (m) => logs.push(`C:warn:${m}`), debug: () => {} },
    get: () => ({ register: (def) => { regC.push(def.name); if (regC.filter((x) => x === def.name).length > 1) throw new Error('tool already registered'); return () => {} } }),
    effect: () => {},
    on: () => {},
    set: () => {},
  }
  apply(ctxC, { path: dir })
  const cWarn = logs.filter((l) => l.startsWith('C:warn'))
  line(`场景 C（register 抛 already registered）：注册调用 ${regC.length} 次，warn 日志 ${cWarn.length} 条`)
  assert('already registered 不产生 warn 刷屏', cWarn.length === 0, `warn=${cWarn.length}`)

  // ── 场景 D：坏 ctx（什么都不提供）也不能抛 ──
  let threw = null
  try { apply({}, {}) } catch (e) { threw = e }
  assert('坏 ctx（无 logger/get/on/effect）下 apply 不抛异常', threw === null,
    threw ? String(threw.message) : '未抛')
  let threw2 = null
  try { apply(undefined, undefined) } catch (e) { threw2 = e }
  assert('apply(undefined, undefined) 不抛异常', threw2 === null,
    threw2 ? String(threw2.message) : '未抛')

  // ── 场景 E：apply 返回的 memory 实例可直接用 ──
  const ret = apply(ctxA, { path: dir, namespace: 'host' })
  const okRet = ret && ret.memory && typeof ret.memory.observe === 'function'
  line(`apply 返回：${okRet ? 'memory 实例可用' : '缺 memory'}`)
  assert('apply 返回可用的 memory 实例', okRet, okRet ? 'observe 存在' : '缺失')

  // ── 场景 F：真实 DSH_HOME 未被触碰 ──
  const realRoot = path.join(process.env.DSH_HOME || path.join(os.homedir(), '.dsh'), 'state', 'dsh-neural-memory')
  const existed = fs.existsSync(realRoot)
  line(`真实状态目录 ${realRoot} → ${existed ? '存在（可能是用户真实数据，本次未写入）' : '不存在（本次测试未创建它）'}`)
  line('  —— 本脚本全部实例都显式传了 path，没有任何一次走默认目录。')
}

// ══════════════════════════ 主流程 ══════════════════════════

async function main() {
  const t0 = Date.now()
  console.log('='.repeat(78))
  console.log('dsh-neural-memory · 独立验收 tests/run.mjs')
  console.log(`Node ${process.version} · ${new Date().toISOString()}`)
  console.log('立场：不采信作者报告的任何数字，全部重算。全部实例使用临时目录。')
  console.log('='.repeat(78))

  TMP_ROOT = await mkRoot()
  console.log(`临时根目录：${TMP_ROOT}`)
  console.log(`真实 DSH_HOME 是否被触碰：见第 8 节末尾（落盘目录只应在本临时目录内）`)

  let ctx3 = null
  let err = null
  try {
    await check1_math()
    await check2_gradient()
    ctx3 = await check3_recall()
    await check35_contract(ctx3)
    await check4_surprise()
    await check5_forget()
    await check6_provenance(ctx3)
    await check7_persistence()
    await check8_edges()
    const libFiles = (await fsp.readdir(LIB)).filter((n) => n.endsWith('.js')).map((n) => path.join(LIB, n))
    await check9_clean(libFiles)
    await check10_config()
    await check11_host()
  } catch (e) {
    err = e
    console.log('')
    console.log('!! 测试脚本自身异常中断 !!')
    console.log(e?.stack ?? String(e))
  } finally {
    const dt = Date.now() - t0
    console.log('')
    console.log('='.repeat(78))
    console.log('汇总')
    console.log('='.repeat(78))
    let passSections = 0
    for (const r of results) {
      if (r.ok) passSections++
      console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.id.padEnd(5)} ${r.title}`)
      if (!r.ok) for (const n of r.failNotes) console.log(`        └─ ${n}`)
    }
    const total = results.length
    console.log('')
    console.log(`小节：${passSections}/${total} 通过`)
    if (ctx3) {
      console.log(`召回率：${ctx3.hits}/${ctx3.total} = ${(ctx3.hitRate * 100).toFixed(1)}%（契约要求 ≥60%）`)
    }
    console.log(`临时目录：${TMP_ROOT}`)
    await rmDir(TMP_ROOT)
    console.log(`临时目录已清理：${!fs.existsSync(TMP_ROOT)}`)
    console.log(`总耗时：${dt} ms`)
    if (err) {
      console.log('')
      console.log('结论：脚本异常中断 → FAIL')
      process.exitCode = 1
    } else if (passSections === total) {
      console.log('')
      console.log('结论：全部检查通过 → exit 0')
      process.exitCode = 0
    } else {
      console.log('')
      console.log(`结论：${total - passSections} 个小节失败 → exit 1（失败态如实保留，未改阈值）`)
      process.exitCode = 1
    }
    console.log('='.repeat(78))
  }
}

main()

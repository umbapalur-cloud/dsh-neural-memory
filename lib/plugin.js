/**
 * lib/plugin.js —— dsh-neural-memory 的 Host 入口
 *
 * CCG 六要素:
 *   功能名   dsh-neural-memory：Titans 风格的神经持久化记忆
 *   生效条件 装进 profile，`tools` 服务就绪后注册 6 个 nm_* 工具；开启 autoObserve/autoRecall 后自动接入会话生命周期
 *   子功能   nm_observe（按惊讶度写入）· nm_recall（矩阵读出召回）· nm_forget（权重衰减）
 *            · nm_status（形态与统计）· nm_export（快照）· nm_delete（墓碑抹除+反学习）
 *            · lifecycle（可选自动观察 / 自动联想微注入 / 空闲刷盘与自动遗忘）
 *   执行     apply(ctx, config) 建一个 NeuralMemory 实例，注册工具，按配置接入生命周期，惰性加载 namespace
 *   验证方式 tests/run.mjs（12 节基线验收）/ tests/upgrade.mjs（升级特性验收）
 *   不适用条件 需要毫秒级海量检索（这是 d² 的权重，不是向量库）；
 *              需要跨 namespace 共享权重（每个 namespace 一份 M，故意的）
 *
 * ⚠️ 工具注册为什么要延迟重试（对齐 dsh-mind/lib/tool-register.js 的修复）：
 *   apply 时机 `ctx.get('tools')` 常常拿不到东西 —— 服务还没 provide。
 *   1. 未就绪（!tools?.register）只计 notReady，不消耗 attempts，避免 10 秒就静默放弃；
 *   2. 必须 6/6 全部注册完成（done.size === defs.length）才置 settled，防止部分成功提前停轮询；
 *   3. 「already registered」视为目标已达成，记进 done 集合跳过，绝不刷 warn。
 */

import { NeuralMemory, resolveOptions, DEFAULT_OPTIONS, PLUGIN_NAME } from './model.js'
import { makeToolDefinitions } from './tools.js'
import { attachLifecycle } from './lifecycle.js'

export const name = PLUGIN_NAME

/**
 * 不声明 inject：tools 可能晚于本插件就绪，声明了反而会被 Cordis 卡在 PENDING。
 * 用 ctx.get('tools') 主动探测 + tools/change 补注册，两条路都不依赖注入时序。
 */
export const inject = []

/**
 * Config：**总是**导出一个 standard-schema 形状的校验器。
 *
 * 为什么不直接 import @deepseek-ai/schemastery：
 *   1. 契约 2.1 要求零运行时依赖，往 package.json 加依赖越界（那是 LEAD 的文件）；
 *   2. 宿主若在运行时提供了 schemastery，Cordis 走的是 `Config['~standard'].validate`，
 *      所以只要这个接口在，默认值就一定会被填好；
 *   3. 真拿不到 schemastery 时也不能让 Config 变成 undefined —— 那样配置原样透传、
 *      缺少的键要插件自己兜，等于把"默认值"这件事散在两处。
 * 这里手写一个最小实现，语义与 Schemastery 对齐：给默认值、类型不对就报 issue。
 */
export const Config = {
  '~standard': {
    version: 1,
    vendor: PLUGIN_NAME,
    validate(input) {
      const raw = input && typeof input === 'object' && !Array.isArray(input) ? input : {}
      const issues = []
      const out = {}

      const numberField = (key, min, max) => {
        if (!(key in raw) || raw[key] === undefined || raw[key] === null) {
          out[key] = DEFAULT_OPTIONS[key]
          return
        }
        const v = Number(raw[key])
        if (!Number.isFinite(v)) {
          issues.push({ message: `${key} 必须是数字`, path: [key] })
          out[key] = DEFAULT_OPTIONS[key]
          return
        }
        out[key] = Math.min(max, Math.max(min, v))
      }

      numberField('dim', 8, 256)
      // 2η < 2 是稳定性条件，超过 0.5 会震荡发散，这里直接夹住而不是让用户踩坑。
      numberField('eta', 0.0001, 0.4999)
      numberField('surpriseThreshold', 0, 1e6)
      // 2026-11-17 修复一：相对新颖度门限，与 surpriseThreshold 组成双判据。
      numberField('surpriseThresholdRel', 0, 2)
      numberField('decay', 0, 0.999)
      numberField('minStrength', 0, 1e6)
      numberField('maxEntries', 1, 100000)
      // 2026-11-17 修复二：置信度熔断的阈值与生效门槛。
      numberField('minScore', -1, 1)
      // 2026-11-17 新增：词面覆盖门限（召回的第二道真伪闸门，见 model.js 的 bigramCoverage）。
      numberField('minCoverage', 0, 1)
      numberField('minEntriesForTrust', 1, 10000)
      numberField('decayPerObserve', 0, 0.9)
      out.dim = Math.round(out.dim)
      out.maxEntries = Math.round(out.maxEntries)
      out.minEntriesForTrust = Math.round(out.minEntriesForTrust)

      if (typeof raw.namespace === 'string' && raw.namespace.length > 0) {
        out.namespace = raw.namespace
      } else {
        out.namespace = DEFAULT_OPTIONS.namespace
      }

      // path 用空串表示"走默认目录"（<DSH_HOME>/state/dsh-neural-memory）。
      // 不用 null 默认值，是因为配置文件里写 path: null 比留空更容易被误读成"禁用它"。
      out.path = typeof raw.path === 'string' && raw.path.length > 0 ? raw.path : null

      if (raw.tools === undefined || raw.tools === null) out.tools = true
      else out.tools = raw.tools !== false

      if (typeof raw.maxChars === 'number' && Number.isFinite(raw.maxChars)) {
        out.maxChars = Math.round(Math.min(1 << 20, Math.max(1, raw.maxChars)))
      } else {
        out.maxChars = DEFAULT_OPTIONS.maxChars
      }

      const boolField = (key) => {
        if (!(key in raw) || raw[key] === undefined || raw[key] === null) {
          out[key] = DEFAULT_OPTIONS[key]
        } else if (typeof raw[key] === 'boolean') {
          out[key] = raw[key]
        } else {
          issues.push({ message: `${key} 必须是布尔值`, path: [key] })
          out[key] = DEFAULT_OPTIONS[key]
        }
      }

      if (raw.embedMode === undefined || raw.embedMode === null) {
        out.embedMode = DEFAULT_OPTIONS.embedMode
      } else if (raw.embedMode === 'ngram' || raw.embedMode === 'hybrid') {
        out.embedMode = raw.embedMode
      } else {
        issues.push({ message: 'embedMode 必须是 ngram 或 hybrid', path: ['embedMode'] })
        out.embedMode = DEFAULT_OPTIONS.embedMode
      }

      if (raw.storageFormat === undefined || raw.storageFormat === null) {
        out.storageFormat = DEFAULT_OPTIONS.storageFormat
      } else if (raw.storageFormat === 'json' || raw.storageFormat === 'base64') {
        out.storageFormat = raw.storageFormat
      } else {
        issues.push({ message: 'storageFormat 必须是 json 或 base64', path: ['storageFormat'] })
        out.storageFormat = DEFAULT_OPTIONS.storageFormat
      }

      numberField('momentum', 0, 0.95)
      numberField('recallBoost', 0, 2)
      numberField('persistDebounceMs', 0, 60000)
      numberField('maxLoadedNamespaces', 2, 256)
      numberField('autoObserveThreshold', 0, 1e6)
      numberField('autoObserveMaxSentences', 1, 20)
      numberField('autoRecallMinScore', -1, 1)
      numberField('autoRecallMinCoverage', 0, 1)
      numberField('autoRecallMinMargin', 0, 1)
      numberField('autoRecallTopK', 1, 10)
      numberField('autoRecallCooldownTurns', 0, 100)
      numberField('autoInjectMaxChars', 120, 4000)
      numberField('autoForgetEveryTurns', 0, 1000)
      numberField('autoForgetDecay', 0, 0.5)
      out.persistDebounceMs = Math.round(out.persistDebounceMs)
      out.maxLoadedNamespaces = Math.round(out.maxLoadedNamespaces)
      out.autoObserveMaxSentences = Math.round(out.autoObserveMaxSentences)
      out.autoRecallTopK = Math.round(out.autoRecallTopK)
      out.autoRecallCooldownTurns = Math.round(out.autoRecallCooldownTurns)
      out.autoInjectMaxChars = Math.round(out.autoInjectMaxChars)
      out.autoForgetEveryTurns = Math.round(out.autoForgetEveryTurns)

      boolField('migrateOnDimChange')
      boolField('autoObserve')
      boolField('autoRecall')

      if (issues.length) return { issues }
      return { value: out }
    },
  },
}

/** 只在宿主提供 logger 时记一行；绝不 console.log（契约 6）。 */
function makeLogger(ctx) {
  const sink = ctx?.logger
  const emit = (level, msg) => {
    try {
      if (sink && typeof sink[level] === 'function') sink[level](`[${PLUGIN_NAME}] ${msg}`)
    } catch {
      /* 日志失败不能影响功能 */
    }
  }
  return {
    info: (m) => emit('info', m),
    warn: (m) => emit('warn', m),
    debug: (m) => emit('debug', m),
  }
}

export function apply(ctx, config = {}) {
  const options = resolveOptions(config)
  const log = makeLogger(ctx)
  const memory = new NeuralMemory({ ...options, store: null })

  // 启动就把默认 namespace 读进内存；失败也只是"这一次从空记忆开始"，不影响后续。
  Promise.resolve()
    .then(() => memory.ensureLoaded(options.namespace))
    .catch(() => {})

  // ── 工具注册（未就绪延迟重试 + tools/change，全量到位才结束） ──
  if (options.tools !== false) {
    let settled = false
    let attempts = 0
    let notReady = 0
    const MAX_ATTEMPTS = 20
    const MAX_NOT_READY = 600
    const done = new Set()
    const defs = makeToolDefinitions(memory, { config: options })

    const isAlreadyRegistered = (e) => /already registered|duplicate tool|已注册/i.test(String(e?.message ?? e))

    const tryRegister = () => {
      if (settled || attempts >= MAX_ATTEMPTS) return
      try {
        const tools = typeof ctx?.get === 'function' ? ctx.get('tools') : undefined
        if (!tools || typeof tools.register !== 'function') {
          notReady++
          return // 服务尚未就绪，不消耗 attempts 次数
        }
        attempts++
        for (const def of defs) {
          if (done.has(def.name)) continue
          try {
            const dispose = tools.register(def)
            if (typeof ctx?.effect === 'function' && typeof dispose === 'function') ctx.effect(() => dispose)
            done.add(def.name)
          } catch (e) {
            if (isAlreadyRegistered(e)) {
              done.add(def.name)
            } else {
              log.warn(`工具注册失败 :: ${def?.name} :: ${e?.message ?? e}`)
            }
          }
        }
        if (done.size === defs.length) {
          settled = true
          log.info(`nm_* 工具已注册 :: ${done.size}/${defs.length} 个（第 ${attempts} 次尝试）`)
        }
      } catch (e) {
        attempts++
        log.warn(`tools 注册异常 :: ${e?.message ?? e}`)
      }
    }

    tryRegister()
    if (!settled) {
      log.info('tools 服务尚未就绪，改延迟注册（tools/change + 500ms 轮询）')
      let offChange = null
      try {
        if (typeof ctx?.on === 'function') offChange = ctx.on('tools/change', () => tryRegister())
      } catch {
        /* 事件总线不可用时靠轮询 */
      }
      const timer = setInterval(() => {
        tryRegister()
        if (settled || attempts >= MAX_ATTEMPTS || notReady >= MAX_NOT_READY) {
          clearInterval(timer)
          if (typeof offChange === 'function') {
            try {
              offChange()
            } catch {
              /* 忽略 */
            }
            offChange = null
          }
          if (!settled) {
            log.warn(`工具注册超时放弃 :: 已完成 ${done.size}/${defs.length}（attempts=${attempts}, notReady=${notReady}）`)
          }
        }
      }, 500)
      if (typeof timer.unref === 'function') timer.unref()
      if (typeof ctx?.effect === 'function') {
        ctx.effect(() => () => {
          clearInterval(timer)
          if (typeof offChange === 'function') {
            try {
              offChange()
            } catch {
              /* 忽略 */
            }
          }
        })
      }
    }
  }

  // ── 生命周期接入（仅当启用自动观察 / 自动联想注入 / 自动遗忘 / 防抖刷盘时挂载） ──
  const lifecycleEnabled =
    options.autoObserve ||
    options.autoRecall ||
    options.autoForgetEveryTurns > 0 ||
    options.persistDebounceMs > 0
  if (lifecycleEnabled) {
    const disposeLifecycle = attachLifecycle(ctx, memory, options, log)
    if (typeof ctx?.effect === 'function') {
      ctx.effect(() => disposeLifecycle)
    }
  }

  // 把实例挂到 ctx 上，方便同进程内别的插件/调试工具直接摸（失败就算了）。
  try {
    if (typeof ctx?.set === 'function') ctx.set('neural-memory', memory)
  } catch {
    /* 服务注册不是必需的 */
  }

  return { memory }
}

/**
 * lib/lifecycle.js —— dsh-neural-memory 与 DSH 会话生命周期的轻量集成层
 *
 * 零 DSH 包 import，仅通过宿主事件形状接入：
 *   1. agent/inbox/claimed  → 自动观察（Auto-Observe）：过滤机器/插件消息、去代码块分句、按高惊讶度沉淀
 *   2. agent/pre-step       → 自动联想微注入（Auto-Recall）：先 await next()，三闸门+冷却+眼前去重，未命中 0 字节隐身
 *   3. agent/status (idle)  → 空闲冲刷脏状态 + 按轮次自动轻量遗忘（让遗忘曲线真实运转）
 *   4. agent/disposed       → 清理会话冷却态并冲刷落盘
 *
 * 与 dsh-mind / dsh-strand 的防循环契约：
 *   本模块注入消息的 source.kind 固定为 'dsh-neural-memory-inject'，
 *   匹配 dsh-mind/lib/message.js 的 /^dsh-.*(inject|push|context|brief|mind)/i 通用防线，
 *   同时本模块也过滤一切 dsh-mind / dsh-strand / 系统提醒消息，双方绝不互相放大。
 */

import { randomUUID } from 'node:crypto'

export const INJECT_SOURCE_KIND = 'dsh-neural-memory-inject'
export const INJECT_WATERMARK = '【神经记忆 · 自动联想 · 非用户输入】'

const MAX_SESSIONS = 64
const SELF_KIND_RE = /^dsh-.*(?:inject|push|context|brief|mind|strand|neural)|^agent-instructions$|^system-reminder$/i
const WATERMARK_PREFIXES = [
  INJECT_WATERMARK,
  '【心智',
  '【跨会话',
  '【经验',
  '【自我觉知',
  '【记忆',
  '【压缩',
  '【生命索引',
]
const MACHINE_PREFIX_RE = /^(?:<system-reminder>|<goal_round>|<CONTEXT_SUMMARY>|\[Message\]\s+timestamp=|Stop hook blocked|\[background job)/i
const GREETING_ONLY_RE = /^(?:好|好的|好呀|好哒|行|可以|收到|明白|了解|嗯|嗯嗯|哦|谢谢|感谢|继续|开始|ok|okay|yes|no|thx|thanks|hi|hello|\/[a-z0-9_-]+)[!！。.~～?？\s]*$/i

/** 提取消息里的纯文本（兼容 string 与 content blocks 数组）。 */
export function extractPlainText(msg) {
  if (!msg) return ''
  if (typeof msg === 'string') return msg
  const c = msg.content
  if (typeof c === 'string') return c
  if (Array.isArray(c)) {
    const parts = []
    for (const block of c) {
      if (block && block.type === 'text' && typeof block.text === 'string') {
        parts.push(block.text)
      }
    }
    return parts.join('\n')
  }
  return ''
}

/** 判别是否为本插件或其他 DSH 记忆/系统插件自产消息。 */
export function isSelfOrPluginMessage(msg) {
  if (!msg || typeof msg !== 'object') return false
  const kind = msg.source && typeof msg.source.kind === 'string' ? msg.source.kind : ''
  if (kind && SELF_KIND_RE.test(kind)) return true
  const body = extractPlainText(msg).trim()
  for (const prefix of WATERMARK_PREFIXES) {
    if (body.startsWith(prefix)) return true
  }
  return false
}

/** 判别是否为机器/系统自动生成的播报文本。 */
export function looksMachineText(body) {
  if (typeof body !== 'string') return true
  const trimmed = body.trim()
  if (trimmed.length === 0) return true
  return MACHINE_PREFIX_RE.test(trimmed)
}

/** 清洗并按语义句边界切分适合神经记忆观察的片段（跳过代码块与纯寒暄）。 */
export function extractCandidateSentences(body, maxSentences = 4) {
  if (typeof body !== 'string') return []
  const noCode = body
    .replace(/```[\s\S]*?```/g, '\n')
    .replace(/`[^`\n]{40,}`/g, ' ')
  const pieces = noCode.split(/[。！？；\n]+/u)
  const out = []
  const seen = new Set()
  for (const part of pieces) {
    const cleaned = part.replace(/\s+/gu, ' ').trim()
    if (cleaned.length < 12 || cleaned.length > 220) continue
    if (GREETING_ONLY_RE.test(cleaned)) continue
    if (looksMachineText(cleaned)) continue
    if (/^https?:\/\/\S+$/.test(cleaned)) continue
    if (seen.has(cleaned)) continue
    seen.add(cleaned)
    out.push(cleaned)
    if (out.length >= maxSentences) break
  }
  return out
}

/** 检测当前会话是否发生过新的 compaction/end 事件。 */
function detectCompactionSeq(agent) {
  try {
    const session = agent?.session
    if (!session) return 0
    if (typeof session.lastCompactionSeq === 'number') return session.lastCompactionSeq
    const events = Array.isArray(session.events)
      ? session.events
      : (typeof session.tailEvents === 'function' ? session.tailEvents(60) : null)
    if (!Array.isArray(events)) return 0
    let latest = 0
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i]
      if (ev && ev.type === 'compaction/end') {
        const seq = Number(ev.seq ?? i + 1)
        if (seq > latest) latest = seq
        break
      }
    }
    return latest
  } catch {
    return 0
  }
}

/** 从 messages 列表定位最新一条真人用户消息。 */
function findLatestHumanUserMessage(messages) {
  if (!Array.isArray(messages)) return null
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (!m || m.role !== 'user') continue
    if (isSelfOrPluginMessage(m)) continue
    const body = extractPlainText(m).trim()
    if (looksMachineText(body)) continue
    return { message: m, body }
  }
  return null
}

/** 检查某条候选记忆的原文是否已经出现在最近几条对话里（避免刚说过就复读）。 */
function alreadyVisibleInRecent(messages, candidateText) {
  if (!Array.isArray(messages) || !candidateText) return false
  const needle = candidateText.trim()
  if (needle.length < 8) return false
  const start = Math.max(0, messages.length - 6)
  for (let i = start; i < messages.length; i++) {
    const m = messages[i]
    if (!m || isSelfOrPluginMessage(m)) continue
    const body = extractPlainText(m)
    if (body && body.indexOf(needle) !== -1) return true
  }
  return false
}

/** 构造符合 DSH 宿主契约的神经记忆注入消息，并限制在 maxChars 以内。 */
export function buildRecallMessage(hits, maxChars = 600) {
  if (!Array.isArray(hits) || hits.length === 0) return null
  const lines = [INJECT_WATERMARK]
  let used = INJECT_WATERMARK.length
  let count = 0
  for (const h of hits) {
    const snippet = String(h.text ?? '').replace(/\s+/gu, ' ').trim()
    if (!snippet) continue
    const scoreTag = Number.isFinite(h.score) ? ` (r=${h.score.toFixed(2)})` : ''
    const line = `- ${snippet}${scoreTag}`
    if (used + 1 + line.length > maxChars) {
      if (count === 0) {
        const avail = Math.max(24, maxChars - used - 5 - scoreTag.length)
        lines.push(`- ${snippet.slice(0, avail)}…${scoreTag}`)
        count++
      }
      break
    }
    lines.push(line)
    used += 1 + line.length
    count++
  }
  if (count === 0) return null
  return {
    id: randomUUID(),
    role: 'user',
    content: [{ type: 'text', text: lines.join('\n') }],
    source: { kind: INJECT_SOURCE_KIND, form: 'nm-recall' },
  }
}

/** 按 claimed 消息 id 紧随其后插入，保持上游/系统消息顺序稳定。 */
export function insertAfterClaimed(messages, claimedId, injection) {
  const list = Array.isArray(messages) ? messages.slice() : []
  if (claimedId) {
    const idx = list.findIndex((m) => m && m.id === claimedId)
    if (idx !== -1) {
      list.splice(idx + 1, 0, injection)
      return list
    }
  }
  list.push(injection)
  return list
}

function sessionIdOf(agent) {
  return String(agent?.session?.id ?? agent?.sessionId ?? agent?.id ?? 'default-session')
}

/**
 * 将神经记忆接入 DSH 会话生命周期。
 * 仅在 autoObserve / autoRecall / autoForgetEveryTurns 至少开启一项时调用。
 *
 * @returns {() => void} 卸载清理函数
 */
export function attachLifecycle(ctx, memory, options, log) {
  const disposers = []
  const sessions = new Map() // sid -> { turn: number, cooldown: Map<hash, turn>, lastCompSeq: number }
  const autoRecallStats = {
    calls: 0,
    injected: 0,
    noMatch: 0,
    marginBlocked: 0,
    marginBypassed: 0,
    cooldownOrVisible: 0,
  }
  const getAutoRecallStats = () => ({ ...autoRecallStats })
  if (memory && typeof memory === 'object') {
    memory.getAutoRecallStats = getAutoRecallStats
  }
  let idleTurns = 0

  const getSessionState = (sid) => {
    let st = sessions.get(sid)
    if (!st) {
      if (sessions.size >= MAX_SESSIONS) {
        const oldest = sessions.keys().next().value
        if (oldest !== undefined) sessions.delete(oldest)
      }
      st = { turn: 0, cooldown: new Map(), lastCompSeq: 0 }
      sessions.set(sid, st)
    }
    return st
  }

  const safeOn = (event, handler) => {
    try {
      if (typeof ctx?.on === 'function') {
        const off = ctx.on(event, handler)
        if (typeof off === 'function') disposers.push(off)
      }
    } catch (e) {
      log?.warn?.(`监听 ${event} 失败 :: ${e?.message ?? e}`)
    }
  }

  // 1) 自动观察：捕获真人发言中的高惊讶新事实
  if (options.autoObserve) {
    safeOn('agent/inbox/claimed', (payload) => {
      try {
        const msg = payload?.message
        if (!msg || isSelfOrPluginMessage(msg)) return
        const body = extractPlainText(msg)
        if (looksMachineText(body)) return
        const sentences = extractCandidateSentences(body, options.autoObserveMaxSentences)
        if (sentences.length === 0) return
        // 异步后台观察，绝不阻塞主会话流程
        ;(async () => {
          for (const seg of sentences) {
            await memory.observe(seg, {
              namespace: options.namespace,
              surpriseThreshold: options.autoObserveThreshold,
            })
          }
        })().catch((e) => {
          log?.debug?.(`autoObserve 异常 :: ${e?.message ?? e}`)
        })
      } catch (e) {
        log?.debug?.(`autoObserve 同步过滤异常 :: ${e?.message ?? e}`)
      }
    })
  }

  // 2) 自动联想微注入：仅高置信命中时按需注入 ≤ autoInjectMaxChars
  if (options.autoRecall) {
    safeOn('agent/pre-step', async (payload, next) => {
      const decision = typeof next === 'function' ? await next() : payload
      try {
        if (!decision || decision.kind === 'reject' || payload?.signal?.aborted) return decision
        if (!Array.isArray(decision.messages) || decision.messages.length === 0) return decision

        const sid = sessionIdOf(payload?.agent)
        const sst = getSessionState(sid)
        sst.turn += 1

        // 压缩感知：若会话刚经历压缩，立刻清空冷却，让关键高惊讶记忆可以重新唤醒
        const compSeq = detectCompactionSeq(payload?.agent)
        if (compSeq > sst.lastCompSeq) {
          sst.lastCompSeq = compSeq
          sst.cooldown.clear()
        }

        const latest = findLatestHumanUserMessage(decision.messages)
        if (!latest || latest.body.length < 4) return decision

        autoRecallStats.calls += 1
        const res = await memory.recall(latest.body, {
          namespace: options.namespace,
          topK: options.autoRecallTopK,
          minScore: options.autoRecallMinScore,
          minCoverage: options.autoRecallMinCoverage,
        })
        if (!res?.ok || !res.matched || !Array.isArray(res.results) || res.results.length === 0) {
          autoRecallStats.noMatch += 1
          if (res?.ok && Number.isFinite(res.top1Score) && res.top1Score >= options.autoRecallMinScore - 0.12) {
            log?.debug?.(
              `autoRecall 近阈值未命中 :: top1=${res.top1Score.toFixed(3)} cov=${Number(res.top1Coverage ?? 0).toFixed(2)} margin=${Number(res.margin ?? 0).toFixed(3)}`,
            )
          }
          return decision
        }

        // 唯一峰置信检查：按实际胜出候选的 matchedScore / matchedMargin 判别（避免被词面未过线的假阳性 scored[0] 误导）
        const winnerScore = Number.isFinite(res.matchedScore) ? res.matchedScore : (res.results[0]?.score ?? res.top1Score)
        const winnerMargin = Number.isFinite(res.matchedMargin) ? res.matchedMargin : res.margin
        if (res.entries >= 4 && winnerMargin < options.autoRecallMinMargin) {
          if (winnerScore >= 0.72) {
            autoRecallStats.marginBypassed += 1
            log?.debug?.(
              `autoRecall 高分豁免边际闸门 :: score=${winnerScore.toFixed(3)} margin=${winnerMargin.toFixed(3)}`,
            )
          } else {
            autoRecallStats.marginBlocked += 1
            log?.debug?.(
              `autoRecall 边际闸门拦截 :: score=${winnerScore.toFixed(3)} margin=${winnerMargin.toFixed(3)} < ${options.autoRecallMinMargin}`,
            )
            return decision
          }
        }

        // 冷却 + 眼前上下文去重
        const eligible = []
        for (const hit of res.results) {
          const lastTurn = sst.cooldown.get(hit.hash)
          if (lastTurn !== undefined && sst.turn - lastTurn < options.autoRecallCooldownTurns) continue
          if (alreadyVisibleInRecent(decision.messages, hit.text)) continue
          eligible.push(hit)
        }
        if (eligible.length === 0) {
          autoRecallStats.cooldownOrVisible += 1
          return decision
        }

        const injMsg = buildRecallMessage(eligible, options.autoInjectMaxChars)
        if (!injMsg) return decision

        autoRecallStats.injected += 1
        for (const hit of eligible) sst.cooldown.set(hit.hash, sst.turn)
        return {
          ...decision,
          messages: insertAfterClaimed(decision.messages, latest.message.id, injMsg),
        }
      } catch (e) {
        log?.debug?.(`autoRecall 异常已降级直通 :: ${e?.message ?? e}`)
        return decision
      }
    })
  }

  // 3) 空闲刷盘 + 按轮次自动轻量遗忘
  safeOn('agent/status', (payload) => {
    try {
      if (payload?.status !== 'idle') return
      memory.flushAll().catch(() => {})
      if (options.autoForgetEveryTurns > 0) {
        idleTurns += 1
        if (idleTurns >= options.autoForgetEveryTurns) {
          idleTurns = 0
          memory.forget({
            namespace: options.namespace,
            decay: options.autoForgetDecay,
          }).catch((e) => {
            log?.debug?.(`autoForget 异常 :: ${e?.message ?? e}`)
          })
        }
      }
    } catch {
      /* 状态钩子绝不外抛 */
    }
  })

  // 4) 会话释放：清理会话冷却表并刷盘
  safeOn('agent/disposed', (payload) => {
    try {
      const sid = sessionIdOf(payload?.agent)
      sessions.delete(sid)
      memory.flushAll().catch(() => {})
    } catch {
      /* 忽略 */
    }
  })

  log?.info?.(
    `生命周期已接入 :: autoObserve=${options.autoObserve} autoRecall=${options.autoRecall} autoForgetEveryTurns=${options.autoForgetEveryTurns}`,
  )

  return () => {
    for (const off of disposers) {
      try {
        off()
      } catch {
        /* 忽略 */
      }
    }
    disposers.length = 0
    sessions.clear()
    if (memory && memory.getAutoRecallStats === getAutoRecallStats) {
      delete memory.getAutoRecallStats
    }
    memory.flushAll().catch(() => {})
  }
}

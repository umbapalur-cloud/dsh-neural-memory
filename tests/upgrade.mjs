/**
 * tests/upgrade.mjs —— dsh-neural-memory v1.1.0 升级特性专项验收
 * 全部实例使用临时目录，绝不触碰真实 DSH_HOME。
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { NeuralMemory, embedTextHybrid, frobenius } from '../lib/model.js'
import { apply, Config } from '../lib/plugin.js'
import { INJECT_SOURCE_KIND, INJECT_WATERMARK, extractCandidateSentences, attachLifecycle } from '../lib/lifecycle.js'

let passed = 0
let failed = 0

function assert(name, cond, detail = '') {
  if (cond) {
    passed++
    console.log(`  [PASS] ${name}${detail ? ' :: ' + detail : ''}`)
  } else {
    failed++
    console.error(`  [FAIL] ${name}${detail ? ' :: ' + detail : ''}`)
  }
}

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nm-upgrade-'))

async function main() {
  try {
    console.log('=== 1. hybrid 嵌入 + 标识符/英文词干/自然问句召回 ===')
    {
      const dir = path.join(tmpRoot, 's1-hybrid')
      const mem = new NeuralMemory({ path: dir, dim: 128, embedMode: 'hybrid', momentum: 0.25 })
      const facts = [
        '项目 qingteng.dev 的生产端口已迁移到 8443',
        'Alex prefers Rust servers for low-latency audio pipelines',
        '陈景行的工号是 KX9920 并且负责结算模块',
        '安然在 2026 年改用 C# 开发桌面客户端',
      ]
      for (const f of facts) await mem.observe(f, { surpriseThreshold: 0 })

      const r1 = await mem.recall('请问 qingteng.dev 的生产端口是多少啊')
      assert('hybrid 含自然问句虚词仍命中目标', r1.matched && r1.results[0]?.text.includes('8443'),
        `score=${r1.top1Score.toFixed(3)} cov=${r1.top1Coverage.toFixed(2)} margin=${r1.margin.toFixed(3)}`)

      const r2 = await mem.recall('Rust server audio pipeline')
      assert('hybrid 英文词边界+词干(servers→server)命中', r2.matched && r2.results[0]?.text.includes('Rust'),
        `score=${r2.top1Score.toFixed(3)} cov=${r2.top1Coverage.toFixed(2)}`)

      const r3 = await mem.recall('南极洲的企鹅冬天吃什么')
      assert('hybrid 库外无关问句依然 0 误报', !r3.matched && r3.results.length === 0,
        `top1=${r3.top1Score.toFixed(3)} cov=${r3.top1Coverage.toFixed(2)}`)

      const v1 = embedTextHybrid('Rust server port 8443', 128)
      const v2 = embedTextHybrid('Rust server port 8443', 128)
      let diff = 0
      for (let i = 0; i < 128; i++) diff = Math.max(diff, Math.abs(v1[i] - v2[i]))
      assert('embedTextHybrid 100% 确定性', diff === 0)
    }

    console.log('\n=== 2. Titans 惊讶动量 + 回忆强化 + 删除反学习 ===')
    {
      const dir = path.join(tmpRoot, 's2-dynamics')
      const mNoMom = new NeuralMemory({ path: dir + '-a', dim: 64, momentum: 0 })
      const mMom = new NeuralMemory({ path: dir + '-b', dim: 64, momentum: 0.3, recallBoost: 0.25 })
      await mNoMom.observe('深海声呐协议版本为 v3.4', { surpriseThreshold: 0 })
      await mNoMom.observe('深海声呐协议默认采样率 48kHz', { surpriseThreshold: 0 })
      await mMom.observe('深海声呐协议版本为 v3.4', { surpriseThreshold: 0 })
      await mMom.observe('深海声呐协议默认采样率 48kHz', { surpriseThreshold: 0 })
      const stMom = await mMom.status()
      assert('momentum > 0 产生非零动量矩阵 S', stMom.momentumNorm > 0, `‖S‖=${stMom.momentumNorm.toFixed(4)}`)

      // 回忆强化抗遗忘测试：只召回第一条，然后执行 40 轮衰减，第一条强度应显著高于第二条
      await mMom.observe('冷门一次性备忘：周二下午清理临时缓存目录', { surpriseThreshold: 0 })
      await mMom.recall('深海声呐协议版本 v3.4')
      await mMom.recall('深海声呐协议版本 v3.4')
      const beforeBoosted = mMom.stateOf('default').entries.find((e) => e.text.includes('v3.4'))?.strength ?? 0
      const beforeCold = mMom.stateOf('default').entries.find((e) => e.text.includes('缓存目录'))?.strength ?? 0
      assert('recallBoost 提升高频回忆条目强度', beforeBoosted > beforeCold + 0.3,
        `boosted=${beforeBoosted.toFixed(3)} vs cold=${beforeCold.toFixed(3)}`)

      // 删除反学习：删除后用相同查询读出，其与被删值的投影应显著归零
      const normBeforeDel = frobenius(mMom.stateOf('default').M)
      const delRes = await mMom.deleteEntries({ texts: ['冷门一次性备忘：周二下午清理临时缓存目录'], dryRun: false })
      const normAfterDel = frobenius(mMom.stateOf('default').M)
      assert('deleteEntries 反学习同步降低权重残余', delRes.deleted === 1 && normAfterDel < normBeforeDel,
        `‖M‖: ${normBeforeDel.toFixed(4)} → ${normAfterDel.toFixed(4)}`)
    }

    console.log('\n=== 3. Base64 紧凑编码 + dim 升级无损迁移 + 防抖落盘 + BOM 兼容 ===')
    {
      const dir = path.join(tmpRoot, 's3-storage')
      // 先用旧版 dim=64 + json 落盘 3 条
      const oldMem = new NeuralMemory({ path: dir, dim: 64, embedMode: 'ngram', storageFormat: 'json' })
      await oldMem.observe('小青喜欢喝热桂花乌龙茶', { surpriseThreshold: 0 })
      await oldMem.observe('项目构建目录位于 /opt/demo-app/dist', { surpriseThreshold: 0 })
      await oldMem.observe('服务器主控端口锁定在 9527', { surpriseThreshold: 0 })
      const jsonSize = fs.statSync(path.join(dir, 'default.json')).size

      // 故意给 _deleted.json 写入 UTF-8 BOM 测试兼容
      await fsp.writeFile(path.join(dir, '_deleted.json'), '\uFEFF{"version":1,"namespaces":{}}', 'utf8')

      // 升级到 dim=128 + hybrid + base64 + migrateOnDimChange=true
      const newMem = new NeuralMemory({
        path: dir,
        dim: 128,
        embedMode: 'hybrid',
        storageFormat: 'base64',
        migrateOnDimChange: true,
        persistDebounceMs: 200,
      })
      const rMig = await newMem.recall('热桂花乌龙茶')
      assert('dim 64→128 + ngram→hybrid 无损迁移保留原记忆并可召回',
        rMig.entries === 3 && rMig.matched && rMig.results[0]?.text.includes('桂花乌龙茶'),
        `entries=${rMig.entries} score=${rMig.top1Score.toFixed(3)}`)

      await newMem.flushAll()
      const rawDisk = JSON.parse(fs.readFileSync(path.join(dir, 'default.json'), 'utf8'))
      const b64Size = fs.statSync(path.join(dir, 'default.json')).size
      assert('base64 紧凑格式写入 M_b64 且不写冗长数组 M', typeof rawDisk.M_b64 === 'string' && !rawDisk.M,
        `dim=64 json=${jsonSize}B vs dim=128(4x矩阵) b64=${b64Size}B`)

      // 重新加载验证 base64 逐位无损读回
      const reloadMem = new NeuralMemory({ path: dir, dim: 128, embedMode: 'hybrid', storageFormat: 'base64' })
      const rReload = await reloadMem.recall('项目构建目录 /opt/demo-app/dist')
      assert('base64 状态重启读回召回正常', rReload.matched && rReload.results[0]?.text.includes('/opt/demo-app/dist'),
        `score=${rReload.top1Score.toFixed(3)}`)

      // mergeExternal 补学测试：另一实例写入新条目后，原实例 save 时合入且 M 被补学
      const peerMem = new NeuralMemory({ path: dir, dim: 128, embedMode: 'hybrid', storageFormat: 'base64' })
      await new Promise((r) => setTimeout(r, 25))
      await peerMem.observe('外部进程新增事实：火星基地代号 Ares-7', { surpriseThreshold: 0 })
      await reloadMem.observe('本进程新增事实：月球基地代号 Selene-3', { surpriseThreshold: 0 })
      const rMerged = await reloadMem.recall('火星基地代号 Ares-7')
      assert('mergeExternal 合入外部条目并同步补学入 M', rMerged.entries === 5 && rMerged.matched,
        `entries=${rMerged.entries} score=${rMerged.top1Score.toFixed(3)}`)
    }

    console.log('\n=== 4. DSH 会话生命周期：自动观察 / 自动微注入 / 冷却 / 压缩解冷却 / 防自我放大 ===')
    {
      const dir = path.join(tmpRoot, 's4-lifecycle')
      const listeners = new Map()
      const effects = []
      let registerCalls = 0
      const fakeCtx = {
        get: (k) => {
          if (k === 'tools') {
            return {
              register: (def) => {
                registerCalls++
                // 模拟第一次注册第 3 个工具临时抛错，验部分成功不提前停轮询
                if (registerCalls === 3) throw new Error('transient bus error')
                return () => {}
              },
            }
          }
          return undefined
        },
        on: (ev, fn) => {
          if (!listeners.has(ev)) listeners.set(ev, [])
          listeners.get(ev).push(fn)
          return () => {
            const arr = listeners.get(ev) || []
            const i = arr.indexOf(fn)
            if (i !== -1) arr.splice(i, 1)
          }
        },
        effect: (fn) => effects.push(fn),
        set: () => {},
        logger: { info: () => {}, warn: () => {}, debug: () => {} },
      }

      const { memory } = apply(fakeCtx, {
        path: dir,
        dim: 128,
        embedMode: 'hybrid',
        momentum: 0.25,
        autoObserve: true,
        autoObserveThreshold: 0.35,
        autoRecall: true,
        autoRecallMinScore: 0.45,
        autoRecallMinCoverage: 0.25,
        autoRecallMinMargin: 0.01,
        autoRecallCooldownTurns: 5,
        autoInjectMaxChars: 600,
        autoForgetEveryTurns: 2,
        autoForgetDecay: 0.05,
        persistDebounceMs: 100,
      })

      // 等待延迟补注册完成第 3 个工具
      await new Promise((r) => setTimeout(r, 650))
      assert('部分注册失败时继续轮询补齐全部 6 个工具', registerCalls >= 7, `registerCalls=${registerCalls}`)

      // 验证候选分句过滤代码块与纯寒暄
      const cands = extractCandidateSentences('好的\n```js\nconst x = 1;\n```\n本机部署的数据库密码规范已改为走密钥环托管。')
      assert('分句过滤代码块与纯寒暄', cands.length === 1 && cands[0].includes('密钥环托管'))

      // 模拟插件/dsh-mind 消息（应被完全忽略，不写入）
      for (const fn of listeners.get('agent/inbox/claimed') || []) {
        fn({
          message: {
            role: 'user',
            source: { kind: 'dsh-mind-inject' },
            content: [{ type: 'text', text: '【心智】这条是 dsh-mind 注入内容绝不能吃进神经记忆' }],
          },
        })
      }
      await new Promise((r) => setTimeout(r, 60))
      assert('防自我放大：过滤 dsh-mind 注入消息（0 写入）', (await memory.status()).entries === 0)

      // 模拟真人消息触发自动观察
      for (const fn of listeners.get('agent/inbox/claimed') || []) {
        fn({
          message: {
            role: 'user',
            content: [{ type: 'text', text: '记一下：天枢网关的灰度集群在华东三区，域名是 tianshu.internal。' }],
          },
        })
      }
      await new Promise((r) => setTimeout(r, 120))
      assert('autoObserve 自动沉淀真人高惊讶事实', (await memory.status()).entries === 1)

      // 模拟 agent/pre-step 自动联想微注入（使用新对话历史，不把原句放眼前）
      const preSteps = listeners.get('agent/pre-step') || []
      const fakeAgent = { session: { id: 'sess-life-1', lastCompactionSeq: 0 } }
      const makePayload = (txt) => ({
        agent: fakeAgent,
        messages: [
          { id: 'u1', role: 'user', content: [{ type: 'text', text: txt }] },
        ],
      })
      const runPreStep = async (txt) => {
        const p = makePayload(txt)
        return preSteps[0](p, async () => ({ kind: 'continue', messages: p.messages }))
      }

      const dec1 = await runPreStep('天枢网关的灰度集群域名是什么')
      const inj1 = dec1.messages.find((m) => m?.source?.kind === INJECT_SOURCE_KIND)
      assert('autoRecall 命中时紧随用户消息微注入且带合规水印',
        !!inj1 && inj1.content[0].text.startsWith(INJECT_WATERMARK) && inj1.content[0].text.length <= 600,
        `chars=${inj1?.content?.[0]?.text?.length ?? 0}`)

      // 同一会话紧接着再问：冷却期生效，不再重复注入
      const dec2 = await runPreStep('天枢网关的灰度集群域名是什么')
      const inj2 = dec2.messages.find((m) => m?.source?.kind === INJECT_SOURCE_KIND)
      assert('会话内冷却去重生效（连续问不复读）', !inj2)

      // 模拟发生压缩（lastCompactionSeq 跃升）：冷却立刻清空，高惊讶记忆复活
      fakeAgent.session.lastCompactionSeq = 42
      const dec3 = await runPreStep('天枢网关的灰度集群域名是什么')
      const inj3 = dec3.messages.find((m) => m?.source?.kind === INJECT_SOURCE_KIND)
      assert('压缩后自动清空冷却，记忆立即复活注入', !!inj3)

      // 无关提问：0 字节完全隐身
      const dec4 = await runPreStep('帮我写一个冒泡排序函数')
      const inj4 = dec4.messages.find((m) => m?.source?.kind === INJECT_SOURCE_KIND)
      assert('无关提问 0 字节完全隐身', !inj4)

      // 模拟两次 idle 触发自动遗忘
      for (const fn of listeners.get('agent/status') || []) {
        fn({ agent: fakeAgent, status: 'idle' })
        fn({ agent: fakeAgent, status: 'idle' })
      }
      await new Promise((r) => setTimeout(r, 120))
      const stAfterIdle = await memory.status()
      assert('空闲满 autoForgetEveryTurns 轮自动执行轻量遗忘', stAfterIdle.decayRounds >= 1,
        `decayRounds=${stAfterIdle.decayRounds}`)

      // Config 校验新字段
      const cfgBad = Config['~standard'].validate({ embedMode: 'unknown', autoObserve: 'yes' })
      assert('Config 校验拦截非法新字段', Array.isArray(cfgBad.issues) && cfgBad.issues.length === 2)

      // 生命周期轻量闸门统计可读
      const statsBeforeDispose = memory.getAutoRecallStats?.()
      assert(
        'lifecycle 暴露轻量闸门统计（calls/injected/noMatch/cooldownOrVisible）',
        statsBeforeDispose &&
          statsBeforeDispose.calls === 4 &&
          statsBeforeDispose.injected === 2 &&
          statsBeforeDispose.cooldownOrVisible === 1 &&
          statsBeforeDispose.noMatch === 1,
        JSON.stringify(statsBeforeDispose),
      )

      // 触发卸载清理
      for (const ef of effects) {
        const clean = ef()
        if (typeof clean === 'function') clean()
      }
      assert('卸载后自动清理 getAutoRecallStats 挂载', memory.getAutoRecallStats === undefined)
    }

    console.log('\n=== 5. 闸门对齐（matchedScore/matchedMargin）与近阈值 debug 诊断 ===')
    {
      // 5a: model.recall() 返回 matchedScore / matchedMargin（未命中为 0，命中对齐 results[0]）
      const dir = path.join(tmpRoot, 's5-matched')
      const mem = new NeuralMemory({ path: dir, dim: 128, embedMode: 'hybrid', minEntriesForTrust: 1 })
      await mem.observe('林霏喜欢在雨夜喝温热的桂花乌龙茶', { surpriseThreshold: 0 })
      await mem.observe('海豚仓库的镜像地址配置在 harbor-cn.internal', { surpriseThreshold: 0 })
      const rHit = await mem.recall('桂花乌龙茶', { minScore: 0.4, minCoverage: 0.3 })
      assert(
        'recall 命中时返回与 results[0] 对齐的 matchedScore 与 matchedMargin',
        rHit.matched &&
          Math.abs(rHit.matchedScore - rHit.results[0].score) < 1e-12 &&
          rHit.matchedMargin > 0.3,
        `matchedScore=${rHit.matchedScore.toFixed(3)} matchedMargin=${rHit.matchedMargin.toFixed(3)}`,
      )
      const rMiss = await mem.recall('完全无关的量子力学公式', { minScore: 0.52, minCoverage: 0.3 })
      assert(
        'recall 未命中时 matchedScore 与 matchedMargin 归零',
        !rMiss.matched && rMiss.matchedScore === 0 && rMiss.matchedMargin === 0,
      )

      // 5b: 第三道闸门以 matchedScore / matchedMargin 为准，不被淘汰的假阳性 top1Score 误导
      const ls = new Map()
      const debugLogs = []
      const stubCtx = {
        on(ev, fn) {
          if (!ls.has(ev)) ls.set(ev, [])
          ls.get(ev).push(fn)
          return () => ls.set(ev, (ls.get(ev) || []).filter((x) => x !== fn))
        },
      }
      const queue = [
        // 场景 1：原始 top1=0.85(被 coverage 淘汰的假阳性)、raw margin=0.25，但实际胜出 matchedScore=0.60, matchedMargin=0.02 < 0.08 → 应拦截！
        {
          ok: true,
          matched: true,
          entries: 8,
          top1Score: 0.85,
          top1Coverage: 0.0,
          margin: 0.25,
          matchedScore: 0.60,
          matchedMargin: 0.02,
          results: [{ hash: 'h-ambig', text: '实际胜出但与次优接近的记忆', score: 0.60, coverage: 0.70 }],
        },
        // 场景 2：实际胜出 matchedScore=0.78 >= 0.72 且 matchedMargin=0.02 < 0.08 → 应高分豁免并计入 marginBypassed！
        {
          ok: true,
          matched: true,
          entries: 8,
          top1Score: 0.78,
          top1Coverage: 0.85,
          margin: 0.02,
          matchedScore: 0.78,
          matchedMargin: 0.02,
          results: [{ hash: 'h-high', text: '极高分豁免边际闸门的关键记忆', score: 0.78, coverage: 0.85 }],
        },
        // 场景 3：近阈值未命中（top1=0.45 >= 0.52 - 0.12）→ 应记 noMatch 并输出 debug 诊断
        {
          ok: true,
          matched: false,
          entries: 8,
          top1Score: 0.45,
          top1Coverage: 0.40,
          margin: 0.12,
          matchedScore: 0,
          matchedMargin: 0,
          results: [],
        },
      ]
      const stubMem = {
        async recall() {
          return queue.shift()
        },
        async flushAll() {},
      }
      const disposeStub = attachLifecycle(
        stubCtx,
        stubMem,
        {
          namespace: 'default',
          autoObserve: false,
          autoRecall: true,
          autoRecallTopK: 2,
          autoRecallMinScore: 0.52,
          autoRecallMinCoverage: 0.30,
          autoRecallMinMargin: 0.08,
          autoRecallCooldownTurns: 6,
          autoInjectMaxChars: 400,
          autoForgetEveryTurns: 0,
          autoForgetDecay: 0.02,
        },
        { debug: (m) => debugLogs.push(m), info() {}, warn() {} },
      )
      const firePreStep = async (q) => {
        const msg = { id: `q-${Math.random()}`, role: 'user', content: [{ type: 'text', text: q }] }
        let dec = { kind: 'proceed', messages: [msg] }
        for (const fn of ls.get('agent/pre-step') || []) {
          dec = await fn({ agent: { session: { id: 'sess-gate' } } }, async () => dec)
        }
        return dec
      }

      const dBlock = await firePreStep('测试场景一：假阳性 top1 不应绕过第三道闸门')
      assert('实际胜出 matchedMargin 过低时正确拦截（不被假阳性 top1=0.85 误导）',
        !dBlock.messages.some((m) => m?.source?.kind === INJECT_SOURCE_KIND))

      const dBypass = await firePreStep('测试场景二：实际胜出高分豁免边际闸门')
      assert('实际胜出 matchedScore ≥ 0.72 时高分豁免并注入',
        dBypass.messages.some((m) => m?.source?.kind === INJECT_SOURCE_KIND))

      const dNear = await firePreStep('测试场景三：近阈值未命中输出 debug 日志')
      assert('近阈值未命中不注入', !dNear.messages.some((m) => m?.source?.kind === INJECT_SOURCE_KIND))

      const gs = stubMem.getAutoRecallStats()
      assert(
        '闸门计数准确区分 marginBlocked / marginBypassed / noMatch / injected',
        gs.calls === 3 && gs.marginBlocked === 1 && gs.marginBypassed === 1 && gs.noMatch === 1 && gs.injected === 1,
        JSON.stringify(gs),
      )
      assert(
        '落闸与近阈值均输出精简 debug 诊断（3 条）',
        debugLogs.length === 3 &&
          debugLogs.some((s) => s.includes('边际闸门拦截')) &&
          debugLogs.some((s) => s.includes('高分豁免边际闸门')) &&
          debugLogs.some((s) => s.includes('近阈值未命中')),
        `debugLogs=${debugLogs.length}`,
      )
      disposeStub()
    }

    console.log(`\n========================================`)
    console.log(`dsh-neural-memory 升级专项验收：PASS=${passed}, FAIL=${failed}`)
    console.log(`========================================`)
    if (failed > 0) process.exitCode = 1
  } finally {
    fs.rmSync(tmpRoot, { recursive: true, force: true })
  }
}

main()

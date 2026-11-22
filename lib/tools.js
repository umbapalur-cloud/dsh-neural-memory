/**
 * lib/tools.js —— 六个 nm_* 工具的定义（DSH 依赖只出现"形状"上，不 import 任何宿主包）
 *
 * 契约 2.3 的形状：扁平 parameters + output.schema + output.render。
 * 契约 3.3 的返回字段一个不少，多出来的字段（threshold / namespace / dim 之类）只是给人看的，
 * 因为 output.schema 允许 additionalProperties。
 *
 * 三条纪律：
 *   1. execute 永不抛异常给模型 —— 全部包 try/catch，失败返回 { ok:false, error }。
 *   2. render 永不抛异常 —— 模型最终看到的是它，它挂了等于工具挂了。
 *   3. 不在文件里 console.log —— 需要报告的东西一律放进返回值。
 */

import { compileToolDefinition } from './tool-schema.js'

const textOutput = (value) => [
  { type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) },
]

// ⚠️ 2026-11-17 修：宿主 register() 不编译作者侧 parameters，
//   必须先把扁平 DSL 编译成 { type:'object', properties, required } 再注册，
//   否则模型侧 schema 没有 properties，参数等于不存在。见 lib/tool-schema.js 文件头。
/**
 * 契约 2.3 的形状是 `output: { schema, render }` —— 注意 output 是一个**对象**，
 * 不是把 schema/render 平铺进工具定义。这里用 common 包住再展开，别直接展开 outputShape。
 */
const common = {
  output: {
    schema: { type: 'object', properties: { ok: { type: 'boolean' } }, additionalProperties: true },
    render: (_args, value) => textOutput(value),
  },
}

/** 统一的兜底：把任何异常收成一个可读的失败返回。 */
function fail(e, extra = {}) {
  return { ok: false, error: String(e?.message ?? e), ...extra }
}

const NS_PARAM = {
  type: 'string',
  description: "记忆分区名，默认 default。只允许小写字母、数字、下划线、连字符（1~64 字符）。",
}

/**
 * @param {import('./model.js').NeuralMemory} memory
 * @param {{config?:object}} deps
 */
export function makeToolDefinitions(memory, deps = {}) {
  const config = deps.config ?? {}
  const defaultNS =
    typeof config.namespace === 'string' && config.namespace ? config.namespace : 'default'

  return [
    {
      name: 'nm_observe',
      description:
        '把一段文本交给神经记忆，由「惊讶度」（模型预测误差 ‖M·k−v‖²）决定要不要写进长期权重。' +
        '惊讶度高才写：熟悉的重复内容几乎不写、也不占条目；意外的新信息才会被记住。' +
        '返回 wrote（本次是否真的写入）、surprise（本次预测误差）、entries（记忆条目总数）、' +
        'retained（本次观察是否被留住）、inMemory（这条文本当前是否在记忆里）。' +
        '⚠️ 实现口径：记忆由一张 d×d 关联矩阵 M 加一张**原文条目表**组成——' +
        'M 负责打分排序（容量恒定），条目表保存原文（随条数线性增长，上限由 maxEntries 控制）。' +
        '阅读原文必须经过条目表，所以它不是"不含外部表的纯权重记忆"。',
      parameters: {
        text: {
          type: 'string',
          description: '要观察的文本（一句话事实、结论、偏好都行）。超过 8000 字符会被拒绝。',
          required: true,
        },
        namespace: NS_PARAM,
        surpriseThreshold: {
          type: 'number',
          description: '本次的绝对写入门限，默认取插件配置（0.35）。调高 = 更挑，调低 = 记得更多；传 0 表示强制写入。',
        },
        surpriseThresholdRel: {
          type: 'number',
          description: '本次的相对新颖度门限（1−cos(v̂,v)），默认取插件配置（0.02），用于防止衰减后的旧记忆被误判为新惊喜。',
        },
      },
      ...common,
      async execute(args) {
        try {
          const text = args?.text
          if (typeof text !== 'string' || text.trim().length === 0) {
            return { ok: false, error: 'text 必填且不能为空', wrote: false, surprise: 0, entries: 0, retained: false }
          }
          return await memory.observe(text, {
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
            surpriseThreshold: args?.surpriseThreshold,
            surpriseThresholdRel: args?.surpriseThresholdRel,
          })
        } catch (e) {
          return fail(e, { wrote: false, surprise: 0, entries: 0, retained: false })
        }
      },
    },

    {
      name: 'nm_recall',
      description:
        '从神经记忆里召回：先用查询的键读出 v̂ = M·k，再算 v̂ 与每条记忆的值的余弦相似度，取 topK 条。' +
        '排序完全来自矩阵乘法，不是关键词匹配；但判别"这条是否真的相关"用两道闸门——' +
        '矩阵分数 >= minScore **且** 查询与条目的字符二元组覆盖 >= minCoverage。' +
        '能力边界（实测）：原句/近原句查询 top-5 命中率高；截断、只给唯一实体（如 KX4821）通常可召回；' +
        '**完全不含原句实词的纯语义问法召不回**（例如记的是「岳临川擅长写 Rust」，问「谁最擅长编程语言」）。' +
        '★ 结果为空且 matched=false 时，表示两道闸门都没过（或库是空的），不要据此编造内容。' +
        '容量：可靠性在 100 条内基本无损，300 条级会掉两到三成；条目上限 512。定位是短中期工作记忆。',
      parameters: {
        query: {
          type: 'string',
          description:
            '查询文本。请尽量复用记忆里出现过的词、人名、编号——本工具按字符 n-gram 相似度排序，' +
            '不是语义检索。改写成完全不含原文实词的说法（如把「岳临川擅长写 Rust」问成「谁最擅长编程语言」）召不回。',
          required: true,
        },
        topK: { type: 'number', description: '返回多少条，默认 5，最大 1000。' },
        minScore: {
          type: 'number',
          description: '覆盖矩阵分数门限（默认取插件配置 0.45）。传 -1 可完全关掉过滤，拿到原始候选（调试用）。',
        },
        minCoverage: {
          type: 'number',
          description: '覆盖词面覆盖门限（默认取插件配置 0.25，0~1）。调高 = 更严格，只有查询实词真的出现在条目里才返回。',
        },
        namespace: NS_PARAM,
      },
      ...common,
      async execute(args) {
        try {
          const query = args?.query
          if (typeof query !== 'string' || query.trim().length === 0) {
            return { ok: false, error: 'query 必填且不能为空', results: [], entries: 0 }
          }
          return await memory.recall(query, {
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
            topK: args?.topK,
            minScore: args?.minScore,
            minCoverage: args?.minCoverage,
          })
        } catch (e) {
          return fail(e, { results: [], entries: 0 })
        }
      },
    },

    {
      name: 'nm_forget',
      description:
        '施加一轮遗忘：M ← (1−decay)·M，条目强度同比衰减，低于存活阈值的条目被剔除。' +
        '返回 entriesBefore / entriesAfter / decayed（矩阵是否真的衰减）以及衰减前后的 ‖M‖。' +
        '重复调用可以看到记忆曲线单调下降——这是"遗忘"而不是"删除"。',
      parameters: {
        decay: { type: 'number', description: '本轮衰减系数，默认取插件配置（0.02），取值 0~0.999。' },
        namespace: NS_PARAM,
      },
      ...common,
      async execute(args) {
        try {
          return await memory.forget({
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
            decay: args?.decay,
          })
        } catch (e) {
          return fail(e, { entriesBefore: 0, entriesAfter: 0, decayed: false })
        }
      },
    },

    {
      name: 'nm_status',
      description:
        '查看记忆当前状态：条目数、权重字节数（d×d×8）、平均惊讶度、高惊讶条目数、' +
        '遗忘轮数、‖M‖、当前参数与已存在的 namespace 列表。排查"装了却没记住"时先看它。',
      parameters: {
        namespace: NS_PARAM,
      },
      ...common,
      async execute(args) {
        try {
          return await memory.status({
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
          })
        } catch (e) {
          return fail(e, { entries: 0, weightBytes: 0 })
        }
      },
    },

    {
      name: 'nm_export',
      description:
        '把记忆矩阵 M 与条目表整份导出成一个 JSON 快照（原子写），用于备份、迁移或人工检查。' +
        '返回 path（快照文件绝对路径）、entries（条目数）、bytes（文件字节数）。',
      parameters: {
        namespace: NS_PARAM,
      },
      ...common,
      async execute(args) {
        try {
          return await memory.export({
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
          })
        } catch (e) {
          return fail(e, { entries: 0, bytes: 0 })
        }
      },
    },

    {
      name: 'nm_delete',
      description:
        '显式删除记忆条目（写墓碑，防止被内存缓存写回）。' +
        '按**精确文本**匹配或直接给指纹；默认 dryRun 预览，确认后再传 dryRun:false 真删。' +
        '⚠️ 与 nm_forget 的区别：nm_forget 是"遗忘"（衰减权重、条目按强度自然淘汰，但保留路径）；' +
        'nm_delete 是"抹掉"（写进墓碑表，之后 load/save/merge 都会跳过，不会复活）。' +
        '用于清理测试数据、错误事实、不该留存的私密内容。',
      parameters: {
        namespace: NS_PARAM,
        texts: {
          type: 'array',
          description: '要删除的条目原文列表（精确匹配，宁可漏删不误删）。',
          items: { type: 'string' },
        },
        hashes: {
          type: 'array',
          description: '要删除的条目指纹列表（从 nm_status 或导出文件里拿）。',
          items: { type: 'string' },
        },
        dryRun: {
          type: 'boolean',
          description: '为 true（默认）只预览会删掉哪些，不落盘。确认后传 false 才真删。',
        },
      },
      ...common,
      async execute(args) {
        try {
          const texts = Array.isArray(args?.texts) ? args.texts : undefined
          const hashes = Array.isArray(args?.hashes) ? args.hashes : undefined
          if (!texts && !hashes) {
            return { ok: false, error: '必须给 texts 或 hashes 之一', deleted: 0 }
          }
          return await memory.deleteEntries({
            namespace: typeof args?.namespace === 'string' && args.namespace ? args.namespace : defaultNS,
            texts,
            hashes,
            // 安全默认：没显式传 false 就只预览。
            dryRun: args?.dryRun !== false,
          })
        } catch (e) {
          return fail(e, { deleted: 0 })
        }
      },
    },
  ].map((def) => compileToolDefinition(def))
}

export const TOOL_NAMES = ['nm_observe', 'nm_recall', 'nm_forget', 'nm_status', 'nm_export', 'nm_delete']

/**
 * lib/store.js —— 持久化层（从 core/store.js 复制，本插件的 pluginName = 'dsh-neural-memory'）
 *
 * 相对 core 版本只**加**了一个 writeExport()（导出物单独放 exports/ 子目录，不混进状态目录）；
 * 其余函数的语义一个字符都没改，三个插件共用同一套约定。
 *
 * 为什么这么写：
 *   1. 先写 .tmp 再 rename。直接写目标文件，进程在写一半时被杀会留下半个 JSON，
 *      下次读到 "Unexpected end of JSON input" —— 这种损坏发生一次就很难再信任这块盘。
 *   2. namespace 只允许 [a-z0-9_-]{1,64}。它会被拼进文件路径，
 *      不做白名单就能用 "../../../whatever" 写到任意位置。
 *   3. 读失败一律返回 null，不抛。插件不该因为一个坏文件让整个会话起不来。
 */

import fs from 'node:fs'
import fsp from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

export const NAME_RE = /^[a-z0-9_-]{1,64}$/

/**
 * 临时文件名序号。
 *
 * ⚠️ 2026-11-17 修：原来只用 `pid + Date.now()`，同进程同毫秒的并发写会生成
 * 同名临时文件 → Windows 上 rename EPERM → 内存里有、盘上没有（实测 40 条只剩 37~39）。
 * 加一个进程内自增序号后，同一毫秒内的 N 次写也各有各的文件名。
 * 这是**必要条件不是充分条件**：提交顺序仍可能倒挂，那由 model.js 的写队列解决。
 */
let tmpSeq = 0

function stripBom(str) {
  return typeof str === 'string' && str.charCodeAt(0) === 0xfeff ? str.slice(1) : str
}

async function atomicRename(tmp, dest) {
  const delays = [5, 15, 35]
  for (let i = 0; ; i++) {
    try {
      await fsp.rename(tmp, dest)
      return
    } catch (err) {
      const code = err?.code
      if (i < delays.length && (code === 'EPERM' || code === 'EBUSY' || code === 'EACCES')) {
        await new Promise((r) => setTimeout(r, delays[i]))
        continue
      }
      try {
        await fsp.unlink(tmp)
      } catch {
        /* 清理失败不影响原错误上报 */
      }
      throw err
    }
  }
}

export function dshHome() {
  return process.env.DSH_HOME || path.join(os.homedir(), '.dsh')
}

/**
 * @param {string} pluginName  包名，作为 state 下的子目录
 * @param {string|null} override  配置里指定的目录，优先
 */
export function createStore(pluginName, override) {
  const root = override || path.join(dshHome(), 'state', pluginName)

  function fileFor(key) {
    if (typeof key !== 'string' || !NAME_RE.test(key)) {
      throw new Error(`非法 namespace：${JSON.stringify(key)}（只允许 a-z 0-9 _ -，1~64 字符）`)
    }
    return path.join(root, `${key}.json`)
  }

  async function ensureDir() {
    await fsp.mkdir(root, { recursive: true })
  }

  async function read(key) {
    let file
    try {
      file = fileFor(key)
    } catch {
      return null
    }
    try {
      const raw = await fsp.readFile(file, 'utf8')
      const parsed = JSON.parse(stripBom(raw))
      return parsed && typeof parsed === 'object' ? parsed : null
    } catch {
      return null
    }
  }

  async function write(key, value) {
    const file = fileFor(key) // 非法 key 在这里抛，是调用方传错了，该修调用方
    await ensureDir()
    // ⚠️ 2026-11-17 修：临时文件名必须全局唯一。
    //   原来用 `${pid}.${Date.now()}` —— 同进程同毫秒的并发写会撞名，
    //   实测 Windows 上表现为 rename EPERM，条目静默丢失（40 条只剩 37~39）。
    const tmp = `${file}.${process.pid}.${Date.now()}.${++tmpSeq}.tmp`
    const body = JSON.stringify(value)
    await fsp.writeFile(tmp, body, 'utf8')
    await atomicRename(tmp, file)
    return { bytes: Buffer.byteLength(body, 'utf8'), path: file }
  }

  async function remove(key) {
    try {
      await fsp.unlink(fileFor(key))
      return true
    } catch {
      return false
    }
  }

  /**
   * 读状态文件的修改时间（毫秒）；不存在或读不到返回 null。
   *
   * ⚠️ 2026-11-17 加：多实例/热加载重叠时，用它判断「盘上是不是被别人改过」。
   * 只有 mtime 变了才需要合并，平时一次 stat 的成本可以忽略。
   */
  async function statMtime(key) {
    try {
      const st = await fsp.stat(fileFor(key))
      return st.mtimeMs
    } catch {
      return null
    }
  }

  /**
   * 导出用：把整份 payload 写到 `<root>/exports/<key>.json`。
   *
   * 为什么单独放一个子目录，而不是复用 write()：
   *   state 目录是插件自己的恢复源，它里面的文件会被 load 当成"当前记忆"读回来。
   *   导出物是给人看/给别的工具吃的快照，混进去会出现「导出一次，条目凭空多一倍」。
   *   分开放之后 remove()/list() 看到的仍然只有真正的状态文件。
   */
  async function writeExport(key, value) {
    const file = fileFor(key) // 非法 key 在这里抛，和 write 保持一致
    const dir = path.join(root, 'exports')
    await fsp.mkdir(dir, { recursive: true })
    const dest = path.join(dir, `${key}.json`)
    const tmp = `${dest}.${process.pid}.${Date.now()}.${++tmpSeq}.tmp`
    const body = JSON.stringify(value, null, 2)
    await fsp.writeFile(tmp, body, 'utf8')
    await atomicRename(tmp, dest)
    return { bytes: Buffer.byteLength(body, 'utf8'), path: dest }
  }

  /**
   * 墓碑表文件名（含 `.json` 后缀）。
   *
   * ⚠️ 2026-11-17 修自引入缺陷：墓碑表原来叫 `deleted.json`，和 namespace 文件
   * 放在同一个目录。`list()` 会把所有 `*.json` 当成 namespace，
   * 于是 `nm_status` 的 namespaces 里凭空多出一个 **`deleted`** ——
   * 一个根本不存在的记忆分区（重启后实测复现）。
   * 现在改用下划线前缀 `_deleted.json`：`NAME_RE` 只允许 `[a-z0-9_-]`，
   * 但 `_deleted` 本身是**合法**的，所以光靠白名单挡不住 ——
   * 必须在 `list()` 里显式排除这一份。
   */
  const TOMB_FILE = '_deleted.json'

  async function list() {
    try {
      const names = await fsp.readdir(root)
      return names
        .filter((n) => n.endsWith('.json'))
        .filter((n) => n !== TOMB_FILE)
        // ⚠️ 兜底：旧版墓碑叫 deleted.json，可能还没迁移；
        //   只要文件内容是墓碑形状（version + namespaces），就不当 namespace 报出去。
        .filter((n) => n !== 'deleted.json')
        .map((n) => n.slice(0, -5))
        .filter((n) => NAME_RE.test(n))
        .sort()
    } catch {
      return []
    }
  }

  /**
   * 墓碑表：被**显式删除**的条目指纹，落盘为 `<root>/deleted.json`。
   *
   * ⚠️ 2026-11-17 加（为什么必须有）：
   *   插件进程把整个 namespace 缓存在内存里（states Map）。外部直接改
   *   default.json 删掉假数据是**没用的** —— 下一次 observe 会把整份内存快照
   *   覆盖回磁盘，删掉的条目复活（隔离已复现）。
   *   所以"删除"必须是一个**进程可见的持久事实**，而不是一次文件手术：
   *   写进墓碑表后，任何实例在 load / observe / merge 时都会跳过这些指纹。
   *
   * 形状：`{ version:1, namespaces: { <ns>: [hash, ...] } }`，读失败一律当空表。
   */
  function tombsPath() {
    return path.join(root, TOMB_FILE)
  }

  /**
   * 旧墓碑文件名（`deleted.json`）。读的时候两个都看，写的时候只写新名。
   *
   * ⚠️ 为什么要兼容：上一版已经用旧名字在本机写过一份真实墓碑
   * （7 条测试假数据的指纹）。改名后若只读新名字，那些指纹会被当成"没删过"，
   * 已清理的假数据就可能在下次 observe 时复活。所以**必须向后兼容读**。
   */
  function legacyTombsPath() {
    return path.join(root, 'deleted.json')
  }

  let tombsCache = { legacyKey: null, curKey: null, map: null }

  async function statKey(file) {
    try {
      const st = await fsp.stat(file)
      return `${st.mtimeMs}:${st.size}`
    } catch {
      return 'missing'
    }
  }

  function cloneTombsMap(src) {
    const out = new Map()
    for (const [k, set] of src) out.set(k, new Set(set))
    return out
  }

  async function readTombs() {
    const legacy = legacyTombsPath()
    const cur = tombsPath()
    const [legacyKey, curKey] = await Promise.all([statKey(legacy), statKey(cur)])
    if (tombsCache.map && tombsCache.legacyKey === legacyKey && tombsCache.curKey === curKey) {
      return cloneTombsMap(tombsCache.map)
    }
    const out = new Map()
    // 先读旧名（兼容），再读新名（优先），两边取并集。
    for (const file of [legacy, cur]) {
      try {
        const raw = await fsp.readFile(file, 'utf8')
        const parsed = JSON.parse(stripBom(raw))
        const nsMap = parsed && typeof parsed.namespaces === 'object' ? parsed.namespaces : {}
        for (const [ns, arr] of Object.entries(nsMap)) {
          if (!NAME_RE.test(ns) || !Array.isArray(arr)) continue
          const set = out.get(ns) ?? new Set()
          for (const h of arr) if (typeof h === 'string') set.add(h)
          out.set(ns, set)
        }
      } catch {
        /* 文件不存在或坏了都当没有 */
      }
    }
    tombsCache = { legacyKey, curKey, map: cloneTombsMap(out) }
    return out
  }

  async function writeTombs(nsMap) {
    await ensureDir()
    const namespaces = {}
    for (const [ns, set] of nsMap) namespaces[ns] = [...set]
    const file = tombsPath()
    const tmp = `${file}.${process.pid}.${Date.now()}.${++tmpSeq}.tmp`
    const body = JSON.stringify({ version: 1, namespaces })
    await fsp.writeFile(tmp, body, 'utf8')
    await atomicRename(tmp, file)
    tombsCache = { legacyKey: null, curKey: null, map: null }
    return { bytes: Buffer.byteLength(body, 'utf8'), path: file }
  }

  /** 把一批指纹加入墓碑（幂等）。 */
  async function addTombs(ns, hashes) {
    if (!NAME_RE.test(ns)) throw new Error(`非法 namespace：${JSON.stringify(ns)}`)
    const map = await readTombs()
    const set = map.get(ns) ?? new Set()
    let added = 0
    for (const h of hashes) {
      if (typeof h === 'string' && !set.has(h)) {
        set.add(h)
        added++
      }
    }
    map.set(ns, set)
    await writeTombs(map)
    return { added, total: set.size }
  }

  return { root, fileFor, read, write, writeExport, remove, list, ensureDir, statMtime, readTombs, writeTombs, addTombs }
}

/** 同步版读取，插件 apply 里初始化用得上（此时还不该做 IO 阻塞太久，只用于小配置）。 */
export function readSyncOrNull(file) {
  try {
    return JSON.parse(stripBom(fs.readFileSync(file, 'utf8')))
  } catch {
    return null
  }
}

/**
 * Host 薄壳：真正的实现在 lib/plugin.js。
 *
 * 为什么要单独留这一层：bundle patch 里的 name 指向包名，Loader 走 package.json 的 exports["."]。
 * 这一层保持在 4 行，实现者重构 lib/ 时不会碰到安装配置。
 *
 * name 必须转出：Cordis 的 fiber.name 取自 plugin.name，
 * 不转出的话插件在宿主日志和 Loader 树里会显示成 'root'——功能不受影响，
 * 但排查问题时看不出是哪个插件在报错。这是 2026-11-17 nm-impl 交付时发现的。
 */
export { apply, inject, Config, name } from './lib/plugin.js'

/**
 * 统一启动器 —— 解决一个真实会咬人的环境坑。
 *
 * 问题：若宿主环境带着 ELECTRON_RUN_AS_NODE=1（本机 IDE 会注入），
 *  Electron 会退化成纯 Node 运行，主进程 `require('electron').app` 为 undefined，
 *  报 “Cannot read properties of undefined (reading 'whenReady')”，
 *  且该错误看着像代码 bug，极易误判。
 *
 * 做法：启动子进程前剔除该变量，其他环境变量原样透传。
 *
 * 用法：
 *   node scripts/run.mjs dev       # 开发模式（HMR）
 *   node scripts/run.mjs build     # 仅构建
 *   node scripts/run.mjs smoke     # 冒烟自检（构建后跑一遍并退出）
 *   node scripts/run.mjs start     # 以已构建产物启动
 */

import { spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const root = dirname(dirname(fileURLToPath(import.meta.url)))

const mode = process.argv[2] ?? 'dev'
const passthrough = process.argv.slice(3)

/** 剔除会让 Electron 退化为纯 Node 的变量，其余透传 */
const env = { ...process.env }
delete env['ELECTRON_RUN_AS_NODE']

const electronViteCli = join(root, 'node_modules', 'electron-vite', 'bin', 'electron-vite.js')
/** electron 包导出的是可执行文件路径 */
const electronBin = require('electron')

function run(command, args, label) {
  console.log(`[run] ${label}：${command} ${args.join(' ')}`)
  const child = spawn(command, args, { cwd: root, env, stdio: 'inherit', shell: false })
  child.on('exit', (code) => process.exit(code ?? 0))
  child.on('error', (err) => {
    console.error(`[run] 启动失败：${err.message}`)
    process.exit(1)
  })
  return child
}

switch (mode) {
  case 'dev':
    run(process.execPath, [electronViteCli, 'dev', ...passthrough], '开发模式')
    break

  case 'build':
    run(process.execPath, [electronViteCli, 'build', ...passthrough], '构建')
    break

  case 'start':
    run(electronBin, ['.', ...passthrough], '启动（已构建产物）')
    break

  case 'smoke': {
    // 先构建，构建成功后再跑冒烟；串行执行，失败即停
    const build = spawn(process.execPath, [electronViteCli, 'build'], {
      cwd: root,
      env,
      stdio: 'inherit',
      shell: false
    })
    build.on('exit', (code) => {
      if (code !== 0) {
        console.error('[run] 构建失败，冒烟自检取消')
        process.exit(code ?? 1)
      }
      run(electronBin, ['.', '--smoke', '--smoke-harness', ...passthrough], '冒烟自检')
    })
    break
  }

  default:
    console.error(`[run] 未知模式：${mode}（可用：dev / build / start / smoke）`)
    process.exit(1)
}

import { resolve } from 'node:path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'

/**
 * @openintj/* 是搬过来的主干包，走 paths 映射（与 tsconfig.node.json 一致）。
 *
 * tsconfig 的 paths 只让 **tsc** 认得这些名字；真正打包是 vite 在解析，
 * 两边只配一边的结果很典型：类型检查全绿，一 build 就报 "Failed to resolve import"。
 */
const openintjAlias = {
  '@openintj/core': resolve(__dirname, 'src/vendor/openintj/core/index.ts'),
  '@openintj/shared': resolve(__dirname, 'src/vendor/openintj/shared/index.ts'),
  '@openintj/llm-hunyuan': resolve(__dirname, 'src/vendor/openintj/llm-hunyuan/index.ts')
}

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    resolve: { alias: openintjAlias },
    build: {
      rollupOptions: {
        input: {
          // P-01 主进程
          index: resolve(__dirname, 'src/main/index.ts'),
          // P-03 Agent 主循环宿主（Utility Process）
          'agent-host': resolve(__dirname, 'src/agent-host/index.ts'),
          // P-04 工具宿主（Utility Process）
          'tool-host': resolve(__dirname, 'src/tool-host/index.ts')
        }
      }
    }
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, 'src/preload/index.ts') }
      }
    }
  },
  renderer: {
    root: resolve(__dirname, 'src/renderer'),
    plugins: [react()],
    build: {
      rollupOptions: {
        input: { index: resolve(__dirname, 'src/renderer/index.html') }
      }
    }
  }
})

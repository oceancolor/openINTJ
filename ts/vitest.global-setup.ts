/**
 * Vitest globalSetup —— 自动修复 better-sqlite3 ABI 不匹配。
 *
 * 背景：
 *   - 桌面端在 `pnpm desktop:dev` / `pnpm desktop:package` 前会把 better-sqlite3
 *     重编成 Electron 的 NODE_MODULE_VERSION（Electron 33 = 130）。
 *   - 一旦切换过，pure Node 的 vitest（NODE_MODULE_VERSION = 127）就 dlopen 失败。
 *   - 反之，跑完 vitest 之后 binding 又被 `pnpm rebuild` 切回 127，desktop dev 就崩。
 *
 * 这里在所有 spec 跑之前：
 *   1. 尝试 require('better-sqlite3')
 *   2. 如果失败且原因是 NODE_MODULE_VERSION mismatch，则执行 `pnpm -w rebuild better-sqlite3`
 *      把 binding 切回 Node ABI，然后继续。
 *   3. 任何其它错误（包括 better-sqlite3 没装）一律忽略——下游 spec 自己 skip 即可。
 *
 * 这是单向修复：跑测试一定切回 Node ABI；要继续 desktop dev 时再跑一次 `pnpm desktop:dev`
 * （其 `predev` 钩子会把 binding 切回 Electron ABI）。
 */

import { execSync, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";

const resolveBetterSqlitePkg = (): string | undefined => {
  try {
    return createRequire(import.meta.url).resolve("better-sqlite3/package.json");
  } catch {
    return undefined;
  }
};

/**
 * 在子进程里探测 better-sqlite3 是否能加载。
 *
 * 关键：本进程绝不能 require('better-sqlite3')，否则 Windows 下 .node 文件
 * 会被本进程持有文件句柄，后续重编时 prebuild-install 写不动 (EBUSY/EPERM)。
 */
const tryLoadBetterSqlite = (): { ok: boolean; reason?: string; modulePath?: string } => {
  const modulePath = resolveBetterSqlitePkg();
  if (!modulePath) {
    return { ok: false, reason: "Cannot find module 'better-sqlite3'" };
  }
  const probe =
    "try{const M=require('better-sqlite3');new M(':memory:').close();process.stdout.write('OK');}" +
    "catch(e){process.stdout.write('ERR:'+(e&&e.message?String(e.message):String(e)));}";
  const r = spawnSync(process.execPath, ["-e", probe], {
    cwd: path.dirname(modulePath),
    stdio: ["ignore", "pipe", "ignore"],
  });
  const out = (r.stdout ?? "").toString();
  if (out === "OK") return { ok: true, modulePath };
  if (out.startsWith("ERR:")) return { ok: false, reason: out.slice(4), modulePath };
  return { ok: false, reason: out || "probe failed", modulePath };
};

const isAbiMismatch = (msg: string | undefined): boolean =>
  Boolean(msg && /NODE_MODULE_VERSION/i.test(msg));

const isMissingPackage = (msg: string | undefined): boolean =>
  Boolean(msg && /Cannot find module|MODULE_NOT_FOUND|ERR_MODULE_NOT_FOUND/i.test(msg));

export default async function globalSetup(): Promise<void> {
  // 显式 opt-out（CI 不希望 vitest 内部 npm rebuild）：
  if (process.env["OPENINTJ_SKIP_ABI_FIX"] === "1") return;

  const first = tryLoadBetterSqlite();
  if (first.ok) return;
  if (isMissingPackage(first.reason)) {
    // better-sqlite3 没装：worker 端的 spec 该 skip 就 skip。
    return;
  }
  if (!isAbiMismatch(first.reason)) {
    // 其它失败原因不在此处理；spec 自行报错。
    return;
  }

  if (!first.modulePath) {
    throw new Error(
      "[vitest global-setup] 找不到 better-sqlite3 的安装路径，无法自动 rebuild。",
    );
  }
  const pkgDir = path.dirname(first.modulePath);

  // eslint-disable-next-line no-console
  console.warn(
    `[vitest global-setup] better-sqlite3 ABI mismatch detected → 自动 prebuild-install 切回 Node ABI ...\n` +
      `  package: ${pkgDir}`,
  );
  try {
    // better-sqlite3 的 install 脚本是 `prebuild-install || node-gyp rebuild --release`；
    // 不带额外 env 时跑当前 process 的 Node ABI 预编译。
    const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
    execSync(`${npmCmd} run install`, { stdio: "inherit", env: process.env, cwd: pkgDir });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `[vitest global-setup] 自动修复 better-sqlite3 失败：${msg}\n` +
        '请手动执行 "(cd ' +
        pkgDir +
        ' && npm run install)" 后重试。',
    );
  }

  // 再用子进程探一次（不能在本进程 require，否则 .node 句柄锁住）
  const second = tryLoadBetterSqlite();
  if (!second.ok) {
    throw new Error(
      `[vitest global-setup] 重编后子进程加载仍失败：${second.reason ?? "unknown"}\n` +
        "请手动检查 node_modules/.pnpm/better-sqlite3 状态。",
    );
  }
}

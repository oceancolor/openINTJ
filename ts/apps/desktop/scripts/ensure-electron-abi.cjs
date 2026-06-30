#!/usr/bin/env node
/**
 * predev / prepackage 钩子：确保 better-sqlite3 的 native binding 处于 Electron ABI。
 *
 * - 已经是 Electron ABI（marker 文件存在）→ 直接跳过，省 ~8s。
 * - 否则跑 `electron-builder install-app-deps` 让 @electron/rebuild 把
 *   binding 重编成 Electron 33 的 NODE_MODULE_VERSION。
 *
 * 反向修复（切回 Node ABI 跑 vitest）由根 vitest.global-setup.ts 做。
 */

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const SLEEP_MS = 800;

const sleepSync = (ms) => {
  // Atomics.wait 是同步 sleep，跨平台无依赖
  const sab = new SharedArrayBuffer(4);
  Atomics.wait(new Int32Array(sab), 0, 0, ms);
};

const tryLoadBetterSqliteInNode = () => {
  // 关键：必须在子进程里测，否则本进程持有 .node 文件句柄，后续 prebuild-install 写不动 (EBUSY)。
  const probe =
    "try{const M=require('better-sqlite3');new M(':memory:').close();process.stdout.write('OK');}" +
    "catch(e){process.stdout.write('ERR:'+(e&&e.message?String(e.message):String(e)));}";
  const r = spawnSync(process.execPath, ["-e", probe], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "ignore"],
  });
  const out = (r.stdout || "").toString();
  if (out === "OK") return { ok: true };
  if (out.startsWith("ERR:")) return { ok: false, reason: out.slice(4) };
  return { ok: false, reason: out || "probe subprocess failed" };
};

const getElectronVersion = () => {
  const electronPkg = require.resolve("electron/package.json", { paths: [ROOT] });
  return require(electronPkg).version;
};

const getBetterSqlitePkgDir = () => {
  const bsPkg = require.resolve("better-sqlite3/package.json", { paths: [ROOT] });
  return path.dirname(bsPkg);
};

const getPrebuildInstallBin = (pkgDir) => {
  // pnpm 把 prebuild-install 装到 better-sqlite3 自己的 node_modules 里（透过 .pnpm 软链）
  return require.resolve("prebuild-install/bin.js", { paths: [pkgDir] });
};

const runPrebuildInstall = (binPath, pkgDir, opts) => {
  const args = [
    binPath,
    `--runtime=${opts.runtime}`,
    `--target=${opts.target}`,
    `--arch=${opts.arch}`,
    `--platform=${opts.platform}`,
    "--force",
  ];
  return spawnSync(process.execPath, args, {
    cwd: pkgDir,
    stdio: "inherit",
    env: process.env,
  });
};

const main = () => {
  const loaded = tryLoadBetterSqliteInNode();
  if (!loaded.ok && /NODE_MODULE_VERSION/i.test(loaded.reason || "")) {
    console.log(
      "[ensure-electron-abi] better-sqlite3 已是 Electron ABI（Node require 命中 ABI 错误 = 预期），跳过",
    );
    return;
  }
  if (!loaded.ok) {
    console.log(`[ensure-electron-abi] better-sqlite3 未装或异常，跳过 rebuild：${loaded.reason}`);
    return;
  }

  const electronVer = getElectronVersion();
  const pkgDir = getBetterSqlitePkgDir();
  const arch = process.arch;
  const platform = process.platform;
  const prebuildBin = getPrebuildInstallBin(pkgDir);

  console.log(
    `[ensure-electron-abi] 当前是 Node ABI，重抓 Electron ${electronVer}/${platform}-${arch} 预编译 ...`,
  );

  // 注意 Windows 下 .node 可能被前一个 Electron 进程残留句柄锁住 (EBUSY/EPERM)。
  // 这里 ~1s 间隔重试 3 次，足以让 OS 释放句柄。
  const maxAttempts = 3;
  let lastStatus = -1;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const r = runPrebuildInstall(prebuildBin, pkgDir, {
      runtime: "electron",
      target: electronVer,
      arch,
      platform,
    });
    lastStatus = r.status ?? -1;
    if (lastStatus === 0) break;
    if (attempt < maxAttempts) {
      console.warn(
        `[ensure-electron-abi] prebuild-install 第 ${attempt} 次返回 ${lastStatus}，${SLEEP_MS}ms 后重试 ...`,
      );
      sleepSync(SLEEP_MS);
    }
  }

  if (lastStatus !== 0) {
    console.error(
      `[ensure-electron-abi] prebuild-install 重试 ${maxAttempts} 次仍失败 (status=${lastStatus})。\n` +
        "  常见原因：旧 Electron 进程还持有 better_sqlite3.node 句柄；关掉所有 electron.exe 再试。",
    );
    process.exit(lastStatus);
  }

  console.log(
    "[ensure-electron-abi] 完成。下次 `pnpm test` 时 vitest globalSetup 会自动切回 Node ABI。",
  );
};

main();

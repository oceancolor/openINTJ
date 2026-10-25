/**
 * 凭据保险箱（IC-06）
 *
 * 一条硬规则：**明文密钥不落盘**。
 *
 * Electron 的 `safeStorage` 用 OS 级密钥链（Windows DPAPI / macOS Keychain / Linux libsecret）
 * 派生密钥做对称加密，加密后的密文可以随便存，没有当前用户的登录态解不开。
 *
 * 三个刻意的决定：
 *   ① 保险箱不可用时**拒绝写入**（fail-stop），绝不「降级成明文文件」。
 *      降级是最常见的安全事故起点：用户以为加密了，其实没有。
 *   ② 解密失败与文件不存在**分开报**。混在一起报「未配置」会让用户以为配置丢了，
 *      而实际是换了机器／密钥环被重置 —— 这两种情况的处置完全不同。
 *   ③ 对外只给掩码。明文只在「保存那一瞬间」和「下发给子进程」两处出现，
 *      既不回传渲染层，也不进日志与事件流。
 */

import { safeStorage } from 'electron'
import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { LlmConfig, LlmProtocol } from '../shared/llm'
import { parseProtocol } from '../shared/llm'

/** 落盘的凭据内容（明文形态，仅在内存中出现） */
export interface VaultCredential {
  protocol: LlmProtocol
  baseUrl: string
  model: string
  apiKey: string
  disableThinking: boolean
  /** 保存时间，界面用来显示「什么时候配的」 */
  savedAt: number
}

export type VaultReadResult =
  | { kind: 'ok'; credential: VaultCredential }
  /** 从未保存过 —— 正常的初始状态 */
  | { kind: 'absent' }
  /**
   * 文件在，但解不开。
   * 常见原因：换了用户/机器（DPAPI 与用户绑定）、系统密钥环被重置、文件被改过。
   * 这种必须让用户看到，而不是当「没配过」糊弄过去。
   */
  | { kind: 'unreadable'; reason: string }

const FILE_NAME = 'credentials.enc'

export function credentialFilePath(dataDir: string): string {
  return join(dataDir, FILE_NAME)
}

/**
 * 密钥掩码。
 *
 * 显示前 3 后 4 是为了让用户能认出「这是哪一把 key」，同时不足以还原。
 * 短于 16 字符的一律全掩 —— 那种长度掩了也等于没掩。
 */
export function maskKey(key: string): string {
  if (!key) return ''
  if (key.length < 16) return '•'.repeat(Math.min(key.length, 8))
  return `${key.slice(0, 3)}…${key.slice(-4)}`
}

export function readVault(dataDir: string): VaultReadResult {
  const file = credentialFilePath(dataDir)
  if (!existsSync(file)) return { kind: 'absent' }

  if (!safeStorage.isEncryptionAvailable()) {
    return { kind: 'unreadable', reason: '当前环境没有可用的系统密钥链，已有密文无法解开' }
  }

  let cipher: Buffer
  try {
    cipher = readFileSync(file)
  } catch (err) {
    return { kind: 'unreadable', reason: `读取失败：${(err as Error).message}` }
  }

  let plain: string
  try {
    plain = safeStorage.decryptString(cipher)
  } catch (err) {
    return {
      kind: 'unreadable',
      reason: `解密失败：${(err as Error).message}（多见于换机器、换用户或系统密钥环被重置）`
    }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(plain)
  } catch {
    return { kind: 'unreadable', reason: '解密成功但内容不是合法 JSON，文件可能被改动过' }
  }

  const cred = normalizeCredential(parsed)
  if (!cred) return { kind: 'unreadable', reason: '凭据字段不完整（缺少协议 / 地址 / 模型 / 密钥）' }
  return { kind: 'ok', credential: cred }
}

/**
 * 校验并归一一个来自界面的凭据。
 * 任何一项不合格都返回 null —— 宁可不存，也不存一份跑不起来的配置。
 */
export function normalizeCredential(input: unknown): VaultCredential | null {
  if (typeof input !== 'object' || input === null) return null
  const o = input as Record<string, unknown>

  // 协议判定走 shared 的唯一入口 —— 这里曾经自己写了一份二元三元表达式，
  // 加 Responses 时就会变成「界面能选、落盘却是 anthropic」的隐性故障。
  const protocol: LlmProtocol | null = parseProtocol(String(o['protocol'] ?? ''))
  const baseUrl = String(o['baseUrl'] ?? '').trim().replace(/\/+$/, '')
  const model = String(o['model'] ?? '').trim()
  const apiKey = String(o['apiKey'] ?? '').trim()

  if (!protocol || !baseUrl || !model || !apiKey) return null

  // 只接受 http(s)，避免 file:// 之类把凭据发到奇怪的地方
  if (!/^https?:\/\//i.test(baseUrl)) return null

  const savedAt = typeof o['savedAt'] === 'number' ? o['savedAt'] : Date.now()
  return { protocol, baseUrl, model, apiKey, disableThinking: o['disableThinking'] !== false, savedAt }
}

export type VaultWriteResult = { ok: true; file: string } | { ok: false; reason: string }

export function writeVault(dataDir: string, credential: VaultCredential): VaultWriteResult {
  // fail-stop：没有系统密钥链就不写。绝不退回明文。
  if (!safeStorage.isEncryptionAvailable()) {
    return { ok: false, reason: '系统密钥链不可用，拒绝写入（不会降级成明文存储）' }
  }

  let cipher: Buffer
  try {
    cipher = safeStorage.encryptString(JSON.stringify(credential))
  } catch (err) {
    return { ok: false, reason: `加密失败：${(err as Error).message}` }
  }

  const file = credentialFilePath(dataDir)
  try {
    mkdirSync(dirname(file), { recursive: true })
    writeFileSync(file, cipher)
    // 不是所有平台都支持 chmod，失败不影响正确性（内容本身已是密文）
    try {
      chmodSync(file, 0o600)
    } catch {
      /* Windows 上 ACL 由目录继承决定，忽略 */
    }
    return { ok: true, file }
  } catch (err) {
    return { ok: false, reason: `写入失败：${(err as Error).message}` }
  }
}

export function clearVault(dataDir: string): { ok: boolean; reason?: string } {
  const file = credentialFilePath(dataDir)
  if (!existsSync(file)) return { ok: true }
  try {
    rmSync(file)
    return { ok: true }
  } catch (err) {
    return { ok: false, reason: `删除失败：${(err as Error).message}` }
  }
}

/** 保险箱里的凭据 → LLM 配置（子进程直接用这个形状） */
export function toLlmConfig(cred: VaultCredential): LlmConfig {
  return {
    protocol: cred.protocol,
    baseUrl: cred.baseUrl,
    apiKey: cred.apiKey,
    model: cred.model,
    disableThinking: cred.disableThinking
  }
}

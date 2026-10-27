/**
 * 设置与凭据中心（IC-06）
 *
 * 这个页面的全部意义是：**让用户不必再手敲环境变量**，同时不把密钥摊在明文文件里。
 *
 * 两条界面纪律：
 *   ① 保存之后只显示掩码。界面上永远拿不到明文 —— 想改就重填，想核对就看后 4 位。
 *   ② 必须显示「当前生效来源」。保险箱与环境变量都可能有值，
 *      不写清楚的话，用户改了半天不知道改的是哪一份，这种困惑比不能用更耗人。
 */

import { useCallback, useEffect, useState } from 'react'
import type {
  CredentialInput,
  CredentialStatus,
  ImageInput,
  McpStatusSnapshot
} from '../shared/ipc'
import type { ImageStatus, LlmProtocol } from '../shared/llm'

const PROTOCOLS: { id: LlmProtocol; label: string; hint: string }[] = [
  { id: 'anthropic', label: 'Anthropic Messages', hint: '/v1/messages · x-api-key' },
  { id: 'openai', label: 'OpenAI Chat Completions', hint: '/v1/chat/completions · Bearer' },
  // Responses 与 Chat Completions 同属 OpenAI 家族但线格式不同（output[] / 无 finish_reason），
  // 所以单独列一项，而不是藏在「OpenAI」里面自动嗅探 —— 自动嗅探会把排障变成猜谜。
  { id: 'openai-responses', label: 'OpenAI Responses', hint: '/v1/responses · Bearer · instructions' }
]

const SOURCE_LABEL: Record<string, string> = {
  vault: '保险箱（界面配置）',
  env: '环境变量',
  none: '未配置'
}

const SOURCE_TONE: Record<string, string> = {
  vault: 'ok',
  env: 'warn',
  none: 'danger'
}

export interface SettingsPageProps {
  status: CredentialStatus | null
  /** 保存 / 清除后让上层重新拉一次全局模型状态（顶栏要跟着变） */
  onChanged: () => Promise<void>
}

export default function SettingsPage({ status, onChanged }: SettingsPageProps) {
  const [protocol, setProtocol] = useState<LlmProtocol>('anthropic')
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [disableThinking, setDisableThinking] = useState(true)

  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [msg, setMsg] = useState<{ tone: 'ok' | 'warn' | 'error'; text: string } | null>(null)

  const [mcp, setMcp] = useState<McpStatusSnapshot | null>(null)
  const [mcpBusy, setMcpBusy] = useState(false)

  // 把已生效的配置填进表单，省得用户重打一遍地址和模型名。
  // apiKey 刻意不回填 —— 明文本来就不在渲染层，留空即「不修改」。
  useEffect(() => {
    if (!status) return
    if (status.protocol) setProtocol(status.protocol)
    if (status.baseUrl) setBaseUrl(status.baseUrl)
    if (status.model) setModel(status.model)
    setDisableThinking(status.disableThinking)
  }, [status])

  const vaultHasKey = Boolean(status?.vaultHasCredential)
  /** 当前有可用密钥，但可能来自环境变量 —— 用于给输入框写对提示语 */
  const hasAnyKey = Boolean(status?.maskedKey)

  const loadMcp = useCallback(async () => {
    try {
      setMcp(await window.openintj.mcpStatus())
    } catch (e) {
      setMcp({
        enabled: false,
        loading: false,
        configPath: null,
        configError: (e as Error).message,
        servers: []
      })
    }
  }, [])

  useEffect(() => {
    void loadMcp()
  }, [loadMcp])

  const onMcpReload = useCallback(async () => {
    setMcpBusy(true)
    try {
      setMcp(await window.openintj.mcpReload())
    } catch (e) {
      setMcp({
        enabled: false,
        loading: false,
        configPath: null,
        configError: (e as Error).message,
        servers: []
      })
    } finally {
      setMcpBusy(false)
    }
  }, [])

  const onSave = useCallback(async () => {
    setSaving(true)
    setMsg(null)
    try {
      const input: CredentialInput = {
        protocol,
        baseUrl: baseUrl.trim(),
        model: model.trim(),
        // 留空 = 沿用已有密钥
        apiKey: apiKey.trim() || undefined,
        disableThinking
      }
      const res = await window.openintj.credentialSave(input)
      if (res.ok) {
        setApiKey('') // 明文只在这一瞬间存在，存完立刻从输入框清掉
        setMsg({ tone: 'ok', text: '已加密保存到系统保险箱' })
        await onChanged()
      } else {
        setMsg({ tone: 'error', text: res.reason ?? '保存失败' })
      }
    } catch (e) {
      setMsg({ tone: 'error', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }, [protocol, baseUrl, model, apiKey, disableThinking, onChanged])

  const onTest = useCallback(async () => {
    setTesting(true)
    setMsg(null)
    try {
      const res = await window.openintj.llmTest()
      if (res.ok) {
        setMsg({
          tone: 'ok',
          text: `连通 · ${res.latencyMs}ms · ${res.protocol} · 模型实际回复「${res.sample || '(空)'}」`
        })
      } else {
        setMsg({ tone: 'error', text: `不通：${res.reason ?? '未知原因'}` })
      }
    } catch (e) {
      setMsg({ tone: 'error', text: (e as Error).message })
    } finally {
      setTesting(false)
    }
  }, [])

  const onClear = useCallback(async () => {
    setMsg(null)
    try {
      const res = await window.openintj.credentialClear()
      if (res.ok) {
        setApiKey('')
        setMsg({
          tone: 'warn',
          text: vaultHasKey ? '已清除保险箱凭据' : '保险箱里本来就没有凭据'
        })
        await onChanged()
      } else {
        setMsg({ tone: 'error', text: res.reason ?? '清除失败' })
      }
    } catch (e) {
      setMsg({ tone: 'error', text: (e as Error).message })
    }
  }, [vaultHasKey, onChanged])

  const vaultBlocked = status ? !status.vaultAvailable : false
  const canSave = Boolean(baseUrl.trim() && model.trim()) && !vaultBlocked && !saving

  return (
    <div className="settings">
      <div className="grid-3">
        <div className="card">
          <div className="metric-label">凭据保险箱</div>
          <div className="metric-value">
            <span className={'dot ' + (status?.vaultAvailable ? 'online' : 'error')} />
            {status ? (status.vaultAvailable ? '可用' : '不可用') : '探测中'}
          </div>
          <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
            {status?.vaultBackend ?? '—'} · 策略 {status?.vaultPolicy ?? '—'}
          </div>
        </div>

        <div className="card">
          <div className="metric-label">当前生效来源</div>
          <div className="metric-value">
            <span className={'state-chip ' + (SOURCE_TONE[status?.source ?? 'none'] ?? 'pending')}>
              {SOURCE_LABEL[status?.source ?? 'none']}
            </span>
          </div>
          <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
            {status?.maskedKey ?? '无密钥'}
          </div>
        </div>

        <div className="card">
          <div className="metric-label">生效模型</div>
          <div className="metric-value mono">{status?.model ?? '未配置'}</div>
          <div className="hint" style={{ color: 'var(--text-faint)' }}>
            {status?.protocol ?? '—'}
            {status?.savedAt ? ` · 保存于 ${new Date(status.savedAt).toLocaleString('zh-CN')}` : ''}
          </div>
        </div>
      </div>

      {/* 保险箱不可用：必须说清楚「为什么不能存」，而不是只把按钮变灰 */}
      {vaultBlocked && (
        <div className="notice danger">
          <strong>保险箱不可用 · 拒绝写入</strong>
          <span>
            当前系统没有可用的密钥链（Windows DPAPI / macOS Keychain / Linux libsecret），
            因此这里<b>不允许保存</b>凭据 —— 本产品不会把密钥退回明文文件。
            此时只能继续用环境变量 `OPENINTJ_LLM_*`。
          </span>
        </div>
      )}

      {/* 密文解不开：这是最容易让人以为「配置丢了」的情况，必须点名 */}
      {status?.vaultError && (
        <div className="notice danger">
          <strong>保险箱里的凭据无法解密</strong>
          <span>{status.vaultError}</span>
        </div>
      )}

      <div className="card">
        <div className="card-title">模型接入</div>

        <div className="field">
          <label>协议</label>
          <div className="tier-row">
            {PROTOCOLS.map((p) => (
              <button
                key={p.id}
                className={'pill' + (protocol === p.id ? ' active' : '')}
                onClick={() => setProtocol(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
            {PROTOCOLS.find((p) => p.id === protocol)?.hint}
          </div>
        </div>

        <div className="field">
          <label>Base URL</label>
          <input
            className="mono"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={
              protocol === 'anthropic' ? 'https://api.anthropic.com/v1' : 'https://api.openai.com/v1'
            }
          />
          <div className="hint">兼容网关也可以填，例如 https://tokenhub.tencentmaas.com/v1</div>
        </div>

        <div className="field">
          <label>模型</label>
          <input
            className="mono"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="deepseek/deepseek-flash"
          />
        </div>

        <div className="field">
          <label>API Key</label>
          <input
            className="mono"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={
              vaultHasKey
                ? `保险箱已存 ${status?.maskedKey}，留空则不修改`
                : hasAnyKey
                  ? `当前用环境变量的 ${status?.maskedKey}，填写可覆盖`
                  : '必填'
            }
            autoComplete="off"
            spellCheck={false}
          />
          <div className="hint">
            密钥只在保存的一瞬间经过界面，落盘前由系统密钥链加密；保存后这里再也读不到明文。
          </div>
        </div>

        <div className="field">
          <label>思考开关</label>
          <div className="tier-row">
            <button
              className={'pill' + (disableThinking ? ' active' : '')}
              onClick={() => setDisableThinking(true)}
            >
              请求关闭思考
            </button>
            <button
              className={'pill' + (!disableThinking ? ' active' : '')}
              onClick={() => setDisableThinking(false)}
            >
              保留思考
            </button>
          </div>
          <div className="hint">
            关掉思考能省一个数量级的时间与 token。少数模型不认这个参数（会直接 400），
            代码会自动去掉该参数重试一次。
          </div>
        </div>

        <div className="contract-actions">
          <button className="pill" onClick={() => void onTest()} disabled={testing || status?.source === 'none'}>
            {testing ? '测试中…' : '测试连接'}
          </button>
          <button className="pill primary" onClick={() => void onSave()} disabled={!canSave}>
            {saving ? '保存中…' : '保存到保险箱'}
          </button>
          <button className="pill ghost" onClick={() => void onClear()} disabled={!vaultHasKey}>
            清除凭据
          </button>
        </div>

        {msg && <div className={'result ' + msg.tone}>{msg.text}</div>}

        {status?.envMissing.length ? (
          <div className="hint">
            环境变量还缺：{status.envMissing.join(' / ')}。不配保险箱时，这组变量是另一条可用路径。
          </div>
        ) : null}
      </div>

      {/* ---- 生图服务（独立凭据） ---- */}
      <ImageCredentialCard vaultBlocked={vaultBlocked} />

      {/* ---- MCP（D6） ---- */}
      <div className="card" data-testid="mcp-card">
        <div className="card-title">
          MCP 工具
          <button className="pill ghost" onClick={() => void onMcpReload()} disabled={mcpBusy}>
            {mcpBusy ? '重新连接中…' : '重新加载'}
          </button>
        </div>

        <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
          {mcp?.configPath ?? '—'}
          {mcp?.loading ? ' · 连接中' : ''}
        </div>

        {mcp?.configError && (
          <div className="notice danger">
            <strong>MCP 配置有问题</strong>
            <span>{mcp.configError}</span>
          </div>
        )}

        {mcp && mcp.servers.length === 0 && !mcp.configError ? (
          <div className="hint">
            还没有配置任何 MCP server。在 <span className="mono">mcp.json</span> 里写一段{' '}
            <span className="mono">mcpServers</span> 即可，格式与其他 MCP 客户端一致；改完点「重新加载」。
          </div>
        ) : null}

        {mcp && mcp.servers.length > 0 && (
          <div className="mcp-list">
            {mcp.servers.map((s) => (
              <div className="mcp-row" key={s.name}>
                <span className="mcp-name mono">{s.name}</span>
                <span className={'state-chip ' + MCP_TONE[s.status]}>{MCP_LABEL[s.status]}</span>
                <span className="mcp-meta mono">
                  {s.toolCount} 个工具 · {s.trust === 'readonly' ? '只读，自动放行' : '每次调用需审批'}
                </span>
                {/* 失败原因必须显示：否则用户只看到「工具少了一个」，会以为是产品坏了 */}
                {s.error && <span className="mcp-error mono">{s.error}</span>}
              </div>
            ))}
          </div>
        )}

        <div className="hint">
          MCP 工具与 shell 命令走同一条审批与审计链。server 声明为只读的自动放行，
          其余每次调用都要你点头 —— 外部进程能干什么，静态判断不了。
        </div>
      </div>
    </div>
  )
}

/**
 * 生图服务凭据。
 *
 * 与对话模型分成两张卡，是因为它们**不是同一件事**：端点不同、计费不同、
 * 连失败的原因也不同（生图 402 是「没开通后付费」，与「模型没配」毫无关系）。
 * 合成一张卡的后果是：生图跑不起来时，用户会去改对话模型而改不好。
 */
function ImageCredentialCard({ vaultBlocked }: { vaultBlocked: boolean }): React.ReactElement {
  const [st, setSt] = useState<ImageStatus | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [size, setSize] = useState('1280x720')
  const [apiKey, setApiKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ tone: 'ok' | 'warn' | 'danger'; text: string } | null>(null)

  const refresh = useCallback(async (): Promise<void> => {
    const s = await window.openintj.imageStatus()
    setSt(s)
    if (s.model) setModel(s.model)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const save = useCallback(async (): Promise<void> => {
    setBusy(true)
    setMsg(null)
    try {
      const input: ImageInput = {
        provider: 'hunyuan',
        baseUrl: baseUrl.trim(),
        model: model.trim(),
        size: size.trim(),
        // 留空 = 沿用保险箱里已有的那一把
        apiKey: apiKey.trim() || undefined
      }
      const res = await window.openintj.imageSave(input)
      if (res.ok) {
        setApiKey('') // 与对话凭据同一条规矩：存完立刻从输入框清掉
        setMsg({ tone: 'ok', text: '生图凭据已加密保存' })
        if (res.status) setSt(res.status)
      } else {
        setMsg({ tone: 'danger', text: res.reason ?? '保存失败' })
      }
    } finally {
      setBusy(false)
    }
  }, [baseUrl, model, size, apiKey])

  const clear = useCallback(async (): Promise<void> => {
    setBusy(true)
    try {
      const res = await window.openintj.imageClear()
      setMsg({ tone: res.ok ? 'ok' : 'danger', text: res.ok ? '生图凭据已清除' : (res.reason ?? '清除失败') })
      await refresh()
    } finally {
      setBusy(false)
    }
  }, [refresh])

  const canSave = Boolean(baseUrl.trim()) && !vaultBlocked && !busy

  return (
    <div className="card" data-testid="image-card">
      <div className="card-title">
        生图服务
        <span className={'state-chip ' + (st?.configured ? 'done' : 'pending')}>
          {st?.configured ? '已配置' : '未配置'}
        </span>
      </div>

      <div className="hint">
        生图走独立凭据，不复用对话模型那把 key：端点、计费与失败原因都不同，
        混在一起只会让「生图为什么用不了」查不清。没配时生图任务会落一张
        <b>写着失败原因的占位图</b>（同样要过审批才准落盘），而不是假装成功。
      </div>

      <div className="field">
        <label>Base URL</label>
        <input
          className="mono"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          placeholder="https://tokenhub.tencentmaas.com/v1"
        />
        <div className="hint mono" style={{ color: 'var(--text-faint)' }}>
          实际请求 {baseUrl.trim() ? baseUrl.trim().replace(/\/+$/, '') : '{baseUrl'}
          /wand/hunyuan-image/v3-generation
        </div>
      </div>

      <div className="field">
        <label>模型 / 尺寸</label>
        <div className="tier-row">
          <input
            className="mono"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="hy-image-v3"
          />
          <input
            className="mono"
            value={size}
            onChange={(e) => setSize(e.target.value)}
            placeholder="1280x720"
          />
        </div>
      </div>

      <div className="field">
        <label>API Key</label>
        <input
          className="mono"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={st?.configured ? '已保存，留空则不修改' : '必填'}
          autoComplete="off"
          spellCheck={false}
        />
      </div>

      {(st?.missing.length ?? 0) > 0 && (
        <div className="hint" style={{ color: 'var(--text-faint)' }}>
          还缺：{st!.missing.join(' / ')}
        </div>
      )}

      <div className="contract-actions">
        <button className="pill primary" onClick={() => void save()} disabled={!canSave}>
          {busy ? '保存中…' : '保存生图凭据'}
        </button>
        <button className="pill ghost" onClick={() => void clear()} disabled={!st?.configured || busy}>
          清除
        </button>
      </div>

      {msg && <div className={'result ' + msg.tone}>{msg.text}</div>}

      <div className="hint">
        需要先保存过对话模型凭据 —— 生图片段存在同一份密文里，不单独落一个文件。
        产物一律落在工作区的 <span className="mono">artifacts/</span> 下。
      </div>
    </div>
  )
}

const MCP_LABEL: Record<string, string> = {
  connected: '已连接',
  connecting: '连接中',
  failed: '连接失败',
  disabled: '已禁用'
}

const MCP_TONE: Record<string, string> = {
  connected: 'done',
  connecting: 'pending',
  failed: 'danger',
  disabled: 'pending'
}

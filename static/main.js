// ============================================================
// OpenINTJ Agent IDE — 主入口模块
// ============================================================

// ===== 全局状态 =====
const state = {
  isLoading: false,
  shaderMode: '',
  reasoningTab: 'chain',
  previewTab: 'output',
  lastResponse: null,
  chainSteps: [],
  eventLog: [],
  memories: [],
  auditEvents: [],
  planProgress: 0,
  stats: null,
  llmStatus: null,
  // 图片会话状态
  pendingImage: null,       // {base64, mimeType, fileName, fileSize, dataUrl}
  visionSupported: false,   // LLM 是否支持 Vision
};

// ===== 快捷命令 =====
const QUICK_COMMANDS = [
  { label: '框架介绍', query: '请简单介绍一下 OpenINTJ 框架的核心特点', icon: 'ri-information-line' },
  { label: '架构分析', query: '请分析 OpenINTJ 的四平面分层架构设计', icon: 'ri-layout-grid-line' },
  { label: '着色器原理', query: '请详细解释记忆着色器的工作原理', icon: 'ri-brush-line' },
  { label: '代码示例', query: '帮我编写一个记忆着色器管线处理的代码示例', icon: 'ri-code-s-slash-line' },
  { label: 'Agent Loop', query: '请描述 Agent Loop 的完整闭环流程', icon: 'ri-loop-left-line' },
  { label: '治理机制', query: '请分析框架的安全治理机制', icon: 'ri-shield-check-line' },
];

// ===== 文件树数据 =====
const FILE_TREE = [
  { name: 'main.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  { name: 'agent_loop.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  { name: 'framework_core.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  { name: 'context_engine.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  { name: 'control_plane/', icon: 'ri-folder-line', color: 'text-amber-400', type: 'dir', children: [
    { name: '__init__.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  ]},
  { name: 'execution_plane/', icon: 'ri-folder-line', color: 'text-amber-400', type: 'dir', children: [
    { name: '__init__.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  ]},
  { name: 'memory_plane/', icon: 'ri-folder-line', color: 'text-amber-400', type: 'dir', children: [
    { name: '__init__.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  ]},
  { name: 'governance_plane/', icon: 'ri-folder-line', color: 'text-amber-400', type: 'dir', children: [
    { name: '__init__.py', icon: 'ri-file-code-line', color: 'text-blue-400', type: 'file' },
  ]},
  { name: 'static/', icon: 'ri-folder-line', color: 'text-emerald-400', type: 'dir', children: [
    { name: 'index.html', icon: 'ri-html5-line', color: 'text-orange-400', type: 'file' },
    { name: 'style.css', icon: 'ri-css3-line', color: 'text-cyan-400', type: 'file' },
    { name: 'main.js', icon: 'ri-javascript-line', color: 'text-yellow-400', type: 'file' },
  ]},
];

// ============================================================
// 初始化
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initChat();
  initImageUpload();
  initQuickCommands();
  initWorkspace();
  initReasoningTabs();
  initPreviewTabs();
  initResizers();
  refreshStats();
  refreshLLMStatus();
  renderReasoningPanel();
  renderPreviewPanel();
});

// ============================================================
// 1. 对话面板
// ============================================================
function initChat() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');
  const clearBtn = document.getElementById('btn-clear-chat');
  const shaderSelect = document.getElementById('shader-select');

  if (!input || !sendBtn) return;

  // 输入监听 - 启用/禁用发送按钮（文本或图片任一存在即可发送）
  input.addEventListener('input', () => {
    updateSendButtonState();
  });

  // 发送
  sendBtn.addEventListener('click', () => sendMessage());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 清空
  if (clearBtn) {
    clearBtn.addEventListener('click', clearChat);
  }

  // 着色器模式
  if (shaderSelect) {
    shaderSelect.addEventListener('change', (e) => {
      state.shaderMode = e.target.value;
      updateFooterShader();
    });
  }
}

async function sendMessage(queryOverride) {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');
  const query = queryOverride || (input ? input.value.trim() : '');
  const hasImage = !!state.pendingImage;
  
  // 需要文本或图片至少有一个
  if ((!query && !hasImage) || state.isLoading) return;

  state.isLoading = true;
  if (sendBtn) sendBtn.disabled = true;
  if (input) input.value = '';

  // 保存并清除待发送图片
  const imageToSend = state.pendingImage;
  clearPendingImage();

  // 隐藏快捷命令
  const quickEl = document.getElementById('quick-cmds');
  if (quickEl) quickEl.style.display = 'none';

  // 显示用户消息（含图片）
  appendMessage('user', query || '', imageToSend ? imageToSend.dataUrl : null);

  // 显示加载
  const loadingEl = appendLoading();

  try {
    const body = { query: query || (hasImage ? '[图片]' : '') };
    if (state.shaderMode) body.shader_mode = state.shaderMode;
    
    // 附加图片数据
    if (imageToSend) {
      body.image = imageToSend.base64;
      body.image_mime_type = imageToSend.mimeType;
    }

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: '请求失败' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    state.lastResponse = data;

    // 移除加载
    if (loadingEl) loadingEl.remove();

    // 显示AI响应（打字机效果）
    await appendAgentMessage(data);

    // 更新推理链
    updateChainFromResponse(data);

    // 更新所有面板
    refreshStats();
    refreshMemory();
    refreshAudit();
    renderReasoningPanel();
    renderPreviewPanel();

  } catch (err) {
    if (loadingEl) loadingEl.remove();
    appendMessage('error', err.message);
  } finally {
    state.isLoading = false;
    updateSendButtonState();
    if (input) input.focus();
  }
}

function appendMessage(type, content, imageDataUrl) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const div = document.createElement('div');
  div.className = 'msg';
  div.style.animation = 'msgIn .3s ease';

  if (type === 'user') {
    const textHtml = content ? `<div>${escapeHtml(content)}</div>` : '';
    const imageHtml = imageDataUrl
      ? `<img src="${imageDataUrl}" class="msg-image-thumb" alt="用户图片" onclick="openImageLightbox(this.src)">`
      : '';
    div.innerHTML = `
      <div style="flex:1;display:flex;justify-content:flex-end">
        <div class="msg-bubble msg-bubble-user">
          ${textHtml}
          ${imageHtml}
        </div>
      </div>
      <div class="msg-avatar bg-gradient-to-br from-emerald-500 to-teal-600">
        <i class="ri-user-line text-white text-[10px]"></i>
      </div>`;
  } else if (type === 'error') {
    div.innerHTML = `
      <div class="msg-avatar bg-gradient-to-br from-red-500 to-rose-600">
        <i class="ri-error-warning-line text-white text-[10px]"></i>
      </div>
      <div class="msg-body">
        <div class="msg-bubble msg-bubble-error"><i class="ri-error-warning-line"></i> ${escapeHtml(content)}</div>
      </div>`;
  }

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendLoading() {
  const container = document.getElementById('chat-messages');
  if (!container) return null;

  const div = document.createElement('div');
  div.className = 'msg';
  div.innerHTML = `
    <div class="msg-avatar bg-gradient-to-br from-violet-500 to-indigo-600">
      <i class="ri-brain-line text-white text-[10px]"></i>
    </div>
    <div class="msg-body">
      <div class="msg-bubble msg-bubble-ai">
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span style="font-size:11px;color:var(--text-muted);margin-left:4px">Agent Loop 运行中...</span>
      </div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

async function appendAgentMessage(data) {
  const container = document.getElementById('chat-messages');
  if (!container) return;

  const div = document.createElement('div');
  div.className = 'msg';
  div.style.animation = 'msgIn .3s ease';

  const shaderInfo = getShaderInfo(data.shader_mode);
  const taskLabel = getTaskLabel(data.task_type);

  div.innerHTML = `
    <div class="msg-avatar bg-gradient-to-br from-violet-500 to-indigo-600">
      <i class="ri-brain-line text-white text-[10px]"></i>
    </div>
    <div class="msg-body">
      <span class="msg-name">OpenINTJ</span>
      <div class="msg-bubble msg-bubble-ai">
        <span class="typewriter-target"></span><span class="typing-cursor" style="color:var(--accent-violet)">|</span>
      </div>
      <div class="flex flex-wrap items-center gap-1.5 mt-1.5 px-1" style="opacity:0;transition:opacity .5s" id="msg-meta-latest">
        <span class="msg-tag" style="background:rgba(139,92,246,.1);color:#c4b5fd;border:1px solid rgba(139,92,246,.2)"><i class="${shaderInfo.icon}"></i> ${shaderInfo.text}</span>
        <span class="msg-tag" style="background:rgba(99,102,241,.1);color:#a5b4fc;border:1px solid rgba(99,102,241,.2)"><i class="ri-focus-3-line"></i> ${taskLabel}</span>
        <span class="msg-tag" style="background:rgba(107,114,128,.1);color:#9ca3af;border:1px solid rgba(107,114,128,.2)"><i class="ri-timer-line"></i> ${data.duration_ms.toFixed(1)}ms</span>
        <span class="msg-tag" style="background:rgba(16,185,129,.1);color:#6ee7b7;border:1px solid rgba(16,185,129,.2)"><i class="ri-progress-3-line"></i> ${(data.plan_progress * 100).toFixed(0)}%</span>
      </div>
    </div>`;

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;

  // 打字机效果
  const textEl = div.querySelector('.typewriter-target');
  const cursorEl = div.querySelector('.typing-cursor');
  const metaEl = div.querySelector('#msg-meta-latest');
  const text = data.response;

  for (let i = 0; i < text.length; i += 3) {
    textEl.textContent += text.slice(i, i + 3);
    container.scrollTop = container.scrollHeight;
    await sleep(15);
  }

  if (cursorEl) cursorEl.style.display = 'none';
  if (metaEl) { metaEl.style.opacity = '1'; metaEl.removeAttribute('id'); }
}

function clearChat() {
  const container = document.getElementById('chat-messages');
  if (!container) return;
  // 保留欢迎消息
  const children = Array.from(container.children);
  children.slice(1).forEach(el => el.remove());
  // 显示快捷命令
  const quickEl = document.getElementById('quick-cmds');
  if (quickEl) quickEl.style.display = '';
  // 清除待发送图片
  clearPendingImage();
  // 重置状态
  state.chainSteps = [];
  state.eventLog = [];
  state.lastResponse = null;
  state.planProgress = 0;
  renderReasoningPanel();
  renderPreviewPanel();
}

// ============================================================
// 2. 快捷命令
// ============================================================
function initQuickCommands() {
  const list = document.getElementById('quick-cmds-list');
  if (!list) return;

  list.innerHTML = QUICK_COMMANDS.map(cmd =>
    `<button class="quick-cmd" data-query="${escapeAttr(cmd.query)}">
      <i class="${cmd.icon}"></i> ${cmd.label}
    </button>`
  ).join('');

  list.addEventListener('click', (e) => {
    const btn = e.target.closest('.quick-cmd');
    if (btn) sendMessage(btn.dataset.query);
  });
}

// ============================================================
// 3. 文件工作区
// ============================================================
function initWorkspace() {
  renderFileTree();
  const refreshBtn = document.getElementById('btn-refresh-ws');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', renderFileTree);
  }
}

function renderFileTree(items, container, depth) {
  items = items || FILE_TREE;
  container = container || document.getElementById('workspace-tree');
  depth = depth || 0;
  if (!container) return;

  if (depth === 0) container.innerHTML = '';

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'tree-item';
    div.style.paddingLeft = `${8 + depth * 16}px`;
    div.innerHTML = `
      <i class="tree-icon ${item.icon} ${item.color}"></i>
      <span class="tree-label">${item.name}</span>`;

    div.addEventListener('click', () => {
      container.querySelectorAll('.tree-item').forEach(el => el.classList.remove('active'));
      div.classList.add('active');
    });

    container.appendChild(div);

    if (item.children) {
      renderFileTree(item.children, container, depth + 1);
    }
  });
}

// ============================================================
// 4. 推理/调试面板
// ============================================================
function initReasoningTabs() {
  const tabsEl = document.getElementById('reasoning-tabs');
  if (!tabsEl) return;

  tabsEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.sub-tab');
    if (!btn) return;
    tabsEl.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    state.reasoningTab = btn.dataset.sub;
    renderReasoningPanel();
  });
}

function renderReasoningPanel() {
  const body = document.getElementById('reasoning-body');
  if (!body) return;

  switch (state.reasoningTab) {
    case 'chain': renderChainView(body); break;
    case 'events': renderEventsView(body); break;
    case 'memory': renderMemoryView(body); break;
    case 'govern': renderGovernView(body); break;
  }
}

function renderChainView(container) {
  if (state.chainSteps.length === 0) {
    container.innerHTML = `<div class="empty-state"><i class="ri-route-line"></i><p>发送消息后将显示推理链</p></div>`;
    return;
  }

  container.innerHTML = state.chainSteps.map(step => `
    <div class="chain-step">
      <div class="chain-dot ${step.status}"></div>
      <div class="chain-content">
        <div class="chain-action">${step.action}</div>
        <div class="chain-desc">${step.description}</div>
      </div>
    </div>`).join('');
}

function renderEventsView(container) {
  if (state.eventLog.length === 0) {
    container.innerHTML = `<div class="empty-state"><i class="ri-list-check-2"></i><p>暂无事件记录</p></div>`;
    return;
  }

  container.innerHTML = state.eventLog.map(ev => {
    const typeColor = ev.type.includes('fail') || ev.type.includes('error')
      ? 'background:rgba(239,68,68,.1);color:#fca5a5'
      : ev.type.includes('compacted') || ev.type.includes('loaded')
        ? 'background:rgba(6,182,212,.1);color:#67e8f9'
        : 'background:rgba(139,92,246,.1);color:#c4b5fd';
    return `<div class="event-item">
      <span class="event-time">${ev.time}</span>
      <span class="event-type" style="${typeColor}">${ev.type}</span>
      <span class="event-msg">${ev.message}</span>
    </div>`;
  }).join('');
}

function renderMemoryView(container) {
  if (state.memories.length === 0) {
    container.innerHTML = `<div class="empty-state"><i class="ri-brain-line"></i><p>加载记忆中...</p></div>`;
    refreshMemory();
    return;
  }

  container.innerHTML = state.memories.map(mem => {
    const typeColor = mem.memory_type === 'long_term'
      ? 'background:rgba(139,92,246,.15);color:#c4b5fd'
      : 'background:rgba(6,182,212,.15);color:#67e8f9';
    return `<div class="memory-card">
      <div style="display:flex;align-items:center;gap:6px">
        <span class="mem-type" style="${typeColor}">${mem.memory_type === 'long_term' ? '长期' : '短期'}</span>
        <span style="font-size:9px;color:var(--text-muted)">重要度: ${(mem.importance * 100).toFixed(0)}%</span>
      </div>
      <div class="mem-content">${escapeHtml(mem.content.slice(0, 200))}${mem.content.length > 200 ? '...' : ''}</div>
      <div class="mem-meta">
        <span><i class="ri-price-tag-3-line"></i> ${(mem.task_tags || []).join(', ') || '无标签'}</span>
      </div>
    </div>`;
  }).join('');
}

function renderGovernView(container) {
  if (state.auditEvents.length === 0) {
    container.innerHTML = `<div class="empty-state"><i class="ri-shield-check-line"></i><p>加载治理数据中...</p></div>`;
    refreshAudit();
    return;
  }

  container.innerHTML = state.auditEvents.map(ev => `
    <div class="audit-row">
      <span class="audit-badge ${ev.result}">${ev.result}</span>
      <span style="color:var(--text-secondary);flex:1">${ev.action} → ${ev.target}</span>
      <span style="font-size:9px;color:var(--text-muted)">${ev.risk_level}</span>
    </div>`).join('');
}

// ============================================================
// 5. 输出预览面板
// ============================================================
function initPreviewTabs() {
  const tabsEl = document.getElementById('preview-tabs');
  if (!tabsEl) return;

  tabsEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.sub-tab');
    if (!btn) return;
    tabsEl.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    state.previewTab = btn.dataset.sub;
    renderPreviewPanel();
  });
}

function renderPreviewPanel() {
  const body = document.getElementById('preview-body');
  if (!body) return;

  switch (state.previewTab) {
    case 'output': renderOutputView(body); break;
    case 'plan': renderPlanView(body); break;
    case 'metrics': renderMetricsView(body); break;
  }
}

function renderOutputView(container) {
  if (!state.lastResponse) {
    container.innerHTML = `<div class="empty-state"><i class="ri-eye-line"></i><p>Agent 的输出结果将在此显示</p></div>`;
    return;
  }

  const d = state.lastResponse;
  container.innerHTML = `
    <div class="output-block">
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:6px;display:flex;align-items:center;gap:6px">
        <i class="ri-robot-2-line text-violet-400"></i> Agent 响应
        <span style="margin-left:auto;font-size:10px">${d.trace_id.slice(0, 8)}</span>
      </div>
      <pre>${escapeHtml(d.response)}</pre>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <div class="metric-card" style="flex:1;min-width:100px">
        <div class="metric-value" style="color:var(--accent-violet)">${d.shader_mode}</div>
        <div class="metric-label">着色器模式</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:100px">
        <div class="metric-value" style="color:var(--accent-cyan)">${d.task_type}</div>
        <div class="metric-label">任务类型</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:100px">
        <div class="metric-value" style="color:var(--accent-emerald)">${d.duration_ms.toFixed(1)}<span style="font-size:11px">ms</span></div>
        <div class="metric-label">耗时</div>
      </div>
      <div class="metric-card" style="flex:1;min-width:100px">
        <div class="metric-value" style="color:var(--accent-amber)">${d.events_count}</div>
        <div class="metric-label">事件数</div>
      </div>
    </div>`;
}

function renderPlanView(container) {
  if (!state.lastResponse) {
    container.innerHTML = `<div class="empty-state"><i class="ri-list-ordered-2"></i><p>执行计划将在此显示</p></div>`;
    return;
  }

  const progress = state.lastResponse.plan_progress;
  container.innerHTML = `
    <div style="margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:4px">
        <span style="color:var(--text-secondary)">计划进度</span>
        <span style="color:var(--accent-violet);font-weight:600">${(progress * 100).toFixed(0)}%</span>
      </div>
      <div class="plan-bar"><div class="plan-bar-fill" style="width:${progress * 100}%"></div></div>
    </div>
    <div>
      ${state.chainSteps.map((step, i) => `
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.03)">
          <span style="font-size:10px;color:var(--text-muted);width:20px">#${i + 1}</span>
          <span class="chain-dot ${step.status}" style="margin-top:0"></span>
          <span style="font-size:11px;color:var(--text-primary);flex:1">${step.action}</span>
          <span style="font-size:10px;color:var(--text-muted)">${step.status}</span>
        </div>`).join('')}
    </div>`;
}

function renderMetricsView(container) {
  if (!state.stats) {
    container.innerHTML = `<div class="empty-state"><i class="ri-bar-chart-2-line"></i><p>加载指标中...</p></div>`;
    return;
  }

  const s = state.stats;
  const llm = state.llmStatus || {};
  const llmColor = llm.available ? 'var(--accent-emerald)' : 'var(--accent-amber)';
  const llmText = llm.available ? `${llm.provider || '混元'}` : 'Mock';

  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px">
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-emerald)">${s.state === 'idle' ? '就绪' : s.state}</div>
        <div class="metric-label">Agent 状态</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:${llmColor}">${llmText}</div>
        <div class="metric-label">LLM 模式</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-violet)">${s.total_runs}</div>
        <div class="metric-label">总运行次数</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-cyan)">${s.total_iterations}</div>
        <div class="metric-label">总迭代数</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-amber)">${s.memory?.total_count ?? 0}</div>
        <div class="metric-label">记忆总数</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-indigo)">${(s.context?.budget?.usage_ratio * 100 || 0).toFixed(1)}%</div>
        <div class="metric-label">Token 使用率</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-rose)">${s.governance?.audit?.total_events ?? 0}</div>
        <div class="metric-label">审计事件</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:var(--accent-emerald)">${s.tools?.length ?? 0}</div>
        <div class="metric-label">已注册工具</div>
      </div>
    </div>`;
}

// ============================================================
// 6. 分割线拖拽
// ============================================================
function initResizers() {
  document.querySelectorAll('.resizer').forEach(resizer => {
    let startPos = 0;
    let startSize = 0;
    let target = null;
    const dir = resizer.dataset.dir;

    function onMouseDown(e) {
      e.preventDefault();
      const targetId = resizer.dataset.target;
      target = document.getElementById(targetId);
      if (!target) return;

      resizer.classList.add('active');
      startPos = dir === 'v' ? e.clientX : e.clientY;
      startSize = dir === 'v' ? target.offsetWidth : target.offsetHeight;

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    }

    function onMouseMove(e) {
      if (!target) return;
      const delta = (dir === 'v' ? e.clientX : e.clientY) - startPos;
      const newSize = Math.max(150, startSize + delta);
      if (dir === 'v') {
        target.style.width = newSize + 'px';
      } else {
        target.style.height = newSize + 'px';
      }
    }

    function onMouseUp() {
      resizer.classList.remove('active');
      target = null;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }

    resizer.addEventListener('mousedown', onMouseDown);
  });
}

// ============================================================
// 7. 数据刷新
// ============================================================
async function refreshStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    state.stats = await res.json();
    updateHeader();
    updateFooter();
    if (state.previewTab === 'metrics') renderPreviewPanel();
  } catch (e) { /* 静默 */ }
}

async function refreshLLMStatus() {
  try {
    const res = await fetch('/api/llm/status');
    if (!res.ok) return;
    const data = await res.json();
    state.llmStatus = data;
    state.visionSupported = !!data.vision_supported;
    updateLLMIndicator(data);
    updateVisionIndicator(data);
  } catch (e) {
    updateLLMIndicator({ available: false, mode: 'error', provider: '未知' });
    updateVisionIndicator({ vision_supported: false });
  }
}

function updateLLMIndicator(status) {
  const el = document.getElementById('hdr-llm');
  if (!el) return;
  const dot = el.querySelector('span:first-child');
  const label = el.querySelector('span:last-child');
  if (status.available) {
    dot.className = 'w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse';
    label.textContent = `${status.provider || '混元'} · ${status.model || ''}`;
    label.style.color = '#6ee7b7';
    el.title = `LLM 已连接 | ${status.model} | ${status.mode}`;
  } else if (status.mode === 'unauthorized') {
    dot.className = 'w-1.5 h-1.5 rounded-full bg-red-400';
    label.textContent = 'LLM 未授权';
    label.style.color = '#fca5a5';
    el.title = status.last_error || '当前 HUNYUAN_API_KEY 未通过鉴权或没有模型权限。';
  } else {
    dot.className = 'w-1.5 h-1.5 rounded-full bg-amber-400';
    label.textContent = 'Mock 模式';
    label.style.color = '#fcd34d';
    el.title = status.last_error || 'LLM 未连接，使用模拟响应。请设置 HUNYUAN_API_KEY 环境变量。';
  }
}

function updateVisionIndicator(status) {
  const hdrVision = document.getElementById('hdr-vision');
  const footVision = document.getElementById('foot-vision');
  const attachBtn = document.getElementById('btn-attach-image');
  const visionSupported = !!status.vision_supported;
  
  state.visionSupported = visionSupported;

  if (hdrVision) {
    hdrVision.style.display = 'flex';
    if (visionSupported) {
      hdrVision.className = 'flex items-center gap-1 cursor-pointer';
      hdrVision.querySelector('span').textContent = `Vision · ${status.vision_model || ''}`;
      hdrVision.title = `多模态能力已启用 | ${status.vision_model || 'hunyuan-vision'}`;
    } else {
      hdrVision.className = 'flex items-center gap-1 cursor-pointer unsupported';
      hdrVision.querySelector('span').textContent = 'Vision 不可用';
      hdrVision.title = '当前模型不支持图片理解';
    }
  }

  if (footVision) {
    footVision.style.display = 'inline-flex';
    if (visionSupported) {
      footVision.className = '';
      footVision.innerHTML = `<i class="ri-image-line" style="color:#06b6d4"></i> Vision: ${status.vision_model || '就绪'}`;
      footVision.style.color = '#06b6d4';
    } else {
      footVision.className = 'unsupported';
      footVision.innerHTML = '<i class="ri-image-line"></i> Vision: 不可用';
      footVision.style.color = '';
    }
  }

  // 更新图片上传按钮状态
  if (attachBtn) {
    if (visionSupported) {
      attachBtn.classList.remove('disabled');
      attachBtn.title = '上传图片 (支持拖拽/粘贴)';
    } else {
      attachBtn.classList.add('disabled');
      attachBtn.title = '当前模型不支持图片理解';
    }
  }
}

async function refreshMemory() {
  try {
    const res = await fetch('/api/memory/stats');
    if (!res.ok) return;
    const data = await res.json();
    // 从统计中构建记忆列表
    state.memories = (data.recent_fragments || []).map(f => ({
      content: f.content || '',
      memory_type: f.memory_type || 'long_term',
      importance: f.importance || 0.5,
      task_tags: f.task_tags || [],
    }));
    if (state.memories.length === 0 && data.total_count > 0) {
      // 如果API没返回fragments，构造占位数据
      state.memories = [
        { content: 'OpenINTJ 四平面分层架构：控制、执行、记忆、治理', memory_type: 'long_term', importance: 0.9, task_tags: ['architecture'] },
        { content: '记忆着色器：借鉴3D Shader，动态调整记忆细节', memory_type: 'long_term', importance: 0.95, task_tags: ['shader', 'innovation'] },
        { content: 'Agent Loop：感知→决策→行动→观察→反馈闭环', memory_type: 'long_term', importance: 0.85, task_tags: ['agent_loop'] },
        { content: '上下文引擎：token预算监控 + JIT加载 + 自动压缩', memory_type: 'long_term', importance: 0.8, task_tags: ['context', 'budget'] },
      ];
    }
    if (state.reasoningTab === 'memory') renderReasoningPanel();
  } catch (e) { /* 静默 */ }
}

async function refreshAudit() {
  try {
    const res = await fetch('/api/governance/audit');
    if (!res.ok) return;
    const data = await res.json();
    state.auditEvents = (data.recent_events || []).map(ev => ({
      action: ev.action || '',
      target: ev.target || '',
      result: ev.result || 'allowed',
      risk_level: ev.risk_level || 'low',
    }));
    if (state.reasoningTab === 'govern') renderReasoningPanel();
  } catch (e) { /* 静默 */ }
}

function updateChainFromResponse(data) {
  // 根据响应构建推理链
  const steps = [
    { action: '感知输入', description: `接收查询: "${data.task_type}"`, status: 'done' },
    { action: '任务分类', description: `类型: ${getTaskLabel(data.task_type)}`, status: 'done' },
    { action: '记忆检索', description: `着色器: ${data.shader_mode}`, status: 'done' },
    { action: '上下文构建', description: `事件数: ${data.events_count}`, status: 'done' },
    { action: '计划执行', description: `进度: ${(data.plan_progress * 100).toFixed(0)}%`, status: data.plan_progress >= 1 ? 'done' : 'running' },
    { action: '生成响应', description: `耗时: ${data.duration_ms.toFixed(1)}ms`, status: 'done' },
  ];
  state.chainSteps = steps;

  // 添加事件日志
  const now = new Date();
  const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
  state.eventLog.unshift(
    { time: timeStr, type: 'response', message: `Agent 响应完成 (${data.duration_ms.toFixed(1)}ms)` },
    { time: timeStr, type: 'memory_loaded', message: `记忆着色器: ${data.shader_mode}` },
    { time: timeStr, type: 'plan_executed', message: `计划进度: ${(data.plan_progress * 100).toFixed(0)}%` },
  );
  // 限制日志数量
  if (state.eventLog.length > 50) state.eventLog = state.eventLog.slice(0, 50);
}

// ============================================================
// 8. 头部/底部状态更新
// ============================================================
function updateHeader() {
  if (!state.stats) return;
  const statusEl = document.querySelector('#hdr-status span:last-child');
  const memoryEl = document.querySelector('#hdr-memory span');
  const tokenEl = document.querySelector('#hdr-token span');

  if (statusEl) statusEl.textContent = state.stats.state === 'idle' ? '就绪' : state.stats.state;
  if (memoryEl) memoryEl.textContent = state.stats.memory?.total_count ?? 0;
  if (tokenEl) tokenEl.textContent = ((state.stats.context?.budget?.usage_ratio || 0) * 100).toFixed(1) + '%';
}

function updateFooter() {
  if (!state.stats) return;
  const runsEl = document.getElementById('foot-runs');
  const loopEl = document.getElementById('foot-loop');
  const llmEl = document.getElementById('foot-llm');
  if (runsEl) runsEl.innerHTML = `<i class="ri-play-circle-line"></i> ${state.stats.total_runs} 次运行`;
  if (loopEl) loopEl.textContent = `循环: ${state.stats.state}`;
  if (llmEl && state.llmStatus) {
    const s = state.llmStatus;
    if (s.available) {
      llmEl.innerHTML = `<i class="ri-cloud-line" style="color:#6ee7b7"></i> ${s.provider}: ${s.model}`;
      llmEl.style.color = '#6ee7b7';
    } else if (s.mode === 'unauthorized') {
      llmEl.innerHTML = `<i class="ri-error-warning-line" style="color:#fca5a5"></i> LLM 未授权`;
      llmEl.style.color = '#fca5a5';
      llmEl.title = s.last_error || '当前 HUNYUAN_API_KEY 未通过鉴权或没有模型权限。';
    } else {
      llmEl.innerHTML = `<i class="ri-cloud-off-line" style="color:#fcd34d"></i> Mock 模式`;
      llmEl.style.color = '#fcd34d';
      llmEl.title = s.last_error || '当前未连接真实模型。';
    }
  }
}

function updateFooterShader() {
  const el = document.getElementById('foot-shader');
  if (!el) return;
  const labels = { '': '自适应', 'high_fidelity': '高保真', 'low_fidelity': '低保真', 'hybrid': '混合' };
  el.innerHTML = `<i class="ri-brush-line"></i> ${labels[state.shaderMode] || '自适应'}`;
}

// ============================================================
// 工具函数
// ============================================================
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function escapeAttr(text) {
  return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function getShaderInfo(mode) {
  const map = {
    high_fidelity: { text: '高保真', icon: 'ri-hd-line', cls: 'msg-tag', style: 'background:rgba(139,92,246,.1);color:#c4b5fd;border:1px solid rgba(139,92,246,.2)' },
    low_fidelity: { text: '低保真', icon: 'ri-speed-mini-line', cls: 'msg-tag', style: 'background:rgba(16,185,129,.1);color:#6ee7b7;border:1px solid rgba(16,185,129,.2)' },
    hybrid: { text: '混合', icon: 'ri-contrast-line', cls: 'msg-tag', style: 'background:rgba(99,102,241,.1);color:#a5b4fc;border:1px solid rgba(99,102,241,.2)' },
    adaptive: { text: '自适应', icon: 'ri-equalizer-line', cls: 'msg-tag', style: 'background:rgba(245,158,11,.1);color:#fcd34d;border:1px solid rgba(245,158,11,.2)' },
  };
  const info = map[mode] || { text: mode || '自适应', icon: 'ri-question-line', cls: 'msg-tag', style: 'background:rgba(107,114,128,.1);color:#9ca3af;border:1px solid rgba(107,114,128,.2)' };
  // 将style合并到cls中
  info.cls = `msg-tag`;
  return info;
}

function getTaskLabel(taskType) {
  const map = {
    code_generation: '代码生成',
    technical_writing: '技术文档',
    general_chat: '一般对话',
    quick_response: '快速响应',
    analysis: '分析任务',
    planning: '规划任务',
  };
  return map[taskType] || taskType;
}

// ============================================================
// 9. 图片上传与预览
// ============================================================

const MAX_IMAGE_SIZE = 5 * 1024 * 1024; // 5MB
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];

function initImageUpload() {
  const attachBtn = document.getElementById('btn-attach-image');
  const fileInput = document.getElementById('image-file-input');
  const removeBtn = document.getElementById('btn-remove-image');
  const chatPanel = document.getElementById('panel-chat');
  const inputWrapper = document.getElementById('chat-input-wrapper');
  const chatInput = document.getElementById('chat-input');

  if (!attachBtn || !fileInput) return;

  // 点击上传按钮
  attachBtn.addEventListener('click', () => {
    if (attachBtn.classList.contains('disabled')) return;
    fileInput.click();
  });

  // 文件选择
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
    fileInput.value = ''; // 重置以允许重复选择同一文件
  });

  // 移除图片
  if (removeBtn) {
    removeBtn.addEventListener('click', () => {
      clearPendingImage();
    });
  }

  // 拖拽上传
  if (chatPanel && inputWrapper) {
    chatPanel.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!state.visionSupported) return;
      inputWrapper.classList.add('drag-over');
      chatPanel.classList.add('drag-over-panel');
    });

    chatPanel.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      inputWrapper.classList.remove('drag-over');
      chatPanel.classList.remove('drag-over-panel');
    });

    chatPanel.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      inputWrapper.classList.remove('drag-over');
      chatPanel.classList.remove('drag-over-panel');
      if (!state.visionSupported) return;

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (ALLOWED_IMAGE_TYPES.includes(file.type)) {
          handleImageFile(file);
        }
      }
    });
  }

  // 粘贴上传
  if (chatInput) {
    chatInput.addEventListener('paste', (e) => {
      if (!state.visionSupported) return;
      const items = e.clipboardData?.items;
      if (!items) return;

      for (const item of items) {
        if (item.type.startsWith('image/')) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) handleImageFile(file);
          break;
        }
      }
    });
  }
}

function handleImageFile(file) {
  // 类型校验
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
    showImageError(`不支持的图片格式: ${file.type}，支持 JPG/PNG/GIF/WebP`);
    return;
  }

  // 大小校验
  if (file.size > MAX_IMAGE_SIZE) {
    showImageError('图片过大，请选择 5MB 以内的图片');
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    // 提取 Base64 部分（去掉 data:image/xxx;base64, 前缀）
    const base64 = dataUrl.split(',')[1];

    state.pendingImage = {
      base64: base64,
      mimeType: file.type,
      fileName: file.name,
      fileSize: file.size,
      dataUrl: dataUrl,
    };

    showImagePreview();
    updateSendButtonState();
  };
  reader.readAsDataURL(file);
}

function showImagePreview() {
  const area = document.getElementById('image-preview-area');
  const img = document.getElementById('image-preview-img');
  const nameEl = document.getElementById('image-preview-name');
  const sizeEl = document.getElementById('image-preview-size');
  const attachBtn = document.getElementById('btn-attach-image');

  if (!area || !state.pendingImage) return;

  img.src = state.pendingImage.dataUrl;
  nameEl.textContent = state.pendingImage.fileName;
  sizeEl.textContent = formatFileSize(state.pendingImage.fileSize);
  area.style.display = '';

  if (attachBtn) attachBtn.classList.add('has-image');
}

function clearPendingImage() {
  state.pendingImage = null;

  const area = document.getElementById('image-preview-area');
  const img = document.getElementById('image-preview-img');
  const attachBtn = document.getElementById('btn-attach-image');

  if (area) area.style.display = 'none';
  if (img) img.src = '';
  if (attachBtn) attachBtn.classList.remove('has-image');

  updateSendButtonState();
}

function updateSendButtonState() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');
  if (!sendBtn) return;

  const hasText = input && input.value.trim().length > 0;
  const hasImage = !!state.pendingImage;
  sendBtn.disabled = state.isLoading || (!hasText && !hasImage);
}

function showImageError(message) {
  // 在对话中显示错误提示
  appendMessage('error', message);
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================================
// 10. 图片灯箱
// ============================================================

function openImageLightbox(src) {
  const lightbox = document.createElement('div');
  lightbox.className = 'image-lightbox';
  lightbox.innerHTML = `<img src="${src}" alt="图片预览">`;
  lightbox.addEventListener('click', () => lightbox.remove());
  document.addEventListener('keydown', function handler(e) {
    if (e.key === 'Escape') {
      lightbox.remove();
      document.removeEventListener('keydown', handler);
    }
  });
  document.body.appendChild(lightbox);
}

// 将 openImageLightbox 挂载到 window 以便 onclick 调用
window.openImageLightbox = openImageLightbox;
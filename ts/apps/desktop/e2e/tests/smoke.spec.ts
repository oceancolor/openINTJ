/**
 * Desktop E2E —— smoke 套件
 *
 * 覆盖默认装配（mock LLM + no-persist + 无 Dormant）下的最关键 5 个路径：
 *  1. App 启动 + window + header
 *  2. StatusBar 拿到 status 推送
 *  3. Chat 全链路：发送 → 用户气泡 → mock 回答
 *  4. Trajectory 面板拿到 hook 事件
 *  5. Tab 切换到 Dormant 显示"未启用"提示（默认装配下 dormant 关）
 */
import { expect, test } from "../fixtures.js";

test.describe("desktop smoke (mock provider, no persist)", () => {
  test("boots and renders workbench", async ({ page }) => {
    await expect(page).toHaveTitle(/OpenINTJ v0\.3\.1/);
    await expect(page.getByTestId("title-bar")).toBeVisible();
    await expect(page.getByTestId("title-bar-logo")).toBeVisible();
    await expect(page.getByTestId("title-bar")).not.toContainText("OpenINTJ");
    await expect(page.getByTestId("title-bar").getByRole("button", { name: "File" })).toBeVisible();
    await expect(page.getByTestId("workspace-workbench")).toBeVisible();
    await expect(page.getByLabel("对话模型")).toHaveValue("hunyuan-hy4-preview");
    await expect(page.getByTestId("context-usage-ring")).toBeVisible();
  });

  test("keeps the workspace stage left of the agent rail", async ({ page }) => {
    const stage = page.getByTestId("workspace-workbench");
    const input = page.locator('textarea[placeholder*="说点什么"]');
    const [stageBox, inputBox] = await Promise.all([stage.boundingBox(), input.boundingBox()]);
    expect(stageBox).not.toBeNull();
    expect(inputBox).not.toBeNull();
    expect(inputBox!.x).toBeGreaterThan(stageBox!.x + stageBox!.width - 1);
  });

  test("status bar populates within 8s", async ({ page }) => {
    const statusBar = page.locator("text=/LLM:/").first();
    await expect(statusBar).toBeVisible({ timeout: 10_000 });
    // 工具区一定有内容（builtin tools 至少 1 个）
    await expect(page.locator("text=/工具:/").first()).toBeVisible();
  });

  test("chat round-trip: 你好 → mock greet answer", async ({ page }) => {
    await page.getByLabel("对话模型").selectOption("mock");
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("你好");
    await page.getByRole("button", { name: "发送" }).click();

    // 对话在右侧 Agent 栏；左侧工作台底色相近，必须圈在 chat-panel 内。
    const chat = page.getByTestId("chat-panel");
    await expect(chat.getByText("你好", { exact: false }).first()).toBeVisible({ timeout: 5_000 });
    await expect(chat.getByText(/\[mock\] 收到：你好/)).toBeVisible({
      timeout: 15_000,
    });
  });

  test("shows understanding card and waits for material clarification", async ({ page }) => {
    await page.getByLabel("对话模型").selectOption("mock");
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("部署到生产。");
    await page.getByRole("button", { name: "发送" }).click();

    const card = page.getByTestId("understanding-card");
    await expect(card).toBeVisible({ timeout: 15_000 });
    await expect(card).toContainText("等待补充");
    await expect(card).toContainText(/环境|集群|域名/);
  });

  test("task tree creates a conversation and switches its model", async ({ page }) => {
    await page.getByRole("button", { name: "任务", exact: true }).click();
    await expect(page.getByText("Inbox", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "+ 任务" }).click();
    await expect(page.getByText("新任务", { exact: true })).toBeVisible();
    await page.getByRole("button", { name: "对话", exact: true }).click();

    const selector = page.getByLabel("对话模型");
    await selector.selectOption("mock");
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("对比测试");
    await page.getByRole("button", { name: "发送" }).click();
    const chat = page.getByTestId("chat-panel");
    await expect(chat.getByText(/\[mock\] 收到：对比测试/)).toBeVisible({ timeout: 15_000 });
  });

  test("trajectory tab counts after chat", async ({ page }) => {
    await page.getByLabel("对话模型").selectOption("mock");
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("hello");
    await page.getByRole("button", { name: "发送" }).click();

    // 不直接断 mock 文本（会撞 trajectory JSON）；等对话栏里出现 assistant 气泡即可
    const chat = page.getByTestId("chat-panel");
    const assistantBubble = chat.locator("div.bg-\\[\\#313244\\]");
    await expect(assistantBubble.first()).toBeVisible({ timeout: 15_000 });

    // 推理轨迹 tab 上应显示一个数字小计（trajectory.length > 0）
    const trajectoryTab = page.getByRole("button", { name: /推理轨迹/ });
    await expect(trajectoryTab).toBeVisible();
    await expect(trajectoryTab).toContainText(/\d+/, { timeout: 10_000 });
  });

  test("workspace workbench is present and workspace actions sit in the agent rail", async ({
    page,
  }) => {
    const stage = page.getByTestId("workspace-workbench");
    await expect(stage).toBeVisible();
    // 只读强化：快速打开过滤框在资源管理器上方常驻。
    await expect(page.getByLabel("过滤或快速打开文件")).toBeVisible();
    await expect(page.getByRole("button", { name: "打开工作区", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "关闭工作区" })).toBeVisible();
    await expect(page.getByRole("button", { name: "打开工作区目录" })).toBeVisible();
  });

  test("dormant tab shows '未启用' when not enabled", async ({ page }) => {
    await page.getByRole("button", { name: /Dormant/ }).click();
    await expect(page.getByText(/Dormant 子系统未启用/)).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText(/OPENINTJ_DORMANT=1/)).toBeVisible();

    // 切回推理轨迹仍然可用
    await page.getByRole("button", { name: /推理轨迹/ }).click();
    await expect(page.getByText(/Dormant 子系统未启用/)).not.toBeVisible();
  });

  test("skills tab exposes enable controls when not enabled", async ({ page }) => {
    await page.getByRole("button", { name: /^技能/ }).click();
    const panel = page.getByTestId("skill-panel");
    await expect(panel.getByText(/技能系统未启用/)).toBeVisible({ timeout: 5_000 });
    await expect(panel.getByRole("button", { name: "启用技能系统" })).toBeVisible();
    await expect(panel.getByRole("button", { name: "开启自学习" })).toBeVisible();
  });
});

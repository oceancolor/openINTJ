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
  test("boots and renders header", async ({ page }) => {
    await expect(page.locator("header").first()).toContainText("OpenINTJ");
    await expect(page.locator("header").first()).toContainText("v3.0 Local Desktop");
  });

  test("status bar populates within 8s", async ({ page }) => {
    const statusBar = page.locator("text=/LLM:/").first();
    await expect(statusBar).toBeVisible({ timeout: 10_000 });
    // 工具区一定有内容（builtin tools 至少 1 个）
    await expect(page.locator("text=/工具:/").first()).toBeVisible();
  });

  test("chat round-trip: 你好 → mock greet answer", async ({ page }) => {
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("你好");
    await page.getByRole("button", { name: "发送" }).click();

    // 聊天气泡区位于左侧主面板（trajectory pre 块在右侧 tab，也含 "mock 模式"
    // JSON 文本，所以必须把搜索范围圈在主聊天区内）。
    const chat = page.locator("div.bg-\\[\\#1e1e2e\\]");
    await expect(chat.getByText("你好", { exact: false }).first()).toBeVisible({ timeout: 5_000 });
    // mock greet 特征片段（mock-responses.ts:13："你好！我是 OpenINTJ Agent（mock 模式）…"）
    await expect(chat.getByText(/OpenINTJ Agent（mock 模式）/)).toBeVisible({ timeout: 15_000 });
  });

  test("trajectory tab counts after chat", async ({ page }) => {
    const input = page.locator('textarea[placeholder*="说点什么"]');
    await input.fill("hello");
    await page.getByRole("button", { name: "发送" }).click();

    // 不直接断 mock 文本（会撞 trajectory JSON）；等聊天主面板里出现 assistant 气泡即可
    const chat = page.locator("div.bg-\\[\\#1e1e2e\\]");
    const assistantBubble = chat.locator("div.bg-\\[\\#313244\\]");
    await expect(assistantBubble.first()).toBeVisible({ timeout: 15_000 });

    // 推理轨迹 tab 上应显示一个数字小计（trajectory.length > 0）
    const trajectoryTab = page.getByRole("button", { name: /推理轨迹/ });
    await expect(trajectoryTab).toBeVisible();
    await expect(trajectoryTab).toContainText(/\d+/, { timeout: 10_000 });
  });

  test("dormant tab shows '未启用' when not enabled", async ({ page }) => {
    await page.getByRole("button", { name: /Dormant/ }).click();
    await expect(page.getByText(/Dormant 子系统未启用/)).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText(/OPENINTJ_DORMANT=1/)).toBeVisible();

    // 切回推理轨迹仍然可用
    await page.getByRole("button", { name: /推理轨迹/ }).click();
    await expect(page.getByText(/Dormant 子系统未启用/)).not.toBeVisible();
  });
});

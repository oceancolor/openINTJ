/**
 * Desktop E2E —— Dormant 面板
 *
 * `OPENINTJ_DORMANT=1` 启用 Dormant 子系统（内存模式，no-persist）。
 * 验证：
 *  1. Dormant tab 不再显示"未启用"提示
 *  2. Mine 按钮可点 → 出现扫描摘要文本
 *  3. 默认 pending 过滤下显示"暂无待审批 proposal"占位
 */
import { expect, test } from "../fixtures.js";

test.describe("desktop dormant (OPENINTJ_DORMANT=1, mock provider, no persist)", () => {
  test.use({
    extraEnv: {
      OPENINTJ_DORMANT: "1",
    },
  });

  test("dormant tab shows mine button + pending filter (not '未启用')", async ({ page }) => {
    await page.getByRole("button", { name: /Dormant/ }).click();

    // 不应再显示未启用提示
    await expect(page.getByText(/Dormant 子系统未启用/)).toHaveCount(0, { timeout: 5_000 });

    // Mine 按钮存在
    const mine = page.getByRole("button", { name: /^Mine$/ });
    await expect(mine).toBeVisible();

    // 默认过滤为"待审批"
    await expect(page.getByRole("button", { name: "待审批" })).toBeVisible();
    await expect(page.getByText(/暂无待审批 proposal/)).toBeVisible();
  });

  test("clicking Mine produces a summary line", async ({ page }) => {
    await page.getByRole("button", { name: /Dormant/ }).click();
    await page.getByRole("button", { name: /^Mine$/ }).click();
    // mine 完成后会在面板顶部展示一行摘要："扫描 N 条事件 · 产出 M 个 pattern · K 个新 proposal"
    // 没有 user input 的情况下 N=0；但 pattern 行一定出现
    await expect(page.getByText(/扫描 \d+ 条事件/)).toBeVisible({ timeout: 15_000 });
  });
});

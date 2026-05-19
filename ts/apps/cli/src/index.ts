#!/usr/bin/env node
import { Command } from "commander";
import kleur from "kleur";
import { type LlmProvider, assembleAgent } from "./agent.js";

const program = new Command();
program
  .name("openintj")
  .description("OpenINTJ Agent — TAO/ReAct 本地优先 Agent 框架")
  .version("3.0.0-alpha.0");

program
  .command("chat")
  .alias("c")
  .description("发起一次 Agent 对话")
  .argument("<query...>", "用户查询（可空格分隔）")
  .option("-p, --provider <provider>", "LLM 提供方: auto | hunyuan | ollama | mock", "auto")
  .option("-i, --max-iter <n>", "TAO 宏循环最大轮数", (v: string) => Number.parseInt(v, 10), 1)
  .option("--show-trajectory", "打印完整 trajectory（用于调试）", false)
  .option("--system <prompt>", "自定义系统提示", "")
  .action(async (queryParts: string[], rawOpts: unknown) => {
    const opts = rawOpts as {
      provider: LlmProvider;
      maxIter: number;
      showTrajectory: boolean;
      system: string;
    };
    const query = queryParts.join(" ");
    const agentOpts: Parameters<typeof assembleAgent>[0] = {
      llmProvider: opts.provider,
      maxTaoIterations: opts.maxIter,
    };
    if (opts.system) agentOpts.systemPrompt = opts.system;
    const agent = assembleAgent(agentOpts);

    const status = agent.llm.getStatus();
    process.stderr.write(
      kleur.gray(`[llm] provider=${status.provider} mode=${status.mode} status=${status.status}\n`),
    );

    const t0 = Date.now();
    const result = await agent.run(query);

    process.stdout.write(`${result.finalAnswer}\n`);

    process.stderr.write(
      kleur.gray(
        `[tao] status=${result.status} taoIter=${result.iterations} reactSteps=${result.reactTotalSteps} duration=${Date.now() - t0}ms taskType=${result.taskType} shader=${result.shaderMode}\n`,
      ),
    );

    if (opts.showTrajectory) {
      process.stderr.write(kleur.cyan("=== trajectory ===\n"));
      for (const t of result.trajectory as ReadonlyArray<(typeof result.trajectory)[number]>) {
        const s = t.state;
        const tag =
          s.type === "thought"
            ? "💭 THOUGHT"
            : s.type === "action"
              ? "⚙️  ACTION "
              : s.type === "observation"
                ? "👁  OBSERVE"
                : "✅ FINAL  ";
        const body =
          s.type === "thought"
            ? s.content
            : s.type === "action"
              ? `${s.tool} ← ${JSON.stringify(s.params)}`
              : s.type === "observation"
                ? `success=${s.toolResult.success} ${
                    s.toolResult.success
                      ? JSON.stringify(s.toolResult.output).slice(0, 200)
                      : s.toolResult.error
                  }`
                : s.answer;
        process.stderr.write(kleur.gray(`  ${tag}  `) + body.slice(0, 200) + "\n");
      }
    }
  });

program
  .command("status")
  .description("查看 LLM/Plane 状态")
  .option("-p, --provider <provider>", "LLM 提供方", "auto")
  .action((rawOpts: unknown) => {
    const opts = rawOpts as { provider: LlmProvider };
    const agent = assembleAgent({ llmProvider: opts.provider });
    const llm = agent.llm.getStatus();
    const gov = agent.governance.getStats();
    const mem = agent.memory.getStats();
    const out = {
      llm,
      governance: gov,
      memory: mem,
      tools: agent.execution.toolHub.list().map((t) => t.name),
    };
    process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
  });

program.parseAsync().catch((err) => {
  console.error(kleur.red("[error]"), err);
  process.exit(1);
});

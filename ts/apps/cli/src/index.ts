#!/usr/bin/env node
import type { EmbedProviderId } from "@openintj/model-runtime";
import { loadOpenintjEnv } from "@openintj/shared";
import { Command } from "commander";
import kleur from "kleur";
import { type LlmProvider, assembleAgentAsync } from "./agent.js";
import { parseProductBehaviorCohort } from "./product-behavior-option.js";

// CLI 是开发者直接调用的入口，最常忘记 export $env:HUNYUAN_API_KEY；
// 把 .env / .env.local 自动注入（不覆盖 shell env），让 cp .env.example .env 真正生效。
loadOpenintjEnv({ logPrefix: "[openintj cli env]" });

const program = new Command();
program
  .name("openintj")
  .description("OpenINTJ Agent — TAO/ReAct 本地优先 Agent 框架")
  .version("0.3.0-alpha.0");

program
  .command("chat")
  .alias("c")
  .description("发起一次 Agent 对话")
  .argument("<query...>", "用户查询（可空格分隔）")
  .option(
    "-p, --provider <provider>",
    "LLM 提供方: auto | ollama | hunyuan | kimi | minimax | glm | mock",
    "auto",
  )
  .option(
    "--embedding-provider <provider>",
    "Embedding 提供方: auto | simple | ollama | xenova | mock",
    "auto",
  )
  .option("-i, --max-iter <n>", "TAO 宏循环最大轮数", (v: string) => Number.parseInt(v, 10), 1)
  .option("--show-trajectory", "打印完整 trajectory（用于调试）", false)
  .option(
    "--task-pool",
    "为 planning/analysis 启用 RFC-007 TaskPool（自动启用必需的 classifier）",
    false,
  )
  .option("--system <prompt>", "自定义系统提示", "")
  .option(
    "--product-behavior <cohort>",
    "Product Behavior A/B: treatment | control（未指定则沿用 env/default）",
    parseProductBehaviorCohort,
  )
  .action(async (queryParts: string[], rawOpts: unknown) => {
    const opts = rawOpts as {
      provider: LlmProvider;
      embeddingProvider: EmbedProviderId;
      maxIter: number;
      showTrajectory: boolean;
      system: string;
      productBehavior?: boolean;
      taskPool: boolean;
    };
    const query = queryParts.join(" ");
    const agentOpts: Parameters<typeof assembleAgentAsync>[0] = {
      llmProvider: opts.provider,
      embedProvider: opts.embeddingProvider,
      maxTaoIterations: opts.maxIter,
    };
    if (opts.taskPool) agentOpts.enableTaskPool = true;
    if (opts.system) agentOpts.systemPrompt = opts.system;
    if (opts.productBehavior !== undefined) {
      agentOpts.enableProductBehavior = opts.productBehavior;
    }
    const agent = await assembleAgentAsync(agentOpts);

    const modelRuntime = agent.refreshModelRuntime
      ? await agent.refreshModelRuntime()
      : agent.modelRuntime;
    const status = modelRuntime?.llm ?? agent.llm.getStatus();
    const embedStatus = modelRuntime?.embed;
    process.stderr.write(
      kleur.gray(
        `[llm] provider=${status.provider} mode=${status.mode} status=${status.status}${status.mode === "mock" ? " (visible mock)" : ""}\n`,
      ),
    );
    if (embedStatus) {
      process.stderr.write(
        kleur.gray(
          `[embed] provider=${embedStatus.provider} model=${embedStatus.model} mode=${embedStatus.mode}${embedStatus.fallbackFrom ? ` fallbackFrom=${embedStatus.fallbackFrom}` : ""}\n`,
        ),
      );
    }
    process.stderr.write(
      kleur.gray(
        `[product-behavior] version=${agent.productBehavior.version} cohort=${agent.productBehavior.cohort}\n`,
      ),
    );
    process.stderr.write(
      kleur.gray(
        `[taskpool] requested=${agent.taskPoolActivation.requested} active=${agent.taskPoolActivation.active} classifier=${agent.taskPoolActivation.classifierEnabled} reason=${agent.taskPoolActivation.reason}\n`,
      ),
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
  .option(
    "-p, --provider <provider>",
    "LLM 提供方: auto | ollama | hunyuan | kimi | minimax | glm | mock",
    "auto",
  )
  .option(
    "--embedding-provider <provider>",
    "Embedding 提供方: auto | simple | ollama | xenova | mock",
    "auto",
  )
  .option("--task-pool", "显示 TaskPool 激活状态（自动启用 classifier）", false)
  .option(
    "--product-behavior <cohort>",
    "Product Behavior A/B: treatment | control（未指定则沿用 env/default）",
    parseProductBehaviorCohort,
  )
  .action(async (rawOpts: unknown) => {
    const opts = rawOpts as {
      provider: LlmProvider;
      embeddingProvider: EmbedProviderId;
      productBehavior?: boolean;
      taskPool: boolean;
    };
    const agentOpts: Parameters<typeof assembleAgentAsync>[0] = {
      llmProvider: opts.provider,
      embedProvider: opts.embeddingProvider,
    };
    if (opts.productBehavior !== undefined) {
      agentOpts.enableProductBehavior = opts.productBehavior;
    }
    if (opts.taskPool) agentOpts.enableTaskPool = true;
    const agent = await assembleAgentAsync(agentOpts);
    const modelRuntime = agent.refreshModelRuntime
      ? await agent.refreshModelRuntime()
      : agent.modelRuntime;
    const llm = agent.llm.getStatus();
    const gov = agent.governance.getStats();
    const mem = agent.memory.getStats();
    const out = {
      llm,
      embed: modelRuntime?.embed,
      modelRuntime,
      governance: gov,
      memory: mem,
      productBehavior: agent.productBehavior,
      classifier: agent.classifierStatus,
      taskPool: agent.taskPoolActivation,
      tools: agent.execution.toolHub.list().map((t) => t.name),
    };
    process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
  });

program.parseAsync().catch((err) => {
  console.error(kleur.red("[error]"), err);
  process.exit(1);
});

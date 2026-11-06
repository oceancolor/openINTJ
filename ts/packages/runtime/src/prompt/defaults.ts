import { buildProductBehaviorPrompt } from "@openintj/shared";
import { PROMPT_SECTION_ORDERS, type PromptSection, type PromptSectionProvider } from "./types.js";

export interface DefaultPromptSectionProviders {
  /** Default: RFC-006 product behavior (empty when disabled). */
  productBehavior?: PromptSectionProvider;
  productBehaviorEnabled?: boolean;
  persona?: PromptSectionProvider;
  skills?: PromptSectionProvider;
  /** Default: `ctx.planAdviceBlock`. */
  planAdvice?: PromptSectionProvider;
  /**
   * Optional memory renderer. Production hosts usually let ContextEngine
   * append `[记忆参考]` after the prefix; registering it here is mainly for
   * snapshot tests and hosts without ContextEngine.
   */
  memory?: PromptSectionProvider;
}

/**
 * Default RFC-006 stack. `assembleSystemPromptPrefix` in @openintj/shared
 * remains the pure-function implementation for unit tests; production
 * contextProviders render through this registry.
 */
export const createDefaultPromptSections = (
  providers: DefaultPromptSectionProviders = {},
): PromptSection[] => {
  const sections: PromptSection[] = [
    {
      id: "product-behavior",
      order: PROMPT_SECTION_ORDERS.productBehavior,
      render:
        providers.productBehavior ??
        (() =>
          providers.productBehaviorEnabled === false
            ? undefined
            : buildProductBehaviorPrompt({ enabled: true }) || undefined),
    },
    {
      id: "plan-advice",
      order: PROMPT_SECTION_ORDERS.planAdvice,
      render: providers.planAdvice ?? ((ctx) => ctx.planAdviceBlock),
    },
  ];
  if (providers.persona) {
    sections.push({
      id: "user-persona",
      order: PROMPT_SECTION_ORDERS.userPersona,
      render: providers.persona,
    });
  }
  if (providers.skills) {
    sections.push({
      id: "skills",
      order: PROMPT_SECTION_ORDERS.skills,
      render: providers.skills,
    });
  }
  if (providers.memory) {
    sections.push({
      id: "memory",
      order: PROMPT_SECTION_ORDERS.memory,
      render: providers.memory,
    });
  }
  return sections;
};

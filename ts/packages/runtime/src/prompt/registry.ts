import type { PromptSection, PromptSectionRegistry } from "./types.js";

export const createPromptSectionRegistry = (
  initial: readonly PromptSection[] = [],
): PromptSectionRegistry => {
  const sections: PromptSection[] = [...initial];
  const sorted = (): PromptSection[] =>
    [...sections].sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
  return {
    register(section) {
      const idx = sections.findIndex((item) => item.id === section.id);
      if (idx >= 0) sections[idx] = section;
      else sections.push(section);
    },
    list() {
      return sorted();
    },
    async render(ctx) {
      const blocks: string[] = [];
      for (const section of sorted()) {
        const block = (await section.render(ctx))?.trim();
        if (block) blocks.push(block);
      }
      return blocks.join("\n\n");
    },
  };
};

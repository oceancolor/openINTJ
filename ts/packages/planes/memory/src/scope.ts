/**
 * RFC-021 F04: read-scope eligibility for memory fragments.
 *
 * Scope is carried in fragment taskTags as `workspace:<id>`, `task:<id>`,
 * `conversation:<id>` (written by hosts from verified context, never from
 * model text). Eligibility is fail-closed: a fragment tagged at a level is
 * visible only inside that exact level; untagged fragments are user-level
 * and visible everywhere. This runs BEFORE any ranking truncation — an
 * invisible fragment must not consume a topK slot or bump access stats.
 */
export interface MemoryScope {
  principalId?: string;
  workspaceId?: string;
  taskId?: string;
  conversationId?: string;
}

const scopeTagValue = (fragment: { taskTags: readonly string[] }, prefix: string): string | undefined => {
  for (const tag of fragment.taskTags) {
    if (tag.startsWith(prefix)) return tag.slice(prefix.length);
  }
  return undefined;
};

export const isFragmentVisibleToScope = (
  fragment: { taskTags: readonly string[] },
  scope: MemoryScope | undefined,
): boolean => {
  if (!scope) return true;
  const workspace = scopeTagValue(fragment, "workspace:");
  if (workspace !== undefined && workspace !== scope.workspaceId) return false;
  const task = scopeTagValue(fragment, "task:");
  if (task !== undefined && task !== scope.taskId) return false;
  const conversation = scopeTagValue(fragment, "conversation:");
  if (conversation !== undefined && conversation !== scope.conversationId) return false;
  return true;
};

/** Build a MemoryScope from verified host context tags (`workspace:x` …). */
export const scopeFromContextTags = (tags: readonly string[] | undefined): MemoryScope => {
  const scope: MemoryScope = {};
  for (const tag of tags ?? []) {
    if (tag.startsWith("workspace:")) scope.workspaceId = tag.slice("workspace:".length);
    else if (tag.startsWith("task:")) scope.taskId = tag.slice("task:".length);
    else if (tag.startsWith("conversation:")) {
      scope.conversationId = tag.slice("conversation:".length);
    }
  }
  return scope;
};

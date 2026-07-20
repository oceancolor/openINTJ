/**
 * 面向轻量检索的无词典 tokenizer。
 *
 * 拉丁文本按单词切分；连续 CJK 文本生成字符二元组，使“所在的城市”和“在哪个城市”
 * 即使没有空格也能共享“城市”。这不是语言学分词器，但比把整句中文视为单个 token
 * 更适合本地、零依赖的 keyword/BM25 fallback。
 */
export const tokenizeLexical = (text: string): string[] => {
  const normalized = text.toLowerCase().normalize("NFKC");
  const segments = normalized.match(/[\p{Script=Han}]+|[\p{L}\p{N}]+/gu) ?? [];
  const tokens: string[] = [];
  for (const segment of segments) {
    if (/^[\p{Script=Han}]+$/u.test(segment)) {
      if (segment.length === 1) tokens.push(segment);
      else {
        for (let i = 0; i < segment.length - 1; i++) tokens.push(segment.slice(i, i + 2));
      }
    } else {
      tokens.push(segment);
    }
  }
  return tokens;
};

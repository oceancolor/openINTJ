/**
 * 单调 ULID 生成器（无外部依赖）
 *
 * 格式：48 位时间戳（Crockford Base32，10 字符）+ 80 位随机（16 字符）= 26 字符
 * 同毫秒内递增，保证字典序 == 时间序，满足事件流按序落盘与位点定位需求。
 */

const ENCODING = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
const ENCODING_LEN = ENCODING.length
const TIME_LEN = 10
const RANDOM_LEN = 16

let lastTime = -1
let lastRandom: number[] = []

function encodeTime(now: number): string {
  let str = ''
  for (let i = TIME_LEN - 1; i >= 0; i--) {
    const mod = now % ENCODING_LEN
    str = ENCODING[mod] + str
    now = (now - mod) / ENCODING_LEN
  }
  return str
}

function randomChars(): number[] {
  const out: number[] = []
  for (let i = 0; i < RANDOM_LEN; i++) {
    out.push(Math.floor(Math.random() * ENCODING_LEN))
  }
  return out
}

function incrementRandom(chars: number[]): number[] {
  const next = chars.slice()
  for (let i = next.length - 1; i >= 0; i--) {
    if (next[i]! < ENCODING_LEN - 1) {
      next[i] = next[i]! + 1
      return next
    }
    next[i] = 0
  }
  return randomChars()
}

export function ulid(now: number = Date.now()): string {
  if (now === lastTime) {
    lastRandom = incrementRandom(lastRandom)
  } else {
    lastTime = now
    lastRandom = randomChars()
  }
  return encodeTime(now) + lastRandom.map((n) => ENCODING[n]).join('')
}

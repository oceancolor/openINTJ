import type { OpenIntjApi } from '../preload/index'

declare global {
  interface Window {
    openintj: OpenIntjApi
  }
}

export {}

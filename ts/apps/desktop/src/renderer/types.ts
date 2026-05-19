import type { OpenintjAPI } from "../preload/index.js";

declare global {
  interface Window {
    openintj: OpenintjAPI;
  }
}

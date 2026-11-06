/// <reference types="vite/client" />

/** Injected by electron.vite.config from package.json "version". */
declare const __OPENINTJ_APP_VERSION__: string;

declare module "*.png" {
  const src: string;
  export default src;
}

declare module "*?worker" {
  const workerConstructor: {
    new (): Worker;
  };
  export default workerConstructor;
}

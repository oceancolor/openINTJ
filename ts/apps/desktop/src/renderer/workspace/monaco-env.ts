import * as monaco from "monaco-editor";
import "monaco-editor/min/vs/editor/editor.main.css";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

let configured = false;

export const ensureMonaco = (): typeof monaco => {
  if (!configured) {
    self.MonacoEnvironment = {
      getWorker: (_workerId: string, label: string): Worker => {
        if (label === "json") return new jsonWorker();
        if (label === "css" || label === "scss" || label === "less") return new cssWorker();
        if (label === "html" || label === "handlebars" || label === "razor") {
          return new htmlWorker();
        }
        if (label === "typescript" || label === "javascript") return new tsWorker();
        return new editorWorker();
      },
    };
    monaco.editor.defineTheme("openintj-dark", {
      base: "vs-dark",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#1e1e2e",
        "editor.foreground": "#cdd6f4",
        "editorLineNumber.foreground": "#6c7086",
        "editorCursor.foreground": "#cba6f7",
        "editor.selectionBackground": "#45475a",
        "editorWidget.background": "#181825",
      },
    });
    configured = true;
  }
  return monaco;
};

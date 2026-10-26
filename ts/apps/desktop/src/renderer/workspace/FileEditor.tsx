import type { editor as MonacoNs } from "monaco-editor";
import React from "react";
import { monacoLanguageForPath } from "../../shared/workspace-view.js";
import { ensureMonaco } from "./monaco-env.js";

export const FileEditor: React.FC<{
  path: string;
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
}> = ({ path, value, onChange, readOnly = false }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const editorRef = React.useRef<MonacoNs.IStandaloneCodeEditor>();
  const onChangeRef = React.useRef(onChange);
  const initialValueRef = React.useRef(value);
  onChangeRef.current = onChange;

  React.useEffect(() => {
    const host = containerRef.current;
    if (!host) return;
    const monaco = ensureMonaco();
    const editor = monaco.editor.create(host, {
      value: initialValueRef.current,
      language: monacoLanguageForPath(path),
      theme: "openintj-dark",
      automaticLayout: true,
      minimap: { enabled: false },
      fontSize: 13,
      lineHeight: 20,
      readOnly,
      wordWrap: "on",
      scrollBeyondLastLine: false,
      renderLineHighlight: "line",
      padding: { top: 8 },
    });
    editorRef.current = editor;
    const sub = editor.onDidChangeModelContent(() => {
      onChangeRef.current(editor.getValue());
    });
    return () => {
      sub.dispose();
      editor.dispose();
      editorRef.current = undefined;
    };
  }, [path, readOnly]);

  React.useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    if (editor.getValue() !== value) {
      const position = editor.getPosition();
      editor.setValue(value);
      if (position) editor.setPosition(position);
    }
  }, [value]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full min-h-0"
      data-testid="workspace-monaco"
    />
  );
};

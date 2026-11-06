import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App.js";
import "./styles.css";

// Keep the document title in sync with the BrowserWindow title set in main
// (`OpenINTJ v<version>`), which the static HTML title would otherwise lose.
document.title = `OpenINTJ v${__OPENINTJ_APP_VERSION__}`;

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

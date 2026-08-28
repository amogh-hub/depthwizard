import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { bootstrapStandaloneRuntime } from "./standalone";
import "./styles/tokens.css";
import "./styles/app.css";
import "./styles/validation.css";
import "./styles/workstation.css";
import "./styles/toolrail-compact.css";
import "./styles/workstation_final.css";
import "./styles/brand.css";

const root = createRoot(document.getElementById("root")!);

function renderStartupFailure(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  root.render(
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "48px",
        background: "#f7f8fa",
        color: "#172033",
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      }}
    >
      <section
        style={{
          width: "min(640px, 100%)",
          padding: "28px",
          border: "1px solid #d9dee8",
          borderRadius: "12px",
          background: "#ffffff",
          boxShadow: "0 8px 28px rgba(24, 39, 75, 0.08)",
        }}
      >
        <div style={{ fontSize: "12px", fontWeight: 700, letterSpacing: "0.08em", color: "#53627a" }}>
          DEPTHWIZARD LOCAL CORE
        </div>
        <h1 style={{ margin: "10px 0 8px", fontSize: "22px" }}>Scientific runtime did not start</h1>
        <p style={{ margin: 0, lineHeight: 1.6, color: "#53627a" }}>{message}</p>
        <p style={{ margin: "16px 0 0", lineHeight: 1.6, color: "#53627a" }}>
          No project processing has started. Close and reopen DepthWizard after checking the packaged
          runtime installation.
        </p>
      </section>
    </main>,
  );
}

async function start(): Promise<void> {
  try {
    await bootstrapStandaloneRuntime();
    root.render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
  } catch (error) {
    renderStartupFailure(error);
  }
}

void start();

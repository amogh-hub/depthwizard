import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    // DepthWizard ships as a Tauri desktop application on modern system WebViews.
    // Targeting legacy Safari 13 forces esbuild 0.28+ to attempt destructuring
    // downlevel transforms it intentionally no longer performs. ES2022 matches
    // our desktop runtime while preserving a deterministic production build.
    target: "es2022",
    minify: process.env.TAURI_DEBUG ? false : "esbuild",
    sourcemap: !!process.env.TAURI_DEBUG,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three")) return "vendor-three";
          if (id.includes("node_modules/react") || id.includes("node_modules/scheduler")) {
            return "vendor-react";
          }
          if (id.includes("node_modules/@tauri-apps")) return "vendor-tauri";
          return undefined;
        },
      },
    },
  },
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

describe("RT5 final workstation closure", () => {
  it("keeps reopened projects usable without fabricating missing optical imagery", () => {
    const app = source("../App.tsx");
    expect(app).toContain("sourceAvailable");
    expect(app).toContain('dtype: "source unavailable"');
    expect(app).toContain('if (view === "Optical") return Boolean(metadata) && sourceAvailable');
    expect(app).toContain("persistedSurface");
  });

  it("distinguishes scientific completion from display-layer availability and provides recovery", () => {
    const app = source("../App.tsx");
    expect(app).toContain("previewError");
    expect(app).toContain("Scientific layer unavailable");
    expect(app).toContain("Retry layer");
    expect(app).toContain("comparisonError");
    expect(app).toContain("Retry comparison");
    expect(app).toContain("terrainOverlayError");
    expect(app).toContain("Retry overlay");
    expect(app).toContain("rasterSurfaceReady");
  });

  it("binds a 3D analytical-layer claim to the renderer-validated overlay frame", () => {
    const app = source("../App.tsx");
    const terrain = source("./TerrainViewport.tsx");
    expect(app).toContain("terrainOverlayRenderState");
    expect(app).toContain("onOverlayState={setTerrainOverlayRenderState}");
    expect(app).toContain("analytical overlay frame-validated");
    expect(app).not.toContain('"analytical overlay active"');
    expect(terrain).toContain("onOverlayState");
    expect(terrain).toContain("Analytical overlay rendered and frame-validated");
    expect(terrain).toContain("source texture restored");
  });

  it("preserves an already-active 3D workspace while keeping raster-analysis fallbacks on the DSM", () => {
    const app = source("../App.tsx");
    expect(app).toContain('(activeTool === "Measure" || activeTool === "Profiles") && geometryReady && activeView !== "3D Terrain"');
    expect(app).toContain('activeTool === "Structures" && calibrationReady && activeView !== "3D Terrain"');
    expect(app).not.toContain('(activeTool === "Measure" || activeTool === "Profiles") && geometryReady && !meshArtifactReady');
  });

  it("enables evidence-native 3D point selection only for supported analyst tools", () => {
    const app = source("../App.tsx");
    expect(app).toContain("const terrainToolInteractive = projectAnalystInteractive");
    expect(app).toContain('activeTool === "Project"');
    expect(app).toContain('activeTool === "Measure"');
    expect(app).toContain('activeTool === "Profiles"');
    expect(app).toContain('(activeTool === "Structures" && calibrationReady)');
    expect(app).toContain('onSelectPoint={terrainToolInteractive ? analyzeRasterPoint : undefined}');
    expect(app).toContain('result.add("Measure")');
    expect(app).toContain('result.add("Profiles")');
    expect(app).toContain('result.add("Structures")');
  });

  it("keeps durable recent-project recovery fail-closed", () => {
    const app = source("../App.tsx");
    expect(app).toContain("forgetProject(selectedDir)");
    expect(app).toContain("Original source imagery is required to resume processing");
  });
});

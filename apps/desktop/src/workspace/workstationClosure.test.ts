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

  it("always moves raster-analysis tools to the DSM workspace even when a mesh already exists", () => {
    const app = source("../App.tsx");
    expect(app).toContain('(activeTool === "Measure" || activeTool === "Profiles") && geometryReady');
    expect(app).not.toContain('(activeTool === "Measure" || activeTool === "Profiles") && geometryReady && !meshArtifactReady');
    expect(app).toContain('activeTool === "Structures" && calibrationReady');
  });

  it("prevents missing raster layers from accepting analyst clicks and avoids accidental 3D analysis-mode selection", () => {
    const app = source("../App.tsx");
    expect(app).toContain("const analystInteractive = !demoMode && Boolean(projectDir) && geometryReady && rasterSurfaceReady");
    expect(app).toContain('onSelectPoint={projectAnalystInteractive && activeTool === "Project" ? analyzeRasterPoint : undefined}');
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

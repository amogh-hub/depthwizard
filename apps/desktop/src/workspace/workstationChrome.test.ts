import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

describe("final workstation chrome", () => {
  it("keeps the tool rail icon-only and removes the redundant Layers mode", () => {
    const toolRail = source("../components/ToolRail.tsx");
    expect(toolRail).not.toContain("dw-tool-label");
    expect(toolRail).not.toContain('id: "Layers"');
    expect(toolRail).toContain('aria-label={label}');
  });

  it("never relies on a visible horizontal command scrollbar", () => {
    const css = source("../styles/workstation_final.css");
    expect(css).toContain("flex-wrap: wrap");
    expect(css).toContain("scrollbar-width: none");
    expect(css).toContain(".dw-toolbar-group::-webkit-scrollbar");
    expect(css).toContain("display: none");
    expect(css).not.toContain("overflow-x: auto");
  });

  it("separates scientific context from navigation help", () => {
    const css = source("../styles/workstation_final.css");
    expect(css).toContain(".dw-canvas-context");
    expect(css).toContain(".dw-terrain-mode-help");
    expect(css).toContain("top: 58px");
  });

  it("ships focused free-camera help, true first-person entry and overlay recovery", () => {
    const terrain = source("./TerrainViewport.tsx");
    expect(terrain).toContain("Orbit · drag to rotate");
    expect(terrain).toContain("Fly · click terrain · WASD move");
    expect(terrain).toContain("First person · click terrain · WASD move");
    expect(terrain).toContain("Top down · drag to pan");
    expect(terrain).toContain("const freeCamera = new FlyControls");
    expect(terrain).not.toContain("FirstPersonControls");
    expect(terrain).toContain("canvasFocused");
    expect(terrain).toContain("enterFirstPerson");
    expect(terrain).toContain("Analytical overlay unavailable");
    expect(terrain).toContain("Retry overlay");
    expect(terrain).toContain("Terrain bytes received · preparing GPU resources");
  });

  it("loads the final workstation override last", () => {
    const main = source("../main.tsx");
    const workstation = main.indexOf('import "./styles/workstation.css"');
    const compactRail = main.indexOf('import "./styles/toolrail-compact.css"');
    const final = main.indexOf('import "./styles/workstation_final.css"');
    expect(workstation).toBeGreaterThanOrEqual(0);
    expect(compactRail).toBeGreaterThan(workstation);
    expect(final).toBeGreaterThan(compactRail);
  });
});

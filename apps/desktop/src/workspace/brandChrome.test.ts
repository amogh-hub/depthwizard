import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

describe("DepthWizard product identity chrome", () => {
  it("keeps the top bar logo-only and centers the project filename in available space", () => {
    const css = source("../styles/brand.css");
    expect(css).toContain("grid-template-columns: 48px minmax(0, 1fr) auto");
    expect(css).toContain(".dw-brand > span:not(.dw-mark)");
    expect(css).toContain("width: 38px");
    expect(css).toContain("height: 38px");
    expect(css).toContain("transform: translateX(-7px)");
    expect(css).toContain("display: none");
    expect(css).toContain('url(\"/depthwizard-mark.png\")');
    expect(css).toContain("justify-self: center");
    expect(css).toContain(".dw-project-title > span");
  });

  it("loads the product-identity override after all workstation chrome", () => {
    const main = source("../main.tsx");
    const finalChrome = main.indexOf('import \"./styles/workstation_final.css\"');
    const brand = main.indexOf('import \"./styles/brand.css\"');
    expect(finalChrome).toBeGreaterThanOrEqual(0);
    expect(brand).toBeGreaterThan(finalChrome);
  });

  it("uses the product mark for web and packaged application identity", () => {
    const html = source("../../index.html");
    const tauri = source("../../src-tauri/tauri.conf.json");
    expect(html).toContain('href=\"/depthwizard-mark.png\"');
    expect(tauri).toContain('\"icons/icon.png\"');
  });
});

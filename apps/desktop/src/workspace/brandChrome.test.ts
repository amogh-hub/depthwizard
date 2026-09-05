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

  it("keeps raw Cargo compile-safe while packaging the complete generated icon family", () => {
    const html = source("../../index.html");
    const tauri = JSON.parse(source("../../src-tauri/tauri.conf.json")) as {
      bundle?: { icon?: string[] };
    };
    const baseIcons = tauri.bundle?.icon ?? [];
    const wrapper = source("../../scripts/tauri-with-macos-icon.sh");

    expect(html).toContain('href=\"/depthwizard-mark.png\"');

    // Raw cargo fmt/clippy/test must be able to compile from a clean checkout.
    expect(baseIcons).toEqual(["icons/icon.png"]);

    // Actual Tauri packaging generates and injects the full Retina/native family.
    expect(wrapper).toContain('"32x32.png"');
    expect(wrapper).toContain('"128x128.png"');
    expect(wrapper).toContain('"128x128@2x.png"');
    expect(wrapper).toContain('"icon.icns"');
    expect(wrapper).toContain('"icon.ico"');
    expect(wrapper).toContain("BUNDLE_OVERRIDE=");
    expect(wrapper).toContain('build --config "$BUNDLE_OVERRIDE"');
    expect(wrapper).toContain('APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:--}"');
    expect(wrapper).toContain('CI="${CI:-true}"');
    expect(wrapper).toContain("TAURI_BUNDLER_DMG_IGNORE_CI=true");
    expect(wrapper).toContain("raw Cargo uses tracked icons/icon.png");
  });
});

import { describe, expect, it } from "vitest";
import {
  DEFAULT_RASTER_VIEW_STATE,
  niceScaleDistance,
  normalizedRasterPoint,
  oneToOneScale,
  panRasterView,
  rasterViewportGeometry,
  zoomRasterViewAt,
} from "./rasterViewport";

const base = { left: 100, top: 0, width: 800, height: 800 };
const hostWidth = 1000;
const hostHeight = 800;

describe("raster viewport navigation", () => {
  it("fits the raster at the default state", () => {
    const geometry = rasterViewportGeometry(DEFAULT_RASTER_VIEW_STATE, hostWidth, hostHeight, base);
    expect(geometry).toEqual({ left: 100, top: 0, width: 800, height: 800 });
  });

  it("keeps the cursor-anchored raster point stable while zooming", () => {
    const anchor = { x: 700, y: 240 };
    const before = normalizedRasterPoint(
      DEFAULT_RASTER_VIEW_STATE,
      anchor.x,
      anchor.y,
      hostWidth,
      hostHeight,
      base,
    );
    const zoomed = zoomRasterViewAt(
      DEFAULT_RASTER_VIEW_STATE,
      4,
      anchor.x,
      anchor.y,
      hostWidth,
      hostHeight,
      base,
    );
    const after = normalizedRasterPoint(zoomed, anchor.x, anchor.y, hostWidth, hostHeight, base);
    expect(after?.x).toBeCloseTo(before?.x ?? 0, 8);
    expect(after?.y).toBeCloseTo(before?.y ?? 0, 8);
  });

  it("pans without allowing the raster to disappear beyond the viewport", () => {
    const zoomed = { scale: 4, centerX: 0.5, centerY: 0.5 };
    const moved = panRasterView(zoomed, 200, -150, hostWidth, hostHeight, base);
    expect(moved.centerX).toBeLessThan(0.5);
    expect(moved.centerY).toBeGreaterThan(0.5);
    const clamped = panRasterView(moved, 100000, 100000, hostWidth, hostHeight, base);
    expect(clamped.centerX).toBeGreaterThanOrEqual(0);
    expect(clamped.centerX).toBeLessThanOrEqual(1);
    expect(clamped.centerY).toBeGreaterThanOrEqual(0);
    expect(clamped.centerY).toBeLessThanOrEqual(1);
  });

  it("rejects analytical clicks outside the transformed raster", () => {
    expect(
      normalizedRasterPoint(DEFAULT_RASTER_VIEW_STATE, 20, 20, hostWidth, hostHeight, base),
    ).toBeNull();
  });

  it("derives deterministic one-to-one and scale-bar values", () => {
    expect(oneToOneScale(1600, 800)).toBe(2);
    expect(niceScaleDistance(136)).toBe(100);
    expect(niceScaleDistance(2870)).toBe(2000);
  });
});

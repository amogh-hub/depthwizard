export type RasterViewState = {
  scale: number;
  centerX: number;
  centerY: number;
};

export type RasterBaseRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type RasterViewportGeometry = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export const DEFAULT_RASTER_VIEW_STATE: RasterViewState = {
  scale: 1,
  centerX: 0.5,
  centerY: 0.5,
};

export const MIN_RASTER_ZOOM = 1;
export const MAX_RASTER_ZOOM = 32;

export function clampRasterZoom(scale: number): number {
  if (!Number.isFinite(scale)) return 1;
  return Math.min(MAX_RASTER_ZOOM, Math.max(MIN_RASTER_ZOOM, scale));
}

function axisCenterBounds(hostSize: number, contentSize: number): [number, number] {
  if (contentSize <= hostSize + 0.5) return [0.5, 0.5];
  const halfVisible = hostSize / (2 * contentSize);
  return [halfVisible, 1 - halfVisible];
}

export function clampRasterView(
  state: RasterViewState,
  hostWidth: number,
  hostHeight: number,
  baseRect: RasterBaseRect,
): RasterViewState {
  const scale = clampRasterZoom(state.scale);
  const width = Math.max(baseRect.width * scale, 1);
  const height = Math.max(baseRect.height * scale, 1);
  const [minX, maxX] = axisCenterBounds(Math.max(hostWidth, 1), width);
  const [minY, maxY] = axisCenterBounds(Math.max(hostHeight, 1), height);
  return {
    scale,
    centerX: Math.min(maxX, Math.max(minX, Number.isFinite(state.centerX) ? state.centerX : 0.5)),
    centerY: Math.min(maxY, Math.max(minY, Number.isFinite(state.centerY) ? state.centerY : 0.5)),
  };
}

export function rasterViewportGeometry(
  state: RasterViewState,
  hostWidth: number,
  hostHeight: number,
  baseRect: RasterBaseRect,
): RasterViewportGeometry {
  const safe = clampRasterView(state, hostWidth, hostHeight, baseRect);
  const width = baseRect.width * safe.scale;
  const height = baseRect.height * safe.scale;
  return {
    left: hostWidth / 2 - safe.centerX * width,
    top: hostHeight / 2 - safe.centerY * height,
    width,
    height,
  };
}

export function panRasterView(
  state: RasterViewState,
  dx: number,
  dy: number,
  hostWidth: number,
  hostHeight: number,
  baseRect: RasterBaseRect,
): RasterViewState {
  const geometry = rasterViewportGeometry(state, hostWidth, hostHeight, baseRect);
  if (geometry.width <= 0 || geometry.height <= 0) return state;
  return clampRasterView(
    {
      ...state,
      centerX: state.centerX - dx / geometry.width,
      centerY: state.centerY - dy / geometry.height,
    },
    hostWidth,
    hostHeight,
    baseRect,
  );
}

export function zoomRasterViewAt(
  state: RasterViewState,
  nextScale: number,
  anchorX: number,
  anchorY: number,
  hostWidth: number,
  hostHeight: number,
  baseRect: RasterBaseRect,
): RasterViewState {
  const before = rasterViewportGeometry(state, hostWidth, hostHeight, baseRect);
  const normalizedX = (anchorX - before.left) / Math.max(before.width, 1);
  const normalizedY = (anchorY - before.top) / Math.max(before.height, 1);
  const scale = clampRasterZoom(nextScale);
  const width = baseRect.width * scale;
  const height = baseRect.height * scale;
  const nextLeft = anchorX - normalizedX * width;
  const nextTop = anchorY - normalizedY * height;
  return clampRasterView(
    {
      scale,
      centerX: (hostWidth / 2 - nextLeft) / Math.max(width, 1),
      centerY: (hostHeight / 2 - nextTop) / Math.max(height, 1),
    },
    hostWidth,
    hostHeight,
    baseRect,
  );
}

export function normalizedRasterPoint(
  state: RasterViewState,
  localX: number,
  localY: number,
  hostWidth: number,
  hostHeight: number,
  baseRect: RasterBaseRect,
): { x: number; y: number } | null {
  const geometry = rasterViewportGeometry(state, hostWidth, hostHeight, baseRect);
  const x = (localX - geometry.left) / Math.max(geometry.width, 1);
  const y = (localY - geometry.top) / Math.max(geometry.height, 1);
  if (x < 0 || y < 0 || x > 1 || y > 1) return null;
  return { x, y };
}

export function oneToOneScale(naturalWidth: number, baseWidth: number): number {
  if (naturalWidth <= 0 || baseWidth <= 0) return 1;
  return clampRasterZoom(naturalWidth / baseWidth);
}

export function niceScaleDistance(targetMetres: number): number {
  if (!Number.isFinite(targetMetres) || targetMetres <= 0) return 0;
  const magnitude = 10 ** Math.floor(Math.log10(targetMetres));
  const normalized = targetMetres / magnitude;
  const multiplier = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return multiplier * magnitude;
}

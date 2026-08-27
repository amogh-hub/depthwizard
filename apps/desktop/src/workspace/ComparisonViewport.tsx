import { useEffect, useMemo, useRef, useState } from "react";
import type { NormalizedPoint } from "../api";
import {
  DEFAULT_RASTER_VIEW_STATE,
  niceScaleDistance,
  normalizedRasterPoint,
  oneToOneScale,
  panRasterView,
  rasterViewportGeometry,
  zoomRasterViewAt,
  type RasterBaseRect,
  type RasterViewState,
} from "./rasterViewport";

type ComparisonViewportProps = {
  predictionUrl: string;
  referenceUrl: string;
  viewState: RasterViewState;
  onViewStateChange: (state: RasterViewState) => void;
  sourceWidth?: number;
  groundSampleDistanceM?: number | null;
  cursorPoint?: NormalizedPoint | null;
  onSelectPoint?: (point: NormalizedPoint) => void;
};

type DragState = {
  pointerId: number;
  lastX: number;
  lastY: number;
  startX: number;
  startY: number;
};

const FIT_MARGIN = 12;
const CLICK_DRAG_THRESHOLD = 5;

function fittedRect(host: HTMLDivElement, image: HTMLImageElement): RasterBaseRect | null {
  if (!image.naturalWidth || !image.naturalHeight || !host.clientWidth || !host.clientHeight) return null;
  const availableWidth = Math.max(host.clientWidth - FIT_MARGIN * 2, 1);
  const availableHeight = Math.max(host.clientHeight - FIT_MARGIN * 2, 1);
  const imageAspect = image.naturalWidth / image.naturalHeight;
  const hostAspect = availableWidth / availableHeight;
  if (hostAspect > imageAspect) {
    const height = availableHeight;
    const width = height * imageAspect;
    return { left: (host.clientWidth - width) / 2, top: (host.clientHeight - height) / 2, width, height };
  }
  const width = availableWidth;
  const height = width / imageAspect;
  return { left: (host.clientWidth - width) / 2, top: (host.clientHeight - height) / 2, width, height };
}

function formatScaleDistance(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(metres >= 10000 ? 0 : 1)} km` : `${metres.toFixed(metres >= 10 ? 0 : 1)} m`;
}

export function ComparisonViewport({
  predictionUrl,
  referenceUrl,
  viewState,
  onViewStateChange,
  sourceWidth,
  groundSampleDistanceM,
  cursorPoint,
  onSelectPoint,
}: ComparisonViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const predictionRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const [split, setSplit] = useState(50);
  const [baseRect, setBaseRect] = useState<RasterBaseRect | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    const image = predictionRef.current;
    if (!host || !image) return;
    const update = () => setBaseRect(fittedRect(host, image));
    const observer = new ResizeObserver(update);
    observer.observe(host);
    image.addEventListener("load", update);
    update();
    return () => {
      observer.disconnect();
      image.removeEventListener("load", update);
    };
  }, [predictionUrl]);

  const geometry = useMemo(() => {
    const host = hostRef.current;
    if (!host || !baseRect) return null;
    return rasterViewportGeometry(viewState, host.clientWidth, host.clientHeight, baseRect);
  }, [baseRect, viewState]);

  const localPoint = (clientX: number, clientY: number) => {
    const host = hostRef.current;
    if (!host) return null;
    const bounds = host.getBoundingClientRect();
    return { x: clientX - bounds.left, y: clientY - bounds.top };
  };

  const zoomAt = (clientX: number, clientY: number, scale: number) => {
    const host = hostRef.current;
    const local = localPoint(clientX, clientY);
    if (!host || !baseRect || !local) return;
    onViewStateChange(zoomRasterViewAt(viewState, scale, local.x, local.y, host.clientWidth, host.clientHeight, baseRect));
  };

  const pointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.button !== 1) return;
    dragRef.current = {
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY,
      startX: event.clientX,
      startY: event.clientY,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const pointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    const host = hostRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !host || !baseRect) return;
    const dx = event.clientX - drag.lastX;
    const dy = event.clientY - drag.lastY;
    drag.lastX = event.clientX;
    drag.lastY = event.clientY;
    if (dx !== 0 || dy !== 0) {
      onViewStateChange(panRasterView(viewState, dx, dy, host.clientWidth, host.clientHeight, baseRect));
    }
  };

  const pointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag || drag.pointerId !== event.pointerId || !onSelectPoint || !baseRect) return;
    const movement = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
    if (movement > CLICK_DRAG_THRESHOLD) return;
    const host = hostRef.current;
    const local = localPoint(event.clientX, event.clientY);
    if (!host || !local) return;
    const point = normalizedRasterPoint(viewState, local.x, local.y, host.clientWidth, host.clientHeight, baseRect);
    if (point) onSelectPoint(point);
  };

  const scaleBar = useMemo(() => {
    if (!geometry || !sourceWidth || !groundSampleDistanceM || groundSampleDistanceM <= 0) return null;
    const metresPerCssPixel = (sourceWidth * groundSampleDistanceM) / Math.max(geometry.width, 1);
    const metres = niceScaleDistance(metresPerCssPixel * 120);
    if (metres <= 0) return null;
    return { metres, width: Math.min(180, Math.max(36, metres / metresPerCssPixel)) };
  }, [geometry, groundSampleDistanceM, sourceWidth]);

  return (
    <div
      ref={hostRef}
      className="dw-compare-view"
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={() => { dragRef.current = null; }}
      onWheel={(event) => {
        event.preventDefault();
        zoomAt(event.clientX, event.clientY, viewState.scale * Math.exp(-event.deltaY * 0.0015));
      }}
      onDoubleClick={(event) => {
        event.preventDefault();
        zoomAt(event.clientX, event.clientY, viewState.scale * 2);
      }}
    >
      <img
        ref={predictionRef}
        src={predictionUrl}
        alt="Predicted DSM"
        draggable={false}
        style={geometry ? { left: geometry.left, top: geometry.top, width: geometry.width, height: geometry.height } : undefined}
      />
      <img
        className="dw-compare-reference"
        src={referenceUrl}
        alt="Aligned reference DSM"
        draggable={false}
        style={geometry ? {
          left: geometry.left,
          top: geometry.top,
          width: geometry.width,
          height: geometry.height,
          clipPath: `inset(0 ${100 - split}% 0 0)`,
        } : { clipPath: `inset(0 ${100 - split}% 0 0)` }}
      />

      {geometry && (
        <>
          <div className="dw-compare-label dw-compare-label--left" style={{ left: geometry.left + 10, top: geometry.top + 10 }}>Reference</div>
          <div className="dw-compare-label dw-compare-label--right" style={{ left: geometry.left + geometry.width - 10, top: geometry.top + 10 }}>Prediction</div>
          <div
            className="dw-compare-divider"
            style={{ left: geometry.left + geometry.width * split / 100, top: geometry.top, height: geometry.height }}
            aria-hidden="true"
          />
          <input
            className="dw-compare-slider"
            type="range"
            min="0"
            max="100"
            step="1"
            value={split}
            aria-label="Prediction reference swipe position"
            style={{ left: geometry.left + 14, top: geometry.top + geometry.height - 38, width: Math.max(80, geometry.width - 28) }}
            onPointerDown={(event) => event.stopPropagation()}
            onChange={(event) => setSplit(Number(event.target.value))}
          />
          {cursorPoint && (
            <span
              className="dw-analysis-crosshair dw-analysis-crosshair--compare"
              style={{
                left: geometry.left + cursorPoint.x * geometry.width,
                top: geometry.top + cursorPoint.y * geometry.height,
              }}
              aria-hidden="true"
            >
              <i />
              <b />
            </span>
          )}
        </>
      )}

      {baseRect && (
        <div className="dw-map-navigation" onPointerDown={(event) => event.stopPropagation()}>
          <button type="button" onClick={() => onViewStateChange(DEFAULT_RASTER_VIEW_STATE)}>Fit</button>
          <button type="button" onClick={() => {
            const host = hostRef.current;
            if (!host) return;
            const bounds = host.getBoundingClientRect();
            zoomAt(bounds.left + host.clientWidth / 2, bounds.top + host.clientHeight / 2, viewState.scale / 1.35);
          }}>−</button>
          <span className="dw-map-zoom">{Math.round(viewState.scale * 100)}%</span>
          <button type="button" onClick={() => {
            const host = hostRef.current;
            if (!host) return;
            const bounds = host.getBoundingClientRect();
            zoomAt(bounds.left + host.clientWidth / 2, bounds.top + host.clientHeight / 2, viewState.scale * 1.35);
          }}>+</button>
          <button type="button" onClick={() => onViewStateChange({ ...viewState, scale: oneToOneScale(sourceWidth ?? predictionRef.current?.naturalWidth ?? 1, baseRect.width) })}>1:1</button>
        </div>
      )}

      {scaleBar && (
        <div className="dw-map-scale">
          <span style={{ width: scaleBar.width }} />
          <strong>{formatScaleDistance(scaleBar.metres)}</strong>
        </div>
      )}
      <div className="dw-navigation-help" aria-hidden="true">Drag to pan · wheel/pinch to zoom · use the slider to swipe reference ↔ prediction</div>
    </div>
  );
}

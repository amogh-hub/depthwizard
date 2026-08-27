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

export type RasterInteractionMode = "navigate" | "probe" | "measure" | "profile" | "structure";

type RasterAnalysisViewportProps = {
  src: string;
  alt: string;
  interactive?: boolean;
  interactionMode?: RasterInteractionMode;
  viewState: RasterViewState;
  onViewStateChange: (state: RasterViewState) => void;
  sourceWidth?: number;
  groundSampleDistanceM?: number | null;
  cursorPoint?: NormalizedPoint | null;
  lineStart?: NormalizedPoint | null;
  lineEnd?: NormalizedPoint | null;
  polygonPoints?: NormalizedPoint[];
  polygonClosed?: boolean;
  onPolygonChange?: (points: NormalizedPoint[]) => void;
  onSelectPoint?: (point: NormalizedPoint) => void;
};

type PointerSample = { x: number; y: number };
type DragState = {
  pointerId: number;
  lastX: number;
  lastY: number;
  startX: number;
  startY: number;
  panning: boolean;
  suppressClick: boolean;
};

type PinchState = { distance: number };

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

function svgPoints(points: NormalizedPoint[]): string {
  return points.map((point) => `${(point.x * 100).toFixed(3)},${(point.y * 100).toFixed(3)}`).join(" ");
}

function eventLocalPoint(host: HTMLDivElement, clientX: number, clientY: number): PointerSample {
  const bounds = host.getBoundingClientRect();
  return { x: clientX - bounds.left, y: clientY - bounds.top };
}

function pointerDistance(a: PointerSample, b: PointerSample): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatScaleDistance(metres: number): string {
  if (metres >= 1000) return `${(metres / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} km`;
  if (metres >= 1) return `${metres.toLocaleString(undefined, { maximumFractionDigits: 1 })} m`;
  return `${(metres * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })} cm`;
}

function editableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.matches("input, textarea, select, [contenteditable='true']")
    || Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

export function RasterAnalysisViewport({
  src,
  alt,
  interactive = false,
  interactionMode = "probe",
  viewState,
  onViewStateChange,
  sourceWidth,
  groundSampleDistanceM,
  cursorPoint,
  lineStart,
  lineEnd,
  polygonPoints = [],
  polygonClosed = false,
  onPolygonChange,
  onSelectPoint,
}: RasterAnalysisViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const viewStateRef = useRef(viewState);
  const pointersRef = useRef(new Map<number, PointerSample>());
  const dragRef = useRef<DragState | null>(null);
  const pinchRef = useRef<PinchState | null>(null);
  const vertexDragRef = useRef<number | null>(null);
  const [baseRect, setBaseRect] = useState<RasterBaseRect | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<NormalizedPoint | null>(null);
  const navigationEnabled = interactionMode === "navigate" || !interactive;
  viewStateRef.current = viewState;

  const commitViewState = (next: RasterViewState) => {
    viewStateRef.current = next;
    onViewStateChange(next);
  };

  useEffect(() => {
    const host = hostRef.current;
    const image = imageRef.current;
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
  }, [src]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (event.code !== "Space" || editableTarget(event.target)) return;
      setSpaceHeld(true);
      event.preventDefault();
    };
    const keyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") setSpaceHeld(false);
    };
    const blur = () => setSpaceHeld(false);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", blur);
    };
  }, []);

  const geometry = useMemo(() => {
    const host = hostRef.current;
    if (!host || !baseRect) return null;
    return rasterViewportGeometry(viewState, host.clientWidth, host.clientHeight, baseRect);
  }, [baseRect, viewState]);

  const normalizedFromClient = (clientX: number, clientY: number): NormalizedPoint | null => {
    const host = hostRef.current;
    if (!host || !baseRect) return null;
    const local = eventLocalPoint(host, clientX, clientY);
    return normalizedRasterPoint(
      viewStateRef.current,
      local.x,
      local.y,
      host.clientWidth,
      host.clientHeight,
      baseRect,
    );
  };

  const zoomAtClient = (clientX: number, clientY: number, nextScale: number) => {
    const host = hostRef.current;
    if (!host || !baseRect) return;
    const local = eventLocalPoint(host, clientX, clientY);
    commitViewState(
      zoomRasterViewAt(
        viewStateRef.current,
        nextScale,
        local.x,
        local.y,
        host.clientWidth,
        host.clientHeight,
        baseRect,
      ),
    );
  };

  const panBy = (dx: number, dy: number) => {
    const host = hostRef.current;
    if (!host || !baseRect) return;
    commitViewState(
      panRasterView(viewStateRef.current, dx, dy, host.clientWidth, host.clientHeight, baseRect),
    );
  };

  const beginPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const host = hostRef.current;
    if (!host) return;
    const local = eventLocalPoint(host, event.clientX, event.clientY);
    pointersRef.current.set(event.pointerId, local);
    event.currentTarget.setPointerCapture(event.pointerId);

    if (event.pointerType === "touch" && pointersRef.current.size >= 2) {
      const [a, b] = Array.from(pointersRef.current.values()).slice(0, 2);
      pinchRef.current = { distance: Math.max(pointerDistance(a, b), 1) };
      dragRef.current = null;
      return;
    }

    const panning = event.button === 1 || spaceHeld || navigationEnabled || event.pointerType === "touch";
    dragRef.current = {
      pointerId: event.pointerId,
      lastX: event.clientX,
      lastY: event.clientY,
      startX: event.clientX,
      startY: event.clientY,
      panning,
      suppressClick: event.button === 1 || spaceHeld,
    };
  };

  const movePointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const host = hostRef.current;
    if (!host) return;
    const local = eventLocalPoint(host, event.clientX, event.clientY);
    if (pointersRef.current.has(event.pointerId)) pointersRef.current.set(event.pointerId, local);

    if (pointersRef.current.size >= 2 && baseRect) {
      const [a, b] = Array.from(pointersRef.current.values()).slice(0, 2);
      const currentDistance = Math.max(pointerDistance(a, b), 1);
      const midpointX = (a.x + b.x) / 2;
      const midpointY = (a.y + b.y) / 2;
      const previous = pinchRef.current;
      if (previous) {
        const current = viewStateRef.current;
        commitViewState(
          zoomRasterViewAt(
            current,
            current.scale * (currentDistance / Math.max(previous.distance, 1)),
            midpointX,
            midpointY,
            host.clientWidth,
            host.clientHeight,
            baseRect,
          ),
        );
      }
      pinchRef.current = { distance: currentDistance };
      setHoverPoint(null);
      return;
    }

    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId && drag.panning) {
      const dx = event.clientX - drag.lastX;
      const dy = event.clientY - drag.lastY;
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      if (dx !== 0 || dy !== 0) panBy(dx, dy);
      setHoverPoint(null);
      return;
    }

    if ((interactionMode === "measure" || interactionMode === "profile") && lineStart && !lineEnd) {
      setHoverPoint(normalizedFromClient(event.clientX, event.clientY));
    } else {
      setHoverPoint(null);
    }
  };

  const endPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    const movement = drag && drag.pointerId === event.pointerId
      ? Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY)
      : Number.POSITIVE_INFINITY;
    const suppressClick = Boolean(drag?.suppressClick);
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (drag?.pointerId === event.pointerId) dragRef.current = null;

    if (
      interactive
      && !suppressClick
      && movement <= CLICK_DRAG_THRESHOLD
      && event.button === 0
      && onSelectPoint
    ) {
      const point = normalizedFromClient(event.clientX, event.clientY);
      if (point) onSelectPoint(point);
    }
  };

  const cancelPointer = (event: React.PointerEvent<HTMLDivElement>) => {
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  const wheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (!baseRect) return;
    event.preventDefault();
    zoomAtClient(event.clientX, event.clientY, viewStateRef.current.scale * Math.exp(-event.deltaY * 0.0015));
  };

  const doubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!navigationEnabled) return;
    event.preventDefault();
    zoomAtClient(event.clientX, event.clientY, viewStateRef.current.scale * 2);
  };

  const updateVertex = (index: number, clientX: number, clientY: number) => {
    const point = normalizedFromClient(clientX, clientY);
    if (!point || !onPolygonChange) return;
    onPolygonChange(polygonPoints.map((item, itemIndex) => itemIndex === index ? point : item));
  };

  const scaleBar = useMemo(() => {
    if (!geometry || !sourceWidth || !groundSampleDistanceM || groundSampleDistanceM <= 0) return null;
    const metresPerCssPixel = (sourceWidth * groundSampleDistanceM) / Math.max(geometry.width, 1);
    const metres = niceScaleDistance(metresPerCssPixel * 120);
    if (metres <= 0) return null;
    return { metres, width: Math.min(180, Math.max(36, metres / metresPerCssPixel)) };
  }, [geometry, groundSampleDistanceM, sourceWidth]);

  const previewEnd = lineEnd ?? hoverPoint;
  const cursorMode = spaceHeld || navigationEnabled ? "navigate" : interactionMode;

  return (
    <div
      ref={hostRef}
      className="dw-raster-view dw-raster-view--interactive"
      data-interactive={interactive}
      data-cursor-mode={cursorMode}
      onPointerDown={beginPointer}
      onPointerMove={movePointer}
      onPointerUp={endPointer}
      onPointerCancel={cancelPointer}
      onWheel={wheel}
      onDoubleClick={doubleClick}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        draggable={false}
        style={geometry ? { left: geometry.left, top: geometry.top, width: geometry.width, height: geometry.height } : undefined}
      />

      {geometry && (
        <div className="dw-raster-overlay" style={{ left: geometry.left, top: geometry.top, width: geometry.width, height: geometry.height }}>
          {lineStart && previewEnd && (
            <svg className="dw-analysis-line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <line
                x1={lineStart.x * 100}
                y1={lineStart.y * 100}
                x2={previewEnd.x * 100}
                y2={previewEnd.y * 100}
                data-preview={!lineEnd}
              />
            </svg>
          )}
          {polygonPoints.length > 1 && (
            <svg className="dw-structure-polygon" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {polygonClosed && polygonPoints.length >= 3
                ? <polygon points={svgPoints(polygonPoints)} />
                : <polyline points={svgPoints(polygonPoints)} />}
            </svg>
          )}
          {lineStart && <span className="dw-analysis-marker" data-kind="start" style={{ left: `${lineStart.x * 100}%`, top: `${lineStart.y * 100}%` }}>A</span>}
          {lineEnd && <span className="dw-analysis-marker" data-kind="end" style={{ left: `${lineEnd.x * 100}%`, top: `${lineEnd.y * 100}%` }}>B</span>}
          {polygonPoints.map((point, index) => (
            <button
              type="button"
              className="dw-structure-vertex"
              key={`${index}`}
              style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
              aria-label={`Structure footprint vertex ${index + 1}. Drag to edit.`}
              onPointerDown={(event) => {
                event.stopPropagation();
                vertexDragRef.current = index;
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerMove={(event) => {
                if (vertexDragRef.current !== index || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
                updateVertex(index, event.clientX, event.clientY);
              }}
              onPointerUp={(event) => {
                if (vertexDragRef.current === index) updateVertex(index, event.clientX, event.clientY);
                vertexDragRef.current = null;
                event.stopPropagation();
              }}
            >
              {index + 1}
            </button>
          ))}
          {cursorPoint && (
            <span className="dw-analysis-crosshair" style={{ left: `${cursorPoint.x * 100}%`, top: `${cursorPoint.y * 100}%` }} aria-hidden="true">
              <i /><b />
            </span>
          )}
        </div>
      )}

      {baseRect && (
        <div className="dw-map-navigation" onPointerDown={(event) => event.stopPropagation()}>
          <button type="button" onClick={() => commitViewState(DEFAULT_RASTER_VIEW_STATE)} title="Fit the full raster to the viewport">Fit</button>
          <button type="button" onClick={() => {
            const host = hostRef.current;
            if (!host) return;
            const bounds = host.getBoundingClientRect();
            zoomAtClient(bounds.left + host.clientWidth / 2, bounds.top + host.clientHeight / 2, viewStateRef.current.scale / 1.35);
          }} aria-label="Zoom out">−</button>
          <span className="dw-map-zoom">{Math.round(viewState.scale * 100)}%</span>
          <button type="button" onClick={() => {
            const host = hostRef.current;
            if (!host) return;
            const bounds = host.getBoundingClientRect();
            zoomAtClient(bounds.left + host.clientWidth / 2, bounds.top + host.clientHeight / 2, viewStateRef.current.scale * 1.35);
          }} aria-label="Zoom in">+</button>
          <button
            type="button"
            onClick={() => commitViewState({
              ...viewStateRef.current,
              scale: oneToOneScale(sourceWidth ?? imageRef.current?.naturalWidth ?? 1, baseRect.width),
            })}
            title="Display one source raster pixel per CSS pixel"
          >1:1</button>
        </div>
      )}

      {scaleBar && (
        <div className="dw-map-scale" aria-label={`Map scale ${formatScaleDistance(scaleBar.metres)}`}>
          <span style={{ width: scaleBar.width }} />
          <strong>{formatScaleDistance(scaleBar.metres)}</strong>
        </div>
      )}

      <div className="dw-navigation-help" aria-hidden="true">
        {navigationEnabled ? "Drag to pan · wheel/pinch to zoom · click to inspect" : "Space + drag to pan · wheel/pinch to zoom"}
      </div>
    </div>
  );
}

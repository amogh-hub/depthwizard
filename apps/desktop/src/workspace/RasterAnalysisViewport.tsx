import { useEffect, useRef, useState } from "react";
import type { NormalizedPoint } from "../api";

type ContentRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type RasterAnalysisViewportProps = {
  src: string;
  alt: string;
  interactive?: boolean;
  cursorPoint?: NormalizedPoint | null;
  lineStart?: NormalizedPoint | null;
  lineEnd?: NormalizedPoint | null;
  polygonPoints?: NormalizedPoint[];
  polygonClosed?: boolean;
  onSelectPoint?: (point: NormalizedPoint) => void;
};

function fittedRect(host: HTMLDivElement, image: HTMLImageElement): ContentRect | null {
  if (!image.naturalWidth || !image.naturalHeight || !host.clientWidth || !host.clientHeight) {
    return null;
  }
  const imageAspect = image.naturalWidth / image.naturalHeight;
  const hostAspect = host.clientWidth / host.clientHeight;
  if (hostAspect > imageAspect) {
    const height = host.clientHeight;
    const width = height * imageAspect;
    return { left: (host.clientWidth - width) / 2, top: 0, width, height };
  }
  const width = host.clientWidth;
  const height = width / imageAspect;
  return { left: 0, top: (host.clientHeight - height) / 2, width, height };
}

function svgPoints(points: NormalizedPoint[]): string {
  return points.map((point) => `${(point.x * 100).toFixed(3)},${(point.y * 100).toFixed(3)}`).join(" ");
}

export function RasterAnalysisViewport({
  src,
  alt,
  interactive = false,
  cursorPoint,
  lineStart,
  lineEnd,
  polygonPoints = [],
  polygonClosed = false,
  onSelectPoint,
}: RasterAnalysisViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [contentRect, setContentRect] = useState<ContentRect | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    const image = imageRef.current;
    if (!host || !image) return;
    const update = () => setContentRect(fittedRect(host, image));
    const observer = new ResizeObserver(update);
    observer.observe(host);
    image.addEventListener("load", update);
    update();
    return () => {
      observer.disconnect();
      image.removeEventListener("load", update);
    };
  }, [src]);

  const selectPoint = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!interactive || !contentRect || !onSelectPoint) return;
    const host = hostRef.current;
    if (!host) return;
    const bounds = host.getBoundingClientRect();
    const localX = event.clientX - bounds.left - contentRect.left;
    const localY = event.clientY - bounds.top - contentRect.top;
    if (localX < 0 || localY < 0 || localX > contentRect.width || localY > contentRect.height) {
      return;
    }
    onSelectPoint({
      x: Math.min(1, Math.max(0, localX / contentRect.width)),
      y: Math.min(1, Math.max(0, localY / contentRect.height)),
    });
  };

  const overlayPoint = (point: NormalizedPoint) => ({
    left: `${point.x * 100}%`,
    top: `${point.y * 100}%`,
  });

  return (
    <div
      ref={hostRef}
      className="dw-raster-view dw-raster-view--interactive"
      data-interactive={interactive}
      onPointerDown={selectPoint}
    >
      <img ref={imageRef} src={src} alt={alt} draggable={false} />
      {contentRect && (
        <div
          className="dw-raster-overlay"
          style={{
            left: contentRect.left,
            top: contentRect.top,
            width: contentRect.width,
            height: contentRect.height,
          }}
        >
          {lineStart && lineEnd && (
            <svg className="dw-analysis-line" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <line
                x1={lineStart.x * 100}
                y1={lineStart.y * 100}
                x2={lineEnd.x * 100}
                y2={lineEnd.y * 100}
              />
            </svg>
          )}
          {polygonPoints.length > 1 && (
            <svg className="dw-structure-polygon" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {polygonClosed && polygonPoints.length >= 3 ? (
                <polygon points={svgPoints(polygonPoints)} />
              ) : (
                <polyline points={svgPoints(polygonPoints)} />
              )}
            </svg>
          )}
          {lineStart && <span className="dw-analysis-marker" data-kind="start" style={overlayPoint(lineStart)}>A</span>}
          {lineEnd && <span className="dw-analysis-marker" data-kind="end" style={overlayPoint(lineEnd)}>B</span>}
          {polygonPoints.map((point, index) => (
            <span
              className="dw-structure-vertex"
              key={`${point.x}-${point.y}-${index}`}
              style={overlayPoint(point)}
              aria-hidden="true"
            >
              {index + 1}
            </span>
          ))}
          {cursorPoint && (
            <span className="dw-analysis-crosshair" style={overlayPoint(cursorPoint)} aria-hidden="true">
              <i />
              <b />
            </span>
          )}
        </div>
      )}
    </div>
  );
}

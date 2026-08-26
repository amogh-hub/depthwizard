import { useEffect, useRef, useState } from "react";
import type { NormalizedPoint } from "../api";

type ContentRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type ComparisonViewportProps = {
  predictionUrl: string;
  referenceUrl: string;
  cursorPoint?: NormalizedPoint | null;
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

export function ComparisonViewport({
  predictionUrl,
  referenceUrl,
  cursorPoint,
  onSelectPoint,
}: ComparisonViewportProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const predictionRef = useRef<HTMLImageElement>(null);
  const [split, setSplit] = useState(50);
  const [contentRect, setContentRect] = useState<ContentRect | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    const image = predictionRef.current;
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
  }, [predictionUrl]);

  const selectPoint = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!contentRect || !onSelectPoint) return;
    const host = hostRef.current;
    if (!host) return;
    const bounds = host.getBoundingClientRect();
    const localX = event.clientX - bounds.left - contentRect.left;
    const localY = event.clientY - bounds.top - contentRect.top;
    if (localX < 0 || localY < 0 || localX > contentRect.width || localY > contentRect.height) {
      return;
    }
    onSelectPoint({ x: localX / contentRect.width, y: localY / contentRect.height });
  };

  return (
    <div ref={hostRef} className="dw-compare-view" onPointerDown={selectPoint}>
      <img ref={predictionRef} src={predictionUrl} alt="Predicted DSM" draggable={false} />
      <img
        className="dw-compare-reference"
        src={referenceUrl}
        alt="Aligned reference DSM"
        draggable={false}
        style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
      />
      <div className="dw-compare-label dw-compare-label--left">Reference</div>
      <div className="dw-compare-label dw-compare-label--right">Prediction</div>
      <div className="dw-compare-divider" style={{ left: `${split}%` }} aria-hidden="true" />
      <input
        className="dw-compare-slider"
        type="range"
        min="0"
        max="100"
        step="1"
        value={split}
        aria-label="Prediction reference swipe position"
        onPointerDown={(event) => event.stopPropagation()}
        onChange={(event) => setSplit(Number(event.target.value))}
      />
      {contentRect && cursorPoint && (
        <span
          className="dw-analysis-crosshair dw-analysis-crosshair--compare"
          style={{
            left: contentRect.left + cursorPoint.x * contentRect.width,
            top: contentRect.top + cursorPoint.y * contentRect.height,
          }}
          aria-hidden="true"
        >
          <i />
          <b />
        </span>
      )}
    </div>
  );
}

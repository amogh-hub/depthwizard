import type { ProjectLayerLegend } from "../api";

function formatValue(value: number | undefined, units: string | null): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  const digits = magnitude >= 1000 ? 0 : magnitude >= 100 ? 1 : magnitude >= 10 ? 2 : 3;
  const suffix = units && units !== "model-native" && units !== "relative illumination" ? ` ${units}` : "";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: digits })}${suffix}`;
}

export function ScientificLegend({ legend }: { legend: ProjectLayerLegend | null }) {
  if (!legend?.available || legend.minimum === undefined || legend.maximum === undefined) return null;
  return (
    <aside className="dw-scientific-legend" data-ramp={legend.ramp} aria-label={`${legend.title} legend`}>
      <header>
        <strong>{legend.title}</strong>
        <span>{legend.units ?? "display"}</span>
      </header>
      <div className="dw-legend-body">
        <div className="dw-legend-ramp" aria-hidden="true" />
        <div className="dw-legend-ticks">
          <span>{formatValue(legend.maximum, legend.units)}</span>
          <span>{formatValue(legend.midpoint, legend.units)}</span>
          <span>{formatValue(legend.minimum, legend.units)}</span>
        </div>
      </div>
      <footer title={legend.semantics}>
        {legend.ramp === "diverging" ? "symmetric P95 display range" : legend.ramp === "grayscale" && legend.layer === "confidence" ? "model-native · not probability calibrated" : "P02–P98 display range"}
      </footer>
    </aside>
  );
}

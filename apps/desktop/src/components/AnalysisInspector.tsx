import type { ProjectProbeResult, ProjectProfileResult } from "../api";

function valueLabel(value: number | null, units: string | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${units ? ` ${units}` : ""}`;
}

function sampleLabel(available: boolean, value: number | null, units: string | null): string {
  return available ? valueLabel(value, units) : "—";
}

function distanceLabel(profile: ProjectProfileResult): string {
  if (profile.horizontal_distance_m !== null) {
    return profile.horizontal_distance_m >= 1000
      ? `${(profile.horizontal_distance_m / 1000).toFixed(3)} km`
      : `${profile.horizontal_distance_m.toFixed(2)} m`;
  }
  return `${profile.horizontal_distance_pixels.toFixed(2)} px`;
}

function fractionalDistanceLabel(profile: ProjectProfileResult, fraction: number): string {
  if (profile.horizontal_distance_m !== null) {
    const value = profile.horizontal_distance_m * fraction;
    return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`;
  }
  return `${(profile.horizontal_distance_pixels * fraction).toFixed(1)} px`;
}

function ProfileChart({ profile }: { profile: ProjectProfileResult }) {
  const surface = profile.samples.filter((sample) => sample.surface.available && sample.surface.value !== null);
  if (surface.length < 2) return null;
  const reference = profile.samples.filter(
    (sample) => sample.reference.available && sample.reference.value !== null,
  );
  const values = [
    ...surface.map((sample) => sample.surface.value as number),
    ...reference.map((sample) => sample.reference.value as number),
  ];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  const yFor = (value: number) => 54 - ((value - min) / span) * 46;
  const points = surface
    .map((sample) => `${(sample.fraction * 100).toFixed(2)},${yFor(sample.surface.value as number).toFixed(2)}`)
    .join(" ");
  const referencePoints = reference
    .map((sample) => `${(sample.fraction * 100).toFixed(2)},${yFor(sample.reference.value as number).toFixed(2)}`)
    .join(" ");
  const midpoint = min + span / 2;

  return (
    <div className="dw-profile-chart" aria-label="Elevation profile chart">
      <div className="dw-profile-plot">
        <svg viewBox="0 0 100 60" preserveAspectRatio="none" role="img" aria-label="Elevation along selected transect">
          <line x1="0" y1="8" x2="100" y2="8" className="dw-profile-grid" />
          <line x1="0" y1="31" x2="100" y2="31" className="dw-profile-grid" />
          <line x1="0" y1="54" x2="100" y2="54" className="dw-profile-axis" />
          <polyline points={points} className="dw-profile-line" />
          {referencePoints && <polyline points={referencePoints} className="dw-profile-reference-line" />}
        </svg>
        <div className="dw-profile-y-axis" aria-hidden="true">
          <span>{valueLabel(max, profile.vertical_units)}</span>
          <span>{valueLabel(midpoint, profile.vertical_units)}</span>
          <span>{valueLabel(min, profile.vertical_units)}</span>
        </div>
      </div>
      <div className="dw-profile-x-axis" aria-hidden="true">
        <span>A · {fractionalDistanceLabel(profile, 0)}</span>
        <span>{fractionalDistanceLabel(profile, 0.5)}</span>
        <span>B · {fractionalDistanceLabel(profile, 1)}</span>
      </div>
      {referencePoints && (
        <div className="dw-profile-key"><span className="dw-profile-key-swatch" />Prediction <span className="dw-profile-key-swatch dw-profile-key-swatch--reference" />Reference</div>
      )}
    </div>
  );
}

type AnalysisInspectorProps = {
  activeTool: string;
  probe?: ProjectProbeResult | null;
  measurement?: ProjectProfileResult | null;
  profile?: ProjectProfileResult | null;
  analysisBusy?: boolean;
};

export function AnalysisInspector({
  activeTool,
  probe,
  measurement,
  profile,
  analysisBusy = false,
}: AnalysisInspectorProps) {
  const measureMode = activeTool === "Measure";
  const profileMode = activeTool === "Profiles";
  const showProbe = Boolean(probe);
  const showMeasurement = measureMode && Boolean(measurement);
  const showProfile = profileMode && Boolean(profile);
  if (!showProbe && !showMeasurement && !showProfile && !analysisBusy && !measureMode && !profileMode) return null;

  return (
    <>
      {(measureMode || profileMode) && !analysisBusy && !(measureMode ? measurement : profile) && (
        <section className="dw-section">
          <div className="dw-section-title">{measureMode ? "Two-point measurement" : "Elevation profile"}</div>
          <div className="dw-validation-empty">
            <strong>{probe ? "Select endpoint B" : "Select endpoint A"}</strong>
            <p>
              {measureMode
                ? "Click two registered surface locations. DepthWizard reports geodesic ground distance when trustworthy plus signed endpoint elevation change; hold Space while dragging to pan without placing a point."
                : "Click the start and end of a transect. The canvas previews the line before endpoint B is committed, then samples the persisted elevation surface at subpixel positions along the path."}
            </p>
          </div>
        </section>
      )}

      {(showProbe || analysisBusy) && (
        <section className="dw-section">
          <div className="dw-section-title">Analyst cursor</div>
          {analysisBusy && !probe ? (
            <div className="dw-validation-empty"><strong>Sampling project</strong><p>Reading persisted geospatial products at the selected location.</p></div>
          ) : probe ? (
            <dl className="dw-property-list">
              <div className="dw-property"><dt>Pixel</dt><dd>{probe.pixel_col}, {probe.pixel_row}</dd></div>
              <div className="dw-property"><dt>Surface</dt><dd>{sampleLabel(probe.surface.available, probe.surface.value, probe.surface.units)}</dd></div>
              <div className="dw-property"><dt>Slope</dt><dd>{sampleLabel(probe.slope.available, probe.slope.value, probe.slope.units)}</dd></div>
              <div className="dw-property"><dt>Reference</dt><dd>{sampleLabel(probe.reference.available, probe.reference.value, probe.reference.units)}</dd></div>
              <div className="dw-property"><dt>Residual</dt><dd>{sampleLabel(probe.residual.available, probe.residual.value, probe.residual.units)}</dd></div>
              <div className="dw-property"><dt>Confidence</dt><dd>{sampleLabel(probe.confidence.available, probe.confidence.value, probe.confidence.units)}</dd></div>
              <div className="dw-property"><dt>Longitude</dt><dd>{probe.longitude === null ? "—" : probe.longitude.toFixed(6)}</dd></div>
              <div className="dw-property"><dt>Latitude</dt><dd>{probe.latitude === null ? "—" : probe.latitude.toFixed(6)}</dd></div>
            </dl>
          ) : null}
        </section>
      )}

      {showMeasurement && measurement && (
        <section className="dw-section">
          <div className="dw-section-title">Two-point measurement</div>
          <dl className="dw-property-list">
            <div className="dw-property"><dt>{measurement.horizontal_distance_m !== null ? "Ground distance" : "Pixel distance"}</dt><dd>{distanceLabel(measurement)}</dd></div>
            <div className="dw-property"><dt>Endpoint A</dt><dd>{sampleLabel(measurement.samples[0]?.surface.available ?? false, measurement.samples[0]?.surface.value ?? null, measurement.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Endpoint B</dt><dd>{sampleLabel(measurement.samples.at(-1)?.surface.available ?? false, measurement.samples.at(-1)?.surface.value ?? null, measurement.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Signed Δz (B − A)</dt><dd>{valueLabel(measurement.vertical_delta, measurement.vertical_units)}</dd></div>
          </dl>
          <div className="dw-validation-empty dw-analysis-note">
            <strong>Surface-to-surface measurement</strong>
            <p>Δz is endpoint B surface elevation minus endpoint A surface elevation. It is not automatically a building height; use Structures for a footprint-versus-local-ground estimate.</p>
          </div>
        </section>
      )}

      {showProfile && profile && (
        <section className="dw-section">
          <div className="dw-section-title">Elevation transect</div>
          <ProfileChart profile={profile} />
          <dl className="dw-property-list dw-profile-properties">
            <div className="dw-property"><dt>{profile.horizontal_distance_m !== null ? "Ground length" : "Pixel length"}</dt><dd>{distanceLabel(profile)}</dd></div>
            <div className="dw-property"><dt>Endpoint Δz</dt><dd>{valueLabel(profile.vertical_delta, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Cumulative gain</dt><dd>{valueLabel(profile.elevation_gain, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Cumulative loss</dt><dd>{valueLabel(profile.elevation_loss, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Samples</dt><dd>{profile.sample_count}</dd></div>
          </dl>
          <div className="dw-profile-note">Gain/loss is accumulated over the sampled DSM transect and is resolution-dependent; endpoint Δz is simply B − A.</div>
        </section>
      )}
    </>
  );
}

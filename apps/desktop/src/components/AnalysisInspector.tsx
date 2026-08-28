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
  const points = surface
    .map((sample) => {
      const x = sample.fraction * 100;
      const y = 94 - ((sample.surface.value as number) - min) / span * 82;
      return `${x.toFixed(2)},${Math.min(98, Math.max(2, y)).toFixed(2)}`;
    })
    .join(" ");
  const referencePoints = reference
    .map((sample) => {
      const x = sample.fraction * 100;
      const y = 94 - ((sample.reference.value as number) - min) / span * 82;
      return `${x.toFixed(2)},${Math.min(98, Math.max(2, y)).toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="dw-profile-chart" aria-label="Elevation profile chart">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none">
        <line x1="0" y1="94" x2="100" y2="94" className="dw-profile-axis" />
        <polyline points={points} className="dw-profile-line" />
        {referencePoints && <polyline points={referencePoints} className="dw-profile-reference-line" />}
      </svg>
      <div className="dw-profile-range">
        <span>{valueLabel(max, profile.vertical_units)}</span>
        <span>{valueLabel(min, profile.vertical_units)}</span>
      </div>
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
                ? "Click two registered surface locations. DepthWizard reports plan distance and endpoint elevation change; hold Space while dragging to pan without placing a point."
                : "Click the start and end of a transect. The canvas previews the line before endpoint B is committed, then samples the persisted elevation surface along the path."}
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
            <div className="dw-property"><dt>Plan distance</dt><dd>{distanceLabel(measurement)}</dd></div>
            <div className="dw-property"><dt>Surface Δz</dt><dd>{valueLabel(measurement.vertical_delta, measurement.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Minimum</dt><dd>{valueLabel(measurement.minimum_surface, measurement.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Maximum</dt><dd>{valueLabel(measurement.maximum_surface, measurement.vertical_units)}</dd></div>
          </dl>
          <div className="dw-validation-empty dw-analysis-note">
            <strong>Manual surface-height interpretation</strong>
            <p>Δz is the selected endpoint surface-elevation difference. DepthWizard does not automatically label it as building height.</p>
          </div>
        </section>
      )}

      {showProfile && profile && (
        <section className="dw-section">
          <div className="dw-section-title">Elevation transect</div>
          <ProfileChart profile={profile} />
          <dl className="dw-property-list dw-profile-properties">
            <div className="dw-property"><dt>Length</dt><dd>{distanceLabel(profile)}</dd></div>
            <div className="dw-property"><dt>Endpoint Δz</dt><dd>{valueLabel(profile.vertical_delta, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Gain</dt><dd>{valueLabel(profile.elevation_gain, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Loss</dt><dd>{valueLabel(profile.elevation_loss, profile.vertical_units)}</dd></div>
            <div className="dw-property"><dt>Samples</dt><dd>{profile.sample_count}</dd></div>
          </dl>
        </section>
      )}
    </>
  );
}

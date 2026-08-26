export type RasterMetadata = {
  path: string;
  width: number;
  height: number;
  count: number;
  dtype: string;
  crs: string | null;
  transform: [number, number, number, number, number, number] | null;
  nodata: number | null;
  ground_sample_distance_x: number | null;
  ground_sample_distance_y: number | null;
};

export type ProjectRunStatus =
  | "created"
  | "queued"
  | "running"
  | "waiting_for_calibration"
  | "complete"
  | "failed";

export type GroundControlPoint = {
  x: number;
  y: number;
  elevation_m: number;
  weight?: number;
};

export type ProcessingRequest = {
  source: string;
  output_dir: string;
  dem_path?: string | null;
  gcps?: GroundControlPoint[];
  requested_output?: "rdsm" | "dsm" | null;
  band_indices?: [number, number, number];
  tile_size?: number;
  overlap?: number;
  harmonize_overlaps?: boolean;
  low_frequency_sigma_px?: number;
};

export type ProjectJobState = {
  job_id: string;
  project_dir: string;
  status: ProjectRunStatus;
  manifest_path: string;
  submitted_at_utc: string;
  updated_at_utc: string;
  error: string | null;
};

export type ProjectArtifact = {
  path: string;
  semantics: string;
  units: string | null;
  sha256: string;
};

export type ProjectStage = {
  status: "running" | "completed" | "waiting" | "failed" | "skipped" | string;
  started_at_utc?: string;
  updated_at_utc?: string;
  completed_at_utc?: string;
  elapsed_seconds?: number;
  artifacts: Record<string, string>;
  details: Record<string, unknown>;
};

export type EstimatorEvidence = {
  branch_id: string;
  evidence_id: string;
  independently_evaluated: boolean;
  promotion_passed: boolean;
  summary: string;
};

export type EstimatorDecision = {
  selected_path: "calibrated_da3" | "promoted_learned_refiner";
  selected_model_id: string;
  reason: string;
  evidence: EstimatorEvidence[];
};

export type ProjectManifest = {
  schema_version: number;
  project_id: string;
  job_id: string | null;
  status: ProjectRunStatus;
  created_at_utc: string;
  updated_at_utc: string;
  source_path: string;
  source_sha256: string | null;
  input_kind: "non_georeferenced" | "georeferenced" | null;
  geometry_config_sha256: string | null;
  run_config_sha256: string | null;
  estimator: EstimatorDecision | Record<string, unknown>;
  artifacts: Record<string, ProjectArtifact>;
  stages: Record<string, ProjectStage>;
  warnings: string[];
  errors: Array<{ at_utc: string; stage: string | null; message: string }>;
};

export type EvaluationMetrics = {
  valid_pixels: number;
  mae_m: number;
  rmse_m: number;
  pearson_r: number | null;
  mean_bias_m: number;
  median_abs_error_m: number;
  p90_abs_error_m: number;
  p95_abs_error_m: number;
};

export type SlopeMetrics = {
  valid_pixels: number;
  mae_degrees: number;
  rmse_degrees: number;
  p95_abs_error_degrees: number;
};

export type ErrorConfidenceBin = {
  lower_confidence: number;
  upper_confidence: number;
  valid_pixels: number;
  mean_confidence: number;
  mae_m: number;
  rmse_m: number;
};

export type ReliabilityDiagnostics = {
  available: boolean;
  semantics: string;
  valid_pixels: number;
  confidence_abs_error_pearson_r: number | null;
  bins: ErrorConfidenceBin[];
};

export type ReferenceValidationReport = {
  schema_version: number;
  project_id: string;
  prediction_sha256: string;
  reference_path: string;
  reference_sha256: string;
  reference_label: string | null;
  independence_check: string;
  alignment: string;
  valid_pixels: number;
  coverage_fraction: number;
  elevation: EvaluationMetrics;
  slope: SlopeMetrics;
  reliability: ReliabilityDiagnostics;
  artifacts: Record<string, string>;
  warnings: string[];
};

export type NormalizedPoint = {
  x: number;
  y: number;
};

export type RasterSample = {
  available: boolean;
  value: number | null;
  units: string | null;
  semantics: string;
};

export type ProjectProbeResult = {
  project_id: string;
  point: NormalizedPoint;
  pixel_col: number;
  pixel_row: number;
  map_x: number | null;
  map_y: number | null;
  longitude: number | null;
  latitude: number | null;
  surface_product: "dsm" | "rdsm";
  surface: RasterSample;
  slope: RasterSample;
  reference: RasterSample;
  residual: RasterSample;
  confidence: RasterSample;
};

export type ProfileSample = {
  fraction: number;
  point: NormalizedPoint;
  distance_pixels: number;
  distance_m: number | null;
  surface: RasterSample;
  slope: RasterSample;
  reference: RasterSample;
  residual: RasterSample;
  confidence: RasterSample;
};

export type ProjectProfileResult = {
  project_id: string;
  surface_product: "dsm" | "rdsm";
  start: NormalizedPoint;
  end: NormalizedPoint;
  sample_count: number;
  horizontal_distance_pixels: number;
  horizontal_distance_m: number | null;
  vertical_delta: number | null;
  vertical_units: string | null;
  minimum_surface: number | null;
  maximum_surface: number | null;
  elevation_gain: number | null;
  elevation_loss: number | null;
  samples: ProfileSample[];
  semantics: string;
};

export type TerrainLodArtifact = {
  level: number;
  stride: number;
  path: string;
  sha256: string;
  vertices: number;
  faces: number;
  width_samples: number;
  height_samples: number;
};

export type ProjectMeshReport = {
  schema_version: number;
  project_id: string;
  surface_product: "dsm" | "rdsm";
  surface_sha256: string;
  texture_sha256: string;
  build_config_sha256: string;
  horizontal_units: "m" | "px";
  vertical_units: "m" | "relative";
  gsd_x: number;
  gsd_y: number;
  raster_width: number;
  raster_height: number;
  valid_pixels: number;
  minimum_elevation: number;
  maximum_elevation: number;
  relief: number;
  lods: TerrainLodArtifact[];
  mesh_manifest_path: string;
  semantics: string;
};

export type ProjectPreviewLayer =
  | "optical"
  | "rdsm"
  | "dsm"
  | "slope"
  | "reference"
  | "residual"
  | "confidence";

type RuntimeConfig = {
  apiBase?: string;
  sessionToken?: string;
};

declare global {
  interface Window {
    __DEPTHWIZARD_RUNTIME__?: RuntimeConfig;
  }
}

const runtime = () => window.__DEPTHWIZARD_RUNTIME__ ?? {};

function runtimeHeaders(init?: RequestInit): Headers {
  const config = runtime();
  const headers = new Headers(init?.headers);
  if (init?.body !== undefined && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (config.sessionToken) {
    headers.set("x-depthwizard-token", config.sessionToken);
  }
  return headers;
}

async function checkedResponse(path: string, init?: RequestInit): Promise<Response> {
  const config = runtime();
  const response = await fetch(`${config.apiBase ?? "http://127.0.0.1:8765"}${path}`, {
    ...init,
    headers: runtimeHeaders(init),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `DepthWizard core returned HTTP ${response.status}`);
  }
  return response;
}

async function coreFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await checkedResponse(path, init);
  return response.json() as Promise<T>;
}

export function inspectRaster(path: string): Promise<RasterMetadata> {
  return coreFetch<RasterMetadata>("/v1/inspect", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export function submitProject(request: ProcessingRequest): Promise<ProjectJobState> {
  return coreFetch<ProjectJobState>("/v1/projects", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getProjectJob(jobId: string): Promise<ProjectJobState> {
  return coreFetch<ProjectJobState>(`/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function getProjectManifest(projectDir: string): Promise<ProjectManifest> {
  const query = new URLSearchParams({ project_dir: projectDir });
  return coreFetch<ProjectManifest>(`/v1/projects/manifest?${query.toString()}`);
}

export function validateProjectReference(
  projectDir: string,
  referencePath: string,
  referenceLabel?: string,
): Promise<ReferenceValidationReport> {
  return coreFetch<ReferenceValidationReport>("/v1/projects/validate", {
    method: "POST",
    body: JSON.stringify({
      project_dir: projectDir,
      reference_path: referencePath,
      reference_label: referenceLabel ?? null,
    }),
  });
}

export function getProjectValidation(projectDir: string): Promise<ReferenceValidationReport> {
  const query = new URLSearchParams({ project_dir: projectDir });
  return coreFetch<ReferenceValidationReport>(`/v1/projects/validation?${query.toString()}`);
}

export function probeProject(projectDir: string, point: NormalizedPoint): Promise<ProjectProbeResult> {
  return coreFetch<ProjectProbeResult>("/v1/projects/probe", {
    method: "POST",
    body: JSON.stringify({ project_dir: projectDir, point }),
  });
}

export function sampleProjectProfile(
  projectDir: string,
  start: NormalizedPoint,
  end: NormalizedPoint,
  samples = 160,
): Promise<ProjectProfileResult> {
  return coreFetch<ProjectProfileResult>("/v1/projects/profile", {
    method: "POST",
    body: JSON.stringify({ project_dir: projectDir, start, end, samples }),
  });
}

export function buildProjectMesh(
  projectDir: string,
  maxFinestSamples = 512,
  lodLevels = 4,
): Promise<ProjectMeshReport> {
  return coreFetch<ProjectMeshReport>("/v1/projects/mesh", {
    method: "POST",
    body: JSON.stringify({
      project_dir: projectDir,
      max_finest_samples: maxFinestSamples,
      lod_levels: lodLevels,
    }),
  });
}

export function getProjectMesh(projectDir: string): Promise<ProjectMeshReport> {
  const query = new URLSearchParams({ project_dir: projectDir });
  return coreFetch<ProjectMeshReport>(`/v1/projects/mesh?${query.toString()}`);
}

export async function getProjectMeshUrl(projectDir: string, level = 0): Promise<string> {
  const query = new URLSearchParams({ project_dir: projectDir });
  const response = await checkedResponse(`/v1/projects/mesh/lod/${level}?${query.toString()}`);
  return URL.createObjectURL(await response.blob());
}

export async function getProjectPreviewUrl(
  projectDir: string,
  layer: ProjectPreviewLayer,
  maxSide = 1600,
): Promise<string> {
  const query = new URLSearchParams({
    project_dir: projectDir,
    layer,
    max_side: String(maxSide),
  });
  const response = await checkedResponse(`/v1/projects/preview?${query.toString()}`);
  return URL.createObjectURL(await response.blob());
}

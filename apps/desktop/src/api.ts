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

async function coreFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const config = runtime();
  const headers = new Headers(init?.headers);
  if (init?.body !== undefined && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (config.sessionToken) {
    headers.set("x-depthwizard-token", config.sessionToken);
  }
  const response = await fetch(`${config.apiBase ?? "http://127.0.0.1:8765"}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `DepthWizard core returned HTTP ${response.status}`);
  }
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

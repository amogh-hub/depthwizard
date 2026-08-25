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

export async function inspectRaster(path: string): Promise<RasterMetadata> {
  const config = runtime();
  const response = await fetch(`${config.apiBase ?? "http://127.0.0.1:8765"}/v1/inspect`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(config.sessionToken ? { "x-depthwizard-token": config.sessionToken } : {}),
    },
    body: JSON.stringify({ path }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `DepthWizard core returned HTTP ${response.status}`);
  }
  return response.json() as Promise<RasterMetadata>;
}

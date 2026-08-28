import {
  staleRuntimeMessage,
  workstationApiProbePasses,
} from "./workspace/runtimeContract";

export type StandaloneRuntimeConfig = {
  apiBase: string;
  sessionToken: string;
  sidecarPid: number;
  offlineCore: boolean;
};

function isTauriRuntime(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

function validateRuntimeConfig(config: StandaloneRuntimeConfig): StandaloneRuntimeConfig {
  const endpoint = new URL(config.apiBase);
  if (endpoint.protocol !== "http:" || endpoint.hostname !== "127.0.0.1") {
    throw new Error("DepthWizard standalone core must use a 127.0.0.1 loopback endpoint.");
  }
  const port = Number(endpoint.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("DepthWizard standalone core returned an invalid loopback port.");
  }
  if (!/^[0-9a-f]{64}$/i.test(config.sessionToken)) {
    throw new Error("DepthWizard standalone core returned an invalid session token.");
  }
  if (!Number.isInteger(config.sidecarPid) || config.sidecarPid <= 0) {
    throw new Error("DepthWizard standalone core returned an invalid process identifier.");
  }
  if (!config.offlineCore) {
    throw new Error("DepthWizard packaged core must run in offline-after-install mode.");
  }
  return config;
}

async function readDetail(response: Response): Promise<unknown> {
  try {
    const payload = await response.json() as { detail?: unknown };
    return payload.detail;
  } catch {
    return undefined;
  }
}

async function assertWorkstationApiContract(config: StandaloneRuntimeConfig): Promise<void> {
  const headers = { "x-depthwizard-token": config.sessionToken };
  const projectDir = "__depthwizard_workstation_contract_probe__";
  const probes = [
    {
      name: "project preview",
      path: `/v1/projects/preview?project_dir=${encodeURIComponent(projectDir)}&layer=optical&max_side=64`,
    },
    {
      name: "project legend",
      path: `/v1/projects/preview/legend?project_dir=${encodeURIComponent(projectDir)}&layer=dsm`,
    },
  ];

  for (const probe of probes) {
    let response: Response;
    try {
      response = await fetch(`${config.apiBase}${probe.path}`, {
        method: "GET",
        headers,
        cache: "no-store",
      });
    } catch (error) {
      throw new Error(`Unable to verify packaged scientific runtime compatibility at ${probe.name}: ${error instanceof Error ? error.message : String(error)}`);
    }
    const detail = await readDetail(response);
    if (!workstationApiProbePasses(response.status, detail)) {
      throw new Error(staleRuntimeMessage(probe.name, response.status, detail));
    }
  }
}

export async function bootstrapStandaloneRuntime(): Promise<StandaloneRuntimeConfig | null> {
  if (!isTauriRuntime()) {
    return null;
  }
  const { invoke } = await import("@tauri-apps/api/core");
  const config = validateRuntimeConfig(
    await invoke<StandaloneRuntimeConfig>("runtime_config"),
  );
  const runtimeWindow = window as typeof window & {
    __DEPTHWIZARD_RUNTIME__?: { apiBase?: string; sessionToken?: string };
  };
  runtimeWindow.__DEPTHWIZARD_RUNTIME__ = {
    apiBase: config.apiBase,
    sessionToken: config.sessionToken,
  };
  await assertWorkstationApiContract(config);
  return config;
}

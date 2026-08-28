export const WORKSTATION_API_PROBE_DETAIL = "project manifest does not exist";

export function workstationApiProbePasses(status: number, detail: unknown): boolean {
  return status === 404 && detail === WORKSTATION_API_PROBE_DETAIL;
}

export function staleRuntimeMessage(route: string, status: number, detail: unknown): string {
  const rendered = typeof detail === "string" ? detail : "no compatible detail";
  return `Packaged scientific runtime is incompatible with this desktop build (${route}: HTTP ${status}, ${rendered}). Rebuild the exact-head DepthWizard sidecar before packaging the app.`;
}

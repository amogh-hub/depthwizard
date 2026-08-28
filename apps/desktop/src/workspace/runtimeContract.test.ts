import { describe, expect, it } from "vitest";
import {
  WORKSTATION_API_PROBE_DETAIL,
  staleRuntimeMessage,
  workstationApiProbePasses,
} from "./runtimeContract";

describe("packaged workstation API contract", () => {
  it("accepts the intentional manifest-missing response from the workstation routes", () => {
    expect(workstationApiProbePasses(404, WORKSTATION_API_PROBE_DETAIL)).toBe(true);
  });

  it("rejects FastAPI's generic Not Found response from a stale sidecar", () => {
    expect(workstationApiProbePasses(404, "Not Found")).toBe(false);
  });

  it("rejects unexpected success or server responses", () => {
    expect(workstationApiProbePasses(200, WORKSTATION_API_PROBE_DETAIL)).toBe(false);
    expect(workstationApiProbePasses(500, "error")).toBe(false);
  });

  it("produces an actionable stale-runtime diagnostic", () => {
    expect(staleRuntimeMessage("preview", 404, "Not Found")).toContain("Rebuild the exact-head DepthWizard sidecar");
  });
});

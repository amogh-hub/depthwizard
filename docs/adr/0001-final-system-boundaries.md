# ADR 0001 — Final-system boundaries

**Status:** Accepted  
**Scope:** SIH26175 DepthWizard

## Decision

DepthWizard is a permanent monorepo containing:

- `src/depthwizard`: scientific/geospatial core and local service.
- `apps/desktop`: Tauri + React + Three.js standalone analyst workstation.
- `data`: dataset registry/manifests only; licensed datasets remain external.
- `docs`: architecture, traceability, validation and user/deployment documentation.

The scientific service boundary is:

`ingest -> preprocess -> geometry prior -> remote-sensing refinement -> metric calibration -> geospatial export -> validation -> mesh assets`

The desktop application consumes only explicit versioned contracts from the local core. Scientific outputs are files with provenance, not opaque in-memory UI state.

## Rationale

This structure permits the ML, geospatial, evaluation and UI lanes to proceed in parallel without creating disposable product versions. It also makes the final standalone package reproducible and testable.

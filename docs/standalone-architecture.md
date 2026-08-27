# DepthWizard standalone architecture

## Purpose

Release Train 5 turns DepthWizard from a desktop frontend that expects a separately started Python service into a true no-terminal scientific workstation. The Tauri shell owns the local scientific runtime lifecycle and exposes only a session-scoped loopback endpoint to the webview.

## Runtime lifecycle

1. Tauri reserves an ephemeral `127.0.0.1` port.
2. Tauri generates a fresh 256-bit random session token.
3. Tauri resolves the qualified scientific runtime from its bundled resource tree.
4. Tauri launches `depthwizard-core` directly as an owned child process, passing the token through the child environment rather than command-line arguments.
5. The Python sidecar validates that packaged mode is loopback-only and that a session token is present.
6. Tauri polls the minimal `/health` endpoint until the core is ready or the startup timeout expires.
7. The React webview obtains `{apiBase, sessionToken}` through a Tauri IPC command before rendering the main application.
8. Every scientific API request carries `x-depthwizard-token`; missing or incorrect tokens are rejected.
9. Tauri terminates and reaps the owned scientific child when the application exits.

The child is launched directly with Rust `std::process::Command`; a general-purpose shell is not required for the production scientific runtime.

## Frozen-runtime packaging policy

DepthWizard uses **PyInstaller onedir**, not onefile, for the scientific runtime. This is deliberate. The runtime contains PyTorch, DA3, Rasterio/GDAL, PROJ, SciPy and other native scientific dependencies. A onefile executable must extract that native environment on every launch and makes bootstrap failures opaque. The onedir runtime stays directly inspectable and is embedded inside the final Tauri `.app` as a resource tree, so the user still launches one normal desktop application.

The staged runtime lives at:

`apps/desktop/src-tauri/resources/depthwizard-core-runtime/`

The directory is generated, ignored by Git, and bundled by Tauri into:

`Contents/Resources/depthwizard-core-runtime/`

PyInstaller symbolic links are preserved when the runtime tree is staged. The build report records a deterministic tree SHA-256, regular-file count, symlink count, logical bytes, executable SHA-256 and the target triple.

## Build-time frozen-runtime qualification

`make sidecar-build` does not declare success merely because PyInstaller produced files. After staging the exact runtime tree that Tauri will bundle, the build script launches that frozen executable with `--self-check` under offline environment flags.

The self-check must successfully:

- reach the frozen Python entrypoint;
- import `rasterio.serde`;
- initialize Rasterio/GDAL;
- initialize PyProj/PROJ;
- resolve EPSG:32643 through PyProj;
- write and reopen an in-memory GeoTIFF with EPSG:32643 through Rasterio/GDAL;
- complete without model loading or network use.

A build that times out, crashes, cannot resolve bundled geospatial data, or produces malformed evidence is rejected. Only a passing runtime receives `QUALIFIED_DEPTHWIZARD_CORE_RUNTIME` in its bundled `runtime-manifest.json` and `PASS_QUALIFIED_SIDECAR_BUILD` in the external build report.

This closes the failure class where a frozen executable was previously reported as built even though its packaged geospatial runtime could not become healthy.

## Startup diagnostics

When `DEPTHWIZARD_STARTUP_TRACE=<path>` is explicitly set for acceptance or diagnostics, the scientific core emits machine-readable JSONL startup phases such as:

- `python_entry`
- `arguments_parsed`
- `launch_environment_validated`
- `offline_network_guard_installed`
- `uvicorn_import_complete`
- `service_import_start`
- `service_import_complete`
- `server_start`

The trace never records the session token. Normal launches do not write a trace. Acceptance failures include the last observed startup phase instead of returning an opaque connection-refused timeout.

## Offline-after-install contract

Packaged launches set:

- `DEPTHWIZARD_OFFLINE_CORE=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

When `DEPTHWIZARD_OFFLINE_CORE=1`, the sidecar also installs a process-wide Python INET connection guard. IPv4/IPv6 connections are permitted only to explicit loopback destinations (`127.0.0.0/8`, `::1`, or `localhost`). Other hostnames are not DNS-resolved and are rejected before connection. This is an application-layer egress barrier in addition to the Hugging Face/Transformers offline flags; it is not presented as an operating-system firewall.

The final core uses model assets already installed/cached on the machine. It must not silently download model weights during an offline judging run. Dedicated RT5 acceptance must still prove that DA3 inference succeeds in this mode on the finale Mac with the required assets present.

## Acceptance hooks

Normal production launches do not write the session token anywhere. For deterministic bundle-lifecycle acceptance only, the Tauri shell honors two environment variables:

- `DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT=<path>` writes a non-secret boot report after the sidecar is healthy. The report records the loopback endpoint, sidecar PID, offline state and token bit length, but never the token value.
- `DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS=<500..60000>` requests a graceful app exit after the specified delay so the acceptance runner can prove that the owned sidecar terminates with the desktop process.

These variables are unused during ordinary application launches.

## Evidence boundaries

`make release-train-5-sidecar-smoke` validates the **already-qualified** frozen runtime's process/security contract. It refuses to run if the staged executable no longer matches the qualified build report. It checks loopback readiness, startup phases, missing-token rejection, wrong-token rejection, authorized raster inspection, forced offline environment, strict Python non-loopback egress policy and clean process termination.

`make release-train-5-app-smoke` is macOS-only and validates the real packaged `.app`: bundle/executable presence, the exact qualified onedir runtime resource, the carried frozen self-check manifest, Tauri-owned startup, loopback readiness, non-export of the session token, offline/egress policy, no user-visible terminal requirement, and sidecar termination on graceful desktop exit.

`make release-train-5-standalone-acceptance` runs both RT5 lifecycle/security smokes after `make standalone-build`.

Those smokes are **not** sufficient to close standalone Gate G. Final RT5 closure additionally requires evidence for:

- offline DA3 reconstruction after model assets have been installed;
- fresh-image process → validate → export from clean application launch;
- malformed-input and runtime-failure recovery behavior;
- final-laptop performance evidence;
- long-running stability/soak evidence;
- a clean-machine installation/reproducibility run.

No standalone acceptance may be presented as DSM-accuracy, terrain-generalization, cross-sensor, or learned-model-promotion evidence.

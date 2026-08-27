# DepthWizard standalone architecture

## Purpose

Release Train 5 turns DepthWizard from a desktop frontend that expects a separately started Python service into a true no-terminal scientific workstation. The Tauri shell owns the local scientific sidecar lifecycle and exposes only a session-scoped loopback endpoint to the webview.

## Runtime lifecycle

1. Tauri reserves an ephemeral `127.0.0.1` port.
2. Tauri generates a fresh 256-bit random session token.
3. Tauri launches the packaged `depthwizard-core` external binary with the port and offline/security policy in its environment.
4. The Python sidecar validates that packaged mode is loopback-only and that a session token is present.
5. Tauri polls the minimal `/health` endpoint until the core is ready or the startup timeout expires.
6. The React webview obtains `{apiBase, sessionToken}` through a Tauri IPC command before rendering the main application.
7. Every scientific API request carries `x-depthwizard-token`; missing or incorrect tokens are rejected.
8. Tauri terminates the owned sidecar when the application exits.

The token is deliberately passed through the child-process environment rather than command-line arguments so it is not exposed through ordinary process-list argument inspection.

## Offline-after-install contract

Packaged launches set:

- `DEPTHWIZARD_OFFLINE_CORE=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

When `DEPTHWIZARD_OFFLINE_CORE=1`, the sidecar also installs a process-wide Python INET connection guard. IPv4/IPv6 connections are permitted only to explicit loopback destinations (`127.0.0.0/8`, `::1`, or `localhost`). Other hostnames are not DNS-resolved and are rejected before connection. This is an application-layer egress barrier in addition to the Hugging Face/Transformers offline flags; it is not presented as an operating-system firewall.

The final core is expected to use model assets already installed/cached on the machine. It must not silently download model weights during an offline judging run. A dedicated RT5 acceptance must still prove that DA3 inference succeeds in this mode on a machine with the required assets present.

## Packaging

`scripts/build_standalone_sidecar.py` builds a one-file Python scientific sidecar with PyInstaller and copies it to Tauri's target-triple external-binary location:

`apps/desktop/src-tauri/binaries/depthwizard-core-<target-triple>`

The build includes DepthWizard source, the pinned Depth Anything 3 vendor source, and explicit dynamic-import handling. Generated sidecar binaries are build artifacts and are not committed to Git.

`make standalone-build` builds the sidecar, frontend, and Tauri application bundle. Packaging success alone is not clean-machine acceptance.

## Acceptance hooks

Normal production launches do not write the session token anywhere. For deterministic bundle-lifecycle acceptance only, the Tauri shell honors two environment variables:

- `DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT=<path>` writes a non-secret boot report after the sidecar is healthy. The report records the loopback endpoint, sidecar PID, offline state and token bit length, but never the token value.
- `DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS=<500..60000>` requests a graceful app exit after the specified delay so the acceptance runner can prove that the owned sidecar terminates with the desktop process.

These variables are unused during ordinary application launches.

## Evidence boundaries

`make release-train-5-sidecar-smoke` validates the packaged sidecar's process/security contract without rerunning reconstruction or any consumed scientific benchmark. It checks loopback readiness, missing-token rejection, wrong-token rejection, authorized raster inspection, forced offline environment, strict Python non-loopback egress policy and clean process termination.

`make release-train-5-app-smoke` is macOS-only and validates the real packaged `.app`: bundle/executable presence, packaged external sidecar, Tauri-owned startup, loopback readiness, non-export of the session token, offline/egress policy, no user-visible terminal requirement, and sidecar termination on graceful desktop exit.

`make release-train-5-standalone-acceptance` runs both RT5 lifecycle/security smokes after `make standalone-build`.

Those smokes are **not** sufficient to close standalone Gate G. Final RT5 closure additionally requires evidence for:

- offline DA3 reconstruction after model assets have been installed;
- fresh-image process → validate → export from clean application launch;
- malformed-input and runtime-failure recovery behavior;
- final-laptop performance evidence;
- long-running stability/soak evidence;
- a clean-machine installation/reproducibility run.

No standalone acceptance may be presented as DSM-accuracy, terrain-generalization, cross-sensor, or learned-model-promotion evidence.

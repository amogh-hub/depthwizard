# Commit-bound CI evidence

This directory preserves the GitHub Actions evidence behind the presentation claims
`418 primary tests passed` and `Core CI jobs: 8/8 passed`.

## Evidence identity

- Qualified source commit: `012301b9c1910ef4ccde5b3da5d4e4d94da60ce6`
- Release tag: `v0.2.0-sih-final`
- Workflow: `depthwizard-ci`
- Workflow run: [34708065916](https://github.com/amogh-hub/depthwizard/actions/runs/34708065916)
- Run number: `861`
- Event: `push` to `main`
- Started: `2026-09-12T17:23:03Z`
- Completed: `2026-09-12T17:26:22Z`
- Conclusion: `success`

The official GitHub Actions log bundle is retained as
`depthwizard-ci-34708065916-logs.zip`. Verify it with `SHA256SUMS` before use.

## Primary test total

| Primary suite | Passed | Counting rule |
| --- | ---: | --- |
| Python core | 379 | The complete Python suite passed on both Python 3.12 and 3.13; count the suite once. |
| Desktop frontend | 33 | Eight frontend test files passed. |
| Desktop Rust | 6 | Six Rust tests passed; the second Rust test binary contained zero tests. |
| **Total** | **418** | `379 + 33 + 6` |

The portability contract selected 41 Python tests and ran them successfully on both `macos-15` and
`windows-2025`. Those tests are a selected subset of the complete 379-test Python suite, so they are
reported separately and are not added to the 418 total.

## Eight successful core CI jobs

1. `dependency lock integrity`
2. `python-core (3.12)`
3. `python-core (3.13)`
4. `portable geospatial (macos-15)`
5. `portable geospatial (windows-2025)`
6. `desktop-frontend`
7. `desktop-rust`
8. `SBOM and dependency license inventory`

All eight jobs completed successfully in run 34708065916.

## Claim boundary

The defensible presentation wording is:

- `418 primary tests passed`
- `Core CI jobs: 8/8 passed`

The second statement applies only to the `depthwizard-ci` workflow above. A separate
[`depthwizard-release` run 34709363538](https://github.com/amogh-hub/depthwizard/actions/runs/34709363538)
for the same source commit concluded with `failure`; therefore, the broader wording `all GitHub
checks passed` is not supported.


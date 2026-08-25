# ADR 0003 — Scientific workstation UI system

**Status:** Accepted

## Decision

DepthWizard uses a restrained scientific workstation layout rather than a generic AI dashboard.

Permanent layout:

- 56 px top application bar.
- narrow left tool rail.
- dominant central scientific canvas.
- contextual right inspector.
- compact bottom status line.

Permanent visual rules:

- neutral/near-white application chrome;
- restrained blue primary accent;
- semantic color reserved for status and scientific data;
- fixed spacing/radius/type tokens;
- no decorative gradients, oversized cards, neon/cyber styling, glass-heavy effects or ornamental AI motifs;
- official metrics and analytical state are visually prioritized over decoration.

The 3D renderer supports orbit, fly, first-person and top-down camera modes using real mesh assets. Empty states never render fake terrain or fake metrics.

# Third-party evaluation baselines

DepthWizard keeps external research baselines isolated from the production estimator. They are
used only for reproducible comparison and are not represented as DepthWizard-authored models.

## RDAH-Net

- Upstream: `Elenairene/RDAH-Net`
- Pinned source commit: `373bca28299683ab0e5d892dfc87598ab967b564`
- Paper: Jiang et al., *Remote Sensing* 18(7), 1024 (2026), DOI `10.3390/rs18071024`
- Upstream license: MIT
- Published checkpoint/data record: Figshare `10.6084/m9.figshare.31986864`
- Depth prior expected by the paper: frozen Depth Anything v2

DepthWizard ports only the released inference graph required to load the authors' checkpoint.
The legacy RDAH-Net requirements are deliberately not installed into the main DepthWizard
virtual environment.

## Depth Anything v2

- Upstream: `DepthAnything/Depth-Anything-V2`
- Pinned source commit: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`
- Baseline configuration: official V2 Small (`vits`) relative-depth checkpoint

The V2 prior is used only to reproduce the input modality expected by RDAH-Net. DepthWizard's
production geometry prior remains the separately pinned DA3MONO-LARGE adapter.

## Comparison semantics

RDAH-Net is trained to predict nDSM-like surface height, while the OrthoLoC demonstration pair
contains a DSM. Therefore the OrthoLoC RDAH comparison is explicitly a cross-domain *geometry*
comparison after the same sparse affine calibration protocol used for the DA3 baseline. Raw RDAH
output is not presented as an independent absolute-DSM accuracy claim on OrthoLoC.

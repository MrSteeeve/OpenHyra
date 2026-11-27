# Published artifacts

- `feedback-ablation-20260904/`: synthetic four-arm feedback-loop protocol ablation.
- `mathai-bermudan-5r-codex-20260903/`: five-round bounded continuation-artifact pilot with public and hidden-audit records.
- `seed-sweep-4families-20x-20260903/`: four-family, 20-seed-block robustness sweep over those frozen bounded artifacts.

Each bundle contains its own `EXPERIMENT_SUMMARY.md` with the applicable evidence boundaries and interpretation limits.

## Limit order book world-model evidence

The current branch retains the immutable [`p2-m1-validation/`](p2-m1-validation/) and
[`p2-m2-synthetic/`](p2-m2-synthetic/) evidence bundles. Verify the original `SHA256SUMS` in the synthetic bundle.
The earlier validation bundle has no checksum file; its immutable Git tree is
`9e3a28dc846f90fe6ce53638ac1cdfe5b918b923`. The earlier P0 and P1 source, manifests
and evidence are preserved at the annotated Git tags [`tier0-p0-final`](https://github.com/MrSteeeve/OpenHyra/tree/tier0-p0-final)
and [`p1-szse-final`](https://github.com/MrSteeeve/OpenHyra/tree/p1-szse-final), respectively.
The historical Bermudan evidence remains on this branch.

## Publication provenance

The bundles retain their original runtime base commit, `4ad87556c2155bcda4120b76448c5393df9f75a3`.
Before publication, that local commit's message metadata was normalized as
`1c218e13528916f750551e3dcc8dd11de4fe61cb`; both commits have the identical
Git tree `5dd270dda37520524d130a756899944cdf4f91fe`. The recorded experiment inputs
and outputs were not regenerated or rewritten as part of that normalization.

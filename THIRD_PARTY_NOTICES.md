# License Scope and Third-Party Notices

The MIT license in LICENSE applies to authorized original project source code and associated documentation. It does not relicense third-party fonts, imported libraries, or assets excluded from this repository. No raw simulation fields, simulator setup files, executables, weights or prediction caches are included or licensed here.

## Architecture Attribution

The Fourier operator and DeepONet implement published architecture ideas. They are project-specific implementations, not the official NeuralOperator, DeepXDE or original DeepONet software packages. Those packages are not bundled or imported by this snapshot.

- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021: https://openreview.net/forum?id=c8P9NQVtmnO.
- Lu et al., *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*, Nature Machine Intelligence 3, 218-229 (2021): https://doi.org/10.1038/s42256-021-00302-5.

AI assistance with source development and visualization is disclosed in the associated manuscript. Method attribution does not establish independent authorship or replace a source-code license. If an identifiable upstream adaptation is established, its actual copyright and license terms must be retained; the project MIT license does not supersede them.

## Installed Dependencies

Dependency sources and binaries are not bundled. Their terms apply independently when installed or redistributed. These links identify the direct versions used for local tests, not every transitive dependency:

| Dependency | Local tested version | Official license |
|---|---|---|
| NumPy | 1.26.4 | https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt |
| SciPy | 1.13.1 | https://github.com/scipy/scipy/blob/v1.13.1/LICENSE.txt |
| scikit-image | 0.24.0 | https://github.com/scikit-image/scikit-image/blob/v0.24.0/LICENSE.txt |
| Matplotlib | 3.9.4 | https://github.com/matplotlib/matplotlib/blob/v3.9.4/LICENSE/LICENSE |
| PyTorch | 2.1.0 | https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE |
| Pillow | 11.1.0 | https://github.com/python-pillow/Pillow/blob/11.1.0/LICENSE |

The reported GPU quality-inference runtime used PyTorch 2.1.2/CUDA 11.8 and is distinct from this local test stack. No scikit-learn import is used in the shipped source.

## Embedded Figure Fonts

Quantitative PDF figures contain DejaVu Sans subsets. Their DejaVu, Bitstream and Arev terms are retained verbatim in [DejaVu-fonts.txt](DejaVu-fonts.txt). These font components are not relicensed as MIT. No standalone font binaries are distributed.

The workflow PDF contains Arial subsets for document embedding, not extraction or redistribution as font software. Arial font-family references in SVG files do not bundle font binaries. Document embedding does not grant a font-software redistribution license. See [Microsoft's font embedding guidance](https://learn.microsoft.com/en-us/typography/fonts/font-faq).

## Availability

The canonical repository address is https://github.com/loc-k-nguyen/groundwater-surrogate-reliability. The manuscript-associated snapshot is tag `ems-v5-2026-10-10`. No archival DOI or journal submission is asserted. Restricted-asset access remains subject to its rights holders' approval.

"""TSC — Time Series Classification with MiniROCKET + JM-KMeans.

This package collects the algorithms, feature selectors, and pipeline
scripts used by the thesis.

Subpackages
-----------
- `TSC.models`         — MiniROCKET, ROCKET, GA, VAE
- `TSC.comaprisons`    — baseline feature selectors (Fisher, mRMR, ReliefF, DM)
- `TSC.detach_rocket`  — DetachROCKET (Uribarri & Barone)
- `TSC.diffusionMaps`  — diffusion-maps feature ranking
- `TSC.MMD`            — maximum mean discrepancy experiments
- `TSC.utils`          — JM distance, K-Means, dataset IO, scoring
- `TSC.visualizations` — JM/MMD figure scripts
- `TSC.har_datasets`   — HAR stratified cross-validation pipeline
"""

__version__ = "0.1.0"

#!/bin/bash
# One-time: Python environment for the static-FC tests and scripts
# (SciPy >= 1.16 is needed for Welch's ANOVA in scipy.stats.f_oneway).
# Run once on a login node or in an interactive session:  bash hipergator_tests/setup_python_env.sh
set -euo pipefail
ENV=${ENV:-/blue/febo/$USER/envs/fc_static}
module purge
module load conda
MAMBA=$(command -v mamba >/dev/null && echo mamba || echo conda)
$MAMBA create -y -p "$ENV" -c conda-forge python=3.12 "scipy>=1.16" numpy pandas \
    statsmodels networkx matplotlib seaborn python-docx
conda run -p "$ENV" pip install bctpy==0.6.1
conda run -p "$ENV" python -c "import scipy, statsmodels, networkx, bct, docx; \
print('scipy', scipy.__version__, '| statsmodels', statsmodels.__version__, \
'| networkx', networkx.__version__, '| bctpy ok')"
echo "Environment ready: $ENV"

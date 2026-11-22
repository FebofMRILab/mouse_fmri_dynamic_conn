#!/usr/bin/env bash
# Renders the complex-PCA figures of the paper (Figures 7 and 8, Supplementary Figure S6, and the
# optional stand-alone traveling-index figure) from the tables in results/.
# Needs Python 3 with numpy, pandas, scipy, matplotlib and seaborn.
# Run from anywhere:  bash make_figures.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p figures
G=results/cpca_group_lags
COORDS=atlas/brainnet_node_coords_highres.csv
MESH=atlas/civm.g

echo "Figure 7: loadings and traveling index by group"
python3 python/plot_cpca_patterns.py --in_dir "$G" --out_dir figures --formats png tiff pdf

echo "Supplementary Figure S6: component brain maps (node files, then the maps)"
python3 python/cpca_brainnet_files.py --loadings results/cpca_loadings_roi.csv \
    --node_template "$COORDS" --out_dir results/cpca_brainnet
python3 python/plot_cpca_brain_maps.py --mesh "$MESH" --bn_dir results/cpca_brainnet \
    --out figures/Fig_cpca_brain_maps.png
python3 python/plot_cpca_brain_maps.py --mesh "$MESH" --bn_dir results/cpca_brainnet \
    --out figures/Fig_cpca_brain_maps.tif --dpi 600

echo "Figure 8: group phase delay maps"
python3 python/plot_cpca_group_phase_maps.py --loadings "$G/cpca_group_loadings.csv" \
    --mesh "$MESH" --node_template "$COORDS" --out figures/Fig_cpca_group_phase_maps.png
python3 python/plot_cpca_group_phase_maps.py --loadings "$G/cpca_group_loadings.csv" \
    --mesh "$MESH" --node_template "$COORDS" --out figures/Fig_cpca_group_phase_maps.tif --dpi 600

echo "Optional: traveling index on its own"
python3 python/plot_cpca_traveling_index.py --in_dir "$G" --out_dir figures --formats png

echo "Done. Figures in $(pwd)/figures"

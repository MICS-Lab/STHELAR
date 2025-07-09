#!/usr/bin/env bash
#
##### Usage #####
# Only transfer missing or updated files:
#   ./src/_6_analyze_trained_model/_4_get_output.sh training_27
# Force rsync to overwrite everything:
#   ./src/_6_analyze_trained_model/_4_get_output.sh training_27 overwrite
#
# Notes
# -----
#   •  “default” mode copies only new / newer files (rsync --update).
#   •  “overwrite” mode forces every file to be overwritten (rsync --ignore-times).
#   •  The *checkpoints* directory (or any symlink called *checkpoints*) and
#      everything inside it are excluded from transfer.

set -eu                                # stop on first error or unset variable
trap 'echo "Error on line $LINENO"' ERR

# ── 1 – Argument check ───────────────────────────────────────────────────────────
if (( $# < 1 )); then
  echo "Usage: $0 <training_id> [mode]"
  echo "mode: 'default' (only new/updated files) or 'overwrite' (force overwrite all files)"
  exit 1
fi

training_id=$1
mode=${2:-default}                     # default mode = copy only new / updated

# ── 2 – Slide list ──────────────────────────────────────────────
# breast_s0 breast_s1 breast_s3 breast_s6 lung_s1 lung_s3 skin_s1 skin_s2 skin_s3 skin_s4 pancreatic_s0 pancreatic_s1 pancreatic_s2 heart_s0 colon_s1 colon_s2 kidney_s0 kidney_s1 liver_s0 liver_s1 tonsil_s0 tonsil_s1 lymph_node_s0 ovary_s0 ovary_s1 prostate_s0 cervix_s0

slide_ids=(
  breast_s0 breast_s1 breast_s3 breast_s6
  lung_s1 lung_s3
  skin_s1 skin_s2 skin_s3 skin_s4
  pancreatic_s0 pancreatic_s1 pancreatic_s2
  heart_s0
  colon_s1 colon_s2
  kidney_s0 kidney_s1
  liver_s0 liver_s1
  tonsil_s0 tonsil_s1
  lymph_node_s0
  ovary_s0 ovary_s1
  prostate_s0
  cervix_s0
)

# ── 3 – Base rsync options ───────────────────────────────────────────────────────
#   -a  : archive mode (preserves permissions, times, symlinks, etc.)
#   --partial --progress : resume support + progress bar
#   --exclude rules      : skip checkpoints dir *and* its contents
#   --prune-empty-dirs   : remove any empty directories that may result
rsync_opts=(
  --partial
  --progress
  -a
  --exclude='checkpoints'          # skip dir or symlink named “checkpoints”
  --exclude='checkpoints/**'       # skip everything inside such a directory
  --prune-empty-dirs
)

if [[ $mode == overwrite ]]; then
  echo "Mode: overwrite – existing files will be replaced"
  rsync_opts+=( --ignore-times )
else
  echo "Mode: default – only new or updated files will be copied"
  rsync_opts+=( --update )
fi

# ── 4 – Loop over slides ─────────────────────────────────────────────────────────
for slide_id in "${slide_ids[@]}"; do
  echo "→ Processing slide: $slide_id"

  src="jeanzay:/lustre/fsstor/projects/rech/user/ubu16ws/HE2CellType/HE2CT/inferences/${training_id}/${slide_id}/"
  dest="/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/analyze_trained_model/${training_id}/output_model/${slide_id}/"

  # ensure destination directory exists
  mkdir -p "$dest"

  rsync "${rsync_opts[@]}" "$src" "$dest"
done

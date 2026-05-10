#!/usr/bin/env bash
#
# download_data.sh — selectively fetch the BBH_stars datasets.
#
# Run this from inside an existing clone of the repository. It uses
# `git sparse-checkout` for plain-git data and `git lfs pull --include` for
# the LFS-tracked `.bin` harmonics files.
#
# For the smallest possible clone, see the README ("Partial download" section)
# for the recommended `git clone --filter=blob:none` + GIT_LFS_SKIP_SMUDGE=1
# invocation.
#
# Usage:
#   ./download_data.sh                  # interactive menu
#   ./download_data.sh code-only        # no data
#   ./download_data.sh small            # small datasets only (~95 MB)
#   ./download_data.sh V0               # + data-V=0-varying-Tmax (~2.4 GB)
#   ./download_data.sh Vneq0            # + data-Vneq0 LFS files (~3.5 GB)
#   ./download_data.sh all              # everything (~6 GB)
#   ./download_data.sh Vneq0 q=0.001    # only Vneq0 .bin files matching q=0.001

set -euo pipefail

cd "$(dirname "$0")"

DIR_SMALL=(
    "3-body/3-body-code"
    "3-body/binary-evolution"
    "3-body/analysis-scripts"
    "N-body"
    "3-body/3-body-data/convergence-tests"
    "3-body/3-body-data/data-V=0"
    "3-body/3-body-data/data-Vneq0"
    "3-body/3-body-data/stopping-condition-3"
)
DIR_VARTMAX="3-body/3-body-data/data-V=0-varying-Tmax"
DIR_VNEQ0="3-body/3-body-data/data-Vneq0"

apply_sparse() {
    git sparse-checkout init --no-cone >/dev/null
    git sparse-checkout set "$@"
    git checkout HEAD -- . 2>/dev/null || true
}

set_code_only()  { apply_sparse '/*' '!3-body/3-body-data'; }
set_small()      { apply_sparse '/*' "!$DIR_VARTMAX"; }
set_with_v0var() { apply_sparse '/*'; }

pull_vneq0() {
    local pattern="${1:-*}"
    echo "Pulling LFS files in $DIR_VNEQ0 matching: $pattern"
    git lfs pull --include="$DIR_VNEQ0/$pattern"
}

mode="${1:-}"
case "$mode" in
    code-only)
        set_code_only
        echo "Sparse-checkout: code only (data directories excluded)."
        ;;
    small)
        set_small
        echo "Sparse-checkout: code + small datasets (data-V=0-varying-Tmax excluded)."
        ;;
    V0)
        set_with_v0var
        echo "Sparse-checkout: includes data-V=0-varying-Tmax (~2.4 GB)."
        ;;
    Vneq0)
        set_small
        pull_vneq0 "${2:-*}"
        ;;
    all)
        set_with_v0var
        pull_vneq0 "*"
        ;;
    "")
        cat <<'EOF'
Pick what to download:
  1) code-only   — scripts only, no data
  2) small       — code + small datasets (~95 MB)
  3) V0          — small + data-V=0-varying-Tmax  (+~2.4 GB)
  4) Vneq0       — small + all data-Vneq0 LFS bins (+~3.5 GB)
  5) all         — everything (~6 GB)
EOF
        read -rp "Choice [1-5]: " choice
        case "$choice" in
            1) exec "$0" code-only ;;
            2) exec "$0" small ;;
            3) exec "$0" V0 ;;
            4) exec "$0" Vneq0 ;;
            5) exec "$0" all ;;
            *) echo "Invalid choice." >&2; exit 1 ;;
        esac
        ;;
    *)
        echo "Unknown mode: $mode" >&2
        echo "Run with no args for an interactive menu, or see the script header." >&2
        exit 1
        ;;
esac

echo "Done."

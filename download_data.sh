#!/usr/bin/env bash
#
# download_data.sh — selectively fetch the BBH_stars datasets.
#
# Run this from inside an existing clone of the repository. The big datasets
# are both LFS-tracked, so this script just decides which LFS files to pull.
#
# For the smallest possible clone, see the README ("Partial download" section)
# for the recommended `GIT_LFS_SKIP_SMUDGE=1 git clone ...` invocation.
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

DIR_VARTMAX="3-body/3-body-data/data-V=0-varying-Tmax"
DIR_VNEQ0="3-body/3-body-data/data-Vneq0"

pull_lfs() {
    local label="$1"
    local pattern="$2"
    echo "Pulling LFS files matching: $pattern  ($label)"
    git lfs pull --include="$pattern"
}

mode="${1:-}"
case "$mode" in
    code-only)
        echo "Code-only: nothing to fetch (LFS data left unsmudged)."
        ;;
    small)
        echo "Small: small text files are already in your working tree."
        echo "Skipping the two large LFS datasets."
        ;;
    V0)
        pull_lfs "data-V=0-varying-Tmax" "$DIR_VARTMAX/*"
        ;;
    Vneq0)
        pull_lfs "data-Vneq0" "$DIR_VNEQ0/${2:-*}"
        ;;
    all)
        pull_lfs "data-V=0-varying-Tmax" "$DIR_VARTMAX/*"
        pull_lfs "data-Vneq0"            "$DIR_VNEQ0/*"
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

#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# cluster.sh  –  thin wrapper around SSH/rsync/SLURM for the
#                IAS typhon cluster
# ──────────────────────────────────────────────────────────────

REMOTE_HOST="typhon-login1"
REMOTE_DIR="/home/tomaselli/3-body-scattering"
REMOTE_USER="tomaselli"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILE_CMD="g++ -std=c++17 -fopenmp -O3 -march=native"

usage() {
    cat <<'EOF'
Usage: ./cluster.sh <command> [options]

Commands:
  deploy   <files...>                       Upload files to the cluster
  compile  <file.cpp> [output_name]         Compile on the cluster
  submit   <slurm_file>                     Submit a SLURM job
  run      <file.cpp> [opts] -- <args...>   Deploy, compile, generate SLURM script, submit
  sweep    <file.cpp> <slurm_file>          Deploy .cpp + .slurm, compile, submit array job
  status                                    Show job queue (squeue)
  cancel   <job_id|all>                     Cancel a job or all jobs
  fetch    [pattern] [--dest dir]           Download results to local machine
  logs     <job_id>                         Show stdout/stderr for a job
  clean                                     Remove .out and .err files on the cluster
  ls       [subdir]                         List files in the remote directory
  wait     <job_id>                         Poll until a job finishes

'run' options (before --):
  --time   HH:MM:SS   Walltime          (default: 12:00:00)
  --cpus   N          CPUs per task     (default: 96)
  --output NAME       Executable name   (default: scattering)

Examples:
  ./cluster.sh run main.cpp -- 0.2 0.6
  ./cluster.sh run main-3D-velocity.cpp --time 06:00:00 -- 0.3 0.7 2000 10000 8
  ./cluster.sh fetch '*.txt' --dest ./results/
  ./cluster.sh status
EOF
}

ssh_cmd() { ssh -o ConnectTimeout=10 "$REMOTE_HOST" "$@"; }

# ── deploy ────────────────────────────────────────────────────
cmd_deploy() {
    if [[ $# -eq 0 ]]; then
        echo "Usage: cluster.sh deploy <files...>" >&2; return 1
    fi
    for f in "$@"; do
        local base
        base="$(basename "$f")"
        echo ">> Uploading $base ..."
        rsync -avzh "$f" "${REMOTE_HOST}:${REMOTE_DIR}/${base}"
    done
    echo ">> Deploy complete."
}

# ── compile ───────────────────────────────────────────────────
cmd_compile() {
    local src="${1:?Usage: cluster.sh compile <file.cpp> [output_name]}"
    local out="${2:-scattering}"
    local base
    base="$(basename "$src")"
    echo ">> Compiling $base -> $out on cluster ..."
    ssh_cmd "cd ${REMOTE_DIR} && ${COMPILE_CMD} ${base} -o ${out}"
    echo ">> Compilation successful."
}

# ── submit ────────────────────────────────────────────────────
cmd_submit() {
    local slurm="${1:?Usage: cluster.sh submit <slurm_file>}"
    local base
    base="$(basename "$slurm")"
    echo ">> Submitting $base ..."
    ssh_cmd "cd ${REMOTE_DIR} && sbatch ${base}"
}

# ── run (deploy + compile + generate slurm + submit) ──────────
cmd_run() {
    local src=""
    local walltime="12:00:00"
    local cpus="96"
    local outname="scattering"
    local -a exe_args=()
    local past_separator=false

    while [[ $# -gt 0 ]]; do
        if $past_separator; then
            exe_args+=("$1"); shift; continue
        fi
        case "$1" in
            --)           past_separator=true; shift ;;
            --time)       walltime="${2:?--time requires a value}"; shift 2 ;;
            --cpus)       cpus="${2:?--cpus requires a value}"; shift 2 ;;
            --output)     outname="${2:?--output requires a value}"; shift 2 ;;
            *)
                if [[ -z "$src" ]]; then src="$1"; shift
                else echo "Unexpected argument: $1" >&2; return 1; fi
                ;;
        esac
    done

    if [[ -z "$src" ]]; then
        echo "Usage: cluster.sh run <file.cpp> [--time T] [--cpus N] [--output NAME] -- <args...>" >&2
        return 1
    fi

    local base
    base="$(basename "$src")"

    # 1. Deploy
    cmd_deploy "$src"

    # 2. Compile
    cmd_compile "$src" "$outname"

    # 3. Generate a temporary SLURM script on the cluster
    local args_str="${exe_args[*]}"
    local slurm_name="_run_${outname}.slurm"

    echo ">> Generating SLURM script ($slurm_name) ..."
    ssh_cmd "cat > ${REMOTE_DIR}/${slurm_name}" <<SLURM
#!/bin/bash
#SBATCH --job-name=${outname}
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --nodes=1
#SBATCH --time=${walltime}
#SBATCH --cpus-per-task=${cpus}

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

echo "Running ./${outname} ${args_str} with \$OMP_NUM_THREADS threads"
srun --cpu-bind=cores ./${outname} ${args_str}
SLURM

    # 4. Submit
    echo ">> Submitting job ..."
    ssh_cmd "cd ${REMOTE_DIR} && sbatch ${slurm_name}"
}

# ── sweep (deploy + compile + submit array job) ───────────────
cmd_sweep() {
    local src="${1:?Usage: cluster.sh sweep <file.cpp> <slurm_file>}"
    local slurm="${2:?Usage: cluster.sh sweep <file.cpp> <slurm_file>}"
    local outname="${3:-scattering}"

    # Deploy both files
    cmd_deploy "$src" "$slurm"

    # Compile
    cmd_compile "$src" "$outname"

    # Submit
    cmd_submit "$slurm"
}

# ── status ────────────────────────────────────────────────────
cmd_status() {
    ssh_cmd "squeue -u ${REMOTE_USER}"
}

# ── cancel ────────────────────────────────────────────────────
cmd_cancel() {
    local target="${1:?Usage: cluster.sh cancel <job_id|all>}"
    if [[ "$target" == "all" ]]; then
        echo ">> Cancelling all jobs ..."
        ssh_cmd "scancel -u ${REMOTE_USER}"
    else
        echo ">> Cancelling job $target ..."
        ssh_cmd "scancel $target"
    fi
    echo ">> Done."
}

# ── fetch ─────────────────────────────────────────────────────
cmd_fetch() {
    local pattern=""
    local dest="${LOCAL_DIR}/results/"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dest) dest="${2:?--dest requires a path}"; shift 2 ;;
            *)      pattern="$1"; shift ;;
        esac
    done

    mkdir -p "$dest"

    if [[ -n "$pattern" ]]; then
        echo ">> Fetching '${pattern}' -> ${dest} ..."
        rsync -avz "${REMOTE_HOST}:${REMOTE_DIR}/${pattern}" "$dest"
    else
        echo ">> Fetching *.txt, *.bin, and harmonics_* -> ${dest} ..."
        local tmplist
        tmplist="$(ssh_cmd "cd ${REMOTE_DIR} && ls -1 *.txt *.bin harmonics_* 2>/dev/null || true")"
        if [[ -z "$tmplist" ]]; then
            echo ">> No matching files found on the cluster."
            return 0
        fi
        echo "$tmplist"
        while IFS= read -r fname; do
            [[ -n "$fname" ]] && rsync -avz "${REMOTE_HOST}:${REMOTE_DIR}/${fname}" "$dest"
        done <<< "$tmplist"
    fi
    echo ">> Fetch complete. Results in: ${dest}"
}

# ── logs ──────────────────────────────────────────────────────
cmd_logs() {
    local job_id="${1:?Usage: cluster.sh logs <job_id>}"
    echo "=== STDOUT (out_*${job_id}*) ==="
    ssh_cmd "cat ${REMOTE_DIR}/out_*${job_id}* 2>/dev/null || echo '(no stdout file found)'"
    echo ""
    echo "=== STDERR (err_*${job_id}*) ==="
    ssh_cmd "cat ${REMOTE_DIR}/err_*${job_id}* 2>/dev/null || echo '(no stderr file found)'"
}

# ── clean ─────────────────────────────────────────────────────
cmd_clean() {
    echo ">> Removing *.out and *.err on cluster ..."
    ssh_cmd "rm -f ${REMOTE_DIR}/*.out ${REMOTE_DIR}/*.err"
    echo ">> Clean complete."
}

# ── ls ────────────────────────────────────────────────────────
cmd_ls() {
    local subdir="${1:-}"
    ssh_cmd "ls -lh ${REMOTE_DIR}/${subdir}"
}

# ── wait ──────────────────────────────────────────────────────
cmd_wait() {
    local job_id="${1:?Usage: cluster.sh wait <job_id>}"
    local interval=60
    echo ">> Waiting for job $job_id to finish (polling every ${interval}s) ..."
    while true; do
        local queue
        queue="$(ssh_cmd "squeue -j $job_id -h 2>/dev/null" || true)"
        if [[ -z "$queue" ]]; then
            echo ">> Job $job_id is no longer in the queue."
            break
        fi
        echo "   $(date '+%H:%M:%S') – still running ..."
        sleep "$interval"
    done
}

# ── main dispatcher ──────────────────────────────────────────
if [[ $# -eq 0 ]]; then usage; exit 0; fi

command="$1"; shift
case "$command" in
    deploy)   cmd_deploy "$@" ;;
    compile)  cmd_compile "$@" ;;
    submit)   cmd_submit "$@" ;;
    run)      cmd_run "$@" ;;
    sweep)    cmd_sweep "$@" ;;
    status)   cmd_status "$@" ;;
    cancel)   cmd_cancel "$@" ;;
    fetch)    cmd_fetch "$@" ;;
    logs)     cmd_logs "$@" ;;
    clean)    cmd_clean "$@" ;;
    ls)       cmd_ls "$@" ;;
    wait)     cmd_wait "$@" ;;
    help|-h|--help) usage ;;
    *)        echo "Unknown command: $command" >&2; usage; exit 1 ;;
esac

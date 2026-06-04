#!/bin/bash
# Poll progress of `create_nuscenes_infos --version v1.0-trainval`.
# Usage: bash scripts/check_nuscenes_infos.sh
set -euo pipefail

LOG=/root/autodl-tmp/ASAP/logs/create_nuscenes_infos_trainval.log
ROOT=/root/autodl-tmp/ASAP/data/nuscenes/v1.0-trainval

PID=$(pgrep -f 'nuscenes_dataset.*create_nuscenes_infos.*v1.0-trainval' | head -1 || true)

echo "== process =="
if [[ -n "${PID}" ]]; then
    ps -p "$PID" -o pid,etime,pcpu,pmem,stat,cmd | head -2
else
    echo "[!] create_nuscenes_infos is NOT running."
fi

echo
echo "== log size + last 10 lines =="
[[ -f "$LOG" ]] && wc -l "$LOG"
[[ -f "$LOG" ]] && tail -10 "$LOG"

echo
echo "== expected output artefacts =="
for f in nuscenes_infos_10sweeps_train.pkl \
         nuscenes_infos_10sweeps_val.pkl \
         nuscenes_dbinfos_10sweeps_withvelo.pkl; do
    if [[ -f "$ROOT/$f" ]]; then
        sz=$(stat -c %s "$ROOT/$f")
        mt=$(stat -c %y "$ROOT/$f" | cut -d. -f1)
        printf "  [done] %-40s %12d B  (%s)\n" "$f" "$sz" "$mt"
    else
        echo "  [....]  $f"
    fi
done

# gt_database directory grows during the slow gt-sampling step
GTD="$ROOT/gt_database_10sweeps_withvelo"
echo
if [[ -d "$GTD" ]]; then
    n=$(find "$GTD" -maxdepth 1 -type f | wc -l)
    sz=$(du -sh "$GTD" 2>/dev/null | cut -f1)
    echo "== gt_database_10sweeps_withvelo: $n .bin files, $sz =="
else
    echo "== gt_database_10sweeps_withvelo: not yet created =="
fi

#!/bin/bash
# Show live progress of the v1.0-trainval aria2c download.
# Usage: bash scripts/check_nuscenes_download.sh
set -euo pipefail

PID=$(pgrep -f 'aria2c.*nuscenes_urls' | head -1 || true)
LOG=/root/autodl-tmp/ASAP/logs/aria2_nuscenes_trainval.log
DATA_DIR=/root/autodl-tmp/ASAP/data/nuscenes

if [[ -z "${PID}" ]]; then
    echo "[!] aria2c is NOT running for nuScenes_urls.txt."
    echo "    Resume with:"
    echo "      setsid nohup aria2c --input-file=/root/autodl-tmp/ASAP/scripts/nuscenes_urls.txt \\"
    echo "        --dir=/root/autodl-tmp/ASAP/data/nuscenes \\"
    echo "        --max-connection-per-server=4 --split=4 --min-split-size=64M \\"
    echo "        --max-concurrent-downloads=2 --continue=true --auto-file-renaming=false \\"
    echo "        --allow-overwrite=false --console-log-level=warn --summary-interval=60 \\"
    echo "        --log=$LOG --log-level=notice --file-allocation=falloc \\"
    echo "        > /root/autodl-tmp/ASAP/logs/aria2_nuscenes_trainval.console 2>&1 < /dev/null &"
    exit 1
fi

echo "== process =="
ps -p "$PID" -o pid,etime,pcpu,pmem,stat,cmd | head -2

echo
echo "== /proc/${PID}/io (snapshot, 30s sample) =="
W1=$(awk '/^write_bytes/{print $2}' /proc/"$PID"/io)
T1=$(date +%s)
sleep 30
W2=$(awk '/^write_bytes/{print $2}' /proc/"$PID"/io)
T2=$(date +%s)
awk -v dw=$((W2-W1)) -v dt=$((T2-T1)) -v w2="$W2" 'BEGIN{
  mbps=dw/1048576.0/dt
  done_gb=w2/1073741824.0
  rem_gb=315 - done_gb
  printf "  download_rate = %.2f MiB/s  (last %ds avg)\n", mbps, dt
  printf "  downloaded    = %.2f GiB / 315 GiB  (%.1f%%)\n", done_gb, done_gb/315*100
  printf "  remaining     = %.2f GiB\n", rem_gb
  if (mbps>0) printf "  ETA           = %.1f h\n", rem_gb*1024/mbps/3600
}'

echo
echo "== per-file finished so far =="
grep -E "Download complete:" "$LOG" | sed 's/.*Download complete: //'

echo
echo "== last 5 log lines =="
tail -5 "$LOG"

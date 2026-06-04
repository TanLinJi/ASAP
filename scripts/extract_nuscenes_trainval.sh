#!/bin/bash
# Extract the 11 v1.0-trainval archives sequentially under data/nuscenes/v1.0-trainval/.
# Uses pigz for parallel gzip decompression (24 CPUs available).
# Each archive contributes the ~/samples/, ~/sweeps/, ~/maps/, and (for meta) ~/v1.0-trainval/.
set -euo pipefail

DATA_DIR=/root/autodl-tmp/ASAP/data/nuscenes
DEST=$DATA_DIR/v1.0-trainval
LOG=/root/autodl-tmp/ASAP/logs/nuscenes_trainval_extract.log

mkdir -p "$DEST" /root/autodl-tmp/ASAP/logs
cd "$DEST"

total_t0=$(date +%s)
echo "[$(date +%H:%M:%S)] start: extracting 11 archives into $DEST" | tee -a "$LOG"

# Extract meta first so v1.0-trainval/ dir is created early; then blobs in order.
ARCHIVES=(
    "$DATA_DIR/v1.0-trainval_meta.tgz"
    "$DATA_DIR/v1.0-trainval01_blobs.tgz"
    "$DATA_DIR/v1.0-trainval02_blobs.tgz"
    "$DATA_DIR/v1.0-trainval03_blobs.tgz"
    "$DATA_DIR/v1.0-trainval04_blobs.tgz"
    "$DATA_DIR/v1.0-trainval05_blobs.tgz"
    "$DATA_DIR/v1.0-trainval06_blobs.tgz"
    "$DATA_DIR/v1.0-trainval07_blobs.tgz"
    "$DATA_DIR/v1.0-trainval08_blobs.tgz"
    "$DATA_DIR/v1.0-trainval09_blobs.tgz"
    "$DATA_DIR/v1.0-trainval10_blobs.tgz"
)

i=0
for archive in "${ARCHIVES[@]}"; do
    i=$((i + 1))
    f=$(basename "$archive")
    size=$(stat -c %s "$archive")
    size_gib=$(awk -v s="$size" 'BEGIN{printf "%.2f", s/1073741824}')
    t0=$(date +%s)
    echo "[$(date +%H:%M:%S)] [$i/11] extracting $f (${size_gib} GiB) ..." | tee -a "$LOG"
    # Use pigz for parallel decompression; tar then untars the stream serially.
    tar -I pigz -xf "$archive"
    t1=$(date +%s)
    dt=$((t1 - t0))
    rate=$(awk -v s="$size" -v t="$dt" 'BEGIN{printf "%.1f", s/1048576/t}')
    echo "[$(date +%H:%M:%S)] [$i/11] done $f in ${dt}s (${rate} MiB/s)" | tee -a "$LOG"
done

total_t1=$(date +%s)
echo "[$(date +%H:%M:%S)] all 11 archives extracted in $((total_t1 - total_t0))s" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] resulting top-level entries in $DEST :" | tee -a "$LOG"
ls -la "$DEST" | tee -a "$LOG"

# Sanity counts
echo "[$(date +%H:%M:%S)] file counts:" | tee -a "$LOG"
for d in samples sweeps maps v1.0-trainval; do
    if [ -d "$DEST/$d" ]; then
        n=$(find "$DEST/$d" -type f | wc -l)
        printf "  %-18s %d files\n" "$d/" "$n" | tee -a "$LOG"
    fi
done

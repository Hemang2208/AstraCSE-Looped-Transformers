#!/usr/bin/env bash
# Builds corpus v2 on a CPU instance (scripts/expand_corpus.py) and uploads it to S3,
# then powers the instance off. Launched as a one-time spot instance whose shutdown
# behaviour is "terminate", so powering off also ends billing.
#
# Resumable: extract outputs are mirrored to s3://<bucket>/data/corpus_v2_work/ after
# each source and synced back at the start, and expand_corpus.py skips files whose
# output already exists, so a replacement instance continues where this one stopped.
#
# Result in S3:
#   data/shards_v2/train/      v1 train shards (shard_0000..) + new shards (shard_1000..)
#   data/shards_v2/val_new/    0.5% of the new documents, held out
#   data/shards_v2/manifest_new.json, status/corpus_v2_done, status/corpus_v2.log
set -euo pipefail

BUCKET="${BUCKET:-your-bucket-name}"
S3="s3://$BUCKET"
WORK="${WORK:-/data/work}"
OUT="${OUT:-/data/out}"
LOG=/home/ubuntu/corpus_v2.log
WORKERS="${WORKERS:-14}"
cd /home/ubuntu/AstraCSE
export PATH="/opt/pytorch/bin:$PATH"
exec >> "$LOG" 2>&1

log() { echo "[$(date -u +%H:%M:%S)] corpus_v2: $*"; }
push_log() { aws s3 cp "$LOG" "$S3/status/corpus_v2.log" --quiet || true; }
trap 'rc=$?; log "exiting with status $rc"; push_log; sudo shutdown -h now' EXIT

log "setup"
python3 -m pip install -q pyarrow "huggingface_hub>=0.30" hf_xet tokenizers boto3 numpy
mkdir -p "$WORK" "$OUT"
aws s3 sync "$S3/checkpoints/tokenizer" tokenizer --quiet
aws s3 sync "$S3/data/shards/val" data/shards/val --quiet
aws s3 sync "$S3/data/corpus_v2_work" "$WORK" --quiet || true
push_log

# Smallest sources first, so a problem shows up within minutes. The max-tokens values
# are extract-stage limits (before dedup); the pack caps below are the real budget.
for spec in "libretexts 1e12" "arxiv 1e12" "stackexchange 1e12" "cosmopedia 7e8" "fineweb 1.25e9"; do
    read -r source max_tokens <<< "$spec"
    log "extract $source (max $max_tokens tokens)"
    python3 scripts/expand_corpus.py extract --source "$source" --max-tokens "$max_tokens" \
        --workdir "$WORK" --workers "$WORKERS"
    aws s3 sync "$WORK/$source" "$S3/data/corpus_v2_work/$source" --quiet
    push_log
done

# About 3.2B new tokens, so shards_v2 is ~3.8x the v1 corpus. Stack Exchange yielded
# 4.9B CS tokens; 1.2B keeps it near a quarter of the new data.
log "pack"
python3 scripts/expand_corpus.py pack --workdir "$WORK" --out-dir "$OUT" \
    --caps fineweb=1.2e9 cosmopedia=8e8 stackexchange=1.2e9 arxiv=2e8
aws s3 sync "$OUT/train" "$S3/data/shards_v2/train" --quiet
aws s3 sync "$OUT/val_new" "$S3/data/shards_v2/val_new" --quiet
aws s3 cp "$OUT/manifest.json" "$S3/data/shards_v2/manifest_new.json" --quiet
log "copying the v1 train shards into shards_v2/train (server-side)"
aws s3 sync "$S3/data/shards/train" "$S3/data/shards_v2/train" --quiet
N=$(aws s3 ls "$S3/data/shards_v2/train/" | grep -c "\.bin$")
log "shards_v2/train has $N shards"
echo "done $N shards $(date -u +%Y-%m-%dT%H:%M:%SZ)" | aws s3 cp - "$S3/status/corpus_v2_done" --quiet
log "finished"

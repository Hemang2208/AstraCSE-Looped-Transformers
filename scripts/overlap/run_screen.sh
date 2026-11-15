#!/bin/bash
# Overlap screen on the GPU box. Launched at nice 19 and idle I/O.
# Reads the training shards (never writes to them). Writes only under W.
W=/home/ubuntu/astra_overlap
R=us-east-1
B=s3://your-bucket-name
cd "$W" || exit 1
stamp() { date -u +%FT%TZ; }
echo "$(stamp) listing ETags"
aws s3api list-objects-v2 --region "$R" --bucket your-bucket-name --prefix data/shards/train/ --query 'Contents[].[Key,ETag]' --output text > meta/v1_train_etags.txt || exit 2
aws s3api list-objects-v2 --region "$R" --bucket your-bucket-name --prefix data/shards_v2/train/ --query 'Contents[].[Key,ETag]' --output text > meta/v2_train_etags.txt || exit 2
echo "$(stamp) syncing v1 train (control copy)"
aws s3 sync "$B/data/shards/train/" v1/train/ --region "$R" --only-show-errors || exit 3
echo "$(stamp) screening"
/opt/pytorch/bin/python3 -I scripts/screen_box.py --repo /home/ubuntu/AstraCSE --work "$W" || exit 4
echo "$(stamp) finished"

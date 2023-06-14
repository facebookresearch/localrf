# Copyright (c) Meta Platforms, Inc. and affiliates.
mkdir weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt -P weights
./RAFT/download_models.sh
rm models.zip

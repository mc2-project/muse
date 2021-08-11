#!/bin/bash
for j in {0..4}
do
  env RAYON_NUM_THREADS=4 cargo +nightly run --all-features --release --bin resnet32-server
done

for j in {0..4}
do
  env RAYON_NUM_THREADS=4 cargo +nightly run --all-features --release --bin minionn-server
done

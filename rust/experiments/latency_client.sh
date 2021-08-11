#!/bin/bash
mkdir -p latency_data
err="latency_data/err.log"

for i in {0..4}
do
  mkdir -p latency_data/run$i
  echo "ResNet32 on 4 threads - Run #$i"
  run_path="latency_data/run$i/resnet32-4.txt"
  env RAYON_NUM_THREADS=4 CLICOLOR=0 cargo +nightly run --all-features --release --bin resnet32-client 2>$err > $run_path;
  cat "$run_path/resnet-32-4.txt" | egrep "End.*Client online|End.*offline phase|End.*ReLU layer\."
  echo -e "\n"
  sleep 2
done

for i in {0..4}
do
  echo "MiniONN on 4 threads - Run #$i"
  run_path="latency_data/run$i/minionn-4.txt"
  env RAYON_NUM_THREADS=4 CLICOLOR=0 cargo +nightly run --all-features --release --bin minionn-client 2>$err > $run_path;
  cat "$run_path/minionn-4.txt" | egrep "End.*Client online|End.*offline phase|End.*ReLU layer\."
  echo -e "\n"
  sleep 2
done

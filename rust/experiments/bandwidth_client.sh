#! /bin/bash
mkdir -p bandwidth_data
err="bandwidth_data/err.log"

path="bandwidth_data/resnet32.pcap"
tshark -i ens5 -w $path &
sleep 5
pid=$!
cargo +nightly run --all-features --release --bin resnet32-client;
sudo kill $pid

path="bandwidth_data/minionn.pcap"
tshark -i ens5 -w $path &
sleep 5
pid=$!
cargo +nightly run --all-features --release --bin minionn-client;
sudo kill $pid

#! /bin/bash
# Run:
#  $ grep mem_heap_B server.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# To just output the peak memory usage in bytes

mkdir -p memory_data
err="memory_data/err.log"

path="memory_data/resnet32-client-$l.out"
valgrind --tool=massif --pages-as-heap=no --massif-out-file=$path ../target/release/resnet32-client

path="memory_data/minionn-client-$l.out"
valgrind --tool=massif --pages-as-heap=no --massif-out-file=$path ../target/release/minionn-client

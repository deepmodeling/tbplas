#! /bin/bash

for i in examples tests; do
    find $i -name "sim_data" | xargs rm -rf
    find $i -name "*.png" | xargs rm -f
done

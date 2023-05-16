#! /bin/bash

topdir=$(pwd)
for i in examples tests; do
    cd $topdir/$i
    find . -name sim_data | xargs rm -rf
    find . -name *.png | xargs rm -f
done

#! /bin/bash

find . -name *.so | xargs rm

tmp_list="build dist tbplas.egg-info"
for i in $tmp_list; do
    test -d $i && rm -r $i
done

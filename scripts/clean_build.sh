#! /bin/bash

postfix="c so"
for i in $postfix; do
    find . -name *.$i | xargs rm
done

tmp_list="build dist tbplas.egg-info"
for i in $tmp_list; do
    test -d $i && rm -r $i
done

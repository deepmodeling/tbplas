#! /bin/bash

for i in builder fortran; do
	cp build/lib.linux-x86_64-cpython-3*/tbplas/$i/*.so tbplas/$i
done

#! /bin/bash

for i in builder fortran; do
	cp build/lib.linux-*/tbplas/$i/*.so tbplas/$i
done

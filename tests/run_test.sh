#! /bin/bash

scripts="
test_base.py \
test_diag.py \
test_graph.py \
test_hetero.py \
test_lindhard.py \
test_materials.py \
test_prim.py \
test_sample.py \
test_sk_soc.py \
test_super_core.py \
test_super.py"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
for s in $scripts; do
    python $s
    wait
done

#export OMP_NUM_THREADS=2
#export MKL_NUM_THREADS=2
#mpirun -np 2 python ./test_mpi.py
#wait

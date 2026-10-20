#!/bin/bash
# survey_sass.sh LABEL BINARY: instruction forms of the order-7 kick kernel
# around the PC multipole staging (gravity_cuda.hpp) and the cp.async paths of
# barrier.h, after the racecheck_forms survey tools.
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-festive-easley-59df3c/494d385c-f529-4643-8d8d-d3d837c7695a/scratchpad
REPO=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/festive-easley-59df3c
label=$1; bin=$2; d=$S/survey/$label
mkdir -p $d && cd $d
/usr/local/cuda/bin/cuobjdump -xelf all $bin > /dev/null 2>&1
cubin=$(ls $d/*.1.sm_86.cubin | head -1)
idx=$(readelf -sW $cubin | grep FUNC | grep "cudaKickKernelILi7" | awk '{print $1}' | tr -d :)
/usr/local/cuda/bin/nvdisasm -g -fun $idx $cubin > $d/sass.txt 2> $d/nvdisasm.err
python3 -I $REPO/docs/cuda_kick_shared_memory_results/validation/racecheck_forms/scripts/condense2.py $d/sass.txt > $d/c3.txt
echo "== $label ($bin) sha $(sha256sum $bin | cut -c1-12) kernel symbol $idx bytes $(readelf -sW $cubin | grep FUNC | grep cudaKickKernelILi7 | awk '{print $3}')"
echo "   LDGSTS by width: $(grep -o 'LDGSTS\.E[A-Z0-9.]*' $d/c3.txt | sort | uniq -c | tr '\n' ';')"
echo "   LDGSTS by source line: $(grep 'LDGSTS' $d/c3.txt | awk '{print $2}' | sort | uniq -c | tr '\n' ';')"
echo "   barrier.h fallback loop lines (1034-1037) present: $(grep -c 'barrier.h:103[4-7] ' $d/c3.txt)"
echo "   ARRIVES: $(grep -c 'ARRIVES.LDGSTSBAR' $d/c3.txt)  LD.E/ST.E from barrier.h: $(grep 'barrier.h' $d/c3.txt | grep -c 'LD\.E\|ST\.E')"

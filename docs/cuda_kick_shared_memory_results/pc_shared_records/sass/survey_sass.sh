#!/bin/bash
# survey_sass.sh LABEL BINARY: instruction forms of the order-7 kick kernel in
# the PC direct routine (gravity_cuda.hpp, cudaGravityPcDirect and the M2P
# operator it inlines), after the racecheck_forms survey tools: the condensed
# listing c3.txt keeps the memory, barrier and control instructions with their
# source attribution. Paths are the session's scratchpad and worktree.
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-mystifying-carson-b3856a/206b190b-3141-4894-8e51-1e5a2e4f94d6/scratchpad
REPO=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/mystifying-carson-b3856a
label=$1; bin=$(readlink -f "$2"); d=$S/survey/$label
mkdir -p $d && cd $d
/usr/local/cuda/bin/cuobjdump -xelf all $bin > /dev/null 2>&1
cubin=$(ls $d/*.1.sm_86.cubin | head -1)
idx=$(readelf -sW $cubin | grep FUNC | grep "cudaKickKernelILi7" | awk '{print $1}' | tr -d :)
/usr/local/cuda/bin/nvdisasm -g -fun $idx $cubin > $d/sass.txt 2> $d/nvdisasm.err
python3 -I $REPO/docs/cuda_kick_shared_memory_results/validation/racecheck_forms/scripts/condense2.py $d/sass.txt > $d/c3.txt
echo "== $label ($bin) sha $(sha256sum $bin | cut -c1-12) kernel symbol $idx bytes $(readelf -sW $cubin | grep FUNC | grep cudaKickKernelILi7 | awk '{print $3}')"
echo "   LDGSTS by source line: $(grep 'LDGSTS' $d/c3.txt | awk '{print $2}' | sort | uniq -c | tr '\n' ';')"
echo "   generated_operators.hpp loads: $(grep 'generated_operators' $d/c3.txt | awk '{print $3}' | sort | uniq -c | tr '\n' ';')"
echo "   LDL/STL in the kernel: $(grep -c 'LDL\|STL' $d/sass.txt)"

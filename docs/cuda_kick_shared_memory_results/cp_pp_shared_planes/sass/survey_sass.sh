#!/bin/bash
# survey_sass.sh LABEL BINARY CP_LINES PP_LINES: instruction forms of the
# order-7 kick kernel in the CP direct routine's source loop and at the PP
# direct routine's force updates (gravity_cuda.hpp), after the racecheck_forms
# survey tools. At order 7 cudaGravityCpDirect is inlined into the kernel
# (cudaKickKernel<7>) while cudaGravityPpDirect<7, true> and <7, false> are
# separate device functions, so three symbols are disassembled: the condensed
# listings c3_<symbol>.txt keep the memory, barrier and control instructions
# with their source attribution; sass_regions.py prints the CP loop bodies
# enclosing the staged-source reads (CP_LINES, e.g. gravity_cuda.hpp:364-366)
# from the kernel's listing and the memory instructions of the force updates
# (PP_LINES) from the two PP functions' listings. Paths are the session's
# scratchpad and worktree; the outputs <label>_cp_loop.txt,
# <label>_pp_stores_<equal|mixed>.txt and the summary lines are written next
# to this script.
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-amazing-wiles-9d8695/9042a4bd-e4ba-4741-b48d-f8c9e28a0cd5/scratchpad
REPO=/home/dmarce1/workspace/cosmictiger/.claude/worktrees/amazing-wiles-9d8695
CONDENSE=$REPO/docs/cuda_kick_shared_memory_results/validation/racecheck_forms/scripts/condense2.py
HERE=$(cd "$(dirname "$0")" && pwd)
label=$1; bin=$(readlink -f "$2"); cp_lines=$3; pp_lines=$4; d=$S/survey/$label
mkdir -p $d && cd $d
/usr/local/cuda/bin/cuobjdump -xelf all $bin > /dev/null 2>&1
cubin=$(ls $d/*.1.sm_86.cubin | head -1)
disassemble() {    # NAME MANGLED_FRAGMENT
    local idx=$(readelf -sW $cubin | grep FUNC | grep "$2" | awk '{print $1}' | tr -d :)
    /usr/local/cuda/bin/nvdisasm -g -fun $idx $cubin > $d/sass_$1.txt 2> $d/nvdisasm_$1.err
    python3 -I $CONDENSE $d/sass_$1.txt > $d/c3_$1.txt
    echo "   $1: symbol $idx, $(readelf -sW $cubin | grep FUNC | grep "$2" | awk '{print $3}') bytes; LDL/STL $(grep -c 'LDL\|STL' $d/sass_$1.txt), LD.E $(grep -c '\bLD\.E' $d/sass_$1.txt), ST.E $(grep -c '\bST\.E' $d/sass_$1.txt), LDS $(grep -c '\bLDS\b' $d/sass_$1.txt), LDS.64 $(grep -c 'LDS\.64' $d/sass_$1.txt), STS $(grep -c '\bSTS\b' $d/sass_$1.txt), LDGSTS $(grep -c 'LDGSTS' $d/sass_$1.txt)"
}
{
echo "== $label ($bin) sha $(sha256sum $bin | cut -c1-12)"
disassemble kernel "cudaKickKernelILi7E"
disassemble pp_equal "cudaGravityPpDirectILi7ELb1EE"
disassemble pp_mixed "cudaGravityPpDirectILi7ELb0EE"
python3 -I $HERE/sass_regions.py $d/sass_kernel.txt loop "$cp_lines" > $HERE/${label}_cp_loop.txt
python3 -I $HERE/sass_regions.py $d/sass_pp_equal.txt site "$pp_lines" > $HERE/${label}_pp_stores_equal.txt
python3 -I $HERE/sass_regions.py $d/sass_pp_mixed.txt site "$pp_lines" > $HERE/${label}_pp_stores_mixed.txt
echo "   CP loops in the kernel: $(grep '^== loop' $HERE/${label}_cp_loop.txt | sed 's/^== loop //' | tr '\n' ';')"
echo "   CP plane reads by form: $(grep -h 'memory instructions by source line' $HERE/${label}_cp_loop.txt | sed 's/.*source line: //' | tr ';' '\n' | awk '{print $NF-0, $(NF-1)}' | sort | uniq -c | tr '\n' ';')"
for v in equal mixed; do
    echo "   PP force-update memory instructions ($v mass): $(head -1 $HERE/${label}_pp_stores_$v.txt | sed 's/.*, //'); forms: $(grep '^      ' $HERE/${label}_pp_stores_$v.txt | awk '{print $3}' | sort | uniq -c | tr '\n' ';')"
done
} | tee -a $HERE/survey_summary.txt

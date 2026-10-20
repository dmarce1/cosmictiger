#!/bin/bash
# survey_one.sh LABEL BINARY: operand forms of the order-7 kick kernel's mbarrier instructions.
S=/tmp/claude-1000/-home-dmarce1-workspace-cosmictiger--claude-worktrees-nifty-lederberg-7f7b6e/d666ad06-9a49-415b-a5ba-0124ef8a78e3/scratchpad
label=$1; bin=$2; d=$S/survey/$label; OUT=$S/forms_summary.txt
mkdir -p $d && cd $d
/usr/local/cuda/bin/cuobjdump -xelf all $bin > /dev/null 2>&1
cubin=$(ls $d/*.1.sm_86.cubin | head -1)
idx=$(readelf -sW $cubin | grep FUNC | grep "cudaKickKernelILi7" | awk '{print $1}' | tr -d :)
/usr/local/cuda/bin/nvdisasm -g -fun $idx $cubin > $d/sass.txt 2> $d/nvdisasm.err
python3 -I $S/condense2.py $d/sass.txt > $d/c3.txt
{
echo "== $label ($bin) sha $(sha256sum $bin | cut -c1-12) kernel symbol $idx"
echo "   ARRIVES forms: $(grep -o 'ARRIVES.LDGSTSBAR.64 \[[^]]*\]' $d/c3.txt | sed 's/ARRIVES.LDGSTSBAR.64 //; s/R[0-9]\+/R/g; s/UR[0-9]\+/UR/g; s/0x[0-9a-f]*/imm/g' | sort | uniq -c | tr '\n' ';')"
echo "   ATOMS.ARRIVE forms: $(grep -o 'ATOMS.ARRIVE.64 R[0-9]*, \[[^]]*\]' $d/c3.txt | sed 's/ATOMS.ARRIVE.64 R[0-9]*, //; s/R[0-9]\+/R/g; s/UR[0-9]\+/UR/g; s/0x[0-9a-f]*/imm/g' | sort | uniq -c | tr '\n' ';')"
echo "   test_wait LDS forms: $(grep 'barrier.h:249 ' $d/c3.txt | grep -o '\[[^]]*\]' | sed 's/R[0-9]\+/R/g; s/UR[0-9]\+/UR/g; s/0x[0-9a-f]*/imm/g' | sort | uniq -c | tr '\n' ';')"
echo "   LDGSTS count $(grep -c 'LDGSTS.E' $d/c3.txt), WARPSYNC $(grep -c WARPSYNC $d/c3.txt), BAR.SYNC $(grep -c 'BAR.SYNC' $d/c3.txt), kernel bytes $(readelf -sW $cubin | grep FUNC | grep cudaKickKernelILi7 | awk '{print $3}')"
} >> $OUT

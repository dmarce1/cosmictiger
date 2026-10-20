The first form of the change copied each staged record to registers with ten
16-byte shared loads (`detail::cudaLoadStagedMultipole`, frozen as
`build/release/frozen/cosmictiger_b1024_pc_registers`, labelled `shared` in
these files). Its m2P loop body is 234 instructions per sink-multipole pair
(`../sass/pc_region_registers.txt`, `../sass/loop_body_counts.txt`) with the
same floating-point instruction multiset as the head's, but the compiler pairs
the operator's fused multiply-adds differently, so its forces differ from the
head's at roundoff level (`head_vs_registers.md`: relative L2 1e-9 to 4e-7,
largest on the lattice, whose forces nearly cancel). The whole-kick timing
here (`paired_speedup.md`, five pairs of 15 samples, buckets 64 and 512) is
that variant's; the landed form reads the record in place with scalar shared
loads and is timed in the parent directory. `../variants/` compares the
exports of all three forms against the head in one run.

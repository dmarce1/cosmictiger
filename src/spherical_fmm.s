	.file	"spherical_fmm.cpp"
	.text
	.section	.text._ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0,"axG",@progbits,_Z8test_M2LILi7EEff,comdat
	.align 2
	.p2align 4
	.type	_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0, @function
_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0:
.LFB10613:
	.cfi_startproc
	movq	$0, 32(%rdi)
	vmovss	32(%rsi), %xmm1
	vxorps	%xmm0, %xmm0, %xmm0
	vmovss	.LC3(%rip), %xmm2
	vfmadd132ss	.LC0(%rip), %xmm0, %xmm1
	vmovss	%xmm1, 32(%rdi)
	vmovss	44(%rsi), %xmm3
	vmulss	%xmm2, %xmm1, %xmm1
	movq	$0, 40(%rdi)
	vfmadd132ss	.LC2(%rip), %xmm0, %xmm3
	vmovss	%xmm1, 32(%rdi)
	vmulss	%xmm2, %xmm3, %xmm1
	vmovss	%xmm1, 36(%rdi)
	vaddss	24(%rsi), %xmm0, %xmm3
	vmovss	.LC4(%rip), %xmm1
	vmovss	%xmm3, 40(%rdi)
	vfmadd231ss	36(%rsi), %xmm1, %xmm0
	vmovss	%xmm0, 44(%rdi)
	vfmadd132ss	40(%rsi), %xmm3, %xmm1
	vmulss	%xmm2, %xmm0, %xmm0
	vmovss	%xmm0, 44(%rdi)
	vmulss	%xmm2, %xmm1, %xmm1
	vmovss	%xmm1, 40(%rdi)
	ret
	.cfi_endproc
.LFE10613:
	.size	_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0, .-_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0
	.set	_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb1ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0,_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0
	.section	.text._ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0,"axG",@progbits,_Z8test_M2LILi7EEff,comdat
	.align 2
	.p2align 4
	.type	_ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0, @function
_ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0:
.LFB10801:
	.cfi_startproc
	vmovss	28(%rsi), %xmm0
	vmovss	108(%rdi), %xmm4
	vmovss	24(%rsi), %xmm1
	vmovss	104(%rdi), %xmm3
	vmulss	%xmm4, %xmm0, %xmm2
	vfmsub231ss	%xmm3, %xmm1, %xmm2
	vmulss	%xmm4, %xmm1, %xmm1
	vmovss	116(%rdi), %xmm4
	vfmadd132ss	%xmm3, %xmm1, %xmm0
	vmovss	112(%rdi), %xmm3
	vmovss	%xmm2, 104(%rdi)
	vmovss	%xmm0, 108(%rdi)
	vmovss	36(%rsi), %xmm0
	vmovss	32(%rsi), %xmm1
	vmulss	%xmm4, %xmm0, %xmm2
	vfmsub231ss	%xmm3, %xmm1, %xmm2
	vmulss	%xmm4, %xmm1, %xmm1
	vfmadd132ss	%xmm3, %xmm1, %xmm0
	vmovss	%xmm2, 112(%rdi)
	vmovss	%xmm0, 116(%rdi)
	ret
	.cfi_endproc
.LFE10801:
	.size	_ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0, .-_ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0
	.section	.text._ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0,"axG",@progbits,_Z8test_M2LILi7EEff,comdat
	.align 2
	.p2align 4
	.type	_ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0, @function
_ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0:
.LFB11034:
	.cfi_startproc
	vmovss	44(%rsi), %xmm0
	vmovss	212(%rdi), %xmm4
	vmovss	40(%rsi), %xmm1
	vmovss	208(%rdi), %xmm3
	vmulss	%xmm4, %xmm0, %xmm2
	vfmsub231ss	%xmm3, %xmm1, %xmm2
	vmulss	%xmm4, %xmm1, %xmm1
	vmovss	220(%rdi), %xmm4
	vfmadd132ss	%xmm3, %xmm1, %xmm0
	vmovss	216(%rdi), %xmm3
	vmovss	%xmm2, 208(%rdi)
	vmovss	%xmm0, 212(%rdi)
	vmovss	52(%rsi), %xmm0
	vmovss	48(%rsi), %xmm1
	vmulss	%xmm4, %xmm0, %xmm2
	vfmsub231ss	%xmm3, %xmm1, %xmm2
	vmulss	%xmm4, %xmm1, %xmm1
	vfmadd132ss	%xmm3, %xmm1, %xmm0
	vmovss	%xmm2, 216(%rdi)
	vmovss	%xmm0, 220(%rdi)
	ret
	.cfi_endproc
.LFE11034:
	.size	_ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0, .-_ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0
	.text
	.p2align 4
	.globl	_Z4Brotiii
	.type	_Z4Brotiii, @function
_Z4Brotiii:
.LFB8869:
	.cfi_startproc
	endbr64
	movl	%esi, %eax
	vmovsd	.LC5(%rip), %xmm0
	orl	%edx, %eax
	orl	%edi, %eax
	je	.L14
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	vxorpd	%xmm0, %xmm0, %xmm0
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	movl	%edx, %r12d
	sarl	$31, %edx
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	movl	%edx, %eax
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	xorl	%r12d, %eax
	subl	%edx, %eax
	cmpl	%edi, %eax
	jg	.L12
	leal	-1(%r12), %edx
	leal	-1(%rdi), %ebp
	leal	1(%r12), %r14d
	testl	%esi, %esi
	je	.L17
	jle	.L8
	leal	-1(%rsi), %r13d
	movl	%ebp, %edi
	movl	%r13d, %esi
	call	_Z4Brotiii
	movl	%r14d, %edx
	movl	%r13d, %esi
	movl	%ebp, %edi
	vmovsd	%xmm0, 8(%rsp)
	call	_Z4Brotiii
	vaddsd	8(%rsp), %xmm0, %xmm1
	movl	%r12d, %edx
	movl	%r13d, %esi
	movl	%ebp, %edi
	vmovsd	%xmm1, 8(%rsp)
	call	_Z4Brotiii
	vmovsd	8(%rsp), %xmm1
	vfmadd132sd	.LC8(%rip), %xmm1, %xmm0
	vmulsd	.LC7(%rip), %xmm0, %xmm0
.L12:
	addq	$24, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L14:
	.cfi_restore 6
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 14
	ret
	.p2align 4
	.p2align 3
.L8:
	.cfi_def_cfa_offset 64
	.cfi_offset 6, -40
	.cfi_offset 12, -32
	.cfi_offset 13, -24
	.cfi_offset 14, -16
	leal	1(%rsi), %r13d
	movl	%ebp, %edi
	movl	%r13d, %esi
	call	_Z4Brotiii
	movl	%r14d, %edx
	movl	%r13d, %esi
	movl	%ebp, %edi
	vmovsd	%xmm0, 8(%rsp)
	call	_Z4Brotiii
	vaddsd	8(%rsp), %xmm0, %xmm1
	movl	%r12d, %edx
	movl	%r13d, %esi
	movl	%ebp, %edi
	vmovsd	%xmm1, 8(%rsp)
	call	_Z4Brotiii
	vmovsd	8(%rsp), %xmm1
	vfnmadd132sd	.LC8(%rip), %xmm1, %xmm0
	vmulsd	.LC7(%rip), %xmm0, %xmm0
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.p2align 4
	.p2align 3
.L17:
	.cfi_restore_state
	movl	%ebp, %edi
	call	_Z4Brotiii
	movl	%r14d, %edx
	movl	%ebp, %edi
	xorl	%esi, %esi
	vmovsd	%xmm0, 8(%rsp)
	call	_Z4Brotiii
	vmovsd	8(%rsp), %xmm3
	vsubsd	%xmm0, %xmm3, %xmm0
	vmulsd	.LC7(%rip), %xmm0, %xmm0
	addq	$24, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8869:
	.size	_Z4Brotiii, .-_Z4Brotiii
	.p2align 4
	.globl	_Z9factoriali
	.type	_Z9factoriali, @function
_Z9factoriali:
.LFB8894:
	.cfi_startproc
	endbr64
	vmovsd	.LC5(%rip), %xmm0
	testl	%edi, %edi
	jne	.L27
	ret
	.p2align 4
	.p2align 3
.L27:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	movl	%edi, %ebx
	leal	-1(%rdi), %edi
	call	_Z9factoriali
	vmovaps	%xmm0, %xmm1
	vxorps	%xmm0, %xmm0, %xmm0
	vcvtsi2sdl	%ebx, %xmm0, %xmm0
	popq	%rbx
	.cfi_def_cfa_offset 8
	vmulsd	%xmm1, %xmm0, %xmm0
	ret
	.cfi_endproc
.LFE8894:
	.size	_Z9factoriali, .-_Z9factoriali
	.p2align 4
	.globl	_Z11random_unitRfS_S_
	.type	_Z11random_unitRfS_S_, @function
_Z11random_unitRfS_S_:
.LFB8907:
	.cfi_startproc
	endbr64
	pushq	%r14
	.cfi_def_cfa_offset 16
	.cfi_offset 14, -16
	pushq	%r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
	pushq	%r12
	.cfi_def_cfa_offset 32
	.cfi_offset 12, -32
	pushq	%rbp
	.cfi_def_cfa_offset 40
	.cfi_offset 6, -40
	movq	%rdi, %r12
	movq	%rsi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
	movq	%rdx, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	leaq	12(%rsp), %r13
	leaq	8(%rsp), %r14
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	.LC7(%rip), %xmm0, %xmm0
	vmulsd	.LC9(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vaddss	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vsubsd	.LC5(%rip), %xmm0, %xmm0
	call	acos@PLT
	movq	%r14, %rsi
	movq	%r13, %rdi
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	call	sincosf@PLT
	vmovss	8(%rsp), %xmm2
	vmovss	12(%rsp), %xmm1
	vmovss	%xmm2, 4(%rsp)
	vmovss	%xmm1, (%rsp)
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	movq	%r14, %rsi
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	.LC7(%rip), %xmm0, %xmm0
	movq	%r13, %rdi
	vmulsd	.LC9(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
	vmulsd	.LC10(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	call	sincosf@PLT
	vmovss	(%rsp), %xmm1
	vmulss	8(%rsp), %xmm1, %xmm0
	vmulss	12(%rsp), %xmm1, %xmm1
	vmovss	4(%rsp), %xmm2
	vmovss	%xmm0, (%r12)
	vmovss	%xmm1, 0(%rbp)
	vmovss	%xmm2, (%rbx)
	addq	$16, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%rbp
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r13
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8907:
	.size	_Z11random_unitRfS_S_, .-_Z11random_unitRfS_S_
	.p2align 4
	.globl	_Z13random_vectorRfS_S_
	.type	_Z13random_vectorRfS_S_, @function
_Z13random_vectorRfS_S_:
.LFB8908:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rdx, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rsi, %rbx
	.p2align 4
	.p2align 3
.L31:
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	vmovss	.LC11(%rip), %xmm4
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	.LC7(%rip), %xmm0, %xmm0
	vmulsd	.LC9(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	.LC4(%rip), %xmm4, %xmm0
	vmovss	%xmm0, 0(%rbp)
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	.LC7(%rip), %xmm0, %xmm0
	vmovss	.LC11(%rip), %xmm5
	vmulsd	.LC9(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	.LC4(%rip), %xmm5, %xmm0
	vmovss	%xmm0, (%rbx)
	call	rand@PLT
	vxorps	%xmm3, %xmm3, %xmm3
	vcvtsi2ssl	%eax, %xmm3, %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vaddsd	.LC7(%rip), %xmm0, %xmm0
	vmovss	.LC11(%rip), %xmm6
	vmulsd	.LC9(%rip), %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vfmadd132ss	.LC4(%rip), %xmm6, %xmm0
	vmovss	%xmm0, (%r12)
	vmovss	(%rbx), %xmm2
	vmovss	0(%rbp), %xmm1
	vmulss	%xmm2, %xmm2, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vfmadd132ss	%xmm0, %xmm1, %xmm0
	vcomiss	.LC12(%rip), %xmm0
	ja	.L31
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8908:
	.size	_Z13random_vectorRfS_S_, .-_Z13random_vectorRfS_S_
	.section	.text._Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_,"axG",@progbits,_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_,comdat
	.p2align 4
	.weak	_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_
	.type	_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_, @function
_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_:
.LFB9347:
	.cfi_startproc
	endbr64
	vmulss	%xmm1, %xmm1, %xmm7
	vmovss	.LC12(%rip), %xmm8
	vmovss	.LC13(%rip), %xmm9
	vxorps	%xmm6, %xmm6, %xmm6
	movq	%rdi, %r8
	movq	$1065353216, (%rdi)
	movl	$3, %esi
	xorl	%eax, %eax
	vfmadd231ss	%xmm0, %xmm0, %xmm7
	vfmadd231ss	%xmm2, %xmm2, %xmm7
	jmp	.L38
	.p2align 4
	.p2align 3
.L35:
	leal	-3(%rsi), %edx
	leal	-1(%rax), %ecx
	vcvtsi2ssl	%edx, %xmm6, %xmm3
	movl	%ecx, %edx
	vmovaps	%xmm3, %xmm5
	imull	%eax, %edx
	sarl	%edx
	addl	%ecx, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %rdx
	vmovss	4(%rdx), %xmm3
	vmovss	(%rdx), %xmm4
	vmulss	%xmm3, %xmm1, %xmm10
	vmulss	%xmm3, %xmm0, %xmm3
	vfmsub231ss	%xmm4, %xmm0, %xmm10
	vfmadd231ss	%xmm4, %xmm1, %xmm3
	vdivss	%xmm5, %xmm10, %xmm10
	vdivss	%xmm5, %xmm3, %xmm3
	vmovss	%xmm10, (%r10)
	vmovss	%xmm3, 4(%r10)
	cmpl	$6, %eax
	je	.L34
	leal	2(%rax), %edx
	movl	%edi, %ecx
	vmulss	%xmm10, %xmm2, %xmm10
	vmulss	%xmm3, %xmm2, %xmm3
	imull	%edx, %ecx
	sarl	%ecx
	addl	%eax, %ecx
	movslq	%ecx, %rcx
	leaq	(%r8,%rcx,8), %r9
	vmovss	%xmm10, (%r9)
	vmovss	%xmm3, 4(%r9)
	cmpl	$6, %edx
	ja	.L40
.L36:
	movl	%eax, %ecx
	vcvtsi2ssl	%esi, %xmm6, %xmm11
	vmulss	%xmm2, %xmm11, %xmm11
	imull	%eax, %ecx
	vmulss	%xmm11, %xmm3, %xmm3
	vmulss	%xmm11, %xmm10, %xmm10
	vcvtsi2ssl	%ecx, %xmm6, %xmm5
	movl	%edx, %ecx
	imull	%edx, %ecx
	vfnmadd231ss	(%r10), %xmm7, %xmm10
	vfnmadd231ss	4(%r10), %xmm7, %xmm3
	vcvtsi2ssl	%ecx, %xmm6, %xmm4
	vsubss	%xmm5, %xmm4, %xmm4
	leal	3(%rax), %ecx
	imull	%ecx, %edx
	vdivss	%xmm4, %xmm8, %xmm4
	sarl	%edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r10
	vmulss	%xmm10, %xmm4, %xmm10
	vmulss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm10, (%r10)
	vmovss	%xmm3, 4(%r10)
	cmpl	$6, %ecx
	ja	.L39
	movl	%ecx, %edx
	imull	%ecx, %edx
	vcvtsi2ssl	%edx, %xmm6, %xmm11
	leal	2(%rsi), %edx
	vsubss	%xmm5, %xmm11, %xmm11
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	vmulss	%xmm2, %xmm4, %xmm4
	vdivss	%xmm11, %xmm8, %xmm11
	vmulss	%xmm3, %xmm4, %xmm3
	vmulss	%xmm10, %xmm4, %xmm4
	vfnmadd231ss	4(%r9), %xmm7, %xmm3
	vfnmadd231ss	(%r9), %xmm7, %xmm4
	leal	4(%rax), %r9d
	imull	%r9d, %ecx
	sarl	%ecx
	movl	%ecx, %edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r11
	vmulss	%xmm11, %xmm4, %xmm4
	vmulss	%xmm11, %xmm3, %xmm3
	vmovss	%xmm4, (%r11)
	vmovss	%xmm3, 4(%r11)
	cmpl	$6, %r9d
	ja	.L39
	movl	%r9d, %edx
	leal	5(%rax), %ecx
	imull	%r9d, %edx
	imull	%ecx, %r9d
	vcvtsi2ssl	%edx, %xmm6, %xmm11
	leal	4(%rsi), %edx
	vsubss	%xmm5, %xmm11, %xmm11
	vcvtsi2ssl	%edx, %xmm6, %xmm10
	vmulss	%xmm2, %xmm10, %xmm10
	movl	%r9d, %edx
	sarl	%edx
	vdivss	%xmm11, %xmm8, %xmm11
	addl	%eax, %edx
	vmulss	%xmm3, %xmm10, %xmm3
	vmulss	%xmm4, %xmm10, %xmm10
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r9
	vfnmadd231ss	(%r10), %xmm7, %xmm10
	vfnmadd231ss	4(%r10), %xmm7, %xmm3
	vmulss	%xmm11, %xmm10, %xmm10
	vmulss	%xmm11, %xmm3, %xmm3
	vmovss	%xmm10, (%r9)
	vmovss	%xmm3, 4(%r9)
	cmpl	$6, %ecx
	ja	.L39
	movl	%ecx, %edx
	leal	6(%rax), %r10d
	imull	%ecx, %edx
	imull	%r10d, %ecx
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	leal	6(%rsi), %edx
	vsubss	%xmm5, %xmm4, %xmm4
	vcvtsi2ssl	%edx, %xmm6, %xmm12
	vmulss	%xmm2, %xmm12, %xmm12
	sarl	%ecx
	movl	%ecx, %edx
	vdivss	%xmm4, %xmm8, %xmm4
	addl	%eax, %edx
	vmulss	%xmm12, %xmm3, %xmm3
	vmulss	%xmm12, %xmm10, %xmm10
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %rdx
	vfnmadd231ss	(%r11), %xmm7, %xmm10
	vfnmadd231ss	4(%r11), %xmm7, %xmm3
	vmulss	%xmm10, %xmm4, %xmm10
	vmulss	%xmm3, %xmm4, %xmm4
	vmovss	%xmm10, (%rdx)
	vmovss	%xmm4, 4(%rdx)
	cmpl	$6, %r10d
	jne	.L39
	leal	8(%rsi), %edx
	vsubss	%xmm5, %xmm9, %xmm5
	addl	$21, %eax
	vcvtsi2ssl	%edx, %xmm6, %xmm3
	vmulss	%xmm2, %xmm3, %xmm3
	cltq
	leaq	(%r8,%rax,8), %rax
	vdivss	%xmm5, %xmm8, %xmm5
	vmulss	%xmm4, %xmm3, %xmm4
	vmulss	%xmm10, %xmm3, %xmm3
	vfnmadd231ss	4(%r9), %xmm7, %xmm4
	vfnmadd231ss	(%r9), %xmm7, %xmm3
	vmulss	%xmm3, %xmm5, %xmm3
	vmulss	%xmm4, %xmm5, %xmm5
	vmovss	%xmm3, (%rax)
	vmovss	%xmm5, 4(%rax)
.L39:
	cmpl	$7, %edi
	je	.L34
.L40:
	addl	$2, %esi
	movl	%edi, %eax
.L38:
	leal	1(%rax), %edi
	movl	%eax, %edx
	imull	%edi, %edx
	sarl	%edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r10
	testl	%eax, %eax
	jne	.L35
	vmulss	(%r10), %xmm2, %xmm10
	vmulss	4(%r10), %xmm2, %xmm3
	leal	(%rdi,%rdi), %ecx
	movl	$2, %edx
	sarl	%ecx
	movslq	%ecx, %rcx
	leaq	(%r8,%rcx,8), %r9
	vmovss	%xmm10, (%r9)
	vmovss	%xmm3, 4(%r9)
	jmp	.L36
	.p2align 4
	.p2align 3
.L34:
	movq	%r8, %rax
	ret
	.cfi_endproc
.LFE9347:
	.size	_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_, .-_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_
	.section	.text._Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_,"axG",@progbits,_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_,comdat
	.p2align 4
	.weak	_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_
	.type	_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_, @function
_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_:
.LFB9397:
	.cfi_startproc
	endbr64
	vmulss	%xmm1, %xmm1, %xmm7
	vmovss	.LC12(%rip), %xmm8
	vmovss	.LC14(%rip), %xmm9
	vxorps	%xmm6, %xmm6, %xmm6
	movq	%rdi, %r8
	movq	$1065353216, (%rdi)
	movl	$3, %esi
	xorl	%eax, %eax
	vfmadd231ss	%xmm0, %xmm0, %xmm7
	vfmadd231ss	%xmm2, %xmm2, %xmm7
	jmp	.L48
	.p2align 4
	.p2align 3
.L45:
	leal	-3(%rsi), %edx
	leal	-1(%rax), %ecx
	vcvtsi2ssl	%edx, %xmm6, %xmm3
	movl	%eax, %edx
	vmovaps	%xmm3, %xmm5
	imull	%ecx, %edx
	sarl	%edx
	addl	%ecx, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %rdx
	vmovss	4(%rdx), %xmm3
	vmovss	(%rdx), %xmm4
	vmulss	%xmm3, %xmm1, %xmm10
	vmulss	%xmm3, %xmm0, %xmm3
	vfmsub231ss	%xmm4, %xmm0, %xmm10
	vfmadd231ss	%xmm4, %xmm1, %xmm3
	vdivss	%xmm5, %xmm10, %xmm10
	vdivss	%xmm5, %xmm3, %xmm3
	vmovss	%xmm10, (%r10)
	vmovss	%xmm3, 4(%r10)
	cmpl	$7, %eax
	je	.L44
	leal	2(%rax), %edx
	vmulss	%xmm10, %xmm2, %xmm10
	vmulss	%xmm3, %xmm2, %xmm3
	movl	%edx, %ecx
	imull	%edi, %ecx
	sarl	%ecx
	addl	%eax, %ecx
	movslq	%ecx, %rcx
	leaq	(%r8,%rcx,8), %r9
	vmovss	%xmm10, (%r9)
	vmovss	%xmm3, 4(%r9)
	cmpl	$7, %edx
	ja	.L50
.L46:
	movl	%eax, %ecx
	vcvtsi2ssl	%esi, %xmm6, %xmm11
	vmulss	%xmm2, %xmm11, %xmm11
	imull	%eax, %ecx
	vmulss	%xmm11, %xmm3, %xmm3
	vmulss	%xmm11, %xmm10, %xmm10
	vcvtsi2ssl	%ecx, %xmm6, %xmm5
	movl	%edx, %ecx
	imull	%edx, %ecx
	vfnmadd231ss	(%r10), %xmm7, %xmm10
	vfnmadd231ss	4(%r10), %xmm7, %xmm3
	vcvtsi2ssl	%ecx, %xmm6, %xmm4
	vsubss	%xmm5, %xmm4, %xmm4
	leal	3(%rax), %ecx
	imull	%ecx, %edx
	vdivss	%xmm4, %xmm8, %xmm4
	sarl	%edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r10
	vmulss	%xmm10, %xmm4, %xmm10
	vmulss	%xmm3, %xmm4, %xmm3
	vmovss	%xmm10, (%r10)
	vmovss	%xmm3, 4(%r10)
	cmpl	$7, %ecx
	ja	.L49
	movl	%ecx, %edx
	imull	%ecx, %edx
	vcvtsi2ssl	%edx, %xmm6, %xmm11
	leal	2(%rsi), %edx
	vsubss	%xmm5, %xmm11, %xmm11
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	vmulss	%xmm2, %xmm4, %xmm4
	vdivss	%xmm11, %xmm8, %xmm11
	vmulss	%xmm3, %xmm4, %xmm3
	vmulss	%xmm10, %xmm4, %xmm4
	vfnmadd231ss	4(%r9), %xmm7, %xmm3
	vfnmadd231ss	(%r9), %xmm7, %xmm4
	leal	4(%rax), %r9d
	imull	%r9d, %ecx
	sarl	%ecx
	movl	%ecx, %edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r11
	vmulss	%xmm11, %xmm4, %xmm4
	vmulss	%xmm11, %xmm3, %xmm3
	vmovss	%xmm4, (%r11)
	vmovss	%xmm3, 4(%r11)
	cmpl	$7, %r9d
	ja	.L49
	movl	%r9d, %edx
	leal	5(%rax), %ecx
	imull	%r9d, %edx
	imull	%ecx, %r9d
	vcvtsi2ssl	%edx, %xmm6, %xmm11
	leal	4(%rsi), %edx
	vsubss	%xmm5, %xmm11, %xmm11
	vcvtsi2ssl	%edx, %xmm6, %xmm10
	vmulss	%xmm2, %xmm10, %xmm10
	movl	%r9d, %edx
	sarl	%edx
	vdivss	%xmm11, %xmm8, %xmm11
	addl	%eax, %edx
	vmulss	%xmm3, %xmm10, %xmm3
	vmulss	%xmm4, %xmm10, %xmm10
	movslq	%edx, %rdx
	vfnmadd231ss	(%r10), %xmm7, %xmm10
	vfnmadd231ss	4(%r10), %xmm7, %xmm3
	leaq	(%r8,%rdx,8), %r10
	vmulss	%xmm11, %xmm10, %xmm10
	vmulss	%xmm11, %xmm3, %xmm3
	vmovss	%xmm10, (%r10)
	vmovss	%xmm3, 4(%r10)
	cmpl	$7, %ecx
	ja	.L49
	movl	%ecx, %edx
	leal	6(%rax), %r9d
	imull	%ecx, %edx
	imull	%r9d, %ecx
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	vsubss	%xmm5, %xmm4, %xmm4
	leal	6(%rsi), %edx
	sarl	%ecx
	vdivss	%xmm4, %xmm8, %xmm12
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	vmulss	%xmm2, %xmm4, %xmm4
	movl	%ecx, %edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	vmulss	%xmm4, %xmm3, %xmm3
	vmulss	%xmm4, %xmm10, %xmm10
	leaq	(%r8,%rdx,8), %rcx
	vfnmadd231ss	(%r11), %xmm7, %xmm10
	vfnmadd231ss	4(%r11), %xmm7, %xmm3
	vmulss	%xmm10, %xmm12, %xmm10
	vmulss	%xmm3, %xmm12, %xmm3
	vmovss	%xmm10, (%rcx)
	vmovss	%xmm3, 4(%rcx)
	cmpl	$7, %r9d
	ja	.L49
	movl	%r9d, %edx
	imull	%r9d, %edx
	vcvtsi2ssl	%edx, %xmm6, %xmm4
	leal	8(%rsi), %edx
	vsubss	%xmm5, %xmm4, %xmm4
	vcvtsi2ssl	%edx, %xmm6, %xmm11
	vmulss	%xmm2, %xmm11, %xmm11
	vdivss	%xmm4, %xmm8, %xmm4
	vmulss	%xmm11, %xmm10, %xmm10
	vmulss	%xmm11, %xmm3, %xmm12
	vfnmadd231ss	4(%r10), %xmm7, %xmm12
	vmovaps	%xmm10, %xmm3
	vfnmadd231ss	(%r10), %xmm7, %xmm3
	leal	7(%rax), %r10d
	imull	%r10d, %r9d
	movl	%r9d, %edx
	sarl	%edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %rdx
	vmulss	%xmm3, %xmm4, %xmm10
	vmulss	%xmm12, %xmm4, %xmm4
	vmovss	%xmm10, (%rdx)
	vmovss	%xmm4, 4(%rdx)
	cmpl	$7, %r10d
	jne	.L49
	leal	10(%rsi), %edx
	vsubss	%xmm5, %xmm9, %xmm5
	addl	$28, %eax
	vcvtsi2ssl	%edx, %xmm6, %xmm3
	vmulss	%xmm2, %xmm3, %xmm3
	cltq
	leaq	(%r8,%rax,8), %rax
	vdivss	%xmm5, %xmm8, %xmm5
	vmulss	%xmm4, %xmm3, %xmm4
	vmulss	%xmm10, %xmm3, %xmm3
	vfnmadd231ss	4(%rcx), %xmm7, %xmm4
	vfnmadd231ss	(%rcx), %xmm7, %xmm3
	vmulss	%xmm3, %xmm5, %xmm3
	vmulss	%xmm4, %xmm5, %xmm5
	vmovss	%xmm3, (%rax)
	vmovss	%xmm5, 4(%rax)
.L49:
	cmpl	$8, %edi
	je	.L44
.L50:
	addl	$2, %esi
	movl	%edi, %eax
.L48:
	leal	1(%rax), %edi
	movl	%eax, %edx
	imull	%edi, %edx
	sarl	%edx
	addl	%eax, %edx
	movslq	%edx, %rdx
	leaq	(%r8,%rdx,8), %r10
	testl	%eax, %eax
	jne	.L45
	vmulss	(%r10), %xmm2, %xmm10
	vmulss	4(%r10), %xmm2, %xmm3
	leal	(%rdi,%rdi), %ecx
	movl	$2, %edx
	sarl	%ecx
	movslq	%ecx, %rcx
	leaq	(%r8,%rcx,8), %r9
	vmovss	%xmm10, (%r9)
	vmovss	%xmm3, 4(%r9)
	jmp	.L46
	.p2align 4
	.p2align 3
.L44:
	movq	%r8, %rax
	ret
	.cfi_endproc
.LFE9397:
	.size	_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_, .-_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_
	.section	.rodata._Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_.str1.8,"aMS",@progbits,1
	.align 8
.LC16:
	.string	"Bounds error - %i is not between 0 and %i\n"
	.align 8
.LC17:
	.string	"/home/dmarce1/cuda-workspace/cosmictiger/cosmictiger/containers.hpp"
	.section	.rodata._Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_.str1.1,"aMS",@progbits,1
.LC18:
	.string	"Error in %s on line %i\n"
	.section	.rodata._Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_.str1.8
	.align 8
.LC19:
	.string	"/home/dmarce1/cuda-workspace/cosmictiger/cosmictiger/safe_io.hpp"
	.section	.rodata._Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_.str1.1
.LC20:
	.string	"false"
	.section	.text._Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_,"axG",@progbits,_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_,comdat
	.p2align 4
	.weak	_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_
	.type	_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_, @function
_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_:
.LFB9352:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rdi, %rbx
	subq	$696, %rsp
	.cfi_def_cfa_offset 752
	movq	%rdi, 88(%rsp)
	vmovss	.LC15(%rip), %xmm3
	leaq	384(%rsp), %r14
	vxorps	%xmm3, %xmm2, %xmm2
	vxorps	%xmm3, %xmm1, %xmm1
	vxorps	%xmm3, %xmm0, %xmm0
	movq	%r14, %r15
	movq	%fs:40, %rax
	movq	%rax, 680(%rsp)
	xorl	%eax, %eax
	leaq	96(%rsp), %rax
	movq	%rax, %rdi
	movq	%rax, 56(%rsp)
	call	_Z26spherical_regular_harmonicIfLi7EE19spherical_expansionIT_XT0_EES1_S1_S1_
	movl	$288, %edx
	movq	%rbx, %rsi
	movq	%r14, %rdi
	call	memcpy@PLT
	movl	$7, 68(%rsp)
	movl	$0, 76(%rsp)
	movl	$0, 84(%rsp)
	vmovss	.LC15(%rip), %xmm3
.L76:
	movl	84(%rsp), %eax
	xorl	%r13d, %r13d
	movq	%r15, %r8
	movl	%eax, %edi
	movl	%eax, 80(%rsp)
	incl	%eax
	movl	%eax, 84(%rsp)
	imull	%edi, %eax
	movl	%edi, %esi
	movq	88(%rsp), %rdi
	negl	%esi
	movl	%esi, 64(%rsp)
	sarl	%eax
	cltq
	leaq	(%rdi,%rax,8), %r12
	movl	76(%rsp), %eax
	movq	%r12, %r14
	movl	%r13d, %r12d
	incl	%eax
	movl	%eax, 72(%rsp)
	.p2align 4
	.p2align 3
.L75:
	movl	76(%rsp), %eax
	xorl	%r13d, %r13d
	movl	$0x00000000, (%r14)
	movl	$0x00000000, 4(%r14)
	subl	%r12d, %eax
	movl	%eax, 48(%rsp)
	movl	%r13d, %eax
	movl	%r12d, %r13d
	.p2align 4
	.p2align 3
.L57:
	movl	64(%rsp), %edx
	movl	%eax, %ebx
	movl	48(%rsp), %ecx
	negl	%ebx
	subl	%eax, %edx
	cmpl	%edx, %ebx
	cmovl	%edx, %ebx
	cmpl	%eax, %ecx
	cmovg	%eax, %ecx
	movl	%ecx, 12(%rsp)
	cmpl	%ecx, %ebx
	jg	.L94
	movl	72(%rsp), %esi
	leal	1(%rax), %edi
	movl	%edi, 52(%rsp)
	leal	(%rsi,%rax), %r15d
	movl	48(%rsp), %esi
	imull	%edi, %eax
	sarl	%eax
	leal	(%rsi,%r13), %edx
	movl	%eax, 28(%rsp)
	imull	%edx, %r15d
	movl	%r15d, %edx
	shrl	$31, %edx
	addl	%edx, %r15d
	sarl	%r15d
	testl	%ebx, %ebx
	js	.L87
	cltq
	movq	%rax, 32(%rsp)
.L58:
	leal	0(%r13,%rbx), %r9d
	movq	32(%rsp), %rcx
	movslq	%ebx, %rbp
	movl	%r13d, 32(%rsp)
	addl	28(%rsp), %ebx
	movl	%r15d, 24(%rsp)
	movl	%r9d, %r12d
	movq	%r8, %r15
	movq	56(%rsp), %rax
	leaq	(%rax,%rcx,8), %r11
	movl	%ebx, %r13d
	movq	%r11, %rbx
	jmp	.L74
	.p2align 4
	.p2align 3
.L98:
	movl	24(%rsp), %eax
	leal	(%rax,%r12), %r10d
	cmpl	$35, %r10d
	jg	.L95
.L68:
	movslq	%r10d, %r10
	leaq	(%r15,%r10,8), %rax
	vmovss	(%rax), %xmm5
	vmovss	4(%rax), %xmm4
	cmpl	$35, %r13d
	jg	.L96
.L72:
	vmovss	4(%rbx,%rbp,8), %xmm1
	vmovss	(%rbx,%rbp,8), %xmm0
	incl	%r12d
	incq	%rbp
	incl	%r13d
	vxorps	%xmm3, %xmm1, %xmm1
	vmulss	%xmm4, %xmm1, %xmm2
	vmulss	%xmm5, %xmm1, %xmm1
	vfmsub231ss	%xmm5, %xmm0, %xmm2
	vfmadd132ss	%xmm4, %xmm1, %xmm0
	vaddss	(%r14), %xmm2, %xmm1
	vaddss	4(%r14), %xmm0, %xmm0
	vmovss	%xmm1, (%r14)
	vmovss	%xmm0, 4(%r14)
	cmpl	%ebp, 12(%rsp)
	jl	.L97
.L74:
	testl	%r12d, %r12d
	jns	.L98
	vmovss	.LC11(%rip), %xmm0
	testb	$1, %r12b
	jne	.L70
	vmovss	.LC12(%rip), %xmm0
.L70:
	movl	24(%rsp), %r10d
	subl	%r12d, %r10d
	cmpl	$35, %r10d
	jg	.L99
.L71:
	movslq	%r10d, %r10
	leaq	(%r15,%r10,8), %rax
	vmovss	4(%rax), %xmm4
	vmulss	(%rax), %xmm0, %xmm5
	vxorps	%xmm3, %xmm4, %xmm4
	vmulss	%xmm0, %xmm4, %xmm4
	cmpl	$35, %r13d
	jle	.L72
.L96:
	movl	$36, %ecx
	movl	%r13d, %edx
	leaq	.LC16(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	vmovss	%xmm5, 28(%rsp)
	vmovss	%xmm4, 16(%rsp)
	call	__printf_chk@PLT
	movl	$75, %ecx
	leaq	.LC17(%rip), %rdx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movl	$73, %ecx
	leaq	.LC19(%rip), %rdx
	xorl	%esi, %esi
	leaq	.LC20(%rip), %rdi
	call	_Z18cosmictiger_assertPKcbS0_i@PLT
	vmovss	28(%rsp), %xmm5
	vmovss	.LC15(%rip), %xmm3
	vmovss	16(%rsp), %xmm4
	jmp	.L72
	.p2align 4
	.p2align 3
.L95:
	movl	%r10d, %edx
	movl	$36, %ecx
	leaq	.LC16(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	movl	%r10d, 16(%rsp)
	call	__printf_chk@PLT
	movl	$75, %ecx
	leaq	.LC17(%rip), %rdx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movl	$73, %ecx
	leaq	.LC19(%rip), %rdx
	xorl	%esi, %esi
	leaq	.LC20(%rip), %rdi
	call	_Z18cosmictiger_assertPKcbS0_i@PLT
	movl	16(%rsp), %r10d
	vmovss	.LC15(%rip), %xmm3
	jmp	.L68
	.p2align 4
	.p2align 3
.L99:
	movl	%r10d, %edx
	movl	$36, %ecx
	leaq	.LC16(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	vmovss	%xmm0, 28(%rsp)
	movl	%r10d, 16(%rsp)
	call	__printf_chk@PLT
	movl	$75, %ecx
	leaq	.LC17(%rip), %rdx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movl	$73, %ecx
	leaq	.LC19(%rip), %rdx
	xorl	%esi, %esi
	leaq	.LC20(%rip), %rdi
	call	_Z18cosmictiger_assertPKcbS0_i@PLT
	vmovss	28(%rsp), %xmm0
	vmovss	.LC15(%rip), %xmm3
	movl	16(%rsp), %r10d
	jmp	.L71
	.p2align 4
	.p2align 3
.L87:
	leal	1(%rcx), %r11d
	movl	$0, %eax
	movq	56(%rsp), %rdx
	movl	%r15d, %r9d
	testl	%r11d, %r11d
	vmovss	.LC12(%rip), %xmm5
	cmovg	%eax, %r11d
	movslq	28(%rsp), %rax
	movl	%r11d, %r15d
	movq	%rax, %rsi
	movq	%rax, 32(%rsp)
	movslq	%ebx, %rax
	subq	%rax, %rsi
	leaq	(%rdx,%rsi,8), %r12
	jmp	.L78
	.p2align 4
	.p2align 3
.L102:
	leal	(%r9,%r13), %ebp
	addl	%ebx, %ebp
	cmpl	$35, %ebp
	jg	.L100
.L60:
	movslq	%ebp, %rbp
	leaq	(%r8,%rbp,8), %rax
	vmovss	(%rax), %xmm4
	vmovss	4(%rax), %xmm1
.L61:
	vmovaps	%xmm5, %xmm2
	testb	$1, %bl
	je	.L64
	vmovss	.LC11(%rip), %xmm2
.L64:
	vmovss	4(%r12), %xmm0
	vmulss	(%r12), %xmm2, %xmm6
	incl	%ebx
	subq	$8, %r12
	vxorps	%xmm3, %xmm0, %xmm0
	vmulss	%xmm2, %xmm0, %xmm0
	vxorps	%xmm3, %xmm0, %xmm0
	vmulss	%xmm0, %xmm1, %xmm2
	vmulss	%xmm0, %xmm4, %xmm0
	vfmsub231ss	%xmm6, %xmm4, %xmm2
	vfmadd132ss	%xmm6, %xmm0, %xmm1
	vaddss	(%r14), %xmm2, %xmm0
	vaddss	4(%r14), %xmm1, %xmm1
	vmovss	%xmm0, (%r14)
	vmovss	%xmm1, 4(%r14)
	cmpl	%r15d, %ebx
	jge	.L101
.L78:
	movl	%ebx, %eax
	addl	%r13d, %eax
	jns	.L102
	vmovss	.LC11(%rip), %xmm1
	testb	$1, %al
	jne	.L62
	vmovss	.LC12(%rip), %xmm1
.L62:
	movl	%r9d, %ebp
	subl	%r13d, %ebp
	subl	%ebx, %ebp
	cmpl	$35, %ebp
	jg	.L103
.L63:
	movslq	%ebp, %rbp
	leaq	(%r8,%rbp,8), %rax
	vmovss	4(%rax), %xmm2
	vmulss	(%rax), %xmm1, %xmm4
	vxorps	%xmm3, %xmm2, %xmm2
	vmulss	%xmm1, %xmm2, %xmm1
	jmp	.L61
	.p2align 4
	.p2align 3
.L100:
	movl	$36, %ecx
	movl	%ebp, %edx
	leaq	.LC16(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	movq	%r8, 16(%rsp)
	movl	%r9d, 24(%rsp)
	call	__printf_chk@PLT
	movl	$75, %ecx
	leaq	.LC17(%rip), %rdx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movl	$73, %ecx
	leaq	.LC19(%rip), %rdx
	xorl	%esi, %esi
	leaq	.LC20(%rip), %rdi
	call	_Z18cosmictiger_assertPKcbS0_i@PLT
	vmovss	.LC12(%rip), %xmm5
	vmovss	.LC15(%rip), %xmm3
	movq	16(%rsp), %r8
	movl	24(%rsp), %r9d
	jmp	.L60
	.p2align 4
	.p2align 3
.L103:
	movl	$36, %ecx
	movl	%ebp, %edx
	leaq	.LC16(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	movq	%r8, 40(%rsp)
	movl	%r9d, 16(%rsp)
	vmovss	%xmm1, 24(%rsp)
	call	__printf_chk@PLT
	movl	$75, %ecx
	leaq	.LC17(%rip), %rdx
	leaq	.LC18(%rip), %rsi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk@PLT
	movq	stdout(%rip), %rdi
	call	fflush@PLT
	movl	$73, %ecx
	leaq	.LC19(%rip), %rdx
	xorl	%esi, %esi
	leaq	.LC20(%rip), %rdi
	call	_Z18cosmictiger_assertPKcbS0_i@PLT
	vmovss	.LC12(%rip), %xmm5
	vmovss	.LC15(%rip), %xmm3
	vmovss	24(%rsp), %xmm1
	movq	40(%rsp), %r8
	movl	16(%rsp), %r9d
	jmp	.L63
	.p2align 4
	.p2align 3
.L101:
	movl	%r9d, %r15d
	cmpl	12(%rsp), %ebx
	jle	.L58
	incl	48(%rsp)
	movl	52(%rsp), %eax
	cmpl	68(%rsp), %eax
	jle	.L57
	.p2align 4
	.p2align 3
.L105:
	decl	64(%rsp)
	leal	1(%r13), %eax
	addq	$8, %r14
	cmpl	%r13d, 80(%rsp)
	je	.L104
	movl	%eax, %r12d
	jmp	.L75
	.p2align 4
	.p2align 3
.L97:
	incl	48(%rsp)
	movl	32(%rsp), %r13d
	movq	%r15, %r8
	movl	52(%rsp), %eax
	cmpl	68(%rsp), %eax
	jle	.L57
	jmp	.L105
	.p2align 4
	.p2align 3
.L94:
	incl	%eax
	incl	48(%rsp)
	movl	%eax, 52(%rsp)
	movl	52(%rsp), %eax
	cmpl	68(%rsp), %eax
	jle	.L57
	jmp	.L105
.L104:
	decl	68(%rsp)
	cmpl	$8, 84(%rsp)
	movq	%r8, %r15
	movl	72(%rsp), %eax
	movl	%eax, 76(%rsp)
	jne	.L76
	movq	680(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L106
	addq	$696, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L106:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE9352:
	.size	_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_, .-_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_
	.section	.text._ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE,"axG",@progbits,_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE,comdat
	.align 2
	.p2align 4
	.weak	_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE
	.type	_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE, @function
_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE:
.LFB9467:
	.cfi_startproc
	endbr64
	subq	$568, %rsp
	.cfi_def_cfa_offset 576
	vmovss	4(%rdx), %xmm1
	vmovss	(%rdx), %xmm0
	vxorps	%xmm2, %xmm2, %xmm2
	movq	%fs:40, %rax
	movq	%rax, 552(%rsp)
	xorl	%eax, %eax
	movq	$1065353216, 496(%rsp)
	vpermilps	$160, 8(%rsi), %xmm8
	vpermilps	$160, 24(%rsi), %xmm9
	vpermilps	$160, 40(%rsi), %xmm11
	vmovaps	%xmm1, %xmm4
	vmovaps	%xmm1, %xmm5
	vfnmadd132ss	%xmm2, %xmm0, %xmm4
	vfmadd132ss	%xmm0, %xmm1, %xmm2
	vmovaps	%xmm8, 32(%rsp)
	vmovaps	%xmm9, (%rsp)
	vmulss	%xmm2, %xmm1, %xmm3
	vmovss	%xmm2, 508(%rsp)
	vmovss	%xmm4, 504(%rsp)
	vmulss	%xmm2, %xmm0, %xmm2
	vfmsub231ss	%xmm4, %xmm0, %xmm3
	vfmadd132ss	%xmm1, %xmm2, %xmm4
	vmovss	%xmm3, 512(%rsp)
	vmulss	%xmm4, %xmm1, %xmm2
	vmovss	%xmm4, 516(%rsp)
	vmulss	%xmm4, %xmm0, %xmm4
	vfmsub231ss	%xmm3, %xmm0, %xmm2
	vfmadd132ss	%xmm1, %xmm4, %xmm3
	vmovss	%xmm2, 520(%rsp)
	vmulss	%xmm3, %xmm1, %xmm4
	vmovss	%xmm3, 524(%rsp)
	vmulss	%xmm3, %xmm0, %xmm3
	vfmsub231ss	%xmm2, %xmm0, %xmm4
	vfmadd132ss	%xmm1, %xmm3, %xmm2
	vmovss	%xmm4, 528(%rsp)
	vmovss	%xmm2, 532(%rsp)
	vmulss	%xmm2, %xmm1, %xmm3
	vmulss	%xmm2, %xmm0, %xmm2
	vfmsub231ss	%xmm4, %xmm0, %xmm3
	vfmadd231ss	%xmm4, %xmm1, %xmm2
	vmovss	%xmm3, 536(%rsp)
	vmulss	%xmm2, %xmm1, %xmm4
	vmovaps	512(%rsp), %xmm1
	vmovss	%xmm2, 540(%rsp)
	vpermilps	$177, 528(%rsp), %xmm10
	vmovaps	%xmm4, %xmm7
	vshufps	$78, 528(%rsp), %xmm1, %xmm4
	vfmsub231ss	%xmm3, %xmm0, %xmm7
	vmulss	%xmm2, %xmm0, %xmm0
	vfmadd132ss	%xmm3, %xmm0, %xmm5
	vmovaps	496(%rsp), %xmm0
	vmovss	%xmm7, 264(%rsp)
	vmovss	%xmm7, 544(%rsp)
	vmovaps	%xmm4, 16(%rsp)
	vmovss	%xmm5, 268(%rsp)
	vmovss	%xmm5, 548(%rsp)
	vmovlhps	%xmm0, %xmm1, %xmm2
	vshufps	$78, %xmm0, %xmm1, %xmm6
	vshufps	$27, %xmm1, %xmm0, %xmm3
	vmovaps	%xmm2, 272(%rsp)
	vmovaps	%xmm6, 288(%rsp)
	vmovaps	%xmm11, 48(%rsp)
	vpermilps	$160, 72(%rsi), %xmm12
	vpermilps	$177, %xmm1, %xmm6
	vpermilps	$177, %xmm0, %xmm2
	vpermilps	$245, 88(%rsi), %xmm5
	vpermilps	$160, 56(%rsi), %xmm13
	vshufps	$27, 528(%rsp), %xmm1, %xmm9
	vpermilps	$160, 88(%rsi), %xmm14
	vpermilps	$160, 104(%rsi), %xmm15
	vpermilps	$160, 120(%rsi), %xmm11
	vpermilps	$160, 200(%rsi), %xmm4
	vmovaps	%xmm12, 80(%rsp)
	vpermilps	$160, 152(%rsi), %xmm12
	vmulps	%xmm3, %xmm5, %xmm5
	vmovaps	%xmm13, 64(%rsp)
	vpermilps	$160, 136(%rsi), %xmm13
	vmovaps	%xmm14, 96(%rsp)
	vpermilps	$160, 168(%rsi), %xmm14
	vmovaps	%xmm15, 112(%rsp)
	vmovaps	%xmm11, 128(%rsp)
	vpermilps	$160, 184(%rsi), %xmm15
	vmovaps	%xmm4, 208(%rsp)
	vshufps	$17, %xmm0, %xmm1, %xmm4
	vpermilps	$245, 72(%rsi), %xmm11
	vmovaps	%xmm12, 160(%rsp)
	vpermilps	$245, 56(%rsi), %xmm12
	vmovaps	%xmm13, 144(%rsp)
	vpermilps	$245, 40(%rsi), %xmm13
	vmovaps	%xmm14, 176(%rsp)
	vpermilps	$245, 24(%rsi), %xmm14
	vmovaps	%xmm15, 192(%rsp)
	vpermilps	$245, 8(%rsi), %xmm15
	vmulps	%xmm3, %xmm12, %xmm12
	vpermilps	$245, 104(%rsi), %xmm3
	vmulps	%xmm13, %xmm4, %xmm13
	vshufps	$27, %xmm0, %xmm1, %xmm4
	vmulps	%xmm11, %xmm4, %xmm11
	vpermilps	$245, 120(%rsi), %xmm4
	vmulps	%xmm2, %xmm15, %xmm15
	vmulps	%xmm2, %xmm14, %xmm14
	vmulps	%xmm3, %xmm9, %xmm9
	vpermilps	$245, 136(%rsi), %xmm3
	vmulps	%xmm2, %xmm4, %xmm4
	vmulps	%xmm6, %xmm3, %xmm8
	vmovaps	%xmm8, 224(%rsp)
	vpermilps	$245, 152(%rsi), %xmm8
	vpermilps	$245, 168(%rsi), %xmm7
	vmovaps	(%rsp), %xmm3
	vmulps	%xmm2, %xmm7, %xmm7
	vmulps	%xmm10, %xmm8, %xmm8
	vpermilps	$245, 184(%rsi), %xmm2
	vmulps	%xmm6, %xmm2, %xmm6
	vmovaps	%xmm3, %xmm2
	vfmsub132ps	%xmm0, %xmm14, %xmm2
	vfmadd231ps	%xmm3, %xmm0, %xmm14
	vmovaps	%xmm6, 240(%rsp)
	vpermilps	$245, 200(%rsi), %xmm6
	vmovaps	%xmm2, 320(%rsp)
	vmovlhps	%xmm0, %xmm1, %xmm2
	vfmsub132ps	48(%rsp), %xmm13, %xmm2
	vmulps	%xmm10, %xmm6, %xmm6
	vmovaps	32(%rsp), %xmm10
	vmovaps	%xmm2, 336(%rsp)
	vfmsub132ps	%xmm0, %xmm15, %xmm10
	vmovaps	%xmm10, 304(%rsp)
	vshufps	$78, %xmm1, %xmm0, %xmm10
	vmovaps	%xmm10, %xmm2
	vfmsub132ps	64(%rsp), %xmm12, %xmm2
	vmovaps	%xmm10, (%rsp)
	vmovaps	%xmm2, 352(%rsp)
	vshufps	$78, %xmm0, %xmm1, %xmm2
	vfmsub132ps	80(%rsp), %xmm11, %xmm2
	vmovaps	%xmm2, 368(%rsp)
	vmovaps	%xmm10, %xmm2
	vmovaps	16(%rsp), %xmm10
	vfmsub132ps	96(%rsp), %xmm5, %xmm2
	vfmsub132ps	112(%rsp), %xmm9, %xmm10
	vmovaps	%xmm2, 384(%rsp)
	vmovaps	%xmm10, 400(%rsp)
	vmovaps	128(%rsp), %xmm10
	vfmsub132ps	%xmm0, %xmm4, %xmm10
	vmovaps	%xmm10, 416(%rsp)
	vmovaps	144(%rsp), %xmm10
	vfmsub213ps	224(%rsp), %xmm1, %xmm10
	vmovaps	%xmm10, 432(%rsp)
	vmovaps	160(%rsp), %xmm10
	vfmsub132ps	528(%rsp), %xmm8, %xmm10
	vmovaps	%xmm10, 448(%rsp)
	vmovaps	176(%rsp), %xmm10
	vfmsub132ps	%xmm0, %xmm7, %xmm10
	vmovaps	%xmm10, 464(%rsp)
	vmovaps	192(%rsp), %xmm10
	vfmsub213ps	240(%rsp), %xmm1, %xmm10
	vmovaps	%xmm10, 480(%rsp)
	vfmadd231ps	32(%rsp), %xmm0, %xmm15
	vfmadd231ps	128(%rsp), %xmm0, %xmm4
	vmovaps	208(%rsp), %xmm10
	vfmadd132ps	176(%rsp), %xmm7, %xmm0
	vmovaps	208(%rsp), %xmm7
	vfmsub132ps	528(%rsp), %xmm6, %xmm10
	vfmadd231ps	528(%rsp), %xmm7, %xmm6
	vmovaps	304(%rsp), %xmm7
	vmovaps	48(%rsp), %xmm3
	vfmadd231ps	272(%rsp), %xmm3, %xmm13
	vmovaps	64(%rsp), %xmm3
	vfmadd231ps	(%rsp), %xmm3, %xmm12
	vmovaps	80(%rsp), %xmm3
	vfmadd231ps	288(%rsp), %xmm3, %xmm11
	vmovaps	(%rsp), %xmm2
	vfmadd231ps	96(%rsp), %xmm2, %xmm5
	vmovaps	112(%rsp), %xmm3
	vmovaps	160(%rsp), %xmm2
	vfmadd231ps	16(%rsp), %xmm3, %xmm9
	vfmadd231ps	528(%rsp), %xmm2, %xmm8
	vmovaps	144(%rsp), %xmm3
	vmovaps	240(%rsp), %xmm2
	vfmadd213ps	224(%rsp), %xmm1, %xmm3
	vblendps	$10, %xmm15, %xmm7, %xmm15
	vmovaps	352(%rsp), %xmm7
	vfmadd132ps	192(%rsp), %xmm2, %xmm1
	vmovups	%xmm15, 8(%rsi)
	vmovaps	320(%rsp), %xmm15
	vblendps	$10, %xmm6, %xmm10, %xmm6
	vblendps	$10, %xmm12, %xmm7, %xmm12
	vblendps	$10, %xmm14, %xmm15, %xmm14
	vmovups	%xmm12, 56(%rsi)
	vmovaps	368(%rsp), %xmm12
	vmovups	%xmm14, 24(%rsi)
	vmovaps	336(%rsp), %xmm14
	vblendps	$10, %xmm11, %xmm12, %xmm11
	vblendps	$10, %xmm13, %xmm14, %xmm13
	vmovups	%xmm13, 40(%rsi)
	vmovups	%xmm11, 72(%rsi)
	vmovaps	384(%rsp), %xmm2
	vmovaps	464(%rsp), %xmm14
	vmovups	%xmm6, 200(%rsi)
	vmovaps	480(%rsp), %xmm15
	vmovss	264(%rsp), %xmm7
	vmovaps	416(%rsp), %xmm11
	vmovaps	432(%rsp), %xmm13
	vmovaps	448(%rsp), %xmm12
	vblendps	$10, %xmm5, %xmm2, %xmm5
	vmovaps	400(%rsp), %xmm2
	vblendps	$10, %xmm0, %xmm14, %xmm0
	vmovups	%xmm5, 88(%rsi)
	vmovups	%xmm0, 168(%rsi)
	vmovss	268(%rsp), %xmm5
	vmovss	220(%rsi), %xmm0
	vblendps	$10, %xmm1, %xmm15, %xmm1
	vmovups	%xmm1, 184(%rsi)
	vblendps	$10, %xmm4, %xmm11, %xmm4
	vblendps	$10, %xmm3, %xmm13, %xmm3
	vblendps	$10, %xmm8, %xmm12, %xmm8
	vmovups	%xmm4, 120(%rsi)
	vmovups	%xmm3, 136(%rsi)
	vmovups	%xmm8, 152(%rsi)
	vblendps	$10, %xmm9, %xmm2, %xmm9
	vmovss	216(%rsi), %xmm2
	vmulss	%xmm5, %xmm0, %xmm1
	vmovups	%xmm9, 104(%rsi)
	vmulss	%xmm7, %xmm0, %xmm0
	vfmsub231ss	%xmm7, %xmm2, %xmm1
	vfmadd231ss	%xmm5, %xmm2, %xmm0
	vmovss	%xmm1, 216(%rsi)
	vmovss	%xmm0, 220(%rsi)
	movq	552(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L111
	addq	$568, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 8
	ret
.L111:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE9467:
	.size	_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE, .-_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE
	.section	.text._ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_,"axG",@progbits,_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_,comdat
	.align 2
	.p2align 4
	.weak	_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_
	.type	_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_, @function
_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_:
.LFB9807:
	.cfi_startproc
	endbr64
	vmovss	120(%rdx), %xmm1
	vmovss	128(%rsi), %xmm3
	vfmadd132ss	.LC21(%rip), %xmm3, %xmm1
	vmovss	132(%rsi), %xmm4
	vmovss	.LC22(%rip), %xmm0
	vmovss	%xmm1, 128(%rsi)
	vmovss	132(%rdx), %xmm2
	vfmadd132ss	.LC0(%rip), %xmm4, %xmm2
	vmovss	%xmm2, 132(%rsi)
	vfmadd231ss	136(%rdx), %xmm0, %xmm1
	vmovss	%xmm1, 128(%rsi)
	vfmadd132ss	148(%rdx), %xmm2, %xmm0
	vmovss	%xmm0, 132(%rsi)
	vmovss	152(%rdx), %xmm5
	vfmadd231ss	.LC23(%rip), %xmm5, %xmm1
	vmovss	%xmm1, 128(%rsi)
	vmovss	164(%rdx), %xmm6
	vfmadd231ss	.LC24(%rip), %xmm6, %xmm0
	vmovss	%xmm0, 132(%rsi)
	ret
	.cfi_endproc
.LFE9807:
	.size	_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_, .-_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_
	.section	.text._Z8test_M2LILi7EEff,"axG",@progbits,_Z8test_M2LILi7EEff,comdat
	.p2align 4
	.weak	_Z8test_M2LILi7EEff
	.type	_Z8test_M2LILi7EEff, @function
_Z8test_M2LILi7EEff:
.LFB9314:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$1448, %rsp
	.cfi_def_cfa_offset 1504
	vmovss	%xmm0, 12(%rsp)
	leaq	1136(%rsp), %r12
	leaq	464(%rsp), %r15
	leaq	688(%rsp), %r14
	leaq	912(%rsp), %rbx
	leaq	392(%rsp), %r13
	movq	%r12, %rbp
	movq	%fs:40, %rax
	movq	%rax, 1432(%rsp)
	xorl	%eax, %eax
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	leaq	380(%rsp), %rax
	vcvtss2sd	12(%rsp), %xmm7, %xmm7
	vxorps	%xmm14, %xmm14, %xmm14
	movq	%rax, 312(%rsp)
	leaq	368(%rsp), %rax
	movl	$10000, 260(%rsp)
	vmovss	%xmm14, 200(%rsp)
	movq	%rax, 288(%rsp)
	leaq	356(%rsp), %rax
	movq	%rax, 264(%rsp)
	leaq	384(%rsp), %rax
	movq	%rax, 320(%rsp)
	leaq	372(%rsp), %rax
	movq	%rax, 296(%rsp)
	leaq	360(%rsp), %rax
	movq	%rax, 272(%rsp)
	leaq	388(%rsp), %rax
	movq	%rax, 328(%rsp)
	leaq	376(%rsp), %rax
	movq	%rax, 304(%rsp)
	leaq	364(%rsp), %rax
	movq	%rax, 280(%rsp)
	leaq	355(%rsp), %rax
	movq	%rax, 192(%rsp)
	leaq	400(%rsp), %rax
	vmovsd	%xmm7, 344(%rsp)
	movq	%rax, 336(%rsp)
	.p2align 4
	.p2align 3
.L119:
	movq	312(%rsp), %rdx
	movq	288(%rsp), %rsi
	movq	264(%rsp), %rdi
	call	_Z13random_vectorRfS_S_
	movq	320(%rsp), %rdx
	movq	296(%rsp), %rsi
	movq	272(%rsp), %rdi
	call	_Z11random_unitRfS_S_
	movq	328(%rsp), %rdx
	movq	304(%rsp), %rsi
	movq	280(%rsp), %rdi
	call	_Z13random_vectorRfS_S_
	vmovsd	.LC7(%rip), %xmm7
	movq	%r15, %rdi
	vmulsd	344(%rsp), %xmm7, %xmm0
	vxorps	%xmm7, %xmm7, %xmm7
	vcvtss2sd	360(%rsp), %xmm7, %xmm1
	vdivsd	%xmm0, %xmm1, %xmm1
	vmovaps	%xmm0, %xmm2
	vcvtsd2ss	%xmm1, %xmm1, %xmm1
	vmovss	%xmm1, 360(%rsp)
	vcvtss2sd	372(%rsp), %xmm7, %xmm1
	vdivsd	%xmm0, %xmm1, %xmm1
	vcvtss2sd	384(%rsp), %xmm7, %xmm0
	vcvtsd2ss	%xmm1, %xmm1, %xmm1
	vdivsd	%xmm2, %xmm0, %xmm0
	vmovss	380(%rsp), %xmm2
	vmovss	%xmm1, 372(%rsp)
	vmovss	368(%rsp), %xmm1
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, 384(%rsp)
	vmovss	356(%rsp), %xmm0
	call	_Z26spherical_regular_harmonicIfLi6EE19spherical_expansionIT_XT0_EES1_S1_S1_
	vmovss	384(%rsp), %xmm7
	movl	$224, %edx
	movq	%r15, %rsi
	vmovss	372(%rsp), %xmm2
	movq	%r14, %rdi
	vmovss	%xmm7, 20(%rsp)
	vmovss	360(%rsp), %xmm7
	vmovss	%xmm2, 12(%rsp)
	vmovss	%xmm7, 16(%rsp)
	call	memcpy@PLT
	movl	$224, %edx
	movq	%r14, %rsi
	movq	%rbx, %rdi
	call	memcpy@PLT
	vmovss	12(%rsp), %xmm2
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	16(%rsp), %xmm7
	vmulss	%xmm2, %xmm2, %xmm1
	vfmadd132ss	%xmm7, %xmm1, %xmm7
	vmovaps	%xmm7, %xmm1
	vsqrtss	%xmm7, %xmm7, %xmm7
	vmovss	%xmm7, 24(%rsp)
	vucomiss	%xmm1, %xmm14
	ja	.L124
.L114:
	vmovss	.LC12(%rip), %xmm7
	vmovss	20(%rsp), %xmm6
	vdivss	24(%rsp), %xmm7, %xmm7
	vfmadd132ss	%xmm6, %xmm1, %xmm6
	vsqrtss	%xmm6, %xmm6, %xmm3
	vucomiss	%xmm6, %xmm14
	vmovss	%xmm7, 28(%rsp)
	ja	.L125
.L115:
	vmovss	.LC12(%rip), %xmm7
	movq	%r13, %rdx
	movq	%rbx, %rsi
	movq	192(%rsp), %rdi
	vdivss	%xmm3, %xmm7, %xmm6
	vmovss	28(%rsp), %xmm7
	vmulss	16(%rsp), %xmm7, %xmm0
	vmulss	%xmm7, %xmm2, %xmm2
	vmovss	%xmm6, 12(%rsp)
	vmovss	%xmm2, 32(%rsp)
	vmovss	%xmm2, 392(%rsp)
	vmovss	%xmm0, 396(%rsp)
	call	_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE
	movl	$224, %edx
	movq	%r12, %rdi
	movq	%rbx, %rsi
	call	memcpy@PLT
	vmovss	.LC0(%rip), %xmm4
	vxorps	%xmm14, %xmm14, %xmm14
	movq	%r12, %rsi
	vfmadd132ss	1152(%rsp), %xmm14, %xmm4
	vmovss	.LC25(%rip), %xmm0
	movq	%rbx, %rdi
	vmovss	%xmm14, 940(%rsp)
	vaddss	1144(%rsp), %xmm14, %xmm1
	movl	$0x00000000, 924(%rsp)
	vmulss	.LC25(%rip), %xmm1, %xmm1
	vmulss	%xmm4, %xmm0, %xmm0
	vmovss	.LC4(%rip), %xmm4
	vfmadd132ss	1156(%rsp), %xmm14, %xmm4
	vmovss	%xmm1, 928(%rsp)
	vmovss	%xmm0, 920(%rsp)
	vmulss	.LC25(%rip), %xmm4, %xmm0
	vmovss	.LC26(%rip), %xmm4
	vfmadd132ss	1160(%rsp), %xmm14, %xmm4
	vmovss	%xmm0, 932(%rsp)
	vmovaps	%xmm4, %xmm0
	vmovss	.LC27(%rip), %xmm4
	vfmadd231ss	1176(%rsp), %xmm4, %xmm0
	vmulss	.LC3(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 936(%rsp)
	call	_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0
	vmovss	1192(%rsp), %xmm7
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	.LC28(%rip), %xmm4
	movl	$0x00000000, 964(%rsp)
	movq	%r12, %rdx
	movq	%rbx, %rsi
	vmovss	1208(%rsp), %xmm2
	movq	%r13, %rdi
	vmovss	1196(%rsp), %xmm5
	vmovss	.LC26(%rip), %xmm3
	vmovss	1184(%rsp), %xmm6
	vmovss	1200(%rsp), %xmm1
	vfmadd132ss	%xmm7, %xmm14, %xmm4
	vfmadd132ss	.LC0(%rip), %xmm14, %xmm7
	vfmadd132ss	%xmm5, %xmm14, %xmm3
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm5
	vmovaps	%xmm4, %xmm0
	vfmadd231ss	.LC29(%rip), %xmm2, %xmm0
	vfmadd132ss	.LC27(%rip), %xmm7, %xmm2
	vmovss	.LC31(%rip), %xmm4
	vmulss	.LC30(%rip), %xmm0, %xmm0
	vmulss	.LC30(%rip), %xmm2, %xmm2
	vfmadd132ss	%xmm6, %xmm14, %xmm4
	vaddss	%xmm14, %xmm6, %xmm6
	vfmadd231ss	.LC21(%rip), %xmm1, %xmm4
	vfmadd132ss	.LC4(%rip), %xmm6, %xmm1
	vmulss	.LC30(%rip), %xmm4, %xmm4
	vmulss	.LC30(%rip), %xmm1, %xmm1
	vmovss	%xmm0, 960(%rsp)
	vmovss	1212(%rsp), %xmm0
	vmovss	%xmm2, 976(%rsp)
	vfmadd231ss	.LC32(%rip), %xmm0, %xmm3
	vfmadd132ss	.LC4(%rip), %xmm5, %xmm0
	vmulss	.LC30(%rip), %xmm3, %xmm3
	vmovss	%xmm4, 968(%rsp)
	vmovss	%xmm1, 984(%rsp)
	vmovss	%xmm3, 972(%rsp)
	vmovss	.LC2(%rip), %xmm3
	vfmadd132ss	1204(%rsp), %xmm14, %xmm3
	vmulss	.LC30(%rip), %xmm3, %xmm2
	vmovss	%xmm2, 980(%rsp)
	vmulss	.LC30(%rip), %xmm0, %xmm0
	vmovss	1216(%rsp), %xmm5
	vmovss	%xmm14, 996(%rsp)
	vmovss	.LC33(%rip), %xmm1
	vmovss	1232(%rsp), %xmm4
	vmovss	1236(%rsp), %xmm9
	vmovss	1224(%rsp), %xmm10
	vmovss	1252(%rsp), %xmm2
	vmovss	.LC37(%rip), %xmm3
	vmovss	1228(%rsp), %xmm7
	vmovss	.LC40(%rip), %xmm8
	vfmadd132ss	%xmm5, %xmm14, %xmm1
	vfmadd231ss	.LC34(%rip), %xmm4, %xmm1
	vmovss	%xmm0, 988(%rsp)
	vmovss	1248(%rsp), %xmm0
	vfmadd132ss	%xmm10, %xmm14, %xmm3
	vfmadd132ss	%xmm7, %xmm14, %xmm8
	vfmadd231ss	.LC35(%rip), %xmm0, %xmm1
	vmovaps	%xmm3, %xmm6
	vmovss	1240(%rsp), %xmm3
	vfmadd231ss	.LC38(%rip), %xmm3, %xmm6
	vmulss	.LC36(%rip), %xmm1, %xmm1
	vmulss	.LC36(%rip), %xmm6, %xmm6
	vmovss	%xmm1, 992(%rsp)
	vmovss	.LC28(%rip), %xmm1
	vmovss	%xmm6, 1000(%rsp)
	vfmadd132ss	%xmm9, %xmm14, %xmm1
	vfmadd231ss	.LC39(%rip), %xmm2, %xmm1
	vmulss	.LC36(%rip), %xmm1, %xmm1
	vmovss	%xmm1, 1004(%rsp)
	vmovss	.LC40(%rip), %xmm1
	vfmadd132ss	%xmm5, %xmm14, %xmm1
	vaddss	%xmm14, %xmm5, %xmm5
	vmovaps	%xmm1, %xmm6
	vfmadd231ss	.LC2(%rip), %xmm4, %xmm6
	vmovss	1244(%rsp), %xmm1
	vfmadd231ss	.LC38(%rip), %xmm1, %xmm8
	vfmadd231ss	.LC41(%rip), %xmm0, %xmm6
	vmulss	.LC36(%rip), %xmm6, %xmm6
	vmovss	%xmm6, 1008(%rsp)
	vmulss	.LC36(%rip), %xmm8, %xmm8
	vfmadd132ss	.LC4(%rip), %xmm5, %xmm4
	vmovss	%xmm14, 1044(%rsp)
	movq	$0, 1036(%rsp)
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm7
	vfmadd132ss	.LC0(%rip), %xmm14, %xmm10
	vfmadd132ss	.LC2(%rip), %xmm14, %xmm9
	vfmadd132ss	.LC4(%rip), %xmm4, %xmm0
	vmovss	.LC43(%rip), %xmm4
	vfmadd132ss	1264(%rsp), %xmm14, %xmm4
	vfmadd132ss	.LC4(%rip), %xmm7, %xmm1
	vfmadd132ss	.LC27(%rip), %xmm10, %xmm3
	vfmadd132ss	.LC42(%rip), %xmm9, %xmm2
	vmulss	.LC36(%rip), %xmm0, %xmm0
	vmovss	%xmm8, 1012(%rsp)
	vmulss	.LC36(%rip), %xmm3, %xmm3
	vmulss	.LC36(%rip), %xmm2, %xmm2
	vmovss	%xmm0, 1024(%rsp)
	vmulss	.LC36(%rip), %xmm1, %xmm0
	vmovss	%xmm3, 1016(%rsp)
	vmovss	%xmm2, 1020(%rsp)
	vmovss	%xmm0, 1028(%rsp)
	vmovaps	%xmm4, %xmm0
	vmovss	.LC44(%rip), %xmm4
	vfmadd231ss	1280(%rsp), %xmm4, %xmm0
	vmovss	.LC45(%rip), %xmm4
	vfmadd231ss	1296(%rsp), %xmm4, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1032(%rsp)
	call	_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_
	vmovss	.LC46(%rip), %xmm4
	vxorps	%xmm14, %xmm14, %xmm14
	vmulss	1040(%rsp), %xmm4, %xmm0
	movq	%r13, %rdx
	movq	%rbx, %rsi
	vmovss	1264(%rsp), %xmm12
	vmovss	1280(%rsp), %xmm10
	vmovss	1296(%rsp), %xmm5
	vmovss	1276(%rsp), %xmm11
	vmovss	.LC47(%rip), %xmm1
	vmovss	1268(%rsp), %xmm8
	vmovss	.LC51(%rip), %xmm3
	vmovss	1256(%rsp), %xmm9
	vmovss	1272(%rsp), %xmm7
	vmovss	1284(%rsp), %xmm6
	vmovss	1300(%rsp), %xmm2
	vmovss	%xmm0, 1040(%rsp)
	vmulss	1044(%rsp), %xmm4, %xmm0
	vmovss	.LC47(%rip), %xmm4
	vfmadd132ss	%xmm11, %xmm14, %xmm1
	vfmadd132ss	%xmm8, %xmm14, %xmm3
	vfmadd132ss	%xmm12, %xmm14, %xmm4
	vmovss	%xmm0, 1044(%rsp)
	vmovaps	%xmm4, %xmm0
	vfmadd231ss	.LC42(%rip), %xmm10, %xmm0
	vmovss	1292(%rsp), %xmm4
	vfmadd231ss	.LC48(%rip), %xmm4, %xmm1
	vfmadd231ss	.LC49(%rip), %xmm5, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1048(%rsp)
	vmulss	.LC46(%rip), %xmm1, %xmm0
	vmovss	.LC50(%rip), %xmm1
	vfmadd132ss	%xmm9, %xmm14, %xmm1
	vfmadd231ss	.LC33(%rip), %xmm7, %xmm1
	vaddss	%xmm14, %xmm9, %xmm9
	vmovss	%xmm0, 1052(%rsp)
	vmovaps	%xmm3, %xmm0
	vmovss	1288(%rsp), %xmm3
	vfmadd231ss	.LC52(%rip), %xmm6, %xmm0
	vfmadd231ss	.LC53(%rip), %xmm3, %xmm1
	vfmadd231ss	.LC54(%rip), %xmm2, %xmm0
	vmulss	.LC46(%rip), %xmm1, %xmm1
	vmovss	%xmm1, 1056(%rsp)
	vfmadd132ss	.LC2(%rip), %xmm14, %xmm11
	vfmadd132ss	.LC4(%rip), %xmm9, %xmm7
	vmovss	%xmm14, 1084(%rsp)
	vmovss	1304(%rsp), %xmm9
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm8
	vmovss	1352(%rsp), %xmm1
	vfmadd132ss	.LC0(%rip), %xmm14, %xmm12
	vmovss	1312(%rsp), %xmm13
	vfmadd132ss	.LC42(%rip), %xmm11, %xmm4
	vfmadd132ss	.LC4(%rip), %xmm7, %xmm3
	vmovss	1336(%rsp), %xmm7
	vfmadd132ss	.LC4(%rip), %xmm8, %xmm6
	vmovss	1320(%rsp), %xmm8
	vfmadd231ss	.LC27(%rip), %xmm10, %xmm12
	vmulss	.LC46(%rip), %xmm4, %xmm4
	vmulss	.LC46(%rip), %xmm3, %xmm3
	vmovss	%xmm0, 1060(%rsp)
	vfmadd132ss	.LC4(%rip), %xmm6, %xmm2
	vfmadd132ss	.LC55(%rip), %xmm12, %xmm5
	vmulss	.LC46(%rip), %xmm2, %xmm2
	vmulss	.LC46(%rip), %xmm5, %xmm5
	vmovss	%xmm4, 1068(%rsp)
	vmovss	.LC34(%rip), %xmm4
	vmovss	%xmm3, 1072(%rsp)
	vfmadd132ss	%xmm9, %xmm14, %xmm4
	vmovss	%xmm2, 1076(%rsp)
	vmovss	%xmm5, 1064(%rsp)
	vmovaps	%xmm4, %xmm0
	vfmadd231ss	.LC41(%rip), %xmm8, %xmm0
	vfmadd231ss	.LC56(%rip), %xmm7, %xmm0
	vfmadd231ss	.LC57(%rip), %xmm1, %xmm0
	vmulss	.LC58(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1080(%rsp)
	vmovss	.LC29(%rip), %xmm4
	vmovss	1324(%rsp), %xmm5
	vmovss	1340(%rsp), %xmm15
	vmovss	1356(%rsp), %xmm3
	vmovss	1328(%rsp), %xmm12
	vmovss	1316(%rsp), %xmm11
	vmovss	.LC21(%rip), %xmm10
	vfmadd132ss	%xmm13, %xmm14, %xmm4
	vfmadd132ss	%xmm11, %xmm14, %xmm10
	vmovaps	%xmm4, %xmm2
	vmovss	.LC42(%rip), %xmm4
	vfmadd231ss	.LC59(%rip), %xmm12, %xmm2
	vmovaps	%xmm10, %xmm6
	vmovss	1332(%rsp), %xmm10
	vfmadd231ss	.LC65(%rip), %xmm10, %xmm6
	vfmadd132ss	%xmm5, %xmm14, %xmm4
	vmovaps	%xmm4, %xmm0
	vfmadd231ss	.LC60(%rip), %xmm15, %xmm0
	vmovss	1344(%rsp), %xmm4
	vfmadd231ss	.LC61(%rip), %xmm4, %xmm2
	vfmadd231ss	.LC62(%rip), %xmm3, %xmm0
	vmulss	.LC58(%rip), %xmm2, %xmm2
	vmulss	.LC58(%rip), %xmm0, %xmm0
	vmovss	%xmm2, 1088(%rsp)
	vmovss	1348(%rsp), %xmm2
	vfmadd231ss	.LC66(%rip), %xmm2, %xmm6
	vmovss	%xmm0, 1092(%rsp)
	vmovss	.LC63(%rip), %xmm0
	vmulss	.LC58(%rip), %xmm6, %xmm6
	vfmadd132ss	%xmm9, %xmm14, %xmm0
	vfmadd231ss	.LC64(%rip), %xmm8, %xmm0
	vmovss	%xmm6, 1100(%rsp)
	vmovss	.LC34(%rip), %xmm6
	vfmadd231ss	.LC32(%rip), %xmm7, %xmm0
	vfmadd231ss	.LC67(%rip), %xmm1, %xmm0
	vfmadd132ss	%xmm13, %xmm14, %xmm6
	vfmadd231ss	.LC0(%rip), %xmm12, %xmm6
	vmulss	.LC58(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1096(%rsp)
	vmovss	.LC68(%rip), %xmm0
	vfmadd132ss	%xmm5, %xmm14, %xmm0
	vfmadd231ss	.LC69(%rip), %xmm15, %xmm0
	vfmadd231ss	.LC70(%rip), %xmm4, %xmm6
	vfmadd132ss	.LC0(%rip), %xmm14, %xmm13
	vfmadd132ss	.LC2(%rip), %xmm14, %xmm5
	vfmadd231ss	.LC71(%rip), %xmm3, %xmm0
	vmulss	.LC58(%rip), %xmm6, %xmm6
	vfmadd231ss	.LC42(%rip), %xmm15, %xmm5
	vfmadd231ss	.LC27(%rip), %xmm12, %xmm13
	vmulss	.LC58(%rip), %xmm0, %xmm0
	vfmadd132ss	.LC43(%rip), %xmm5, %xmm3
	vfmadd231ss	.LC55(%rip), %xmm4, %xmm13
	vmovss	%xmm6, 1104(%rsp)
	vmovss	.LC28(%rip), %xmm6
	vmulss	.LC58(%rip), %xmm3, %xmm3
	vmulss	.LC58(%rip), %xmm13, %xmm13
	vmovss	%xmm0, 1108(%rsp)
	vmovss	.LC51(%rip), %xmm0
	vfmadd132ss	%xmm11, %xmm14, %xmm6
	vfmadd231ss	.LC43(%rip), %xmm10, %xmm6
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm11
	vmovss	%xmm3, 1124(%rsp)
	vmovss	%xmm13, 1120(%rsp)
	vfmadd132ss	%xmm9, %xmm14, %xmm0
	vfmadd231ss	.LC0(%rip), %xmm8, %xmm0
	vaddss	%xmm14, %xmm9, %xmm9
	vfmadd231ss	.LC73(%rip), %xmm2, %xmm6
	vfmadd132ss	.LC4(%rip), %xmm9, %xmm8
	vfmadd132ss	.LC4(%rip), %xmm11, %xmm10
	vfmadd231ss	.LC72(%rip), %xmm7, %xmm0
	vmulss	.LC58(%rip), %xmm6, %xmm6
	vfmadd132ss	.LC4(%rip), %xmm8, %xmm7
	vfmadd231ss	.LC74(%rip), %xmm1, %xmm0
	vmulss	.LC58(%rip), %xmm0, %xmm0
	vmovss	%xmm6, 1116(%rsp)
	vmovss	%xmm0, 1112(%rsp)
	vfmadd132ss	.LC4(%rip), %xmm10, %xmm2
	vfmadd132ss	.LC4(%rip), %xmm7, %xmm1
	vmovss	24(%rsp), %xmm4
	vmovss	12(%rsp), %xmm6
	vxorps	.LC15(%rip), %xmm4, %xmm4
	vmulss	20(%rsp), %xmm6, %xmm12
	movq	192(%rsp), %rdi
	vmulss	.LC58(%rip), %xmm2, %xmm2
	vmulss	.LC58(%rip), %xmm1, %xmm1
	vmulss	%xmm6, %xmm4, %xmm0
	vmovss	%xmm12, 60(%rsp)
	vmovss	%xmm12, 392(%rsp)
	vmovss	%xmm0, 396(%rsp)
	vmovss	%xmm2, 1132(%rsp)
	vmovss	%xmm1, 1128(%rsp)
	call	_ZN18spherical_rotate_zIfLi6ELb0ELb0EEclER19spherical_expansionIfLi6EERK7complexIfE
	movl	$224, %edx
	movq	%r12, %rdi
	movq	%rbx, %rsi
	call	memcpy@PLT
	vmovss	.LC0(%rip), %xmm3
	vxorps	%xmm14, %xmm14, %xmm14
	movq	%r12, %rsi
	vfmadd132ss	1152(%rsp), %xmm14, %xmm3
	vmovss	.LC25(%rip), %xmm0
	movq	%rbx, %rdi
	vmovss	%xmm14, 940(%rsp)
	vaddss	1144(%rsp), %xmm14, %xmm1
	movl	$0x00000000, 924(%rsp)
	vmulss	.LC25(%rip), %xmm1, %xmm1
	vmulss	%xmm3, %xmm0, %xmm0
	vmovss	.LC4(%rip), %xmm3
	vfmadd132ss	1156(%rsp), %xmm14, %xmm3
	vmovss	%xmm1, 928(%rsp)
	vmovss	%xmm0, 920(%rsp)
	vmulss	.LC25(%rip), %xmm3, %xmm0
	vmovss	.LC26(%rip), %xmm3
	vfmadd132ss	1160(%rsp), %xmm14, %xmm3
	vmovss	%xmm0, 932(%rsp)
	vmovaps	%xmm3, %xmm0
	vmovss	.LC27(%rip), %xmm3
	vfmadd231ss	1176(%rsp), %xmm3, %xmm0
	vmulss	.LC3(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 936(%rsp)
	call	_ZN19spherical_swap_xz_mIfLi6ELi2ELi1ELb0ELb1ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_.isra.0
	vmovss	1192(%rsp), %xmm7
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	.LC28(%rip), %xmm3
	movl	$0x00000000, 964(%rsp)
	movq	%r12, %rdx
	movq	%rbx, %rsi
	vmovss	1184(%rsp), %xmm5
	movq	%r13, %rdi
	movq	%rbp, %r12
	vmovss	1208(%rsp), %xmm2
	vmovss	1196(%rsp), %xmm4
	vmovss	1200(%rsp), %xmm1
	vfmadd132ss	%xmm7, %xmm14, %xmm3
	vfmadd132ss	.LC0(%rip), %xmm14, %xmm7
	vmovaps	%xmm3, %xmm0
	vmovss	.LC31(%rip), %xmm3
	vfmadd231ss	.LC29(%rip), %xmm2, %xmm0
	vfmadd132ss	.LC27(%rip), %xmm7, %xmm2
	vmulss	.LC30(%rip), %xmm0, %xmm0
	vmulss	.LC30(%rip), %xmm2, %xmm2
	vfmadd132ss	%xmm5, %xmm14, %xmm3
	vaddss	%xmm14, %xmm5, %xmm5
	vmovaps	%xmm3, %xmm6
	vmovss	.LC26(%rip), %xmm3
	vfmadd231ss	.LC21(%rip), %xmm1, %xmm6
	vmovss	%xmm0, 960(%rsp)
	vmovss	1212(%rsp), %xmm0
	vfmadd132ss	.LC4(%rip), %xmm5, %xmm1
	vmovss	%xmm2, 976(%rsp)
	vmulss	.LC30(%rip), %xmm6, %xmm6
	vmulss	.LC30(%rip), %xmm1, %xmm1
	vfmadd132ss	%xmm4, %xmm14, %xmm3
	vfmadd231ss	.LC32(%rip), %xmm0, %xmm3
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm4
	vmulss	.LC30(%rip), %xmm3, %xmm3
	vfmadd132ss	.LC4(%rip), %xmm4, %xmm0
	vmovss	%xmm6, 968(%rsp)
	vmovss	%xmm1, 984(%rsp)
	vmovss	%xmm3, 972(%rsp)
	vmovss	.LC2(%rip), %xmm3
	vfmadd132ss	1204(%rsp), %xmm14, %xmm3
	vmulss	.LC30(%rip), %xmm3, %xmm2
	vmovss	%xmm2, 980(%rsp)
	vmulss	.LC30(%rip), %xmm0, %xmm0
	vmovss	1216(%rsp), %xmm3
	vmovss	%xmm14, 996(%rsp)
	vmovss	.LC33(%rip), %xmm1
	vmovss	.LC37(%rip), %xmm13
	vfmadd132ss	1224(%rsp), %xmm14, %xmm13
	vmovss	1232(%rsp), %xmm2
	vfmadd132ss	%xmm3, %xmm14, %xmm1
	vmovss	%xmm0, 988(%rsp)
	vfmadd132ss	.LC40(%rip), %xmm14, %xmm3
	vmovaps	%xmm1, %xmm0
	vfmadd231ss	.LC34(%rip), %xmm2, %xmm0
	vmovss	1248(%rsp), %xmm1
	vmovaps	%xmm13, %xmm4
	vmovss	.LC28(%rip), %xmm13
	vfmadd132ss	.LC2(%rip), %xmm3, %xmm2
	vfmadd132ss	1236(%rsp), %xmm14, %xmm13
	vmovss	.LC38(%rip), %xmm3
	vfmadd231ss	.LC35(%rip), %xmm1, %xmm0
	vfmadd132ss	.LC41(%rip), %xmm2, %xmm1
	vmulss	.LC36(%rip), %xmm0, %xmm0
	vmulss	.LC36(%rip), %xmm1, %xmm1
	vmovss	%xmm0, 992(%rsp)
	vmovaps	%xmm13, %xmm0
	vmovss	.LC38(%rip), %xmm13
	vfmadd231ss	1240(%rsp), %xmm13, %xmm4
	vmovss	%xmm1, 1008(%rsp)
	vmovss	.LC39(%rip), %xmm13
	vfmadd231ss	1252(%rsp), %xmm13, %xmm0
	vmovss	.LC40(%rip), %xmm13
	vfmadd132ss	1228(%rsp), %xmm14, %xmm13
	vmulss	.LC36(%rip), %xmm4, %xmm4
	vmulss	.LC36(%rip), %xmm0, %xmm0
	vmovss	%xmm4, 1000(%rsp)
	vmovss	%xmm0, 1004(%rsp)
	vmovaps	%xmm13, %xmm0
	vfmadd231ss	1244(%rsp), %xmm3, %xmm0
	vmulss	.LC36(%rip), %xmm0, %xmm0
	vmovss	.LC43(%rip), %xmm3
	vmovss	%xmm14, 1044(%rsp)
	movq	$0, 1036(%rsp)
	vfmadd132ss	1264(%rsp), %xmm14, %xmm3
	vmovss	%xmm0, 1012(%rsp)
	vmovaps	%xmm3, %xmm0
	vmovss	.LC44(%rip), %xmm3
	vfmadd231ss	1280(%rsp), %xmm3, %xmm0
	vmovss	.LC45(%rip), %xmm3
	vfmadd231ss	1296(%rsp), %xmm3, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1032(%rsp)
	call	_ZN19spherical_swap_xz_lIfLi6ELi5ELi1ELi0ELb0ELb0ELb0ELb0EEclER19spherical_expansionIfLi6EERKS2_
	vmovss	12(%rsp), %xmm0
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	.LC46(%rip), %xmm3
	vmovss	%xmm14, 1140(%rsp)
	movl	$0x00000000, 1084(%rsp)
	movq	%rbp, %rdi
	vmulss	1040(%rsp), %xmm3, %xmm3
	vmovss	.LC46(%rip), %xmm1
	vmulss	1044(%rsp), %xmm1, %xmm13
	vmovss	.LC34(%rip), %xmm1
	vfmadd132ss	1304(%rsp), %xmm14, %xmm1
	vmulss	%xmm0, %xmm0, %xmm7
	vmovaps	%xmm0, %xmm15
	vfmadd132ss	912(%rsp), %xmm14, %xmm15
	vmovss	%xmm3, 20(%rsp)
	vmovss	%xmm3, 1040(%rsp)
	vfmadd231ss	920(%rsp), %xmm7, %xmm15
	vmulss	%xmm0, %xmm7, %xmm3
	vmovss	%xmm13, 1044(%rsp)
	vmulss	%xmm0, %xmm3, %xmm6
	vaddss	%xmm3, %xmm3, %xmm10
	vmulss	.LC33(%rip), %xmm6, %xmm11
	vmovaps	%xmm1, %xmm12
	vmovss	.LC41(%rip), %xmm1
	vfmadd231ss	1320(%rsp), %xmm1, %xmm12
	vmovss	.LC56(%rip), %xmm1
	vfmadd231ss	936(%rsp), %xmm10, %xmm15
	vmulss	%xmm0, %xmm6, %xmm5
	vmulss	%xmm0, %xmm5, %xmm4
	vfmadd231ss	1336(%rsp), %xmm1, %xmm12
	vmovss	.LC57(%rip), %xmm1
	vfmadd231ss	960(%rsp), %xmm11, %xmm15
	vmulss	.LC75(%rip), %xmm4, %xmm9
	vmulss	%xmm0, %xmm4, %xmm2
	vfmadd231ss	1352(%rsp), %xmm1, %xmm12
	vmulss	.LC76(%rip), %xmm2, %xmm8
	vmulss	%xmm0, %xmm2, %xmm1
	vmulss	.LC43(%rip), %xmm5, %xmm0
	vmulss	.LC58(%rip), %xmm12, %xmm12
	vfmadd231ss	992(%rsp), %xmm0, %xmm15
	vfmadd231ss	1032(%rsp), %xmm9, %xmm15
	vmovss	%xmm12, 1080(%rsp)
	vfmadd231ss	%xmm8, %xmm12, %xmm15
	vmovss	%xmm15, 1136(%rsp)
	vmovaps	%xmm7, %xmm15
	vfmadd132ss	912(%rsp), %xmm14, %xmm15
	vfmadd231ss	920(%rsp), %xmm10, %xmm15
	vmulss	.LC77(%rip), %xmm1, %xmm7
	vmulss	.LC26(%rip), %xmm3, %xmm3
	vmulss	.LC51(%rip), %xmm6, %xmm6
	vmulss	.LC68(%rip), %xmm5, %xmm5
	vfmadd231ss	936(%rsp), %xmm11, %xmm15
	vmulss	.LC78(%rip), %xmm4, %xmm4
	vmulss	.LC79(%rip), %xmm2, %xmm2
	vmulss	.LC80(%rip), %xmm1, %xmm1
	vfmadd132ss	912(%rsp), %xmm14, %xmm10
	vfmadd231ss	960(%rsp), %xmm0, %xmm15
	vfmadd231ss	992(%rsp), %xmm9, %xmm15
	vfmadd231ss	1032(%rsp), %xmm8, %xmm15
	vfmadd132ss	%xmm7, %xmm15, %xmm12
	vmovss	928(%rsp), %xmm15
	vmovss	%xmm12, 64(%rsp)
	vmovaps	%xmm3, %xmm12
	vfmadd132ss	932(%rsp), %xmm14, %xmm3
	vfmadd132ss	%xmm15, %xmm14, %xmm12
	vfmadd231ss	944(%rsp), %xmm6, %xmm12
	vfmadd231ss	948(%rsp), %xmm6, %xmm3
	vfmadd231ss	968(%rsp), %xmm5, %xmm12
	vfmadd231ss	972(%rsp), %xmm5, %xmm3
	vfmadd231ss	1000(%rsp), %xmm4, %xmm12
	vfmadd231ss	1004(%rsp), %xmm4, %xmm3
	vfmadd231ss	20(%rsp), %xmm2, %xmm12
	vfmadd231ss	1088(%rsp), %xmm1, %xmm12
	vfmadd231ss	%xmm2, %xmm13, %xmm3
	vfmadd231ss	1092(%rsp), %xmm1, %xmm3
	vmovss	%xmm12, 68(%rsp)
	vmovss	%xmm3, 72(%rsp)
	vmovaps	%xmm10, %xmm3
	vfmadd231ss	920(%rsp), %xmm11, %xmm3
	vmovaps	%xmm15, %xmm10
	vfmadd132ss	%xmm6, %xmm14, %xmm10
	vfmadd231ss	936(%rsp), %xmm0, %xmm3
	vfmadd231ss	960(%rsp), %xmm9, %xmm3
	vfmadd231ss	992(%rsp), %xmm8, %xmm3
	vfmadd231ss	1032(%rsp), %xmm7, %xmm3
	vmovss	%xmm3, 76(%rsp)
	vfmadd132ss	932(%rsp), %xmm14, %xmm6
	vmovaps	%xmm10, %xmm3
	vfmadd231ss	944(%rsp), %xmm5, %xmm3
	vmovaps	%xmm0, %xmm10
	vfmadd132ss	912(%rsp), %xmm14, %xmm11
	vfmadd231ss	948(%rsp), %xmm5, %xmm6
	vfmadd231ss	968(%rsp), %xmm4, %xmm3
	vfmadd231ss	920(%rsp), %xmm0, %xmm11
	vfmadd231ss	972(%rsp), %xmm4, %xmm6
	vfmadd231ss	1000(%rsp), %xmm2, %xmm3
	vfmadd231ss	936(%rsp), %xmm9, %xmm11
	vfmadd231ss	1004(%rsp), %xmm2, %xmm6
	vfmadd231ss	20(%rsp), %xmm1, %xmm3
	vfmadd231ss	960(%rsp), %xmm8, %xmm11
	vfmadd231ss	992(%rsp), %xmm7, %xmm11
	vfmadd132ss	%xmm1, %xmm6, %xmm13
	vmovss	952(%rsp), %xmm6
	vmovss	%xmm3, 80(%rsp)
	vmovss	956(%rsp), %xmm3
	vmovss	%xmm13, 84(%rsp)
	vmovss	%xmm11, 92(%rsp)
	vmovaps	%xmm9, %xmm11
	vfmadd132ss	%xmm6, %xmm14, %xmm10
	vfmadd132ss	%xmm6, %xmm14, %xmm11
	vmovaps	%xmm10, %xmm12
	vfmadd231ss	976(%rsp), %xmm9, %xmm12
	vmovaps	%xmm0, %xmm10
	vfmadd132ss	%xmm3, %xmm14, %xmm10
	vfmadd231ss	980(%rsp), %xmm9, %xmm10
	vfmadd231ss	1008(%rsp), %xmm8, %xmm12
	vfmadd231ss	1012(%rsp), %xmm8, %xmm10
	vfmadd231ss	1052(%rsp), %xmm7, %xmm10
	vmovaps	%xmm12, %xmm13
	vfmadd231ss	1048(%rsp), %xmm7, %xmm13
	vmovaps	%xmm2, %xmm12
	vmovss	%xmm10, 88(%rsp)
	vmovaps	%xmm11, %xmm10
	vmovss	%xmm13, 20(%rsp)
	vmovaps	%xmm15, %xmm13
	vfmadd132ss	%xmm5, %xmm14, %xmm13
	vfmadd132ss	932(%rsp), %xmm14, %xmm5
	vfmadd231ss	944(%rsp), %xmm4, %xmm13
	vfmadd231ss	948(%rsp), %xmm4, %xmm5
	vfmadd231ss	968(%rsp), %xmm2, %xmm13
	vfmadd231ss	972(%rsp), %xmm2, %xmm5
	vfmadd231ss	1000(%rsp), %xmm1, %xmm13
	vfmadd231ss	1004(%rsp), %xmm1, %xmm5
	vfmadd231ss	976(%rsp), %xmm8, %xmm10
	vmovss	%xmm14, 1260(%rsp)
	vfmadd132ss	912(%rsp), %xmm14, %xmm0
	vfmadd231ss	920(%rsp), %xmm9, %xmm0
	vfmadd231ss	936(%rsp), %xmm8, %xmm0
	vmovaps	%xmm10, %xmm11
	vmovss	988(%rsp), %xmm10
	vfmadd231ss	1008(%rsp), %xmm7, %xmm11
	vmovss	%xmm5, 40(%rsp)
	vmovaps	%xmm9, %xmm5
	vfmadd132ss	912(%rsp), %xmm14, %xmm9
	vfmadd132ss	%xmm3, %xmm14, %xmm5
	vfmadd231ss	980(%rsp), %xmm8, %xmm5
	vfmadd231ss	960(%rsp), %xmm7, %xmm0
	vfmadd231ss	920(%rsp), %xmm8, %xmm9
	vfmadd231ss	1012(%rsp), %xmm7, %xmm5
	vfmadd132ss	%xmm10, %xmm14, %xmm12
	vfmadd231ss	1020(%rsp), %xmm1, %xmm12
	vfmadd132ss	%xmm1, %xmm14, %xmm10
	vmovss	%xmm11, 44(%rsp)
	vmovss	984(%rsp), %xmm11
	vmovss	%xmm10, 52(%rsp)
	vmovss	%xmm5, 96(%rsp)
	vmovaps	%xmm2, %xmm5
	vmovss	%xmm12, 36(%rsp)
	vfmadd132ss	%xmm11, %xmm14, %xmm5
	vmovaps	%xmm15, %xmm12
	vfmadd231ss	1016(%rsp), %xmm1, %xmm5
	vfmadd132ss	%xmm4, %xmm14, %xmm12
	vfmadd132ss	932(%rsp), %xmm14, %xmm4
	vfmadd231ss	944(%rsp), %xmm2, %xmm12
	vfmadd132ss	%xmm1, %xmm14, %xmm11
	vfmadd231ss	948(%rsp), %xmm2, %xmm4
	vfmadd231ss	968(%rsp), %xmm1, %xmm12
	vmovss	%xmm11, 48(%rsp)
	vfmadd231ss	972(%rsp), %xmm1, %xmm4
	vmovss	%xmm12, 108(%rsp)
	vmovaps	%xmm8, %xmm12
	vfmadd132ss	%xmm3, %xmm14, %xmm12
	vfmadd231ss	980(%rsp), %xmm7, %xmm12
	vfmadd132ss	%xmm7, %xmm14, %xmm3
	vmovss	%xmm4, 56(%rsp)
	vmovaps	%xmm8, %xmm4
	vfmadd132ss	%xmm6, %xmm14, %xmm4
	vfmadd231ss	976(%rsp), %xmm7, %xmm4
	vfmadd132ss	%xmm7, %xmm14, %xmm6
	vmovss	%xmm4, 104(%rsp)
	vfmadd231ss	936(%rsp), %xmm7, %xmm9
	vmovss	%xmm6, 112(%rsp)
	vfmadd132ss	912(%rsp), %xmm14, %xmm8
	vmovss	%xmm3, 116(%rsp)
	vmovss	%xmm14, 1308(%rsp)
	movq	$0, 1376(%rsp)
	vmovss	%xmm14, 1332(%rsp)
	vmovss	%xmm14, 1348(%rsp)
	vmovss	%xmm14, 1372(%rsp)
	vmovss	%xmm14, 1388(%rsp)
	vmovss	%xmm14, 1404(%rsp)
	vmovss	%xmm14, 1420(%rsp)
	movq	$0, 1392(%rsp)
	movq	$0, 1408(%rsp)
	vfmadd231ss	920(%rsp), %xmm7, %xmm8
	vfmadd132ss	912(%rsp), %xmm14, %xmm7
	vmovss	%xmm9, 100(%rsp)
	vmovaps	%xmm15, %xmm9
	vfmadd132ss	%xmm1, %xmm14, %xmm15
	vfmadd132ss	%xmm2, %xmm14, %xmm9
	vfmadd132ss	932(%rsp), %xmm14, %xmm2
	vfmadd231ss	944(%rsp), %xmm1, %xmm9
	vmovss	%xmm15, 120(%rsp)
	vmovaps	%xmm1, %xmm15
	vfmadd132ss	932(%rsp), %xmm14, %xmm15
	vfmadd231ss	948(%rsp), %xmm1, %xmm2
	vmovss	.LC4(%rip), %xmm1
	vfmadd132ss	68(%rsp), %xmm14, %xmm1
	vmovaps	%xmm15, %xmm4
	vmulss	.LC25(%rip), %xmm1, %xmm15
	vmovss	.LC4(%rip), %xmm1
	vfmadd132ss	72(%rsp), %xmm14, %xmm1
	vmovss	%xmm15, 68(%rsp)
	vmovss	%xmm15, 1144(%rsp)
	vmovss	.LC4(%rip), %xmm15
	vfmadd132ss	64(%rsp), %xmm14, %xmm15
	vmulss	.LC25(%rip), %xmm15, %xmm6
	vmovss	%xmm6, 64(%rsp)
	vmovss	76(%rsp), %xmm6
	vmulss	.LC25(%rip), %xmm1, %xmm15
	vmovss	.LC26(%rip), %xmm1
	vmovss	.LC4(%rip), %xmm11
	vfmadd132ss	%xmm6, %xmm14, %xmm1
	vfmadd231ss	20(%rsp), %xmm11, %xmm1
	vmovss	%xmm15, 72(%rsp)
	vmovaps	%xmm11, %xmm10
	vfmadd132ss	88(%rsp), %xmm14, %xmm10
	vmulss	.LC3(%rip), %xmm1, %xmm15
	vmovss	.LC0(%rip), %xmm1
	vfmadd132ss	80(%rsp), %xmm14, %xmm1
	vmulss	.LC3(%rip), %xmm1, %xmm3
	vmovss	.LC2(%rip), %xmm1
	vfmadd132ss	84(%rsp), %xmm14, %xmm1
	vmovss	%xmm15, 1160(%rsp)
	vmovss	%xmm3, 76(%rsp)
	vmulss	.LC3(%rip), %xmm10, %xmm3
	vmovss	%xmm3, 80(%rsp)
	vmovss	.LC33(%rip), %xmm3
	vfmadd132ss	%xmm6, %xmm14, %xmm3
	vmulss	.LC3(%rip), %xmm1, %xmm6
	vfmadd231ss	20(%rsp), %xmm11, %xmm3
	vmulss	.LC3(%rip), %xmm3, %xmm10
	vmovss	.LC40(%rip), %xmm3
	vmovss	%xmm6, 84(%rsp)
	vmovss	.LC51(%rip), %xmm6
	vmovss	%xmm10, 20(%rsp)
	vmovss	92(%rsp), %xmm10
	vfmadd132ss	%xmm13, %xmm14, %xmm6
	vmovaps	%xmm6, %xmm1
	vfmadd132ss	%xmm10, %xmm14, %xmm3
	vfmadd231ss	%xmm11, %xmm5, %xmm1
	vmovss	.LC26(%rip), %xmm11
	vmulss	.LC30(%rip), %xmm1, %xmm6
	vfmadd132ss	40(%rsp), %xmm14, %xmm11
	vmovss	%xmm6, 1184(%rsp)
	vmovaps	%xmm11, %xmm1
	vmovss	.LC0(%rip), %xmm11
	vfmadd231ss	44(%rsp), %xmm11, %xmm3
	vmovss	.LC4(%rip), %xmm11
	vfmadd231ss	36(%rsp), %xmm11, %xmm1
	vfmadd132ss	.LC21(%rip), %xmm14, %xmm13
	vmulss	.LC30(%rip), %xmm3, %xmm11
	vfmadd231ss	.LC4(%rip), %xmm5, %xmm13
	vmulss	.LC30(%rip), %xmm13, %xmm13
	vmovss	%xmm11, 88(%rsp)
	vmulss	.LC30(%rip), %xmm1, %xmm11
	vmovss	.LC33(%rip), %xmm1
	vfmadd132ss	%xmm0, %xmm14, %xmm1
	vmovss	%xmm11, 92(%rsp)
	vmovss	.LC2(%rip), %xmm11
	vfmadd132ss	96(%rsp), %xmm14, %xmm11
	vmovss	%xmm13, 96(%rsp)
	vmulss	.LC30(%rip), %xmm11, %xmm13
	vmovss	%xmm13, 124(%rsp)
	vmovss	.LC55(%rip), %xmm13
	vfmadd132ss	%xmm10, %xmm14, %xmm13
	vmovaps	%xmm13, %xmm10
	vmovss	.LC32(%rip), %xmm13
	vfmadd132ss	40(%rsp), %xmm14, %xmm13
	vmovaps	%xmm13, %xmm5
	vmovss	.LC27(%rip), %xmm13
	vfmadd231ss	44(%rsp), %xmm13, %xmm10
	vmovss	.LC4(%rip), %xmm13
	vfmadd231ss	36(%rsp), %xmm13, %xmm5
	vmulss	.LC30(%rip), %xmm10, %xmm13
	vmovss	108(%rsp), %xmm10
	vmulss	.LC30(%rip), %xmm5, %xmm11
	vmovss	104(%rsp), %xmm5
	vmovss	%xmm13, 36(%rsp)
	vmovss	.LC28(%rip), %xmm13
	vmovss	%xmm11, 40(%rsp)
	vfmadd231ss	%xmm5, %xmm13, %xmm1
	vmulss	.LC36(%rip), %xmm1, %xmm13
	vmovss	.LC37(%rip), %xmm1
	vfmadd132ss	%xmm10, %xmm14, %xmm1
	vmovss	%xmm13, 1216(%rsp)
	vmovss	.LC40(%rip), %xmm3
	vmovss	48(%rsp), %xmm11
	vfmadd132ss	%xmm14, %xmm14, %xmm13
	vfmadd231ss	.LC0(%rip), %xmm11, %xmm1
	vmulss	.LC36(%rip), %xmm1, %xmm1
	vfmadd132ss	%xmm12, %xmm14, %xmm3
	vmulss	.LC36(%rip), %xmm3, %xmm11
	vmovss	.LC81(%rip), %xmm3
	vfmadd132ss	.LC38(%rip), %xmm14, %xmm12
	vmulss	.LC36(%rip), %xmm12, %xmm12
	vfmadd132ss	%xmm0, %xmm14, %xmm3
	vmovss	%xmm1, 44(%rsp)
	vfmadd132ss	.LC82(%rip), %xmm14, %xmm0
	vmovss	%xmm11, 104(%rsp)
	vmovss	.LC28(%rip), %xmm11
	vfmadd132ss	56(%rsp), %xmm14, %xmm11
	vmovss	%xmm12, 1244(%rsp)
	vmovaps	%xmm11, %xmm1
	vmovss	.LC2(%rip), %xmm11
	vfmadd231ss	52(%rsp), %xmm11, %xmm1
	vfmadd231ss	%xmm5, %xmm11, %xmm3
	vmulss	.LC36(%rip), %xmm1, %xmm11
	vmovss	.LC38(%rip), %xmm1
	vmulss	.LC36(%rip), %xmm3, %xmm3
	vfmadd132ss	%xmm10, %xmm14, %xmm1
	vmovss	.LC39(%rip), %xmm10
	vfmadd132ss	56(%rsp), %xmm14, %xmm10
	vmovss	%xmm11, 128(%rsp)
	vmovss	48(%rsp), %xmm11
	vfmadd132ss	.LC27(%rip), %xmm1, %xmm11
	vmovss	%xmm3, 108(%rsp)
	vmovss	.LC41(%rip), %xmm1
	vmulss	.LC36(%rip), %xmm11, %xmm11
	vfmadd231ss	%xmm5, %xmm1, %xmm0
	vmulss	.LC36(%rip), %xmm0, %xmm0
	vmovss	.LC42(%rip), %xmm5
	vfmadd231ss	52(%rsp), %xmm5, %xmm10
	vmovss	%xmm11, 1240(%rsp)
	vmovss	%xmm0, 1248(%rsp)
	vmulss	.LC36(%rip), %xmm10, %xmm10
	vmovss	.LC55(%rip), %xmm0
	vmovss	100(%rsp), %xmm5
	vmovss	112(%rsp), %xmm11
	vmovss	.LC0(%rip), %xmm1
	vmovss	.LC42(%rip), %xmm12
	vmovss	.LC48(%rip), %xmm3
	vfmadd132ss	%xmm9, %xmm14, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vfmadd132ss	%xmm2, %xmm14, %xmm1
	vmovss	%xmm10, 1252(%rsp)
	vmulss	.LC46(%rip), %xmm1, %xmm1
	vmovss	.LC47(%rip), %xmm10
	vmovss	%xmm0, 1256(%rsp)
	vmovss	.LC27(%rip), %xmm0
	vmovss	%xmm1, 1268(%rsp)
	vmovss	.LC22(%rip), %xmm1
	vfmadd132ss	%xmm5, %xmm14, %xmm0
	vfmadd132ss	%xmm9, %xmm14, %xmm1
	vmulss	.LC46(%rip), %xmm1, %xmm1
	vfmadd132ss	.LC23(%rip), %xmm14, %xmm9
	vfmadd231ss	%xmm11, %xmm10, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vmulss	.LC46(%rip), %xmm9, %xmm9
	vmovss	%xmm1, 1272(%rsp)
	vmovss	.LC22(%rip), %xmm1
	vmovss	%xmm0, 1264(%rsp)
	vmovaps	%xmm10, %xmm0
	vmovss	116(%rsp), %xmm10
	vmovss	%xmm9, 1288(%rsp)
	vfmadd132ss	%xmm2, %xmm14, %xmm1
	vmulss	.LC46(%rip), %xmm1, %xmm1
	vfmadd132ss	%xmm10, %xmm14, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vfmadd132ss	%xmm10, %xmm14, %xmm3
	vmulss	.LC46(%rip), %xmm3, %xmm3
	vmovss	%xmm1, 1284(%rsp)
	vmovss	%xmm0, 1276(%rsp)
	vmovss	.LC22(%rip), %xmm0
	vfmadd132ss	%xmm5, %xmm14, %xmm0
	vfmadd231ss	%xmm11, %xmm12, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 1280(%rsp)
	vmovss	%xmm3, 1292(%rsp)
	vmovss	.LC83(%rip), %xmm9
	vmovss	.LC49(%rip), %xmm10
	vmovss	%xmm14, 1316(%rsp)
	vmovss	.LC58(%rip), %xmm0
	vfmadd132ss	.LC24(%rip), %xmm14, %xmm2
	vfmadd132ss	%xmm5, %xmm14, %xmm9
	vmovss	120(%rsp), %xmm5
	vmulss	.LC46(%rip), %xmm2, %xmm2
	vfmadd231ss	%xmm11, %xmm10, %xmm9
	vmulss	.LC46(%rip), %xmm9, %xmm9
	vmovss	%xmm2, 1300(%rsp)
	vmovss	%xmm9, 1296(%rsp)
	vmovss	.LC34(%rip), %xmm9
	vfmadd132ss	%xmm8, %xmm14, %xmm9
	vmulss	%xmm9, %xmm0, %xmm0
	vmovss	.LC29(%rip), %xmm9
	vmovss	%xmm0, 1304(%rsp)
	vmovss	.LC58(%rip), %xmm0
	vfmadd132ss	%xmm5, %xmm14, %xmm9
	vmulss	%xmm9, %xmm0, %xmm0
	vmovss	.LC38(%rip), %xmm9
	vmovss	%xmm0, 1312(%rsp)
	vfmadd132ss	%xmm8, %xmm14, %xmm9
	vmovaps	%xmm9, %xmm1
	vmovaps	%xmm12, %xmm9
	vmulss	.LC58(%rip), %xmm1, %xmm1
	vfmadd132ss	%xmm4, %xmm14, %xmm9
	vmulss	.LC58(%rip), %xmm9, %xmm0
	vmovss	.LC59(%rip), %xmm9
	vfmadd132ss	%xmm5, %xmm14, %xmm9
	vmovss	%xmm1, 1320(%rsp)
	vmovss	%xmm0, 1324(%rsp)
	vmovss	.LC58(%rip), %xmm0
	vmulss	%xmm9, %xmm0, %xmm0
	vmovss	.LC84(%rip), %xmm9
	vmovss	%xmm0, 1328(%rsp)
	vfmadd132ss	%xmm8, %xmm14, %xmm9
	vmovaps	%xmm9, %xmm1
	vmovss	.LC60(%rip), %xmm9
	vmulss	.LC58(%rip), %xmm1, %xmm1
	vfmadd132ss	%xmm4, %xmm14, %xmm9
	vmulss	.LC58(%rip), %xmm9, %xmm0
	vmovss	.LC61(%rip), %xmm9
	vmovss	%xmm1, 1336(%rsp)
	vfmadd132ss	%xmm5, %xmm14, %xmm9
	vmovss	%xmm0, 1340(%rsp)
	vmovss	.LC58(%rip), %xmm0
	vmulss	%xmm9, %xmm0, %xmm0
	vmovss	%xmm0, 1344(%rsp)
	vfmadd132ss	.LC85(%rip), %xmm14, %xmm8
	vmovss	.LC87(%rip), %xmm0
	movq	$0, 1360(%rsp)
	movq	$1065353216, 400(%rsp)
	vmovss	60(%rsp), %xmm12
	vmovss	.LC62(%rip), %xmm9
	vmulss	.LC58(%rip), %xmm8, %xmm8
	vmovaps	%xmm12, %xmm11
	vfmadd132ss	%xmm4, %xmm14, %xmm9
	vmulss	.LC58(%rip), %xmm9, %xmm1
	vmovaps	%xmm12, %xmm9
	vmovss	%xmm8, 1352(%rsp)
	vmovss	.LC86(%rip), %xmm8
	vmovss	%xmm1, 1356(%rsp)
	vfmadd132ss	%xmm7, %xmm14, %xmm8
	vmulss	%xmm8, %xmm0, %xmm0
	vmovss	.LC88(%rip), %xmm8
	vmovss	%xmm0, 1368(%rsp)
	vmovss	.LC87(%rip), %xmm0
	vfmadd132ss	%xmm7, %xmm14, %xmm8
	vmulss	%xmm8, %xmm0, %xmm0
	vmovss	.LC89(%rip), %xmm8
	vmovss	%xmm0, 1384(%rsp)
	vmovss	.LC87(%rip), %xmm0
	vfmadd132ss	%xmm7, %xmm14, %xmm8
	vfmadd132ss	.LC90(%rip), %xmm14, %xmm7
	vmulss	.LC87(%rip), %xmm7, %xmm7
	vmulss	%xmm8, %xmm0, %xmm0
	vmovss	%xmm0, 1400(%rsp)
	vmovss	%xmm7, 1416(%rsp)
	vmovss	12(%rsp), %xmm7
	vmulss	24(%rsp), %xmm7, %xmm0
	vfmadd132ss	%xmm14, %xmm0, %xmm9
	vfnmadd231ss	%xmm14, %xmm0, %xmm11
	vmulss	%xmm12, %xmm9, %xmm7
	vmulss	%xmm9, %xmm0, %xmm8
	vmovss	%xmm9, 412(%rsp)
	vmovss	%xmm11, 408(%rsp)
	vfmadd231ss	%xmm11, %xmm0, %xmm7
	vfmsub231ss	%xmm12, %xmm11, %xmm8
	vmulss	%xmm12, %xmm7, %xmm10
	vmulss	%xmm7, %xmm0, %xmm5
	vmovss	%xmm7, 420(%rsp)
	vmovss	%xmm8, 416(%rsp)
	vfmadd231ss	%xmm8, %xmm0, %xmm10
	vfmsub231ss	%xmm12, %xmm8, %xmm5
	vmulss	%xmm10, %xmm0, %xmm4
	vmovss	%xmm5, 424(%rsp)
	vmovss	%xmm10, 428(%rsp)
	vmovaps	%xmm4, %xmm1
	vmulss	%xmm12, %xmm10, %xmm4
	vfmsub231ss	%xmm12, %xmm5, %xmm1
	vfmadd231ss	%xmm5, %xmm0, %xmm4
	vmovss	%xmm1, 432(%rsp)
	vmulss	%xmm4, %xmm0, %xmm2
	vmovss	%xmm4, 436(%rsp)
	vmulss	%xmm12, %xmm4, %xmm4
	vmovaps	%xmm2, %xmm3
	vfmadd231ss	%xmm1, %xmm0, %xmm4
	vfmsub231ss	%xmm12, %xmm1, %xmm3
	vmulss	%xmm12, %xmm4, %xmm2
	vmulss	%xmm4, %xmm0, %xmm1
	vmovss	%xmm3, 440(%rsp)
	vmovss	%xmm4, 444(%rsp)
	vmovss	64(%rsp), %xmm4
	vfmadd231ss	%xmm3, %xmm0, %xmm2
	vfmsub231ss	%xmm12, %xmm3, %xmm1
	vmovss	%xmm2, 452(%rsp)
	vmulss	%xmm2, %xmm0, %xmm3
	vmulss	%xmm12, %xmm2, %xmm2
	vmovss	%xmm1, 448(%rsp)
	vfmsub231ss	%xmm12, %xmm1, %xmm3
	vmovss	72(%rsp), %xmm12
	vfmadd231ss	%xmm1, %xmm0, %xmm2
	vmovss	68(%rsp), %xmm1
	vmovss	%xmm3, 456(%rsp)
	vmovss	%xmm2, 460(%rsp)
	vmulss	%xmm12, %xmm9, %xmm0
	vmulss	%xmm12, %xmm11, %xmm3
	vfmadd132ss	%xmm14, %xmm14, %xmm1
	vfmadd231ss	%xmm4, %xmm9, %xmm3
	vfmsub231ss	%xmm4, %xmm11, %xmm0
	vmovss	76(%rsp), %xmm4
	vmovss	%xmm1, 1148(%rsp)
	vmovaps	%xmm15, %xmm1
	vmovss	80(%rsp), %xmm15
	vmovss	%xmm3, 1156(%rsp)
	vfmadd132ss	%xmm14, %xmm14, %xmm1
	vmovss	%xmm0, 1152(%rsp)
	vmovss	%xmm1, 1164(%rsp)
	vmovaps	%xmm6, %xmm1
	vfmadd132ss	%xmm14, %xmm14, %xmm1
	vmulss	%xmm15, %xmm9, %xmm0
	vmulss	%xmm15, %xmm11, %xmm3
	vmovss	84(%rsp), %xmm15
	vfmadd231ss	%xmm4, %xmm9, %xmm3
	vfmsub231ss	%xmm4, %xmm11, %xmm0
	vmovss	20(%rsp), %xmm4
	vmovss	%xmm1, 1188(%rsp)
	vmovss	%xmm0, 1168(%rsp)
	vmulss	%xmm15, %xmm7, %xmm0
	vmovss	%xmm3, 1172(%rsp)
	vmulss	%xmm15, %xmm8, %xmm3
	vmovss	92(%rsp), %xmm15
	vfmadd231ss	%xmm4, %xmm7, %xmm3
	vfmsub231ss	%xmm4, %xmm8, %xmm0
	vmovss	88(%rsp), %xmm4
	vmovss	%xmm0, 1176(%rsp)
	vmovss	%xmm3, 1180(%rsp)
	vmulss	%xmm15, %xmm9, %xmm0
	vmulss	%xmm15, %xmm11, %xmm1
	vmovss	124(%rsp), %xmm15
	vfmadd231ss	%xmm4, %xmm9, %xmm1
	vfmsub231ss	%xmm4, %xmm11, %xmm0
	vmovss	96(%rsp), %xmm4
	vmovss	%xmm0, 1192(%rsp)
	vmovss	%xmm1, 1196(%rsp)
	vmulss	%xmm15, %xmm7, %xmm0
	vmulss	%xmm15, %xmm8, %xmm1
	vmovss	40(%rsp), %xmm15
	vfmadd231ss	%xmm4, %xmm7, %xmm1
	vfmsub231ss	%xmm4, %xmm8, %xmm0
	vmovss	36(%rsp), %xmm4
	vmovss	%xmm0, 1200(%rsp)
	vmovss	%xmm1, 1204(%rsp)
	vmulss	%xmm15, %xmm10, %xmm0
	vfmsub231ss	%xmm4, %xmm5, %xmm0
	vmulss	%xmm15, %xmm5, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm10
	vmovss	%xmm0, 1208(%rsp)
	vmovss	%xmm10, 1212(%rsp)
	vmovss	%xmm13, 1220(%rsp)
	vmovss	104(%rsp), %xmm4
	vmovss	44(%rsp), %xmm6
	movq	336(%rsp), %rsi
	vmulss	%xmm4, %xmm9, %xmm0
	vfmsub231ss	%xmm6, %xmm11, %xmm0
	vmulss	%xmm4, %xmm11, %xmm11
	vmovss	128(%rsp), %xmm4
	vfmadd132ss	%xmm9, %xmm11, %xmm6
	vmovss	%xmm0, 1224(%rsp)
	vmovss	%xmm6, 1228(%rsp)
	vmovss	108(%rsp), %xmm6
	vmulss	%xmm4, %xmm7, %xmm0
	vfmsub231ss	%xmm6, %xmm8, %xmm0
	vmulss	%xmm4, %xmm8, %xmm8
	vfmadd132ss	%xmm7, %xmm8, %xmm6
	vmovss	%xmm0, 1232(%rsp)
	vmovss	%xmm6, 1236(%rsp)
	call	_ZN20spherical_rotate_z_mIfLi7ELi4ELi3ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0
	vpermilps	$245, 1288(%rsp), %xmm1
	vpermilps	$177, 432(%rsp), %xmm7
	vpermilps	$177, 400(%rsp), %xmm0
	vpermilps	$245, 1256(%rsp), %xmm3
	vpermilps	$160, 1304(%rsp), %xmm4
	vpermilps	$160, 1320(%rsp), %xmm13
	vpermilps	$177, 416(%rsp), %xmm8
	vpermilps	$245, 1272(%rsp), %xmm2
	vpermilps	$160, 1272(%rsp), %xmm5
	vpermilps	$160, 1256(%rsp), %xmm6
	vpermilps	$160, 1288(%rsp), %xmm15
	vmulps	%xmm1, %xmm7, %xmm7
	vpermilps	$245, 1304(%rsp), %xmm1
	vmulps	%xmm0, %xmm3, %xmm3
	vmovaps	%xmm4, %xmm9
	vmulps	%xmm8, %xmm2, %xmm2
	vmovaps	%xmm5, %xmm11
	vfmsub132ps	416(%rsp), %xmm2, %xmm11
	vmovaps	%xmm6, %xmm12
	vfmadd231ps	416(%rsp), %xmm5, %xmm2
	vfmsub132ps	400(%rsp), %xmm3, %xmm12
	vmovaps	%xmm15, %xmm10
	vfmadd231ps	400(%rsp), %xmm6, %xmm3
	vfmsub132ps	432(%rsp), %xmm7, %xmm10
	vfmadd231ps	432(%rsp), %xmm15, %xmm7
	vmulps	%xmm0, %xmm1, %xmm1
	vfmsub132ps	400(%rsp), %xmm1, %xmm9
	vpermilps	$245, 1320(%rsp), %xmm0
	vfmadd231ps	400(%rsp), %xmm4, %xmm1
	vmulps	%xmm8, %xmm0, %xmm0
	vmovaps	%xmm13, %xmm8
	vfmsub132ps	416(%rsp), %xmm0, %xmm8
	vfmadd231ps	416(%rsp), %xmm13, %xmm0
	vblendps	$10, %xmm2, %xmm11, %xmm2
	vblendps	$10, %xmm3, %xmm12, %xmm3
	vmovups	%xmm2, 1272(%rsp)
	vmovss	1336(%rsp), %xmm2
	vblendps	$10, %xmm7, %xmm10, %xmm7
	vmovups	%xmm3, 1256(%rsp)
	vmovups	%xmm7, 1288(%rsp)
	vblendps	$10, %xmm1, %xmm9, %xmm1
	vmovups	%xmm1, 1304(%rsp)
	vblendps	$10, %xmm0, %xmm8, %xmm8
	vmovss	1340(%rsp), %xmm0
	vmulss	436(%rsp), %xmm0, %xmm1
	vmovups	%xmm8, 1320(%rsp)
	vfmsub231ss	432(%rsp), %xmm2, %xmm1
	vmulss	432(%rsp), %xmm0, %xmm0
	vfmadd231ss	436(%rsp), %xmm2, %xmm0
	vmovss	%xmm1, 1336(%rsp)
	vmovss	%xmm0, 1340(%rsp)
	call	_ZN20spherical_rotate_z_mIfLi7ELi6ELi5ELb1ELb0ELb0EEclER19spherical_expansionIfLi7EEP7complexIfE.isra.0
	vmovss	1372(%rsp), %xmm5
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	412(%rsp), %xmm7
	vmovss	1368(%rsp), %xmm1
	vmovss	408(%rsp), %xmm0
	vmovss	1388(%rsp), %xmm2
	vmovss	1404(%rsp), %xmm6
	vmovss	1420(%rsp), %xmm3
	vmovss	.LC4(%rip), %xmm12
	vfmadd132ss	1152(%rsp), %xmm14, %xmm12
	vmovss	1216(%rsp), %xmm4
	vmovss	1160(%rsp), %xmm10
	vmovss	1184(%rsp), %xmm8
	vmulss	%xmm5, %xmm7, %xmm13
	vmulss	.LC25(%rip), %xmm12, %xmm12
	vfmsub231ss	%xmm1, %xmm0, %xmm13
	vmulss	%xmm5, %xmm0, %xmm0
	vmovss	428(%rsp), %xmm5
	vfmadd132ss	%xmm1, %xmm0, %xmm7
	vmovss	1384(%rsp), %xmm1
	vmovss	424(%rsp), %xmm0
	vmovss	%xmm12, 204(%rsp)
	vmulss	%xmm2, %xmm5, %xmm11
	vfmsub231ss	%xmm1, %xmm0, %xmm11
	vmulss	%xmm2, %xmm0, %xmm0
	vmovss	444(%rsp), %xmm2
	vfmadd132ss	%xmm1, %xmm0, %xmm5
	vmovss	1400(%rsp), %xmm1
	vmovss	440(%rsp), %xmm0
	vmulss	%xmm6, %xmm2, %xmm9
	vfmsub231ss	%xmm1, %xmm0, %xmm9
	vmulss	%xmm6, %xmm0, %xmm0
	vmovss	460(%rsp), %xmm6
	vfmadd132ss	%xmm1, %xmm0, %xmm2
	vmovss	1416(%rsp), %xmm1
	vmovss	456(%rsp), %xmm0
	vmulss	%xmm3, %xmm6, %xmm15
	vfmsub231ss	%xmm1, %xmm0, %xmm15
	vmulss	%xmm3, %xmm0, %xmm0
	vmovss	1256(%rsp), %xmm3
	vfmadd132ss	%xmm1, %xmm0, %xmm6
	vmovss	1144(%rsp), %xmm0
	vmovss	%xmm12, 1144(%rsp)
	vmovss	.LC4(%rip), %xmm12
	vfmadd132ss	.LC4(%rip), %xmm14, %xmm0
	vfmadd132ss	1156(%rsp), %xmm14, %xmm12
	vmovss	1304(%rsp), %xmm1
	vmulss	.LC25(%rip), %xmm0, %xmm0
	vmovss	%xmm0, 12(%rsp)
	vmulss	.LC25(%rip), %xmm12, %xmm12
	vmovss	.LC26(%rip), %xmm0
	vfmadd132ss	%xmm10, %xmm14, %xmm0
	vfmadd132ss	.LC33(%rip), %xmm14, %xmm10
	vmovss	%xmm12, 20(%rsp)
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1176(%rsp), %xmm12, %xmm0
	vmovss	.LC0(%rip), %xmm12
	vfmadd132ss	1168(%rsp), %xmm14, %xmm12
	vmulss	.LC3(%rip), %xmm0, %xmm0
	vmulss	.LC3(%rip), %xmm12, %xmm12
	vmovss	%xmm0, 208(%rsp)
	vmovss	%xmm0, 1160(%rsp)
	vmovss	.LC4(%rip), %xmm0
	vfmadd132ss	1180(%rsp), %xmm14, %xmm0
	vmovss	%xmm12, 24(%rsp)
	vmulss	.LC3(%rip), %xmm0, %xmm12
	vmovss	%xmm12, 36(%rsp)
	vmovss	.LC2(%rip), %xmm12
	vfmadd132ss	1172(%rsp), %xmm14, %xmm12
	vmovaps	%xmm12, %xmm0
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1176(%rsp), %xmm12, %xmm10
	vmulss	.LC3(%rip), %xmm10, %xmm10
	vmovss	%xmm10, 40(%rsp)
	vmulss	.LC3(%rip), %xmm0, %xmm10
	vmovss	%xmm10, 44(%rsp)
	vmovss	.LC51(%rip), %xmm10
	vfmadd132ss	1192(%rsp), %xmm14, %xmm10
	vmovaps	%xmm10, %xmm0
	vfmadd231ss	1208(%rsp), %xmm12, %xmm0
	vmulss	.LC30(%rip), %xmm0, %xmm10
	vmovss	%xmm10, 216(%rsp)
	vmovss	%xmm10, 1184(%rsp)
	vmovss	.LC40(%rip), %xmm10
	vmovss	.LC26(%rip), %xmm12
	vfmadd132ss	1196(%rsp), %xmm14, %xmm12
	vfmadd132ss	%xmm8, %xmm14, %xmm10
	vfmadd132ss	.LC55(%rip), %xmm14, %xmm8
	vmovaps	%xmm12, %xmm0
	vmovss	.LC0(%rip), %xmm12
	vfmadd231ss	1200(%rsp), %xmm12, %xmm10
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1212(%rsp), %xmm12, %xmm0
	vmovss	.LC21(%rip), %xmm12
	vfmadd132ss	1192(%rsp), %xmm14, %xmm12
	vmulss	.LC30(%rip), %xmm10, %xmm10
	vmovss	%xmm10, 48(%rsp)
	vmulss	.LC30(%rip), %xmm0, %xmm10
	vmovaps	%xmm12, %xmm0
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1208(%rsp), %xmm12, %xmm0
	vmulss	.LC30(%rip), %xmm0, %xmm12
	vmovss	%xmm10, 52(%rsp)
	vmovss	.LC2(%rip), %xmm10
	vfmadd132ss	1204(%rsp), %xmm14, %xmm10
	vmovss	%xmm12, 56(%rsp)
	vmovss	.LC32(%rip), %xmm12
	vmulss	.LC30(%rip), %xmm10, %xmm10
	vfmadd132ss	1196(%rsp), %xmm14, %xmm12
	vmovss	%xmm10, 60(%rsp)
	vmovaps	%xmm12, %xmm0
	vmovss	.LC27(%rip), %xmm12
	vfmadd231ss	1200(%rsp), %xmm12, %xmm8
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1212(%rsp), %xmm12, %xmm0
	vmulss	.LC30(%rip), %xmm8, %xmm8
	vmovss	%xmm8, 64(%rsp)
	vmulss	.LC30(%rip), %xmm0, %xmm8
	vmovss	%xmm8, 68(%rsp)
	vmovss	.LC33(%rip), %xmm8
	vfmadd132ss	%xmm4, %xmm14, %xmm8
	vmovaps	%xmm8, %xmm0
	vmovss	.LC28(%rip), %xmm8
	vfmadd231ss	1232(%rsp), %xmm8, %xmm0
	vfmadd231ss	1248(%rsp), %xmm12, %xmm0
	vmovss	.LC40(%rip), %xmm12
	vfmadd132ss	1236(%rsp), %xmm14, %xmm12
	vmulss	.LC36(%rip), %xmm0, %xmm8
	vmovaps	%xmm12, %xmm0
	vmovss	.LC0(%rip), %xmm12
	vmovss	%xmm8, 212(%rsp)
	vmovss	%xmm8, 1216(%rsp)
	vmovss	.LC37(%rip), %xmm8
	vfmadd132ss	1224(%rsp), %xmm14, %xmm8
	vfmadd231ss	1240(%rsp), %xmm12, %xmm8
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1252(%rsp), %xmm12, %xmm0
	vmovss	.LC2(%rip), %xmm12
	vmulss	.LC36(%rip), %xmm8, %xmm8
	vmovss	%xmm8, 72(%rsp)
	vmulss	.LC36(%rip), %xmm0, %xmm8
	vmovss	%xmm8, 76(%rsp)
	vmovss	.LC81(%rip), %xmm8
	vfmadd132ss	%xmm4, %xmm14, %xmm8
	vmovaps	%xmm8, %xmm0
	vmovss	.LC28(%rip), %xmm8
	vfmadd231ss	1232(%rsp), %xmm12, %xmm0
	vfmadd132ss	1228(%rsp), %xmm14, %xmm8
	vfmadd231ss	1244(%rsp), %xmm12, %xmm8
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1248(%rsp), %xmm12, %xmm0
	vmulss	.LC36(%rip), %xmm8, %xmm8
	vmulss	.LC36(%rip), %xmm0, %xmm12
	vmovss	%xmm8, 84(%rsp)
	vmovss	.LC38(%rip), %xmm8
	vfmadd132ss	1224(%rsp), %xmm14, %xmm8
	vmovss	%xmm12, 80(%rsp)
	vmovaps	%xmm8, %xmm0
	vmovss	.LC38(%rip), %xmm8
	vmovss	.LC39(%rip), %xmm12
	vfmadd132ss	1228(%rsp), %xmm14, %xmm12
	vmovss	.LC27(%rip), %xmm10
	vfmadd231ss	1240(%rsp), %xmm10, %xmm0
	vfmadd132ss	1236(%rsp), %xmm14, %xmm8
	vfmadd132ss	.LC82(%rip), %xmm14, %xmm4
	vmovss	.LC4(%rip), %xmm10
	vfmadd231ss	1252(%rsp), %xmm10, %xmm8
	vmulss	.LC36(%rip), %xmm0, %xmm10
	vmulss	.LC36(%rip), %xmm8, %xmm8
	vmovaps	%xmm12, %xmm0
	vmovss	.LC41(%rip), %xmm12
	vfmadd231ss	1232(%rsp), %xmm12, %xmm4
	vmovss	.LC42(%rip), %xmm12
	vfmadd231ss	1244(%rsp), %xmm12, %xmm0
	vmovss	.LC4(%rip), %xmm12
	vmovss	%xmm10, 88(%rsp)
	vfmadd231ss	1248(%rsp), %xmm12, %xmm4
	vmovss	%xmm8, 92(%rsp)
	vmulss	.LC36(%rip), %xmm4, %xmm4
	vmovss	%xmm4, 96(%rsp)
	vmulss	.LC36(%rip), %xmm0, %xmm4
	vmovss	%xmm4, 100(%rsp)
	vmovss	.LC55(%rip), %xmm4
	vfmadd132ss	1264(%rsp), %xmm14, %xmm4
	vmovaps	%xmm4, %xmm0
	vmovss	.LC81(%rip), %xmm4
	vfmadd231ss	1280(%rsp), %xmm4, %xmm0
	vfmadd231ss	1296(%rsp), %xmm12, %xmm0
	vmulss	.LC46(%rip), %xmm0, %xmm4
	vmovss	%xmm4, 228(%rsp)
	vmovss	%xmm4, 1256(%rsp)
	vmovss	.LC27(%rip), %xmm4
	vmovss	.LC0(%rip), %xmm12
	vfmadd132ss	1268(%rsp), %xmm14, %xmm12
	vmovss	.LC22(%rip), %xmm8
	vfmadd132ss	1268(%rsp), %xmm14, %xmm8
	vfmadd132ss	%xmm3, %xmm14, %xmm4
	vmovaps	%xmm12, %xmm0
	vmovss	.LC47(%rip), %xmm12
	vfmadd231ss	1272(%rsp), %xmm12, %xmm4
	vmovss	.LC51(%rip), %xmm12
	vfmadd231ss	1284(%rsp), %xmm12, %xmm0
	vmovss	.LC0(%rip), %xmm12
	vfmadd231ss	1288(%rsp), %xmm12, %xmm4
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1300(%rsp), %xmm12, %xmm0
	vmovss	.LC33(%rip), %xmm12
	vmulss	.LC46(%rip), %xmm4, %xmm4
	vmovss	%xmm4, 104(%rsp)
	vmulss	.LC46(%rip), %xmm0, %xmm4
	vmovss	%xmm4, 108(%rsp)
	vmovss	.LC22(%rip), %xmm4
	vfmadd132ss	1264(%rsp), %xmm14, %xmm4
	vmovaps	%xmm4, %xmm0
	vmovss	.LC47(%rip), %xmm4
	vfmadd231ss	1280(%rsp), %xmm12, %xmm0
	vfmadd132ss	1276(%rsp), %xmm14, %xmm4
	vmovss	.LC2(%rip), %xmm12
	vfmadd231ss	1292(%rsp), %xmm12, %xmm4
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1296(%rsp), %xmm12, %xmm0
	vmulss	.LC46(%rip), %xmm4, %xmm4
	vmulss	.LC46(%rip), %xmm0, %xmm12
	vmovaps	%xmm8, %xmm0
	vmovss	.LC42(%rip), %xmm8
	vmovss	%xmm4, 116(%rsp)
	vmovss	.LC22(%rip), %xmm4
	vmovss	%xmm12, 112(%rsp)
	vfmadd132ss	%xmm3, %xmm14, %xmm4
	vfmadd231ss	1272(%rsp), %xmm8, %xmm4
	vmovss	.LC52(%rip), %xmm8
	vfmadd231ss	1284(%rsp), %xmm8, %xmm0
	vmovss	.LC27(%rip), %xmm8
	vmovss	.LC83(%rip), %xmm12
	vfmadd231ss	1288(%rsp), %xmm8, %xmm4
	vmovss	.LC4(%rip), %xmm8
	vfmadd231ss	1300(%rsp), %xmm8, %xmm0
	vmovss	.LC48(%rip), %xmm8
	vfmadd132ss	1276(%rsp), %xmm14, %xmm8
	vmulss	.LC46(%rip), %xmm4, %xmm4
	vfmadd132ss	%xmm3, %xmm14, %xmm12
	vmovss	.LC24(%rip), %xmm3
	vfmadd132ss	1268(%rsp), %xmm14, %xmm3
	vmovss	%xmm4, 120(%rsp)
	vmulss	.LC46(%rip), %xmm0, %xmm4
	vmovaps	%xmm8, %xmm0
	vmovss	.LC53(%rip), %xmm8
	vmovss	%xmm4, 124(%rsp)
	vmovss	.LC23(%rip), %xmm4
	vfmadd132ss	1264(%rsp), %xmm14, %xmm4
	vfmadd231ss	1280(%rsp), %xmm8, %xmm4
	vmovss	.LC42(%rip), %xmm8
	vfmadd231ss	1292(%rsp), %xmm8, %xmm0
	vmovss	.LC4(%rip), %xmm8
	vfmadd231ss	1296(%rsp), %xmm8, %xmm4
	vmulss	.LC46(%rip), %xmm0, %xmm10
	vmovaps	%xmm12, %xmm0
	vmulss	.LC46(%rip), %xmm4, %xmm8
	vmovaps	%xmm3, %xmm4
	vmovss	.LC49(%rip), %xmm3
	vfmadd231ss	1272(%rsp), %xmm3, %xmm0
	vmovss	.LC54(%rip), %xmm3
	vfmadd231ss	1284(%rsp), %xmm3, %xmm4
	vmovss	%xmm10, 132(%rsp)
	vmovss	%xmm8, 128(%rsp)
	vmovss	.LC55(%rip), %xmm3
	vmovss	.LC21(%rip), %xmm12
	vfmadd231ss	1288(%rsp), %xmm3, %xmm0
	vmovss	.LC4(%rip), %xmm3
	vfmadd132ss	1324(%rsp), %xmm14, %xmm12
	vfmadd231ss	1300(%rsp), %xmm3, %xmm4
	vmulss	.LC46(%rip), %xmm0, %xmm3
	vmovss	%xmm3, 136(%rsp)
	vmulss	.LC46(%rip), %xmm4, %xmm3
	vmovss	%xmm3, 140(%rsp)
	vmovss	.LC34(%rip), %xmm3
	vfmadd132ss	%xmm1, %xmm14, %xmm3
	vmovaps	%xmm3, %xmm0
	vmovss	.LC32(%rip), %xmm3
	vfmadd231ss	1320(%rsp), %xmm3, %xmm0
	vmovss	.LC37(%rip), %xmm3
	vfmadd231ss	1336(%rsp), %xmm3, %xmm0
	vmovss	.LC4(%rip), %xmm3
	vfmadd231ss	1352(%rsp), %xmm3, %xmm0
	vmulss	.LC58(%rip), %xmm0, %xmm3
	vmovaps	%xmm12, %xmm0
	vmovss	.LC34(%rip), %xmm12
	vmovss	%xmm3, 224(%rsp)
	vmovss	%xmm3, 1304(%rsp)
	vmovss	.LC29(%rip), %xmm3
	vfmadd132ss	1312(%rsp), %xmm14, %xmm3
	vfmadd231ss	1328(%rsp), %xmm12, %xmm3
	vmovss	.LC28(%rip), %xmm12
	vfmadd231ss	1340(%rsp), %xmm12, %xmm0
	vmovss	.LC0(%rip), %xmm12
	vfmadd231ss	1344(%rsp), %xmm12, %xmm3
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1356(%rsp), %xmm12, %xmm0
	vmulss	.LC58(%rip), %xmm3, %xmm3
	vmovss	%xmm3, 144(%rsp)
	vmulss	.LC58(%rip), %xmm0, %xmm3
	vmovss	.LC64(%rip), %xmm12
	vmovss	.LC65(%rip), %xmm4
	vfmadd132ss	1324(%rsp), %xmm14, %xmm4
	vmovss	%xmm3, 148(%rsp)
	vmovss	.LC38(%rip), %xmm3
	vfmadd132ss	%xmm1, %xmm14, %xmm3
	vmovaps	%xmm3, %xmm0
	vmovss	.LC42(%rip), %xmm3
	vfmadd231ss	1320(%rsp), %xmm12, %xmm0
	vfmadd132ss	1316(%rsp), %xmm14, %xmm3
	vmovss	.LC68(%rip), %xmm12
	vfmadd231ss	1332(%rsp), %xmm12, %xmm3
	vmovss	.LC0(%rip), %xmm12
	vfmadd231ss	1336(%rsp), %xmm12, %xmm0
	vmovss	.LC2(%rip), %xmm12
	vfmadd231ss	1348(%rsp), %xmm12, %xmm3
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1352(%rsp), %xmm12, %xmm0
	vmulss	.LC58(%rip), %xmm3, %xmm3
	vmulss	.LC58(%rip), %xmm0, %xmm12
	vmovaps	%xmm4, %xmm0
	vmovss	.LC0(%rip), %xmm4
	vmovss	%xmm3, 156(%rsp)
	vmovss	.LC59(%rip), %xmm3
	vfmadd132ss	1312(%rsp), %xmm14, %xmm3
	vmovss	%xmm12, 152(%rsp)
	vfmadd231ss	1328(%rsp), %xmm4, %xmm3
	vmovss	.LC43(%rip), %xmm4
	vfmadd231ss	1340(%rsp), %xmm4, %xmm0
	vmovss	.LC27(%rip), %xmm4
	vfmadd231ss	1344(%rsp), %xmm4, %xmm3
	vmovss	.LC4(%rip), %xmm4
	vfmadd231ss	1356(%rsp), %xmm4, %xmm0
	vmulss	.LC58(%rip), %xmm3, %xmm3
	vmovss	%xmm3, 160(%rsp)
	vmulss	.LC58(%rip), %xmm0, %xmm3
	vmovss	.LC60(%rip), %xmm4
	vfmadd132ss	1316(%rsp), %xmm14, %xmm4
	vmovss	.LC66(%rip), %xmm8
	vmovss	.LC70(%rip), %xmm10
	vfmadd132ss	1324(%rsp), %xmm14, %xmm8
	vmovss	%xmm3, 164(%rsp)
	vmovss	.LC84(%rip), %xmm3
	vmovaps	%xmm4, %xmm0
	vmovss	.LC32(%rip), %xmm4
	vfmadd132ss	%xmm1, %xmm14, %xmm3
	vfmadd231ss	1320(%rsp), %xmm4, %xmm3
	vmovss	.LC69(%rip), %xmm4
	vfmadd231ss	1332(%rsp), %xmm4, %xmm0
	vmovss	.LC72(%rip), %xmm4
	vfmadd231ss	1336(%rsp), %xmm4, %xmm3
	vmovss	.LC42(%rip), %xmm4
	vfmadd231ss	1348(%rsp), %xmm4, %xmm0
	vmovss	.LC4(%rip), %xmm4
	vfmadd231ss	1352(%rsp), %xmm4, %xmm3
	vmulss	.LC58(%rip), %xmm3, %xmm4
	vmulss	.LC58(%rip), %xmm0, %xmm3
	vmovaps	%xmm8, %xmm0
	vmovss	.LC73(%rip), %xmm8
	vfmadd231ss	1340(%rsp), %xmm8, %xmm0
	vmovss	.LC4(%rip), %xmm8
	vfmadd231ss	1356(%rsp), %xmm8, %xmm0
	vmovss	%xmm3, 172(%rsp)
	vmovss	.LC61(%rip), %xmm3
	vmovss	%xmm4, 168(%rsp)
	vfmadd132ss	1312(%rsp), %xmm14, %xmm3
	vfmadd231ss	1328(%rsp), %xmm10, %xmm3
	vmovss	.LC55(%rip), %xmm10
	vfmadd231ss	1344(%rsp), %xmm10, %xmm3
	vmulss	.LC58(%rip), %xmm3, %xmm8
	vmovss	%xmm8, 176(%rsp)
	vmovss	.LC62(%rip), %xmm12
	vfmadd132ss	.LC85(%rip), %xmm14, %xmm1
	vfmadd132ss	1316(%rsp), %xmm14, %xmm12
	vmulss	.LC58(%rip), %xmm0, %xmm10
	vmovss	%xmm10, 180(%rsp)
	vmovaps	%xmm12, %xmm0
	vmovss	.LC67(%rip), %xmm12
	vfmadd231ss	1320(%rsp), %xmm12, %xmm1
	vmovss	.LC71(%rip), %xmm12
	vfmadd231ss	1332(%rsp), %xmm12, %xmm0
	vmovss	.LC74(%rip), %xmm12
	vfmadd231ss	1336(%rsp), %xmm12, %xmm1
	vmovss	.LC43(%rip), %xmm12
	vfmadd231ss	1348(%rsp), %xmm12, %xmm0
	vmovss	.LC4(%rip), %xmm12
	vfmadd231ss	1352(%rsp), %xmm12, %xmm1
	vmulss	.LC58(%rip), %xmm1, %xmm1
	vmovss	%xmm1, 184(%rsp)
	vmulss	.LC58(%rip), %xmm0, %xmm1
	vmovss	%xmm1, 188(%rsp)
	vmovss	.LC91(%rip), %xmm1
	vfmadd132ss	%xmm13, %xmm14, %xmm1
	vmovaps	%xmm1, %xmm0
	vfmadd231ss	.LC92(%rip), %xmm11, %xmm0
	vfmadd231ss	.LC93(%rip), %xmm9, %xmm0
	vfmadd231ss	%xmm12, %xmm15, %xmm0
	vmulss	.LC87(%rip), %xmm0, %xmm1
	vmovss	%xmm1, 220(%rsp)
	vmovss	%xmm1, 1360(%rsp)
	vmovss	.LC81(%rip), %xmm1
	vfmadd132ss	%xmm7, %xmm14, %xmm1
	vmovaps	%xmm1, %xmm4
	vfmadd231ss	.LC94(%rip), %xmm5, %xmm4
	vfmadd231ss	.LC81(%rip), %xmm2, %xmm4
	vfmadd231ss	%xmm12, %xmm6, %xmm4
	vmulss	.LC87(%rip), %xmm4, %xmm1
	vmovss	%xmm1, 232(%rsp)
	vmovss	.LC54(%rip), %xmm1
	vfmadd132ss	%xmm13, %xmm14, %xmm1
	vmovaps	%xmm1, %xmm3
	vfmadd231ss	.LC95(%rip), %xmm11, %xmm3
	vfmadd231ss	%xmm12, %xmm9, %xmm3
	vfmadd231ss	%xmm12, %xmm15, %xmm3
	vmulss	.LC87(%rip), %xmm3, %xmm1
	vmovss	16(%rsp), %xmm0
	vxorps	.LC15(%rip), %xmm0, %xmm0
	vmovss	%xmm1, 236(%rsp)
	vmovss	.LC53(%rip), %xmm1
	vfmadd132ss	%xmm7, %xmm14, %xmm1
	vmovaps	%xmm1, %xmm8
	vfmadd231ss	.LC96(%rip), %xmm5, %xmm8
	vfmadd231ss	.LC97(%rip), %xmm2, %xmm8
	vfmadd231ss	%xmm12, %xmm6, %xmm8
	vmulss	.LC87(%rip), %xmm8, %xmm1
	vmovss	%xmm1, 240(%rsp)
	vmovss	.LC98(%rip), %xmm1
	vfmadd132ss	%xmm13, %xmm14, %xmm1
	vfmadd132ss	.LC104(%rip), %xmm14, %xmm13
	vmovaps	%xmm1, %xmm10
	vfmadd231ss	.LC99(%rip), %xmm11, %xmm10
	vfmadd231ss	.LC104(%rip), %xmm11, %xmm13
	vfmadd231ss	.LC100(%rip), %xmm9, %xmm10
	vfmadd231ss	.LC105(%rip), %xmm9, %xmm13
	vfmadd231ss	.LC4(%rip), %xmm15, %xmm13
	vmulss	.LC87(%rip), %xmm13, %xmm13
	vfmadd231ss	%xmm12, %xmm15, %xmm10
	vmulss	.LC87(%rip), %xmm10, %xmm1
	vmovss	%xmm13, 252(%rsp)
	vmovss	%xmm1, 244(%rsp)
	vmovss	.LC101(%rip), %xmm1
	vfmadd132ss	%xmm7, %xmm14, %xmm1
	vfmadd132ss	.LC106(%rip), %xmm14, %xmm7
	vmovaps	%xmm1, %xmm12
	vfmadd231ss	.LC107(%rip), %xmm5, %xmm7
	vfmadd231ss	.LC102(%rip), %xmm5, %xmm12
	vfmadd231ss	.LC108(%rip), %xmm2, %xmm7
	vfmadd231ss	.LC103(%rip), %xmm2, %xmm12
	vfmadd231ss	.LC4(%rip), %xmm6, %xmm7
	vfmadd231ss	.LC4(%rip), %xmm6, %xmm12
	vmulss	.LC87(%rip), %xmm7, %xmm5
	vmulss	.LC87(%rip), %xmm12, %xmm1
	vmovss	%xmm5, 256(%rsp)
	vmovss	%xmm1, 248(%rsp)
	vmulss	28(%rsp), %xmm0, %xmm0
	vmovss	32(%rsp), %xmm2
	vmovaps	%xmm2, %xmm5
	vmovaps	%xmm2, %xmm4
	vfmadd132ss	%xmm14, %xmm0, %xmm5
	vfnmadd231ss	%xmm14, %xmm0, %xmm4
	vmulss	%xmm2, %xmm5, %xmm1
	vmulss	%xmm5, %xmm0, %xmm3
	vfmadd231ss	%xmm4, %xmm0, %xmm1
	vfmsub231ss	%xmm2, %xmm4, %xmm3
	vmulss	%xmm2, %xmm1, %xmm9
	vmulss	%xmm1, %xmm0, %xmm8
	vfmadd231ss	%xmm3, %xmm0, %xmm9
	vfmsub231ss	%xmm2, %xmm3, %xmm8
	vmulss	%xmm2, %xmm9, %xmm7
	vmulss	%xmm9, %xmm0, %xmm10
	vfmadd231ss	%xmm8, %xmm0, %xmm7
	vfmsub231ss	%xmm2, %xmm8, %xmm10
	vmulss	%xmm2, %xmm7, %xmm13
	vmulss	%xmm7, %xmm0, %xmm12
	vfmadd231ss	%xmm10, %xmm0, %xmm13
	vfmsub231ss	%xmm2, %xmm10, %xmm12
	vmulss	%xmm2, %xmm13, %xmm11
	vmulss	%xmm13, %xmm0, %xmm15
	vfmadd231ss	%xmm12, %xmm0, %xmm11
	vfmsub231ss	%xmm2, %xmm12, %xmm15
	vmulss	%xmm11, %xmm0, %xmm2
	vmovaps	%xmm2, %xmm6
	vmovss	32(%rsp), %xmm2
	vfmsub231ss	%xmm2, %xmm15, %xmm6
	vmulss	%xmm2, %xmm11, %xmm2
	vmovss	%xmm6, 16(%rsp)
	vmovaps	%xmm0, %xmm6
	vmovss	204(%rsp), %xmm0
	vfmadd132ss	%xmm15, %xmm2, %xmm6
	vmulss	20(%rsp), %xmm5, %xmm2
	vfmsub231ss	12(%rsp), %xmm4, %xmm2
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1148(%rsp)
	vmulss	20(%rsp), %xmm4, %xmm0
	vfmadd231ss	12(%rsp), %xmm5, %xmm0
	vmovss	%xmm2, 1152(%rsp)
	vmulss	36(%rsp), %xmm5, %xmm2
	vfmsub231ss	24(%rsp), %xmm4, %xmm2
	vmovss	%xmm0, 1156(%rsp)
	vmovss	208(%rsp), %xmm0
	vmovss	%xmm2, 1168(%rsp)
	vmulss	44(%rsp), %xmm1, %xmm2
	vfmsub231ss	40(%rsp), %xmm3, %xmm2
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1164(%rsp)
	vmulss	36(%rsp), %xmm4, %xmm0
	vfmadd231ss	24(%rsp), %xmm5, %xmm0
	vmovss	%xmm2, 1176(%rsp)
	vmulss	52(%rsp), %xmm5, %xmm2
	vfmsub231ss	48(%rsp), %xmm4, %xmm2
	vmovss	%xmm0, 1172(%rsp)
	vmulss	44(%rsp), %xmm3, %xmm0
	vfmadd231ss	40(%rsp), %xmm1, %xmm0
	vmovss	%xmm0, 1180(%rsp)
	vmovss	216(%rsp), %xmm0
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1188(%rsp)
	vmulss	52(%rsp), %xmm4, %xmm0
	vfmadd231ss	48(%rsp), %xmm5, %xmm0
	vmovss	%xmm2, 1192(%rsp)
	vmulss	60(%rsp), %xmm1, %xmm2
	vfmsub231ss	56(%rsp), %xmm3, %xmm2
	vmovss	%xmm0, 1196(%rsp)
	vmulss	60(%rsp), %xmm3, %xmm0
	vfmadd231ss	56(%rsp), %xmm1, %xmm0
	vmovss	%xmm2, 1200(%rsp)
	vmulss	68(%rsp), %xmm9, %xmm2
	vfmsub231ss	64(%rsp), %xmm8, %xmm2
	vmovss	%xmm0, 1204(%rsp)
	vmulss	68(%rsp), %xmm8, %xmm0
	vfmadd231ss	64(%rsp), %xmm9, %xmm0
	vmovss	%xmm2, 1208(%rsp)
	vmulss	76(%rsp), %xmm5, %xmm2
	vfmsub231ss	72(%rsp), %xmm4, %xmm2
	vmovss	%xmm0, 1212(%rsp)
	vmovss	212(%rsp), %xmm0
	vmovss	%xmm2, 1224(%rsp)
	vmulss	84(%rsp), %xmm1, %xmm2
	vfmsub231ss	80(%rsp), %xmm3, %xmm2
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1220(%rsp)
	vmulss	76(%rsp), %xmm4, %xmm0
	vfmadd231ss	72(%rsp), %xmm5, %xmm0
	vmovss	%xmm2, 1232(%rsp)
	vmulss	92(%rsp), %xmm9, %xmm2
	vfmsub231ss	88(%rsp), %xmm8, %xmm2
	vmovss	%xmm0, 1228(%rsp)
	vmulss	84(%rsp), %xmm3, %xmm0
	vfmadd231ss	80(%rsp), %xmm1, %xmm0
	vmovss	%xmm2, 1240(%rsp)
	vmovss	%xmm0, 1236(%rsp)
	vmulss	92(%rsp), %xmm8, %xmm0
	vfmadd231ss	88(%rsp), %xmm9, %xmm0
	vmovss	%xmm0, 1244(%rsp)
	vmulss	100(%rsp), %xmm10, %xmm0
	vmulss	100(%rsp), %xmm7, %xmm2
	vfmadd231ss	96(%rsp), %xmm7, %xmm0
	vfmsub231ss	96(%rsp), %xmm10, %xmm2
	vmovss	%xmm0, 1252(%rsp)
	vmovss	228(%rsp), %xmm0
	vmovss	%xmm2, 1248(%rsp)
	vmulss	108(%rsp), %xmm5, %xmm2
	vfmsub231ss	104(%rsp), %xmm4, %xmm2
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1260(%rsp)
	vmulss	108(%rsp), %xmm4, %xmm0
	vmovss	%xmm2, 1264(%rsp)
	vmulss	116(%rsp), %xmm1, %xmm2
	vfmadd231ss	104(%rsp), %xmm5, %xmm0
	vfmsub231ss	112(%rsp), %xmm3, %xmm2
	vmovss	%xmm0, 1268(%rsp)
	vmulss	116(%rsp), %xmm3, %xmm0
	vmovss	%xmm2, 1272(%rsp)
	vmulss	124(%rsp), %xmm9, %xmm2
	vfmadd231ss	112(%rsp), %xmm1, %xmm0
	vfmsub231ss	120(%rsp), %xmm8, %xmm2
	vmovss	%xmm0, 1276(%rsp)
	vmulss	124(%rsp), %xmm8, %xmm0
	vmovss	%xmm2, 1280(%rsp)
	vmulss	132(%rsp), %xmm7, %xmm2
	vfmadd231ss	120(%rsp), %xmm9, %xmm0
	vfmsub231ss	128(%rsp), %xmm10, %xmm2
	vmovss	%xmm0, 1284(%rsp)
	vmulss	132(%rsp), %xmm10, %xmm0
	vmovss	%xmm2, 1288(%rsp)
	vfmadd231ss	128(%rsp), %xmm7, %xmm0
	vmovss	%xmm0, 1292(%rsp)
	vmulss	140(%rsp), %xmm13, %xmm2
	vmulss	140(%rsp), %xmm12, %xmm0
	vfmadd231ss	136(%rsp), %xmm13, %xmm0
	vfmsub231ss	136(%rsp), %xmm12, %xmm2
	vmovss	%xmm0, 1300(%rsp)
	vmovss	224(%rsp), %xmm0
	vmovss	%xmm2, 1296(%rsp)
	vmulss	148(%rsp), %xmm5, %xmm2
	vfmsub231ss	144(%rsp), %xmm4, %xmm2
	vfmadd132ss	%xmm14, %xmm14, %xmm0
	vmovss	%xmm0, 1308(%rsp)
	vmulss	148(%rsp), %xmm4, %xmm0
	vmovss	%xmm2, 1312(%rsp)
	vmulss	156(%rsp), %xmm1, %xmm2
	vfmadd231ss	144(%rsp), %xmm5, %xmm0
	vfmsub231ss	152(%rsp), %xmm3, %xmm2
	vmovss	%xmm0, 1316(%rsp)
	vmulss	156(%rsp), %xmm3, %xmm0
	vmovss	%xmm2, 1320(%rsp)
	vmulss	164(%rsp), %xmm9, %xmm2
	vfmadd231ss	152(%rsp), %xmm1, %xmm0
	vfmsub231ss	160(%rsp), %xmm8, %xmm2
	vmovss	%xmm0, 1324(%rsp)
	vmulss	164(%rsp), %xmm8, %xmm0
	vmovss	%xmm2, 1328(%rsp)
	vmulss	172(%rsp), %xmm7, %xmm2
	vfmadd231ss	160(%rsp), %xmm9, %xmm0
	vfmsub231ss	168(%rsp), %xmm10, %xmm2
	vmovss	%xmm0, 1332(%rsp)
	vmulss	172(%rsp), %xmm10, %xmm0
	vmovss	%xmm2, 1336(%rsp)
	vfmadd231ss	168(%rsp), %xmm7, %xmm0
	vmovss	%xmm0, 1340(%rsp)
	vmulss	180(%rsp), %xmm13, %xmm2
	vmulss	180(%rsp), %xmm12, %xmm0
	vfmadd231ss	176(%rsp), %xmm13, %xmm0
	vfmsub231ss	176(%rsp), %xmm12, %xmm2
	vmovss	%xmm0, 1348(%rsp)
	vmulss	188(%rsp), %xmm15, %xmm0
	vmovss	%xmm2, 1344(%rsp)
	vmulss	188(%rsp), %xmm11, %xmm2
	vfmadd231ss	184(%rsp), %xmm11, %xmm0
	vfmsub231ss	184(%rsp), %xmm15, %xmm2
	vmovss	%xmm0, 1356(%rsp)
	vmulss	220(%rsp), %xmm14, %xmm0
	vmovss	%xmm2, 1352(%rsp)
	vmovss	232(%rsp), %xmm2
	vmulss	%xmm2, %xmm4, %xmm4
	vmovss	%xmm0, 1364(%rsp)
	vmovaps	%xmm2, %xmm0
	vxorps	.LC15(%rip), %xmm0, %xmm0
	vmovss	%xmm4, 1372(%rsp)
	vmovss	236(%rsp), %xmm4
	vmulss	%xmm5, %xmm0, %xmm5
	vmulss	%xmm4, %xmm1, %xmm1
	vmulss	%xmm4, %xmm3, %xmm3
	vmovss	%xmm5, 1368(%rsp)
	vmovss	240(%rsp), %xmm5
	vmovss	%xmm1, 1380(%rsp)
	vmovss	%xmm3, 1376(%rsp)
	vmulss	%xmm5, %xmm8, %xmm8
	vmovaps	%xmm5, %xmm0
	vmovss	244(%rsp), %xmm5
	vxorps	.LC15(%rip), %xmm0, %xmm0
	vmovss	%xmm8, 1388(%rsp)
	vmulss	%xmm5, %xmm7, %xmm7
	vmulss	%xmm9, %xmm0, %xmm9
	vmovss	%xmm7, 1396(%rsp)
	vmovss	248(%rsp), %xmm7
	vmulss	%xmm5, %xmm10, %xmm10
	vmovss	%xmm9, 1384(%rsp)
	vmovss	%xmm10, 1392(%rsp)
	vmovaps	%xmm7, %xmm0
	vxorps	.LC15(%rip), %xmm0, %xmm0
	vmulss	%xmm7, %xmm12, %xmm12
	vmovss	%xmm12, 1404(%rsp)
	vmulss	%xmm13, %xmm0, %xmm13
	vmovss	%xmm13, 1400(%rsp)
	vmovss	252(%rsp), %xmm7
	vmovss	256(%rsp), %xmm5
	vmovss	388(%rsp), %xmm2
	vmovss	376(%rsp), %xmm1
	vmulss	%xmm7, %xmm11, %xmm11
	vmulss	%xmm7, %xmm15, %xmm15
	vmovaps	%xmm5, %xmm0
	vxorps	.LC15(%rip), %xmm0, %xmm0
	vmovss	%xmm11, 1412(%rsp)
	vmovss	%xmm15, 1408(%rsp)
	vmulss	%xmm6, %xmm0, %xmm0
	vmulss	16(%rsp), %xmm5, %xmm6
	vmovss	%xmm0, 1416(%rsp)
	vmovss	364(%rsp), %xmm0
	vmovss	%xmm6, 1420(%rsp)
	call	_Z23spherical_expansion_L2LIfLi7EEvR19spherical_expansionIT_XT0_EES1_S1_S1_
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	376(%rsp), %xmm3
	vmovss	364(%rsp), %xmm2
	vaddss	372(%rsp), %xmm3, %xmm3
	vaddss	360(%rsp), %xmm2, %xmm2
	vmovss	388(%rsp), %xmm1
	vaddss	384(%rsp), %xmm1, %xmm1
	vsubss	368(%rsp), %xmm3, %xmm3
	vsubss	356(%rsp), %xmm2, %xmm2
	vsubss	380(%rsp), %xmm1, %xmm1
	vmulss	%xmm3, %xmm3, %xmm0
	vfmadd231ss	%xmm2, %xmm2, %xmm0
	vfmadd231ss	%xmm1, %xmm1, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm4
	vucomiss	%xmm0, %xmm14
	ja	.L126
.L116:
	vmulss	%xmm4, %xmm4, %xmm0
	vmovss	1152(%rsp), %xmm5
	vmovss	1156(%rsp), %xmm6
	vmulss	%xmm4, %xmm0, %xmm0
	vmovss	1144(%rsp), %xmm4
	vdivss	%xmm0, %xmm3, %xmm3
	vdivss	%xmm0, %xmm2, %xmm2
	vmulss	%xmm3, %xmm3, %xmm3
	vdivss	%xmm0, %xmm1, %xmm1
	vfmadd132ss	%xmm2, %xmm3, %xmm2
	vfmadd132ss	%xmm1, %xmm2, %xmm1
	vsqrtss	%xmm1, %xmm1, %xmm2
	vucomiss	%xmm1, %xmm14
	ja	.L127
.L117:
	vmulss	%xmm6, %xmm6, %xmm6
	vfmadd132ss	%xmm5, %xmm6, %xmm5
	vfmadd132ss	%xmm4, %xmm5, %xmm4
	vsqrtss	%xmm4, %xmm4, %xmm1
	vucomiss	%xmm4, %xmm14
	ja	.L128
.L118:
	vsubss	%xmm1, %xmm2, %xmm0
	decl	260(%rsp)
	vdivss	%xmm2, %xmm0, %xmm0
	vmulss	%xmm0, %xmm0, %xmm0
	vaddss	200(%rsp), %xmm0, %xmm7
	vmovss	%xmm7, 200(%rsp)
	jne	.L119
	call	_ZNSt6chrono3_V212system_clock3nowEv@PLT
	vmovss	200(%rsp), %xmm7
	vxorps	%xmm14, %xmm14, %xmm14
	vdivss	.LC109(%rip), %xmm7, %xmm0
	vsqrtss	%xmm0, %xmm0, %xmm1
	vucomiss	%xmm0, %xmm14
	ja	.L129
.L120:
	movq	1432(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L130
	addq	$1448, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	vmovaps	%xmm1, %xmm0
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L124:
	.cfi_restore_state
	vmovaps	%xmm1, %xmm0
	vmovss	%xmm2, 28(%rsp)
	vmovss	%xmm1, 12(%rsp)
	call	sqrtf@PLT
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	28(%rsp), %xmm2
	vmovss	12(%rsp), %xmm1
	jmp	.L114
.L128:
	vmovaps	%xmm4, %xmm0
	vmovss	%xmm2, 16(%rsp)
	vmovss	%xmm1, 12(%rsp)
	call	sqrtf@PLT
	vmovss	16(%rsp), %xmm2
	vmovss	12(%rsp), %xmm1
	jmp	.L118
.L127:
	vmovaps	%xmm1, %xmm0
	vmovss	%xmm6, 24(%rsp)
	vmovss	%xmm5, 20(%rsp)
	vmovss	%xmm2, 16(%rsp)
	vmovss	%xmm4, 12(%rsp)
	call	sqrtf@PLT
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	24(%rsp), %xmm6
	vmovss	20(%rsp), %xmm5
	vmovss	16(%rsp), %xmm2
	vmovss	12(%rsp), %xmm4
	jmp	.L117
.L126:
	vmovss	%xmm4, 24(%rsp)
	vmovss	%xmm1, 20(%rsp)
	vmovss	%xmm3, 16(%rsp)
	vmovss	%xmm2, 12(%rsp)
	call	sqrtf@PLT
	vxorps	%xmm14, %xmm14, %xmm14
	vmovss	24(%rsp), %xmm4
	vmovss	20(%rsp), %xmm1
	vmovss	16(%rsp), %xmm3
	vmovss	12(%rsp), %xmm2
	jmp	.L116
.L125:
	vmovaps	%xmm6, %xmm0
	vmovss	%xmm3, 32(%rsp)
	vmovss	%xmm2, 12(%rsp)
	call	sqrtf@PLT
	vmovss	32(%rsp), %xmm3
	vmovss	12(%rsp), %xmm2
	jmp	.L115
.L130:
	call	__stack_chk_fail@PLT
.L129:
	vmovss	%xmm1, 12(%rsp)
	call	sqrtf@PLT
	vmovss	12(%rsp), %xmm1
	jmp	.L120
	.cfi_endproc
.LFE9314:
	.size	_Z8test_M2LILi7EEff, .-_Z8test_M2LILi7EEff
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC110:
	.string	"%i %i %e\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LFB8913:
	.cfi_startproc
	endbr64
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
	vmovss	.LC25(%rip), %xmm0
	call	_Z8test_M2LILi7EEff
	movl	$2028, %ecx
	movl	$7, %edx
	leaq	.LC110(%rip), %rsi
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk@PLT
	xorl	%eax, %eax
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE8913:
	.size	main, .-main
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC0:
	.long	1082130432
	.align 4
.LC2:
	.long	1090519040
	.align 4
.LC3:
	.long	1048576000
	.align 4
.LC4:
	.long	1073741824
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC5:
	.long	0
	.long	1072693248
	.align 8
.LC7:
	.long	0
	.long	1071644672
	.align 8
.LC8:
	.long	0
	.long	1073741824
	.align 8
.LC9:
	.long	0
	.long	1040187392
	.align 8
.LC10:
	.long	1413754136
	.long	1074340347
	.section	.rodata.cst4
	.align 4
.LC11:
	.long	3212836864
	.align 4
.LC12:
	.long	1065353216
	.align 4
.LC13:
	.long	1108344832
	.align 4
.LC14:
	.long	1111752704
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC15:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4
	.align 4
.LC21:
	.long	1092616192
	.align 4
.LC22:
	.long	3252682752
	.align 4
.LC23:
	.long	1118306304
	.align 4
.LC24:
	.long	1137836032
	.align 4
.LC25:
	.long	1056964608
	.align 4
.LC26:
	.long	3221225472
	.align 4
.LC27:
	.long	1094713344
	.align 4
.LC28:
	.long	3238002688
	.align 4
.LC29:
	.long	1109393408
	.align 4
.LC30:
	.long	1040187392
	.align 4
.LC31:
	.long	3225419776
	.align 4
.LC32:
	.long	1106247680
	.align 4
.LC33:
	.long	1086324736
	.align 4
.LC34:
	.long	3248488448
	.align 4
.LC35:
	.long	1124859904
	.align 4
.LC36:
	.long	1031798784
	.align 4
.LC37:
	.long	3242196992
	.align 4
.LC38:
	.long	1105199104
	.align 4
.LC39:
	.long	1121976320
	.align 4
.LC40:
	.long	3229614080
	.align 4
.LC41:
	.long	1113587712
	.align 4
.LC42:
	.long	1098907648
	.align 4
.LC43:
	.long	1103101952
	.align 4
.LC44:
	.long	3261071360
	.align 4
.LC45:
	.long	1140588544
	.align 4
.LC46:
	.long	1023410176
	.align 4
.LC47:
	.long	3246391296
	.align 4
.LC48:
	.long	1119879168
	.align 4
.LC49:
	.long	1131413504
	.align 4
.LC50:
	.long	3231711232
	.align 4
.LC51:
	.long	3233808384
	.align 4
.LC52:
	.long	1104150528
	.align 4
.LC53:
	.long	1113063424
	.align 4
.LC54:
	.long	1119092736
	.align 4
.LC55:
	.long	1101004800
	.align 4
.LC56:
	.long	3274178560
	.align 4
.LC57:
	.long	1155989504
	.align 4
.LC58:
	.long	1015021568
	.align 4
.LC59:
	.long	3264217088
	.align 4
.LC60:
	.long	3267362816
	.align 4
.LC61:
	.long	1132724224
	.align 4
.LC62:
	.long	1153826816
	.align 4
.LC63:
	.long	1097859072
	.align 4
.LC64:
	.long	3255304192
	.align 4
.LC65:
	.long	3260547072
	.align 4
.LC66:
	.long	1134886912
	.align 4
.LC67:
	.long	1148682240
	.align 4
.LC68:
	.long	3250585600
	.align 4
.LC69:
	.long	1117782016
	.align 4
.LC70:
	.long	1130102784
	.align 4
.LC71:
	.long	1138491392
	.align 4
.LC72:
	.long	1112539136
	.align 4
.LC73:
	.long	1118830592
	.align 4
.LC74:
	.long	1124335616
	.align 4
.LC75:
	.long	1123024896
	.align 4
.LC76:
	.long	1144258560
	.align 4
.LC77:
	.long	1167949824
	.align 4
.LC78:
	.long	3270508544
	.align 4
.LC79:
	.long	3291742208
	.align 4
.LC80:
	.long	3315433472
	.align 4
.LC81:
	.long	3240099840
	.align 4
.LC82:
	.long	1116471296
	.align 4
.LC83:
	.long	1132199936
	.align 4
.LC84:
	.long	3265789952
	.align 4
.LC85:
	.long	1147600896
	.align 4
.LC86:
	.long	3256877056
	.align 4
.LC87:
	.long	1006632960
	.align 4
.LC88:
	.long	1116733440
	.align 4
.LC89:
	.long	3280207872
	.align 4
.LC90:
	.long	1163296768
	.align 4
.LC91:
	.long	3263954944
	.align 4
.LC92:
	.long	1109917696
	.align 4
.LC93:
	.long	3244294144
	.align 4
.LC94:
	.long	1099956224
	.align 4
.LC95:
	.long	3256352768
	.align 4
.LC96:
	.long	3265003520
	.align 4
.LC97:
	.long	1102053376
	.align 4
.LC98:
	.long	3276144640
	.align 4
.LC99:
	.long	3249537024
	.align 4
.LC100:
	.long	1112014848
	.align 4
.LC101:
	.long	3282370560
	.align 4
.LC102:
	.long	1131544576
	.align 4
.LC103:
	.long	1118568448
	.align 4
.LC104:
	.long	1146519552
	.align 4
.LC105:
	.long	1124204544
	.align 4
.LC106:
	.long	1169928192
	.align 4
.LC107:
	.long	1157251072
	.align 4
.LC108:
	.long	1127612416
	.align 4
.LC109:
	.long	1176256512
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:

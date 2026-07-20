## Hi👋 There 

Hi, I'm Sung-Woo Choi from South Korea.  
I'm a Computer Science student at UNIST.

I'm interested in compilers, systems, and GPUs.  
My work focuses on LLVM-based analysis, low-level performance, and correctness issues in systems software.

## Links
# [Tistory](https://swc0317.tistory.com/): 

https://swc0317.tistory.com/

# [Contact me](swc0317@unist.ac.kr) 

swc0317@unist.ac.kr / swchoi@nshc.net

## Summary of Recent Activity
<!--START_SECTION:activity_summary-->
- Closed issue [#838](https://github.com/exaloop/codon/issues/838) regarding the behavior of `@par(schedule='static')`, which was found to lower to OpenMP chunked static scheduling, negatively impacting locality and throughput for certain workloads.
  
- Merged pull request [#837](https://github.com/exaloop/codon/pull/837) that addressed the absence of symbols in the NumPy vectorized path for GPU. This included adding GPU fill-ins for `cnp_cos_float64` and `cnp_cos_float32`, and fixing the import signature for `cnp_abs_complex64`.

- Opened and merged pull request [#835](https://github.com/exaloop/codon/pull/835) to explicitly export bfloat16 compiler-rt conversion builtins from `codonrt`, ensuring that necessary symbols are default-visible for linking into `libcodonrt.so`.

- Closed issue [#836](https://github.com/exaloop/codon/issues/836) that raised concerns about missing remaps for some NumPy vectorized loops on GPU, particularly regarding the handling of `np.cos` and `np.abs` for complex types.

- Opened issue [#834](https://github.com/exaloop/codon/issues/834) discussing linking failures for GPU kernels using `bfloat16` due to missing conversion builtins, highlighting potential environment-specific issues.
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 🔒 Closed issue [#838](https://github.com/exaloop/codon/issues/838) in [exaloop/codon](https://github.com/exaloop/codon)
2. 💪 Opened PR [#839](https://github.com/exaloop/codon/pull/839) in [exaloop/codon](https://github.com/exaloop/codon)
3. ❗ Opened issue [#838](https://github.com/exaloop/codon/issues/838) in [exaloop/codon](https://github.com/exaloop/codon)
4. 🎉 Merged PR [#837](https://github.com/exaloop/codon/pull/837) in [exaloop/codon](https://github.com/exaloop/codon)
5. 🗣 Commented on [#837](https://github.com/exaloop/codon/pull/837#issuecomment-4890197310) in [exaloop/codon](https://github.com/exaloop/codon)
6. 💪 Opened PR [#837](https://github.com/exaloop/codon/pull/837) in [exaloop/codon](https://github.com/exaloop/codon)
7. ❗ Opened issue [#836](https://github.com/exaloop/codon/issues/836) in [exaloop/codon](https://github.com/exaloop/codon)
8. 🎉 Merged PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
9. 💪 Opened PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
10. ❗ Opened issue [#834](https://github.com/exaloop/codon/issues/834) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

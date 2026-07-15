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
- Opened a pull request in the [exaloop/codon](https://github.com/exaloop/codon/pull/839) repository to lower the default static schedule as non-chunk for OpenMP, addressing issue #838. This change inspects OpenMP decorator arguments during AST-to-CIR translation to improve performance in specific scenarios.
  
- Opened an issue in the [exaloop/codon](https://github.com/exaloop/codon/issues/838) repository regarding the lowering of `@par(schedule='static')` to chunked static scheduling, which negatively impacts locality and throughput for workloads requiring contiguous iteration ownership.

- Merged a pull request in the [exaloop/codon](https://github.com/exaloop/codon/pull/837) repository that added GPU fill-ins for `cnp_cos_float64` and `cnp_cos_float32`, fixing the import signature for `cnp_abs_complex64` and including test workloads for GPU NumPy vectorized loops. This addresses issue #836.

- Merged a pull request in the [exaloop/codon](https://github.com/exaloop/codon/pull/835) repository to export bfloat16 compiler-rt conversion builtins, ensuring they are default-visible in the Codon runtime, which resolves issue #834.

- Merged a pull request in the [exaloop/codon](https://github.com/exaloop/codon/pull/830) repository that updates the NVPTX optimization pipeline to apply LLVM optimization passes to the GPU module instead of the host, addressing issue #829.
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 💪 Opened PR [#839](https://github.com/exaloop/codon/pull/839) in [exaloop/codon](https://github.com/exaloop/codon)
2. ❗ Opened issue [#838](https://github.com/exaloop/codon/issues/838) in [exaloop/codon](https://github.com/exaloop/codon)
3. 🎉 Merged PR [#837](https://github.com/exaloop/codon/pull/837) in [exaloop/codon](https://github.com/exaloop/codon)
4. 🗣 Commented on [#837](https://github.com/exaloop/codon/pull/837#issuecomment-4890197310) in [exaloop/codon](https://github.com/exaloop/codon)
5. 💪 Opened PR [#837](https://github.com/exaloop/codon/pull/837) in [exaloop/codon](https://github.com/exaloop/codon)
6. ❗ Opened issue [#836](https://github.com/exaloop/codon/issues/836) in [exaloop/codon](https://github.com/exaloop/codon)
7. 🎉 Merged PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
8. 💪 Opened PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
9. ❗ Opened issue [#834](https://github.com/exaloop/codon/issues/834) in [exaloop/codon](https://github.com/exaloop/codon)
10. 🎉 Merged PR [#830](https://github.com/exaloop/codon/pull/830) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

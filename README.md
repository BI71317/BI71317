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
- Opened a pull request in the `exaloop/codon` repository titled "[OMP] Add subtraction reduction" (PR #841) to implement subtraction handling in the OpenMP reduction path, addressing part of issue #840. The changes include the addition of `Reduction::Kind::SUB`, a subtraction reduction initializer, and detection of subtraction RHS patterns. The implementation was tested with a minimal reproducible example, confirming that the parallel execution matches the serial result.

- Created an issue in the `exaloop/codon` repository (issue #840) highlighting the lack of OpenMP reduction support for subtraction and logical operations (`and`/`or`). The issue outlines the expected behavior and provides a minimal reproducible example demonstrating the problem.

- Closed issue #838 in the `exaloop/codon` repository, which discussed the behavior of `@par(schedule='static')` lowering to chunked static scheduling, impacting performance due to round-robin iteration assignment. 

- Merged pull request #837 in the `exaloop/codon` repository, which added GPU support for missing symbols in the NumPy vectorized path. This included changes to the handling of `cnp_cos_float64` and `cnp_cos_float32`, along with updates to the import signature for `cnp_abs_complex64` and the addition of test cases for GPU workloads.

- Engaged in discussions on pull request #837 regarding the status of certain mathematical functions (`tan` and `cosh`) in the context of NumPy vectorized-loop producers, clarifying their fallback behavior and inclusion in test cases.
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 🗣 Commented on [#844](https://github.com/exaloop/codon/issues/844#issuecomment-5138231343) in [exaloop/codon](https://github.com/exaloop/codon)
2. 🔒 Closed issue [#840](https://github.com/exaloop/codon/issues/840) in [exaloop/codon](https://github.com/exaloop/codon)
3. 🎉 Merged PR [#841](https://github.com/exaloop/codon/pull/841) in [exaloop/codon](https://github.com/exaloop/codon)
4. 💪 Opened PR [#843](https://github.com/exaloop/codon/pull/843) in [exaloop/codon](https://github.com/exaloop/codon)
5. 🗣 Commented on [#833](https://github.com/exaloop/codon/issues/833#issuecomment-5100457287) in [exaloop/codon](https://github.com/exaloop/codon)
6. 🗣 Commented on [#841](https://github.com/exaloop/codon/pull/841#issuecomment-5088896485) in [exaloop/codon](https://github.com/exaloop/codon)
7. 🗣 Commented on [#841](https://github.com/exaloop/codon/pull/841#issuecomment-5087816177) in [exaloop/codon](https://github.com/exaloop/codon)
8. 🗣 Commented on [#841](https://github.com/exaloop/codon/pull/841#issuecomment-5041229372) in [exaloop/codon](https://github.com/exaloop/codon)
9. 🗣 Commented on [#841](https://github.com/exaloop/codon/pull/841#issuecomment-5031681153) in [exaloop/codon](https://github.com/exaloop/codon)
10. 🗣 Commented on [#841](https://github.com/exaloop/codon/pull/841#issuecomment-5031366518) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

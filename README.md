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
- Opened a pull request in the `exaloop/codon` repository (#826) for implementing a Python GPU DSL decorator. This implementation separates the GPU path from the existing CPU target JIT decorator structure, adding a `codon.gpu` decorator and GPU-specific source rewriting. The pull request is currently open and has not been merged yet. [View PR](https://github.com/exaloop/codon/pull/826).

- Merged a pull request in the `exaloop/codon` repository (#821) that fixed an invalid behavior when using an ordered dictionary in GPU round-trip usage. This fix addressed a segmentation fault that occurred when copying an ordered dictionary with list entries from host to device and back. [View PR](https://github.com/exaloop/codon/pull/821).

- Opened a pull request (#819) to address the limited global variable initialization problem in the `exaloop/codon` repository. This update to the constant propagation pass allows global constants initialized through scalar cast/constructor patterns to be folded, expanding the conditions under which global constants are recognized. The pull request is currently open and has not been merged yet. [View PR](https://github.com/exaloop/codon/pull/819).

- Raised an issue (#818) regarding a complex `sin` operation resulting in `nan` values when executed on the GPU. This issue is believed to be related to the uninitialized global variable problem previously identified. [View Issue](https://github.com/exaloop/codon/issues/818).

- Commented on issue #826, confirming that test cases in `codon/test/transform/kernels.codon` pass when all kernels of Codon native are rewritten in
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 💪 Opened PR [#828](https://github.com/exaloop/codon/pull/828) in [exaloop/codon](https://github.com/exaloop/codon)
2. 🗣 Commented on [#826](https://github.com/exaloop/codon/pull/826#issuecomment-4749537155) in [exaloop/codon](https://github.com/exaloop/codon)
3. 💪 Opened PR [#826](https://github.com/exaloop/codon/pull/826) in [exaloop/codon](https://github.com/exaloop/codon)
4. 💪 Opened PR [#821](https://github.com/exaloop/codon/pull/821) in [exaloop/codon](https://github.com/exaloop/codon)
5. 🗣 Commented on [#819](https://github.com/exaloop/codon/pull/819#issuecomment-4528429692) in [exaloop/codon](https://github.com/exaloop/codon)
6. 💪 Opened PR [#819](https://github.com/exaloop/codon/pull/819) in [exaloop/codon](https://github.com/exaloop/codon)
7. ❗ Opened issue [#818](https://github.com/exaloop/codon/issues/818) in [exaloop/codon](https://github.com/exaloop/codon)
8. 🎉 Merged PR [#804](https://github.com/exaloop/codon/pull/804) in [exaloop/codon](https://github.com/exaloop/codon)
9. 🗣 Commented on [#804](https://github.com/exaloop/codon/pull/804#issuecomment-4473577509) in [exaloop/codon](https://github.com/exaloop/codon)
10. ❗ Opened issue [#811](https://github.com/exaloop/codon/issues/811) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

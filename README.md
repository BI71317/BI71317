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
- Merged pull request [#821](https://github.com/exaloop/codon/pull/821) in the `exaloop/codon` repository addressed an invalid behavior when using an ordered `dict` in GPU round-trip usage, which caused segmentation faults. The fix involved ensuring proper handling of `list entries` during the copy process between host and device.

- Opened pull request [#819](https://github.com/exaloop/codon/pull/819) aims to fix a limited global variable initialization problem by updating the `constant propagation pass` to accept global constants initialized through scalar cast/constructor patterns. This change allows for better handling of global constants in the Codon compiler.

- Reported issue [#818](https://github.com/exaloop/codon/issues/818) highlights a problem where the complex `sin` operation results in `nan` values, which appears to be related to uninitialized global variables. A minimal reproducible example was provided to illustrate the issue.

- Merged pull request [#804](https://github.com/exaloop/codon/pull/804) implemented the `f16` extension in the GPU module, fixing issue [#803](https://github.com/exaloop/codon/issues/803). The implementation included a test suite for kernel scalar type lowering, ensuring that the new functionality works as intended.

- Engaged in discussions on various issues and pull requests, including clarifying the implications of changes in the JIT runtime symbol handling in pull request [#812](https://github.com/exaloop/codon/pull/812) and providing feedback on the necessity of additional test cases for kernel global variable examples in pull request [#819](https://github
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 🎉 Merged PR [#821](https://github.com/exaloop/codon/pull/821) in [exaloop/codon](https://github.com/exaloop/codon)
2. 💪 Opened PR [#821](https://github.com/exaloop/codon/pull/821) in [exaloop/codon](https://github.com/exaloop/codon)
3. 🗣 Commented on [#819](https://github.com/exaloop/codon/pull/819#issuecomment-4528429692) in [exaloop/codon](https://github.com/exaloop/codon)
4. 💪 Opened PR [#819](https://github.com/exaloop/codon/pull/819) in [exaloop/codon](https://github.com/exaloop/codon)
5. ❗ Opened issue [#818](https://github.com/exaloop/codon/issues/818) in [exaloop/codon](https://github.com/exaloop/codon)
6. 🗣 Commented on [#812](https://github.com/exaloop/codon/pull/812#issuecomment-4504037522) in [exaloop/codon](https://github.com/exaloop/codon)
7. 🗣 Commented on [#802](https://github.com/exaloop/codon/pull/802#issuecomment-4484995207) in [exaloop/codon](https://github.com/exaloop/codon)
8. 🎉 Merged PR [#804](https://github.com/exaloop/codon/pull/804) in [exaloop/codon](https://github.com/exaloop/codon)
9. 🗣 Commented on [#804](https://github.com/exaloop/codon/pull/804#issuecomment-4473577509) in [exaloop/codon](https://github.com/exaloop/codon)
10. ❗ Opened issue [#811](https://github.com/exaloop/codon/issues/811) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

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
- Merged pull request [#835](https://github.com/exaloop/codon/pull/835) in the `exaloop/codon` repository added bfloat16 compiler runtime conversion helper symbols to `CMakeLists.txt`, addressing issue #834. This change ensures that the bf16 conversion helpers are default-visible before linking into `libcodonrt.so`.

- Merged pull request [#830](https://github.com/exaloop/codon/pull/830) updated the NVPTX optimization pipeline to apply GPU optimization passes (`gpuopt1` and `gpuopt2`) to the prepared GPU module instead of the host module, resolving issue #829.

- Merged pull request [#828](https://github.com/exaloop/codon/pull/828) implemented a patch in `gpu.cpp` to handle cases where no concrete GPU kernels are realized, fixing issue #827 by ensuring no GPU module is created and no PTX blob is emitted when no definitions exist.

- Opened issue [#834](https://github.com/exaloop/codon/issues/834) raised concerns about linking failures for GPU kernels using bfloat16 scalar operations due to missing compiler runtime conversion helper symbols.

- Opened issue [#829](https://github.com/exaloop/codon/issues/829) highlighted that GPU optimization passes were incorrectly applied to the host module instead of the GPU module, potentially leaving GPU-only helper functions unoptimized.

- Opened pull request [#826](https://github.com/exaloop/codon/pull/826) proposed a Python GPU DSL decorator implementation, which includes GPU-specific source rewriting and a dedicated callback path for launch metadata.
<!--END_SECTION:activity_summary-->

## Recent Activity
<!--START_SECTION:activity-->
1. 🎉 Merged PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
2. 💪 Opened PR [#835](https://github.com/exaloop/codon/pull/835) in [exaloop/codon](https://github.com/exaloop/codon)
3. ❗ Opened issue [#834](https://github.com/exaloop/codon/issues/834) in [exaloop/codon](https://github.com/exaloop/codon)
4. 🎉 Merged PR [#830](https://github.com/exaloop/codon/pull/830) in [exaloop/codon](https://github.com/exaloop/codon)
5. 💪 Opened PR [#830](https://github.com/exaloop/codon/pull/830) in [exaloop/codon](https://github.com/exaloop/codon)
6. ❗ Opened issue [#829](https://github.com/exaloop/codon/issues/829) in [exaloop/codon](https://github.com/exaloop/codon)
7. 💪 Opened PR [#828](https://github.com/exaloop/codon/pull/828) in [exaloop/codon](https://github.com/exaloop/codon)
8. 🗣 Commented on [#826](https://github.com/exaloop/codon/pull/826#issuecomment-4749537155) in [exaloop/codon](https://github.com/exaloop/codon)
9. 💪 Opened PR [#826](https://github.com/exaloop/codon/pull/826) in [exaloop/codon](https://github.com/exaloop/codon)
10. ❗ Opened issue [#811](https://github.com/exaloop/codon/issues/811) in [exaloop/codon](https://github.com/exaloop/codon)
<!--END_SECTION:activity-->

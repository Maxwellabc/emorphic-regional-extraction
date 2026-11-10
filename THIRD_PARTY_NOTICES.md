# Third-party provenance and license scope

## SmoothE artifact

Upstream repository: https://github.com/yaohuicai/smoothe-artifact

Fixed source commit: `c1164eecceccce361766d5b28321bdeef2b3cc87`.

Fixed license source:
https://github.com/yaohuicai/smoothe-artifact/blob/c1164eecceccce361766d5b28321bdeef2b3cc87/LICENSE.txt

Original copyright statement:

> Copyright (c) 2022, Cornell Zhang Research Group
> All rights reserved.

License: BSD-3-Clause. The complete, unmodified upstream license is preserved
as `third_party/smoothe/LICENSE.txt`; its SHA256 is
`c563b26f68450b0e97bf6b57f679ac3313994fbc6d6899ad6597653a246ffc57`.
The original conditions and disclaimer must be preserved when redistributing
covered upstream material. The names of the copyright holder or contributors
must not be used to imply endorsement without prior written permission.

This finite bundle includes that license text as a provenance record. It does
not contain the upstream executable GA code or the full differentiable
optimizer. The broader project's GA comparator is an explicitly budget-adapted
implementation informed by the fixed artifact, not a full SmoothE reproduction
and not claimed to be a clean-room implementation. Any later bundle that adds
copied or adapted upstream source must retain the applicable notices with it.
The upstream license must not be applied to this project's independent code
without an authorized project-license decision.

Academic method citation:

Yaohui Cai, Kaixin Yang, Chenhui Deng, Cunxi Yu, and Zhiru Zhang. 2025.
SmoothE: Differentiable E-Graph Extraction. In Proceedings of the 30th ACM
International Conference on Architectural Support for Programming Languages
and Operating Systems, Volume 1 (ASPLOS '25). ACM, 15 pages.
https://doi.org/10.1145/3669940.3707262

The paper citation records method provenance; it does not substitute for the
full source license. The source license does not authorize redistribution of
the paper's text or figures under that license.

## Other upstream tools and benchmarks

The fixed E-Syn2 and EPFL links in `README.md` are source pointers. This bundle
does not redistribute their tool sources, binaries, library, or circuit files.
No license claim is made here for the cleaned ASAP7 library or the entire
upstream dependency/data tree. Those require a separate inclusion and license
review if a complete physical artifact is prepared.

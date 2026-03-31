# SPDX-License-Identifier: Apache-2.0
"""Early torch compatibility shim for Intel Gaudi.

Imported via ``_vllm_gaudi_torch_compat.pth`` before any other package
so that ``vllm.env_override`` can safely do::

    from torch._dynamo.convert_frame import GraphCaptureOutput

Gaudi's torch fork (2.9.0+hpu) already includes the fix from
pytorch/177558.  The class was renamed/removed, so the import
fails.  We inject a lightweight stub — the monkey-patch that
follows the import in env_override is never actually needed on Gaudi.
"""

import torch._dynamo.convert_frame as _cf

if not hasattr(_cf, "GraphCaptureOutput"):

    class _GraphCaptureOutput:
        """Stub — only exists so the import succeeds."""

        @staticmethod
        def get_runtime_env():  # type: ignore[override]
            raise NotImplementedError("Should never be called on Gaudi")

    _cf.GraphCaptureOutput = _GraphCaptureOutput  # type: ignore[attr-defined]

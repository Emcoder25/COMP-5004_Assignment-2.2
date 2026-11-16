"""NumCompute NumPy-only package."""

import sys

def _alias_stream_module() -> None:
    try:
        from numcompute_stream import stream as stream_module
        sys.modules.setdefault("numcompute.stream", stream_module)
    except ImportError:
        pass

_alias_stream_module()
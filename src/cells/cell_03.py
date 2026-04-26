"""Auto-extracted from notebooks/Crop_y_Discovery.ipynb (code cell 3).
This file mirrors notebook logic for script-style orchestration.
"""

# ---------------------------------------------------------------------------
# Pandas 3.x compatibility patch for cropnet's DataDownloader.
# ---------------------------------------------------------------------------
# DataDownloader was written against pandas 2.x, where pd.read_json() accepted
# raw JSON strings. Pandas 3.x deprecated this and now requires a file-like
# object (StringIO). Without this patch, downloads crash with a TypeError.
#
# We monkey-patch pd.read_json to wrap raw JSON strings in StringIO before
# passing them through. The `_cropnet_patched` flag prevents double-patching
# if this cell is re-run.
# ---------------------------------------------------------------------------
import io
import pandas as pd

if not getattr(pd, '_cropnet_patched', False):
    _orig_read_json = pd.read_json

    def _patched_read_json(path_or_buf, *args, **kwargs):
        # If the input looks like a JSON literal (starts with { or [),
        # wrap it in StringIO so pandas 3.x accepts it.
        if isinstance(path_or_buf, str) and path_or_buf.lstrip().startswith(("{", "[")):
            path_or_buf = io.StringIO(path_or_buf)
        return _orig_read_json(path_or_buf, *args, **kwargs)

    pd.read_json = _patched_read_json
    pd._cropnet_patched = True
    print("pd.read_json patched for pandas 3.x compatibility")

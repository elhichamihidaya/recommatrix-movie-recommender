"""Source package for recommendation optimization with metaheuristics."""

import os
import sys

# Ensure Windows finds conda environment DLLs (OpenBLAS, etc.)
if sys.platform == "win32":
    library_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(library_bin):
        try:
            os.add_dll_directory(library_bin)
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = library_bin + os.pathsep + os.environ.get("PATH", "")

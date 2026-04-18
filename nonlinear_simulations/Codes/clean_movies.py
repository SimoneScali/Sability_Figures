#!/usr/bin/env python3
from pathlib import Path
import shutil

movie = Path(".").resolve()

# Only clean if drift.json exists (in movie_test)
drift = movie / "movie_test" / "drift.json"
if drift.exists():

    # ---- keep last 5 state files ----
    state_files = sorted(movie.glob("state[0-9][0-9][0-9][0-9].hdf5"))

    if len(state_files) > 5:
        for f in state_files[:-5]:
            try:
                if f.is_file() or f.is_symlink():
                    f.unlink()
            except Exception:
                pass

    # ---- keep last 3 Visu folders/files ----
    visu_items = sorted(movie.glob("Visu*"))

    if len(visu_items) > 3:
        for v in visu_items[:-3]:
            try:
                if v.is_dir():
                    shutil.rmtree(v, ignore_errors=True)
                else:
                    v.unlink()
            except Exception:
                pass

"""
Check presence and basic integrity of model files (.pth, .pkl) in the repository.
Prints file path, size and attempts to load .pkl files with joblib and .pth with torch (if available).
"""
import os
import sys
import traceback
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print(f"Repository root: {root}")

model_exts = ['.pth', '.pkl']
found = []
for ext in model_exts:
    for p in root.rglob(f'*{ext}'):
        try:
            size = p.stat().st_size
        except Exception:
            size = -1
        found.append((str(p), ext, size))

if not found:
    print("No .pth or .pkl files found in repository.")
    sys.exit(0)

for path, ext, size in sorted(found):
    print('\n' + '-'*60)
    print(f"File: {path}")
    print(f"Extension: {ext}")
    print(f"Size (bytes): {size}")

    if ext == '.pkl':
        try:
            import joblib
            obj = joblib.load(path)
            print(f"joblib.load() succeeded. Type: {type(obj)}")
            # If it's a dict-like, show keys
            try:
                if hasattr(obj, 'keys'):
                    print(f"Contents keys: {list(obj.keys())[:10]}")
            except Exception:
                pass
        except Exception as e:
            print(f"joblib.load() failed: {e}")
            print(traceback.format_exc())
    elif ext == '.pth':
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu')
            print(f"torch.load() succeeded. Type: {type(checkpoint)}")
            try:
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
            except Exception:
                pass
        except Exception as e:
            print(f"torch.load() failed or torch not installed: {e}")
            # print minimal traceback
            # print(traceback.format_exc())

print('\nDone.')

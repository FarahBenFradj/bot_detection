"""
scripts/resave_models.py
Re-save all .h5 models to the native .keras format.
Usage: python scripts/resave_models.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras

models_dir = Path("models")

h5_files = list(models_dir.glob("*.h5"))
if not h5_files:
    print("❌  No .h5 files found in models/")
    sys.exit(1)

for h5_path in h5_files:
    keras_path = h5_path.with_suffix(".keras")
    print(f"Converting {h5_path.name} → {keras_path.name} …", end=" ")
    try:
        model = keras.models.load_model(str(h5_path))
        model.save(str(keras_path))
        print("✅")
    except Exception as e:
        print(f"❌  {e}")

print("\nDone. Use .keras files from now on.")
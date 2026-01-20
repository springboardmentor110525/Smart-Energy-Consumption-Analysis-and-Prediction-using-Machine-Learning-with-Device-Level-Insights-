# Solution: Fix ModuleNotFoundError for matplotlib

## Problem
You're getting `ModuleNotFoundError: No module named 'matplotlib'` even though pip install shows it's installed.

## Root Cause
This happens when:
1. Jupyter notebook is using a different Python environment than where packages are installed
2. The kernel needs to be restarted after installation
3. Multiple Python installations exist on your system

## Solutions (Try in order)

### Solution 1: Install packages directly in Jupyter notebook
Run this in a Jupyter cell:

```python
import sys
!{sys.executable} -m pip install --upgrade matplotlib scikit-learn pandas numpy seaborn tensorflow joblib flask
```

Then **restart the kernel** (Kernel → Restart Kernel) and run your imports again.

### Solution 2: Use requirements.txt
I've created a `requirements.txt` file. Install all packages at once:

```bash
pip install -r requirements.txt
```

Then restart your Jupyter kernel.

### Solution 3: Install in specific Python environment
If you're using a virtual environment or conda:

**For venv:**
```bash
# Activate your environment first
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Then install
pip install -r requirements.txt
```

**For conda:**
```bash
conda install matplotlib scikit-learn pandas numpy seaborn
conda install -c conda-forge tensorflow
```

### Solution 4: Reinstall Jupyter in the same environment
```bash
pip install --upgrade jupyter ipykernel
python -m ipykernel install --user --name=myenv
```

Then select this kernel in Jupyter.

### Solution 5: Quick fix for immediate testing
Add this at the top of your notebook (temporary solution):

```python
import sys
import subprocess

# Install packages if not available
packages = ['matplotlib', 'scikit-learn', 'pandas', 'numpy', 'seaborn']
for package in packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
```

## Verification
After installation, run `check_imports.py` to verify:

```bash
python check_imports.py
```

This will show which packages are installed and their versions.

## For Your Specific Case
Based on your notebook, you need these packages:
- ✓ pandas (for data manipulation)
- ✓ numpy (for numerical operations)
- ✓ matplotlib (for plotting)
- ✓ seaborn (for advanced visualizations)
- ✓ scikit-learn (for ML models)
- ✓ tensorflow (for LSTM/deep learning)
- ✓ joblib (for model saving)
- ✓ flask (for API deployment)

## Next Steps
1. Try Solution 1 first (install from within Jupyter)
2. **Restart the kernel** - This is crucial!
3. Run the imports again
4. If still failing, check which Python Jupyter is using:
   ```python
   import sys
   print(sys.executable)
   ```
5. Install packages to that specific Python installation

## Common Mistakes to Avoid
- ❌ Not restarting the kernel after installation
- ❌ Installing to wrong Python environment
- ❌ Using `!pip install` without `sys.executable`
- ❌ Having multiple Python versions without knowing which one Jupyter uses

## Still Not Working?
If none of the above work, provide me with:
1. Output of `python --version`
2. Output of `where python` (Windows) or `which python` (Linux/Mac)
3. Output of running this in Jupyter:
   ```python
   import sys
   print(sys.executable)
   print(sys.path)
   ```

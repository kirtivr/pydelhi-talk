#!/bin/sh

# Use Conda for this project without exiting the parent shell when sourced.
# Usage: source ./makevenv.sh CONDA_BASE_PATH
#   CONDA_BASE_PATH: Required path to Conda/Miniconda installation base directory
#                    Example: /home/user/miniconda3 or /opt/conda

ENV_NAME="pydelhi-talk"

# Exit the script early (without killing the caller) if a hard error occurs
_stop_here_ok() { return 0 2>/dev/null || exit 0; }

# Require Conda base location from command line argument
if [ -z "$1" ]; then
    echo "Error: Conda base path is required." >&2
    echo "Usage: source ./makevenv.sh /path/to/conda/or/miniconda" >&2
    echo "Example: source ./makevenv.sh \$HOME/miniconda3" >&2
    _stop_here_ok
fi

CONDA_BASE="$1"

# Initialize conda for the current shell if possible
# Try bin/conda.sh first, then fall back to etc/profile.d/conda.sh
CONDA_SH=""
if [ -r "$CONDA_BASE/bin/conda.sh" ]; then
    CONDA_SH="$CONDA_BASE/bin/conda.sh"
elif [ -r "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_SH" ]; then
    # Store the CONDA_BASE we want to use before sourcing
    _CONDA_BASE_TO_USE="$CONDA_BASE"
    # Unset any existing CONDA_BASE to avoid conflicts
    unset CONDA_BASE
    . "$CONDA_SH"
    # Re-set CONDA_BASE to our explicit path after sourcing
    CONDA_BASE="$_CONDA_BASE_TO_USE"
    unset _CONDA_BASE_TO_USE
else
    echo "Unable to find conda.sh. Checked:" >&2
    echo "  $CONDA_BASE/bin/conda.sh" >&2
    echo "  $CONDA_BASE/etc/profile.d/conda.sh" >&2
    _stop_here_ok
fi

# Use Python from Miniconda to run conda (fixes bad shebang in conda executable)
CONDA_PYTHON="$CONDA_BASE/bin/python"
CONDA_EXE="$CONDA_BASE/bin/conda"
if [ ! -x "$CONDA_PYTHON" ]; then
    echo "Python not found at: $CONDA_PYTHON" >&2
    _stop_here_ok
fi
if [ ! -f "$CONDA_EXE" ]; then
    echo "Conda script not found at: $CONDA_EXE" >&2
    _stop_here_ok
fi

# Create env if missing
if [ ! -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    echo "Creating conda env '$ENV_NAME'..."
    if ! "$CONDA_PYTHON" "$CONDA_EXE" create -y -n "$ENV_NAME" python=3.13 >/dev/null 2>&1; then
        echo "python=3.13 not available, trying python=3.12..."
        "$CONDA_PYTHON" "$CONDA_EXE" create -y -n "$ENV_NAME" python=3.12 || { echo "Failed to create conda env." >&2; _stop_here_ok; }
    fi
fi

# Activate env
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Failed to activate conda env '$ENV_NAME'." >&2
    _stop_here_ok
fi

# Verify activation by checking if we're in the right environment
ENV_PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
ENV_PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

if [ ! -x "$ENV_PYTHON" ]; then
    echo "Conda env '$ENV_NAME' Python not found at: $ENV_PYTHON" >&2
    _stop_here_ok
fi

# Upgrade pip and install deps using the conda environment's pip
"$ENV_PYTHON" -m pip install -U pip >/dev/null 2>&1 || true
if [ -f "requirements.txt" ]; then
    "$ENV_PIP" install -r requirements.txt || true
fi

echo "Activated conda env '$ENV_NAME'."
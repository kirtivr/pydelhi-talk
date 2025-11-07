### PyDelhi Talk — Setup and Usage

This repo contains small Python demos comparing LLM request strategies and using an OpenAI-compatible API.

### Prerequisites

- **Conda or Miniconda** installed on your system
- Linux/macOS shell (examples use bash/zsh)

### 1) Create and activate the Conda environment

From the project root, you **must** provide the path to your Conda/Miniconda installation:

```bash
source ./makevenv.sh /path/to/your/conda/or/miniconda
```

**Examples:**
```bash
# If Miniconda is installed in your home directory
# (The path should point to the directory containing bin/conda)
source ./makevenv.sh $HOME/miniconda3

# Or with an absolute path
source ./makevenv.sh /opt/conda

# Or if you have Anaconda
source ./makevenv.sh $HOME/anaconda3

# To find your Conda base path, run: conda info --base
# Then use that path: source ./makevenv.sh $(conda info --base)
```

**What it does:**
- Creates a Conda environment named `pydelhi-talk` if missing (with Python 3.13 or 3.12)
- Activates the environment
- Upgrades `pip`
- Installs dependencies from `requirements.txt` if present

**Note:** The Conda base path is required. This is typically the directory where Conda/Miniconda is installed (the directory containing `bin/conda`).

### 2) Install dependencies

If you didn't let `makevenv.sh` handle it:

```bash
pip install -r requirements.txt
```

### 3) Set required environment variables

Some scripts require API keys. Export what you need before running:

```bash
# For context_management_with_mem0.py (uses Mem0 for memory management and DeepSeek for LLM)
export MEM0_API_KEY="your-mem0-api-key"  # Get your API key from https://app.mem0.ai/
export DEEPSEEK_API_KEY="your-deepseek-key"
# Optional overrides (defaults shown)
export DEEPSEEK_API_BASE="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"

# For throughput_parallel_vs_sequential.py and ttft_prefix_caching_1.py (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 4) Run the demos

Each script prints metrics to stdout.

- Run Mem0 context management demo:

```bash
python context_management_with_mem0.py
```

- Run parallel vs sequential requests demo:

```bash
python throughput_parallel_vs_sequential.py
```

- Run TTFT and prefix caching strategies demo:

```bash
python ttft_prefix_caching_1.py
```

### Notes

- The file `large_shakespearean_text_dump` is loaded by the Anthropic demos; keep it in the project root.
- To leave the Conda environment: `conda deactivate`.
- The Conda base path argument is required when running `makevenv.sh`.


### PyDelhi Talk — Setup and Usage

This repo contains small Python demos comparing LLM request strategies and using an OpenAI-compatible API.

### Prerequisites

- **Conda or Miniconda** installed on your system
- Linux/macOS shell (examples use bash/zsh)

### 1) Create and activate the Conda environment

From the project root:

```bash
source ./makevenv.sh
```

**If your Conda installation is in a non-standard location**, specify the path:

```bash
source ./makevenv.sh /path/to/your/conda/or/miniconda
```

What it does:
- Creates a Conda environment named `pydelhi-talk` if missing (with Python 3.13 or 3.12)
- Activates the environment
- Upgrades `pip`
- Installs dependencies from `requirements.txt` if present

**Note:** The script will automatically detect Conda if it's in your PATH. Otherwise, you can pass the Conda base directory as an argument.

### 2) Install dependencies

If you didn't let `makevenv.sh` handle it:

```bash
pip install -r requirements.txt
```

### 3) Set required environment variables

Some scripts require API keys. Export what you need before running:

```bash
# For context_management_with_mem0.py (uses an OpenAI-compatible client against DeepSeek)
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
- If Conda is not in your PATH, pass the Conda base directory to `makevenv.sh` as an argument.


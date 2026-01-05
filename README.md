# arxiv-daily

[![PyPI version](https://badge.fury.io/py/arxiv-daily.svg)](https://pypi.org/project/arxiv-daily/)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AI-powered arXiv research assistant** - Beautiful terminal interface for tracking arXiv preprints and generating intelligent summaries with LLMs.

**Key capabilities:**
- **Daily arXiv Updates**: Fetch and filter the latest preprints from any arXiv channel.
- **AI-Powered Summaries**: Generate structured, organized summaries using LLMs.
- **Beautiful Output**: Colorful terminal output, syntax highlighting, and progress bars using the Rich library.
- **Smart Filtering**: Filter by arXiv categories and channels for focused research.

## Quick Start

### Install

Install the package from PyPI:
```bash
pip install arxiv-daily
```

Or install from source for development:
```bash
git clone https://github.com/GZU-MuTian/arxiv-daily.git
cd arxiv-daily
pip install -e .
```

### Environment Setup (Recommended)

To streamline usage and avoid repetitive CLI flags, we recommend configuring environment variables. This approach simplifies command execution and enhances security by avoiding credentials in command history.

```bash
# LLM Configuration (required)
DEEPSEEK_API_KEY="your-deepseek-api-key-here"

# Default arXiv categories (comma-separated)
export ARXIV_CATEGORY="cs.AI,astro-ph.HE,hep-ph"
```

## Usage Guide

### Command-Line Interface

`arxiv-daily` includes a CLI named `arXiv`.
> 🔧 Tip: Run `arXiv --help` for an overview, or `arXiv <command> --help` for command-specific options.

Fetch the latest preprints from any arXiv channel with beautiful terminal formatting:
```bash
# Get the latest papers in Astrophysics
arXiv new

# Specific channel (e.g., Computer Science - AI)
arXiv new --channel cs.AI

# Filter by multiple categories
arXiv new --channel astro-ph --category astro-ph.HE,astro-ph.IM

```

Generate AI Summaries:
```bash
# Basic summary with default model (DeepSeek)
arXiv summarize <arxiv_id>

# Specify model and provider
arXiv summarize <arxiv_id> --model deepseek-chat --provider deepseek

# Short form
arXiv summarize <arxiv_id> -m deepseek-chat -p deepseek -t 0.5
```

Adjust verbosity for debugging or quiet runs:
```bash
# Production - errors only (default)
arXiv --log-level ERROR new

# Short form for detailed debugging
arXiv -v DEBUG new
```

## Project Structure

```text
arxiv_daily/
├── agents.py        # LangGraph agents for complex summarization workflows
├── chains.py        # LangChain chains for LLM interactions
├── cli.py           # Command-line interface built with Typer
├── core.py          # Core functions (_run_new, _run_summarize)
├── llm_client.py    # Unified LLM provider interface
├── utils.py         # Utility functions
└── __init__.py
```

## Related Resources

- [arXiv.org](https://arxiv.org/list/astro-ph/new)
- [arXiv Category Taxonomy](https://arxiv.org/category_taxonomy)
- [LangGraph Guide](https://docs.langchain.com/)

## Contact

For questions and support:

- Author: Yu Liu
- Email: yuliu@gzu.edu.cn
- GitHub Issues: [Report bugs or request features](https://github.com/GZU-MuTian/arxiv-daily/issues)
# arxiv-daily

[![PyPI version](https://badge.fury.io/py/arxiv-daily.svg)](https://pypi.org/project/arxiv-daily/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**AI-powered arXiv research assistant** - Beautiful terminal interface for tracking arXiv preprints and generating intelligent summaries with LLMs.

Build your own personal **RAG (Retrieval-Augmented Generation) knowledge base** - track daily papers, generate structured summaries with rich metadata, and export them to Markdown for seamless integration with vector databases, semantic search engines, and note-taking workflows like Obsidian.

**Key capabilities:**
- **Daily arXiv Updates**: Fetch and filter the latest preprints from any arXiv channel.
- **AI-Powered Summaries**: Generate structured, organized summaries using LLMs.
- **Paper Metadata**: Fetch detailed metadata for any arXiv paper.
- **Beautiful Output**: Colorful terminal output, syntax highlighting, and progress bars using the Rich library.
- **Smart Filtering**: Filter by arXiv categories and channels for focused research.
- **Obsidian Integration**: Export summaries as Markdown with frontmatter for knowledge management.

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

# Default output directory for summaries (optional)
export ARXIV_SUMMARIZE_OUTPUT="/path/to/your/obsidian/vault"

# Default output directory for knowledge graph concepts (optional)
export ARXIV_EXTRACTOR_OUTPUT="/path/to/your/obsidian/vault/concepts"
```

## Usage Guide

### Command-Line Interface

`arxiv-daily` includes a CLI named `arXiv`.
> Tip: Run `arXiv --help` for an overview, or `arXiv <command> --help` for command-specific options.

Fetch the latest preprints from any arXiv channel with beautiful terminal formatting:
```bash
# Get the latest papers in Astrophysics
arXiv new

# Specific channel (e.g., Computer Science - AI)
arXiv new --channel cs.AI

# Filter by multiple categories
arXiv new --channel astro-ph --category astro-ph.HE,astro-ph.IM
```

Fetch Paper Metadata:
```bash
# Get metadata for a specific paper
arXiv meta 2401.12345

# Supports various input formats
arXiv meta arXiv:2401.12345
arXiv meta arXiv:2401.12345v1
```

Generate AI Summaries:
```bash
# Basic summary with default model (DeepSeek)
arXiv summarize 2401.12345

# Specify model and provider
arXiv summarize 2401.12345 --model deepseek-chat --provider deepseek

# Short form
arXiv summarize 2401.12345 -m deepseek-chat -p deepseek -t 0.5

# Save to file (if ARXIV_SUMMARIZE_OUTPUT is set)
arXiv summarize 2401.12345

# Save to specific directory
arXiv summarize 2401.12345 -o /path/to/output
```

**Extract Knowledge Graph Relationships:**
```bash
# Basic extraction with default model (DeepSeek)
arXiv extractor 2401.12345

# Specify model and provider
arXiv extractor 2401.12345 --model deepseek-chat --provider deepseek

# Short form
arXiv extractor 2401.12345 -m deepseek-chat -p deepseek -t 0.5

# Save concept files to directory (if ARXIV_EXTRACTOR_OUTPUT is set)
arXiv extractor 2401.12345

# Save to specific directory
arXiv extractor 2401.12345 -o /path/to/concepts
```

The extractor command analyzes paper summaries and extracts key concepts with their relationships, creating a structured knowledge graph. Each concept is categorized and linked to the source paper, making it perfect for building a personal research knowledge base.

**Obsidian Integration:**
When using the `-o` option, concepts are saved as individual Markdown files with:
- YAML frontmatter for metadata
- Obsidian-style links (`[[arxiv-id]]`)
- Automatic deduplication (same paper won't be added twice)

Adjust verbosity for debugging or quiet runs:
```bash
# Production - errors only (default)
arXiv --log-level ERROR new

# Short form for detailed debugging
arXiv -v DEBUG new
```

## Knowledge Graph Extraction

The `arXiv extractor` command builds a structured knowledge base by extracting key concepts and relationships from academic papers.

### Concept Categories

The extractor classifies concepts into these research domains:

- **galaxy-physics**: Galaxy formation, evolution, dynamics
- **cosmology**: Dark matter, cosmic microwave background, large-scale structure
- **earth-planetary**: Exoplanets, planetary atmospheres, astrobiology
- **high-energy-astrophysics**: Black holes, neutron stars, gamma-ray bursts
- **solar-stellar**: Stellar evolution, solar physics, star formation
- **statistics-ai**: Machine learning, statistical methods, neural networks
- **numerical-simulation**: N-body simulations, hydrodynamics, radiative transfer
- **instrumental-design**: Telescopes, spectrographs, detectors
- **astronomical-events**: Supernovae, gravitational waves, fast radio bursts

### Example Workflow

```bash
# 1. Generate summary first
arXiv summarize 2401.12345 -o ./summaries

# 2. Extract knowledge graph
arXiv extractor 2401.12345 -o ./concepts
```

### Integration with Obsidian

The extractor is designed to work seamlessly with Obsidian:

1. **Backlinks**: Use `[[arxiv-id]]` syntax for paper references
2. **Tags**: Automatic tagging for easy filtering
3. **Graph View**: Visualize connections between papers and concepts
4. **Search**: Find all papers mentioning a specific concept

## Project Structure

```text
arxiv_daily/
├── agents.py        # LangGraph agents for complex summarization workflows
├── chains.py        # LangChain chains for LLM interactions (includes KnowledgeGraphExtractor)
├── cli.py           # Command-line interface built with Typer
├── core.py          # Core functions (_run_new, _run_summarize, _run_extractor)
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
from .core import _run_new, _run_summarize


arxiv_new = _run_new
arxiv_summarize = _run_summarize


__all__ = ["arxiv_new", "arxiv_summarize"]
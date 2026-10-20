"""Бенчмарки сред и рубрика SFT."""
from space_env.eval.benchmarks import run_benchmark, run_registered_benchmarks
from space_env.eval.sft_rubric import score_prediction, term_recall

__all__ = ["run_benchmark", "run_registered_benchmarks", "score_prediction", "term_recall"]

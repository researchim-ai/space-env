"""
Генераторы данных для претренинга и SFT.
"""
from space_env.data_generators.pretraining import generate_pretraining_chunks
from space_env.data_generators.sft import generate_sft_records

__all__ = ["generate_pretraining_chunks", "generate_sft_records"]

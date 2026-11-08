"""Compatibility entry point for the installed planner comparison API."""

from space_env.llm.comparison import METHODS, audit, main, run

__all__ = ["METHODS", "audit", "main", "run"]

if __name__ == "__main__":
    main()

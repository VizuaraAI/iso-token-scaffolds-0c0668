"""Scaffolding at matched total token cost.

Package layout:

    cost.py       the token accountant and the three cost models
    models.py     backends (mock, vLLM, OpenAI-compatible) that report usage
    data.py       dataset loaders
    retrieval.py  BM25, dense and oracle exemplar retrieval
    scaffolds.py  the scaffold library, each with one budget knob
    evaluate.py   answer scoring and difficulty stratification
    run_sweep.py  main experiment entry point
    analysis.py   iso-cost curves, crossover costs, the LaTeX table
    calibrate.py  fit the cost proxies against measured hardware cost
"""

__version__ = "0.1.0"

# The `datasets` formatter behind HuggingFaceDatasets.jl's "julia" format. The Julia package
# runs this file in its `__init__`, so the "julia" format type exists only in Python processes
# that have loaded the package.

from datasets.formatting import NumpyFormatter, _register_formatter


class JuliaFormatter(NumpyFormatter):
    """`NumpyFormatter` that keeps each column's Arrow dtype.

    `NumpyFormatter` casts integer arrays to int64 and float arrays to float32 unless the format
    kwargs set a `dtype`; `dtype=None` keeps the dtype that numpy gets from Arrow.
    """

    def __init__(self, features=None, token_per_repo_id=None, **np_array_kwargs):
        np_array_kwargs.setdefault("dtype", None)
        super().__init__(features=features, token_per_repo_id=token_per_repo_id, **np_array_kwargs)


_register_formatter(JuliaFormatter, "julia")

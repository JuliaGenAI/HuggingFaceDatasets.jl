# The `datasets` formatter behind HuggingFaceDatasets.jl's "julia" format. The Julia package
# runs this file in its `__init__`, so the "julia" format type exists only in Python processes
# that have loaded the package.

import pyarrow as pa
from datasets import Value
from datasets.formatting import NumpyFormatter, _register_formatter


def _strings(column):
    # An object array of `str` (`None` for nulls). The numpy extractor would instead build a
    # fixed-width `<U{maxlen}` array, i.e. 4 bytes per character of the longest string, per row.
    return column.to_numpy(zero_copy_only=False)


class JuliaFormatter(NumpyFormatter):
    """`NumpyFormatter` that keeps each column's Arrow dtype, and reads string columns as
    object arrays of `str`.

    `NumpyFormatter` casts integer arrays to int64 and float arrays to float32 unless the format
    kwargs set a `dtype`; `dtype=None` keeps the dtype that numpy gets from Arrow.
    """

    def __init__(self, features=None, token_per_repo_id=None, **np_array_kwargs):
        np_array_kwargs.setdefault("dtype", None)
        super().__init__(features=features, token_per_repo_id=token_per_repo_id, **np_array_kwargs)

    def _string_columns(self, pa_table):
        # Top-level `Value("string")` columns: nothing to decode or tensorize.
        return {
            field.name
            for field in pa_table.schema
            if (pa.types.is_string(field.type) or pa.types.is_large_string(field.type))
            and (not self.features or isinstance(self.features.get(field.name), Value))
        }

    def _format_split(self, pa_table, format_rest, extract_string):
        names = self._string_columns(pa_table)
        if not names:
            return format_rest(pa_table)
        rest = pa_table.select([name for name in pa_table.column_names if name not in names])
        out = format_rest(rest) if rest.num_columns else {}
        return {
            name: extract_string(pa_table.column(name)) if name in names else out[name]
            for name in pa_table.column_names
        }

    def format_row(self, pa_table):
        return self._format_split(pa_table, super().format_row, lambda column: column[0].as_py())

    def format_column(self, pa_table):
        if self._string_columns(pa_table):
            return _strings(pa_table.column(0))
        return super().format_column(pa_table)

    def format_batch(self, pa_table):
        return self._format_split(pa_table, super().format_batch, _strings)


_register_formatter(JuliaFormatter, "julia")

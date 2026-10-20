# The `datasets` formatter behind HuggingFaceDatasets.jl's "julia" format, registered on import.
# The Julia package appends this directory to `sys.path` and imports the module in its
# `__init__`. Other Python processes import it when they unpickle a "julia"-formatted dataset
# (see `JuliaFormatType`); they never need Julia.

import numpy as np
import pyarrow as pa
from datasets import Value
from datasets.features.features import require_decoding
from datasets.formatting import NumpyFormatter, _register_formatter


def _strings(column):
    # An object array of `str` (`None` for nulls). The numpy extractor would instead build a
    # fixed-width `<U{maxlen}` array, i.e. 4 bytes per character of the longest string, per row.
    return column.to_numpy(zero_copy_only=False)


def _first_string(column):
    return column[0].as_py()


class RaggedArray(np.ndarray):
    """The rows of a list column of numbers, as `NumpyFormatter` returns them (an object array
    of 1-D arrays), that also holds them as one flat `values` array and int64 `offsets` (row `i`
    is `values[offsets[i]:offsets[i + 1]]`), which HuggingFaceDatasets.jl reads instead.

    Arrays derived from it (slices, copies, unpickled ones) have `values = offsets = None`.
    """

    def __array_finalize__(self, obj):
        self.values = None
        self.offsets = None


def _is_numeric_list(column):
    # A non-empty list column of numbers without nulls, at either level.
    return (
        (pa.types.is_list(column.type) or pa.types.is_large_list(column.type))
        and (pa.types.is_integer(column.type.value_type) or pa.types.is_floating(column.type.value_type))
        and len(column) > 0
        and column.null_count == 0
        and all(chunk.flatten().null_count == 0 for chunk in column.chunks)
    )


def _numeric_lists(column):
    # What `NumpyFormatter` returns, built from Arrow's flat buffers: a 2-D array if all rows
    # have the same length, a `RaggedArray` otherwise. Chunk by chunk, since combining them would
    # overflow the int32 offsets of a `list` column past 2^31 values. `values` is a new, writable
    # array, which Julia shares.
    values = np.concatenate([chunk.flatten().to_numpy() for chunk in column.chunks])
    lengths = np.concatenate([np.diff(chunk.offsets.to_numpy()) for chunk in column.chunks])
    if (lengths == lengths[0]).all():
        return values.reshape(len(lengths), lengths[0])
    rows = np.concatenate([chunk.to_numpy(zero_copy_only=False) for chunk in column.chunks])
    rows = rows.view(RaggedArray)
    rows.values = values
    rows.offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    return rows


class JuliaFormatter(NumpyFormatter):
    """`NumpyFormatter` that keeps each column's Arrow dtype, reads string columns as object
    arrays of `str`, and gives a batch of a list column of numbers its flat Arrow buffers.

    `NumpyFormatter` casts integer arrays to int64 and float arrays to float32 unless the format
    kwargs set a `dtype`; `dtype=None` keeps the dtype that numpy gets from Arrow.
    """

    def __init__(self, features=None, token_per_repo_id=None, **np_array_kwargs):
        np_array_kwargs.setdefault("dtype", None)
        super().__init__(features=features, token_per_repo_id=token_per_repo_id, **np_array_kwargs)

    def _extractors(self, pa_table, batch):
        # The columns read straight from Arrow, with their extractor: top-level
        # `Value("string")` columns, and in a batch, list columns of numbers. Neither needs
        # decoding.
        extractors = {}
        for field in pa_table.schema:
            feature = self.features.get(field.name) if self.features else None
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                if not self.features or isinstance(feature, Value):
                    extractors[field.name] = _strings if batch else _first_string
            elif (
                batch
                and self.np_array_kwargs.get("dtype") is None
                and (feature is None or not require_decoding(feature))
                and _is_numeric_list(pa_table.column(field.name))
            ):
                extractors[field.name] = _numeric_lists
        return extractors

    def _format_split(self, pa_table, format_rest, batch):
        extractors = self._extractors(pa_table, batch)
        if not extractors:
            return format_rest(pa_table)
        rest = pa_table.select([name for name in pa_table.column_names if name not in extractors])
        out = format_rest(rest) if rest.num_columns else {}
        return {
            name: extractors[name](pa_table.column(name)) if name in extractors else out[name]
            for name in pa_table.column_names
        }

    def format_row(self, pa_table):
        return self._format_split(pa_table, super().format_row, batch=False)

    def format_column(self, pa_table):
        extractors = self._extractors(pa_table, batch=True)
        if extractors:
            return extractors[pa_table.column_names[0]](pa_table.column(0))
        return super().format_column(pa_table)

    def format_batch(self, pa_table):
        return self._format_split(pa_table, super().format_batch, batch=True)


class JuliaFormatType(str):
    """The "julia" format type name, a `str` subclass so that it pickles as a reference to this
    module.

    `set_format`/`with_format` store the registered name, so a "julia"-formatted dataset pickles
    with that reference, and unpickling it imports this module, registering the formatter. That
    is what lets `datasets`' `num_proc` workers read their shard in the "julia" format when they
    start as fresh interpreters (the default on Windows), since `sys.path` is passed on to them.
    """

    __slots__ = ()


_register_formatter(JuliaFormatter, JuliaFormatType("julia"))


class JuliaFunction:
    """A Julia function given to `Dataset.map`/`filter`, with a fingerprint of its code and data,
    or the generator function given to `Dataset.from_generator`, with a fingerprint of its source.

    `datasets` fingerprints a transform, or a generator, by pickling it, and a Julia function
    would pickle by name. This wrapper pickles as its `fingerprint`, computed by
    HuggingFaceDatasets.jl, so the cache follows what the function does. It does not unpickle
    into a working function.
    """

    __slots__ = ("function", "fingerprint")

    def __init__(self, function, fingerprint):
        self.function = function
        self.fingerprint = fingerprint

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __reduce__(self):
        return (_unpickled_julia_function, (self.fingerprint,))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _unpickled_julia_function(fingerprint):
    def function(*args, **kwargs):
        raise RuntimeError(
            "A Julia function can only be called in the Julia session that passed it to "
            "`map`/`filter`/`from_generator`."
        )

    return function


END = object()  # what the `next_item` functions from Julia return past the last item


def examples(next_item):
    """The examples of a Julia iterable, for `Dataset.from_generator`: what `next_item()`
    returns until it returns `END`."""
    while (example := next_item()) is not END:
        yield example


def batch_rows(next_item):
    """The rows of the batches of a Julia iterable, for `Dataset.from_generator`: of each
    mapping of column names to columns that `next_item()` returns until it returns `END`."""
    while (batch := next_item()) is not END:
        lengths = {name: len(column) for name, column in batch.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"The columns of a batch have different lengths: {lengths}")
        for i in range(next(iter(lengths.values()), 0)):
            yield {name: column[i] for name, column in batch.items()}

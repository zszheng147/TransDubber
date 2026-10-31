import os
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

import pyarrow as pa
import pyarrow.dataset as ds


class BaseDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def collate_fn(self, batch):
        raise NotImplementedError

    # -------- Parquet reading utilities --------
    @staticmethod
    def _normalize_parquet_paths(paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]) -> List[str]:
        """Normalizes input into a list of absolute paths.

        Accepts a single file/dir path or a sequence of paths. Paths may point to
        Parquet files or dataset directories (containing multiple Parquet shards).
        """
        if isinstance(paths, (str, os.PathLike)):
            norm = [os.fspath(paths)]
        else:
            norm = [os.fspath(p) for p in paths]
        return [os.path.abspath(p) for p in norm]

    def load_parquet_columns(
        self,
        paths: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, List]:
        """Reads one or multiple Parquet files/directories and returns columns as Python lists.

        - paths: a path or list of paths to Parquet files or directories (dataset shards)
        - columns: optional subset of columns to read; defaults to known schema

        Known schema (if present in files):
          en_path: string
          en_text: string
          en_codes: list<int32>
          en_length: float32
          zh_path: string
          zh_text: string
          zh_codes: list<int32>
          zh_length: float32
        Missing columns will be filled with None values to preserve row alignment.
        Rows with en_length < 2.0s or > 20s will be filtered out when the column exists.
        """
        expected_columns = [
            "en_path", "en_text", "en_codes", "en_length",
            "zh_path", "zh_text", "zh_codes", "zh_length",
        ]
        read_columns = list(columns) if columns is not None else expected_columns

        norm_paths = self._normalize_parquet_paths(paths)

        dataset = ds.dataset(norm_paths, format="parquet")
        if dataset.count_rows() == 0:
            raise ValueError(f"No rows found in Parquet inputs: {norm_paths}")

        # Intersect requested columns with available columns to avoid errors
        available = {field.name for field in dataset.schema}
        selected = [c for c in read_columns if c in available]
        if not selected:
            raise ValueError(
                f"None of the requested columns are present. Requested={read_columns}, Available={sorted(available)}"
            )

        # Apply filter to drop samples with en/zh_length outside [2.0, 20] when
        # available.  The pushdown comparison is only valid on *scalar* numeric
        # columns: the multi-audio parquet stores en_length/zh_length as
        # list<double> (one entry per concatenated segment), which can't be
        # compared to a scalar.  For those, skip the pushdown and let the
        # dataset filter on the combined duration in Python.
        def _is_scalar_numeric(col: str) -> bool:
            if col not in available:
                return False
            t = dataset.schema.field(col).type
            return pa.types.is_floating(t) or pa.types.is_integer(t)

        filter_expr = None
        if _is_scalar_numeric("en_length"):
            filter_expr = (ds.field("en_length") >= 2.0) & (ds.field("en_length") <= 20.0)

        if _is_scalar_numeric("zh_length"):
            zh_expr = (ds.field("zh_length") >= 2.0) & (ds.field("zh_length") <= 20.0)
            filter_expr = zh_expr if filter_expr is None else (filter_expr & zh_expr)

        # If a max text length is configured, we may need to read en_text/zh_text for filtering
        max_text_length: Optional[int] = getattr(self.config, "max_text_length", None)

        # Build the final column list for scanning: include text columns if needed for filtering
        selected_for_scan: List[str] = list(selected)
        if max_text_length is not None:
            for text_col in ("en_text", "zh_text"):
                if text_col in available and text_col not in selected_for_scan:
                    selected_for_scan.append(text_col)

        scanner = ds.Scanner.from_dataset(dataset, columns=selected_for_scan, filter=filter_expr)

        # Prepare buffers
        buffers: Dict[str, List] = {c: [] for c in selected_for_scan}

        for batch in scanner.to_batches():
            # Convert all needed columns in this batch to Python lists
            batch_lists: Dict[str, List] = {}
            for name in selected_for_scan:
                arr: pa.Array = batch.column(batch.schema.get_field_index(name))
                batch_lists[name] = arr.to_pylist()

            # Build row-level keep mask for text length if configured
            if max_text_length is not None:
                num_rows = batch.num_rows
                keep_mask = [True] * num_rows
                # Apply per-text-column checks if present in this batch
                for text_col in ("en_text", "zh_text"):
                    if text_col in batch_lists:
                        col_values = batch_lists[text_col]
                        for i in range(num_rows):
                            v = col_values[i]
                            # Only character-count scalar string columns; the
                            # multi-audio schema stores en_text/zh_text as
                            # list<string> (len() would count segments, not
                            # characters), so leave those rows untouched here.
                            if isinstance(v, str) and len(v) > max_text_length:
                                keep_mask[i] = False
                # Extend buffers using the mask
                kept_indices = [i for i, k in enumerate(keep_mask) if k]
                for name in selected_for_scan:
                    buffers[name].extend(batch_lists[name][i] for i in kept_indices)
            else:
                # No text-length filtering; append all
                for name in selected_for_scan:
                    buffers[name].extend(batch_lists[name])

        # Fill any expected-but-missing columns with None to keep alignment
        total_rows = len(next(iter(buffers.values()))) if buffers else dataset.count_rows()
        # Build return dict restricted to originally selected columns
        result: Dict[str, List] = {c: buffers[c] for c in selected}
        for col in read_columns:
            if col not in result:
                result[col] = [None] * total_rows

        return result
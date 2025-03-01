"""
Utilities for caching satellite data as numpy arrays.
Provides functionality to convert h5 files to numpy arrays and save them for faster access.
"""

import numpy as np
import datetime as dt
from pathlib import Path

from .file_utils import find_by_date, h5_to_np


def h5_to_np_dir(
    h5_dir_path: str,
    target_var: str,
    h5_start_date: dt.datetime,
    h5_end_date: dt.datetime,
    h5_filename_fmt: str,
    h5_timestep: int,
    np_dir_path: str,
    np_filename_fmt: str,
) -> None:
    """
    Cache h5 satellite data as numpy arrays in a directory.

    Reads h5 files within a specified datetime range, converts them to numpy arrays,
    and saves them for faster future access.

    Args:
        h5_dir_path: Directory containing source h5 files
        target_var: Variable name to extract from h5 files
        h5_start_date: Start datetime for file search (inclusive)
        h5_end_date: End datetime for file search (inclusive)
        h5_filename_fmt: Datetime format pattern for h5 filenames
        h5_timestep: Time difference in minutes between h5 files
        np_dir_path: Directory to save cached numpy arrays
        np_filename_fmt: Datetime format pattern for output npy files

    Raises:
        IOError: If either directory path is invalid

    Note:
        This function will skip files that have already been cached.
    """
    # Convert string paths to Path objects for more robust handling
    h5_dir = Path(h5_dir_path)
    np_dir = Path(np_dir_path)

    # Validate directories
    if not np_dir.is_dir():
        raise IOError(
            f"Output directory '{np_dir}' does not exist or is not a directory"
        )
    if not h5_dir.is_dir():
        raise IOError(
            f"Source directory '{h5_dir}' does not exist or is not a directory"
        )

    # Find all relevant h5 files in the time range
    h5_filenames, h5_timestamps = find_by_date(
        h5_start_date, h5_end_date, str(h5_dir), h5_filename_fmt, "h5", h5_timestep
    )

    # Process each file and convert to numpy array
    num_files = len(h5_filenames)
    num_processed = 0
    num_skipped = 0

    for fn, ts in zip(h5_filenames, h5_timestamps):
        np_filename = np_dir / dt.datetime.strftime(ts, np_filename_fmt)

        # Skip if file is missing or already processed
        if fn is None or np_filename.exists():
            num_skipped += 1
            continue

        # Print progress with carriage return to update in-place
        print(f"\rProcessing file {num_processed+1}/{num_files} ({ts})...", end="")

        # Convert h5 to numpy array and save
        np_arr = h5_to_np(fn, target_var)
        np.save(np_filename, np_arr, fix_imports=False)
        num_processed += 1

    # Final status message
    print(
        f"\rCompleted: {num_processed} files converted, {num_skipped} files skipped in {np_dir}"
    )

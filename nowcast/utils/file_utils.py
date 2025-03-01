"""
Utilities for handling satellite data files from MOSDAC.
Provides functionality to convert h5 files to numpy arrays and
find satellite files within a specific date range.
"""

import os
import h5py
import numpy as np
import datetime as dt
import numpy.typing as npt

from ..config import MOSDACConfig


def h5_to_np(file_path: str | None, target_var: str) -> npt.NDArray[np.float64]:
    """
    Convert MOSDAC h5 file to numpy array for a specific region of interest.

    Extracts data for the Indian subcontinent region based on configuration settings.

    Args:
        file_path: Path to the h5 file, or None to return an array of zeros
        target_var: Name of the variable to extract from the h5 file

    Returns:
        2D numpy array of the extracted data

    Raises:
        IOError: If file_path is not None but file doesn't exist
    """
    if file_path is None:
        return np.zeros(MOSDACConfig.FRAME_SIZE)

    if not os.path.exists(file_path):
        raise IOError(f"File {file_path} not found")

    with h5py.File(file_path, "r") as cur_file:
        # Extract and scale latitude/longitude data
        lat = np.asarray(cur_file["Latitude"]) * 0.01
        lon = np.asarray(cur_file["Longitude"]) * 0.01

        # Extract target variable
        var = np.squeeze(np.asarray(cur_file[target_var]))

        # Create region of interest mask
        lat_filter = (lat >= MOSDACConfig.LAT_MIN) & (lat < MOSDACConfig.LAT_MAX)
        lon_filter = (lon >= MOSDACConfig.LON_MIN) & (lon < MOSDACConfig.LON_MAX)
        mask = lat_filter & lon_filter

        # Apply the mask to all arrays
        lat_filtered = lat[mask]
        lon_filtered = lon[mask]
        var_filtered = var[mask]

    # Initialize output grid and counter arrays
    var_grid = np.zeros(MOSDACConfig.FRAME_SIZE)
    count_grid = np.zeros(MOSDACConfig.FRAME_SIZE)

    # Calculate indices based on configured resolution
    resolution_factor = int(MOSDACConfig.LAT_LON_RESOLUTION**-1)
    lat_indices = np.int32((lat_filtered - MOSDACConfig.LAT_MIN) * resolution_factor)
    lon_indices = np.int32((lon_filtered - MOSDACConfig.LON_MIN) * resolution_factor)

    # Accumulate values and counts
    np.add.at(var_grid, (lat_indices, lon_indices), var_filtered)
    np.add.at(count_grid, (lat_indices, lon_indices), 1)

    # Calculate averages and handle empty cells
    var_grid = np.divide(
        var_grid, count_grid, out=np.full_like(var_grid, np.nan), where=count_grid > 0
    )

    assert (
        var_grid.shape == MOSDACConfig.FRAME_SIZE
    ), f"Expected shape {MOSDACConfig.FRAME_SIZE}, got {var_grid.shape}"
    return var_grid


def find_by_date(
    start_datetime: dt.datetime,
    end_datetime: dt.datetime,
    root_path: str,
    fn_pattern: str,
    fn_ext: str,
    timestep: int,
    error: int = 2,
    upper: bool = True,
    lower: bool = False,
    verbose: bool = False,
) -> tuple[list[str | None], list[dt.datetime]]:
    """
    Find files within a date range that match a specified pattern.

    Args:
        start_datetime: Starting datetime (inclusive)
        end_datetime: Ending datetime (inclusive)
        root_path: Directory to search for files
        fn_pattern: Filename pattern with datetime format codes
        fn_ext: File extension (without the dot)
        timestep: Time difference between consecutive files in minutes
        error: Allowed time error in minutes for finding files
        upper: Convert filename to uppercase
        lower: Convert filename to lowercase
        verbose: Print messages when files are not found

    Returns:
        Tuple containing a list of file paths (or None if not found) and their corresponding datetimes

    Raises:
        ValueError: For invalid parameters
        IOError: If no files are found or the root path doesn't exist
    """
    # Validate input parameters
    if upper and lower:
        raise ValueError("Cannot have both 'upper' and 'lower' set to True")

    if timestep <= 0 or isinstance(timestep, float):
        raise ValueError("'timestep' must be a positive integer")

    if not os.path.exists(root_path):
        raise IOError(f"Root path '{root_path}' does not exist")

    if start_datetime >= end_datetime:
        raise ValueError("'start_datetime' must be earlier than 'end_datetime'")

    # Generate sequence of datetimes
    timestamps = []
    curr_datetime = start_datetime
    while curr_datetime <= end_datetime:
        timestamps.append(curr_datetime)
        curr_datetime += dt.timedelta(minutes=timestep)

    # Find matching files for each timestamp
    filenames = []
    for timestamp in timestamps:
        file_path = _find_matching_file(
            timestamp, root_path, fn_pattern, fn_ext, error, upper, lower, verbose
        )
        filenames.append(file_path)

    # Check if any files were found
    if all(filename is None for filename in filenames):
        raise IOError(f"No files found in '{root_path}' for the given date range")

    return filenames, timestamps


def _find_matching_file(
    timestamp: dt.datetime,
    root_path: str,
    fn_pattern: str,
    fn_ext: str,
    error: int,
    upper: bool,
    lower: bool,
    verbose: bool,
) -> str | None:
    """
    Find a file matching the timestamp, allowing for errors in the timestamp.

    Args:
        timestamp: Target datetime to find a file for
        root_path: Directory to search in
        fn_pattern: Filename pattern with datetime format codes
        fn_ext: File extension (without the dot)
        error: Allowed time error in minutes
        upper: Convert filename to uppercase
        lower: Convert filename to lowercase
        verbose: Print message if file not found

    Returns:
        Path to the found file or None if not found
    """

    # Try exact match first
    file_path = _format_file_path(
        timestamp, root_path, fn_pattern, fn_ext, upper, lower
    )
    if os.path.exists(file_path):
        return file_path

    # Try with negative error margin
    for minutes in range(1, error + 1):
        adjusted_time = timestamp - dt.timedelta(minutes=minutes)
        file_path = _format_file_path(
            adjusted_time, root_path, fn_pattern, fn_ext, upper, lower
        )
        if os.path.exists(file_path):
            return file_path

    # Try with positive error margin
    for minutes in range(1, error + 1):
        adjusted_time = timestamp + dt.timedelta(minutes=minutes)
        file_path = _format_file_path(
            adjusted_time, root_path, fn_pattern, fn_ext, upper, lower
        )
        if os.path.exists(file_path):
            return file_path

    # No match found
    if verbose:
        formatted_time = dt.datetime.strftime(timestamp, fn_pattern)
        if upper:
            formatted_time = formatted_time.upper()
        if lower:
            formatted_time = formatted_time.lower()
        print(f"File not found for datetime: {formatted_time}")

    return None


def _format_file_path(
    timestamp: dt.datetime,
    root_path: str,
    fn_pattern: str,
    fn_ext: str,
    upper: bool,
    lower: bool,
) -> str:
    """
    Format a file path based on the timestamp and pattern.

    Args:
        timestamp: Datetime to format the filename with
        root_path: Directory path
        fn_pattern: Filename pattern with datetime format codes
        fn_ext: File extension (without the dot)
        upper: Convert filename to uppercase
        lower: Convert filename to lowercase

    Returns:
        Formatted file path
    """
    filename = dt.datetime.strftime(timestamp, fn_pattern)

    if upper:
        filename = filename.upper()
    if lower:
        filename = filename.lower()

    filename = f"{filename}.{fn_ext}"
    return os.path.join(root_path, filename)

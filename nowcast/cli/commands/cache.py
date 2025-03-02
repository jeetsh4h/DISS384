"""
Command for caching H5 satellite data as NumPy arrays.
"""

import os
import datetime as dt

from ...config import DataConfig
from ...utils.cache_utils import h5_to_np_dir


def parse_date(date_str: str, start_date: bool) -> dt.datetime:
    """Parse date string in YYYY-MM-DD or YYYY-MM-DD_HH:MM format."""
    date_parsing_fmt = "%Y-%m-%d_%H:%M"
    try:
        if "_" in date_str:
            return dt.datetime.strptime(date_str, date_parsing_fmt)
        else:
            return (
                dt.datetime.strptime(date_str + "_00:00", date_parsing_fmt)
                if start_date
                else dt.datetime.strptime(date_str + "_23:59", date_parsing_fmt)
            )
    except ValueError:
        raise ValueError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD_HH:MM"
        )


def parse_date_range(date_range: str) -> tuple[dt.datetime, dt.datetime]:
    """Parse date range string in the format 'start_date:end_date'."""
    try:
        start_date_str, end_date_str = date_range.split(":")
        start_date = parse_date(start_date_str, True)
        end_date = parse_date(end_date_str, False)

        if start_date >= end_date:
            raise ValueError("Start date must be earlier than end date")

        if start_date.year != end_date.year:
            raise ValueError("Start and end date must be in the same year")

        return start_date, end_date
    except ValueError as e:
        raise ValueError(f"Invalid date range format: {date_range}. {str(e)}")


def setup_parser(subparsers):
    """Set up the argument parser for the cache command."""
    cache_parser = subparsers.add_parser(
        "cache", help="Cache H5 satellite data as NumPy arrays"
    )

    cache_parser.add_argument(
        "--var_name",
        "-v",
        nargs="+",
        type=str,
        choices=DataConfig.VAR_TYPES,
        help="Type of h5 file to process (OLR, HEM)",
    )

    cache_parser.add_argument(
        "--date-ranges",
        "-d",
        nargs="+",
        required=True,
        help="Date ranges in format 'YYYY-MM-DD:YYYY-MM-DD' (assumed to be 00:00 for start and 23:59 for end) or 'YYYY-MM-DD_HH:MM:YYYY-MM-DD_HH:MM'",
    )

    cache_parser.add_argument(
        "--timestep",
        "-t",
        type=int,
        default=30,
        help="Time difference in minutes between files (default: 30)",
    )

    return cache_parser


def execute(args):
    """Execute the cache command with the parsed arguments."""
    # Parse all date ranges
    date_ranges: list[tuple[dt.datetime, dt.datetime]] = []
    for date_range_str in args.date_ranges:
        try:
            date_range = parse_date_range(date_range_str)
            date_ranges.append(date_range)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return 1

    # Initialize cache manager
    for date_range in date_ranges:
        start_date, end_date = date_range
        if start_date.year != end_date.year:
            # TODO: better error message
            print("Error: All individual date ranges must be in the same year")

        try:
            for var_name in args.var_name:
                input_dir = DataConfig.DATA_DIR / var_name / f"May-Sep{start_date.year}"
                output_dir = DataConfig.CACHE_DIR / var_name

                os.makedirs(output_dir, exist_ok=True)
                print(f"Processing and caching {var_name}.")

                h5_to_np_dir(
                    h5_dir_path=input_dir,
                    target_var=var_name,
                    h5_start_date=start_date,
                    h5_end_date=end_date,
                    h5_filename_fmt=DataConfig.H5_FILENAME_FMT.replace(
                        "%s", var_name, 1
                    ),
                    h5_timestep=args.timestep,
                    np_dir_path=str(output_dir),
                    np_filename_fmt=DataConfig.NP_FILENAME_FMT.replace(
                        "%s", var_name, 1
                    ),
                )
        except Exception as e:
            print(
                f"Error processing and caching data for range {start_date} to {end_date}: {str(e)}"
            )
            return 1

        return 0

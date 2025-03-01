"""
Main entry point for nowcast CLI.
"""

import sys
import argparse

# Import command modules
from .commands import cache


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="FLAME Satellite Nowcasting Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # TODO: Add version info automatically
    parser.add_argument("--version", "-v", action="version", version="nowcast 0.1.0")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute", required=False
    )

    # Register commands
    cache_parser = cache.setup_parser(subparsers)

    # TODO: Add more commands here as they're implemented
    # train.setup_parser(subparsers)
    # predict.setup_parser(subparsers)
    # test.setup_parser(subparsers)
    # visualize.setup_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return 1

    # Execute the appropriate command
    if args.command == "cache":
        return cache.execute(args)

    # Command not recognized
    print(f"Error: Unknown command '{args.command}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())

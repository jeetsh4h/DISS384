import sys
import argparse

from .commands import cache, train, visualize, test, metrics


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
    train_parser = train.setup_parser(subparsers)
    visualize_parser = visualize.setup_parser(subparsers)
    test_parser = test.setup_parser(subparsers)
    metric_parser = metrics.setup_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, print help
    if not args.command:
        parser.print_help()
        return 1

    # Execute the appropriate command
    match args.command:
        case "cache":
            return cache.execute(args)
        case "train":
            return train.execute(args)
        case "visualize":
            return visualize.execute(args)
        case "test":
            return test.execute(args)
        case "metrics":
            return metrics.execute(args)
        case _:
            print(f"Error: Unknown command '{args.command}'")
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())

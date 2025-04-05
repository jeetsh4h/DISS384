import sys
import argparse
import tensorflow as tf

from .commands import cache, train, visualize, test, metrics, metric_viz, flow


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="FLAME Satellite Nowcasting Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # TODO: Add version info automatically
    parser.add_argument("--version", "-v", action="version", version="nowcast 0.1.0")

    parser.add_argument(
        "--limit-gpu", "-lg", action="store_true", help="Limit GPU usage"
    )
    parser.add_argument(
        "--no-gpu", "-ng", action="store_true", help="Disable GPU usage"
    )

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
    metric_viz_parser = metric_viz.setup_parser(subparsers)
    flow_parser = flow.setup_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if args.no_gpu and args.limit_gpu:
        print("Error: Cannot use both --no-gpu and --limit-gpu at the same time.")
        return 1

    # Disable GPU if requested
    if args.no_gpu:
        tf.config.set_visible_devices([], "GPU")
        print("GPU usage disabled.")

    # Check if GPU limiting is enabled
    if args.limit_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=6130
                        )
                    ],
                )
            except RuntimeError as e:
                print(f"Error limiting GPU memory: {e}")
                return 1

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
        case "metric_viz":
            return metric_viz.execute(args)
        case "flow":
            return flow.execute(args)
        case _:
            print(f"Error: Unknown command '{args.command}'")
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())

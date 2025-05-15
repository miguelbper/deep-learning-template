from tabulate import tabulate

Metrics = dict[str, float]


def print_metrics(metrics: Metrics, prefix: str) -> None:
    """Pretty print metrics in a table format.

    Args:
        metrics: Dictionary of metric names and values
        prefix: Prefix to use in the title (e.g., 'Validation' or 'Test')
    """
    table = [[name, f"{value:.4f}"] for name, value in metrics.items()]
    print(f"\n{prefix} Metrics:")
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print()

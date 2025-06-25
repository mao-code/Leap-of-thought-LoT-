from dataset import TrainingDataset


def print_portions(ds: TrainingDataset) -> None:
    dataset_counts = {}
    type_counts = {}
    total = len(ds)
    for ex in ds:
        dataset_counts[ex["dataset"]] = dataset_counts.get(ex["dataset"], 0) + 1
        type_counts[ex["type"]] = type_counts.get(ex["type"], 0) + 1

    print("Dataset portions:")
    for name, cnt in sorted(dataset_counts.items()):
        print(f"  {name}: {cnt} ({cnt / total:.1%})")

    print("\nType portions:")
    for name, cnt in sorted(type_counts.items()):
        print(f"  {name}: {cnt} ({cnt / total:.1%})")

if __name__ == "__main__":
    ds = TrainingDataset(total_samples=1000)
    for i, ex in enumerate(ds.get_examples(5)):
        print(f"# {i+1}")
        print("Type:", ex["type"])
        print("Dataset:", ex["dataset"])
        print("Q:", ex["question"])
        print("A:", ex["answers"])
        print()

    print_portions(ds)

    """
    Example usage:
    python test_training_dataset.py
    """

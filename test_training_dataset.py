from dataset import TrainingDataset

if __name__ == "__main__":
    ds = TrainingDataset(total_samples=1000)
    for i, ex in enumerate(ds.get_examples(5)):
        print(f"# {i+1}")
        print("Q:", ex["question"])
        print("A:", ex["answers"])
        print()

    """
    Example usage:
    python test_training_dataset.py
    """

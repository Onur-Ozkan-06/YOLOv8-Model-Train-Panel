import os
import yaml
import numpy as np
from collections import Counter

def main(dataset_dir):
    
    print("Class Weight calculation starting:", dataset_dir)

    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_labels_dir   = os.path.join(dataset_dir, "valid", "labels")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    all_labels = []
    #for labels_dir in [train_labels_dir, val_labels_dir]:
    for labels_dir in [train_labels_dir]:
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} not found, skipping !!!")
            continue

        for file in os.listdir(labels_dir):
            if file.endswith(".txt"):
                with open(os.path.join(labels_dir, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 0:
                            continue  
                        cls = int(parts[0])
                        all_labels.append(cls)

    # Class distribution
    class_counts = Counter(all_labels)
    total = sum(class_counts.values())

    # Calculate Class weights
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = (total / (len(class_counts) * count)) ** 0.5

    # Normalize weights
    sum_weights = sum(weights.values())
    weights = {cls: w/sum_weights for cls, w in weights.items()}

    # Get class names
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data_yaml = yaml.safe_load(f)
        names = data_yaml.get("names", {})
        print("\n Class names:\n")
        for k, name in names.items():
            print(f"{k}: {name}")

    # Print Class distribution
    print("\n Class distribution:\n")
    for k, v in sorted(class_counts.items()):
        print(f"Class {k}: {v} samples ({v/total*100:.2f}%)")


    print("\n Normalized class weights:\n")
    for cls, w in sorted(weights.items()):
        print(f"Class {cls}: {w:.4f}")


    # Save weights
    weights_path = os.path.join(dataset_dir, "Class_Weights.txt")
    with open(weights_path, "w") as f:
        yaml.dump(weights, f)

    print(f"\n Class weights saved -> {weights_path}\n\n")
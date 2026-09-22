"""
Week 3 Experiment Simulation
AI-Driven Media Content Curator Internship

Purpose:
    Simulate dataset loading, model predictions, evaluation metrics,
    latency/throughput benchmarking, and confusion matrices for
    Text, Image, and Video content categorization.

IMPORTANT:
    This script intentionally uses SIMULATED predictions. It does not
    train real deep-learning models and must not be reported as actual
    empirical model performance.

Requirements:
    Python 3.9+
    numpy
    pandas
    scikit-learn
    matplotlib

Install:
    pip install numpy pandas scikit-learn matplotlib

Run:
    python experiment_simulation.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

CLASSES = ["News", "Sports", "Entertainment", "Technology"]

OUTPUT_DIR = Path("experiment_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. SIMULATED DATASET LOADING
# ============================================================

def create_simulated_dataset(modality: str, n_samples: int) -> pd.DataFrame:
    """
    Create a small metadata representation of a media dataset.

    In a real project, this function would be replaced by:
      - pandas.read_csv(...)
      - a PyTorch Dataset/DataLoader
      - image/video file loading
      - tokenizer-based text loading
    """

    if modality == "text":
        sources = ["news_portal", "sports_site", "tech_blog", "entertainment_site"]
        data = {
            "id": [f"T{i:05d}" for i in range(n_samples)],
            "modality": ["text"] * n_samples,
            "source": rng.choice(sources, n_samples),
            "label": rng.choice(
                CLASSES,
                n_samples,
                p=[0.30, 0.25, 0.25, 0.20],
            ),
        }

    elif modality == "image":
        data = {
            "id": [f"I{i:05d}" for i in range(n_samples)],
            "modality": ["image"] * n_samples,
            "width": rng.choice([224, 256, 512], n_samples),
            "height": rng.choice([224, 256, 512], n_samples),
            "label": rng.choice(
                CLASSES,
                n_samples,
                p=[0.30, 0.25, 0.25, 0.20],
            ),
        }

    elif modality == "video":
        data = {
            "id": [f"V{i:05d}" for i in range(n_samples)],
            "modality": ["video"] * n_samples,
            "duration_sec": np.round(
                rng.uniform(5, 120, n_samples), 2
            ),
            "fps": rng.choice([24, 25, 30, 60], n_samples),
            "label": rng.choice(
                CLASSES,
                n_samples,
                p=[0.25, 0.25, 0.25, 0.25],
            ),
        }

    else:
        raise ValueError("modality must be text, image, or video")

    return pd.DataFrame(data)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(df: pd.DataFrame):
    """
    Stratified 70/15/15 split.
    """

    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=RANDOM_SEED,
    )

    return train, validation, test


# ============================================================
# 4. SIMULATED MODEL PREDICTIONS
# ============================================================

def simulate_predictions(
    y_true,
    target_accuracy: float,
    confidence_noise: float = 0.08,
):
    """
    Generate hypothetical predictions with an approximate target
    accuracy.

    This is NOT model inference. It is a controlled simulation used
    to demonstrate the evaluation pipeline.
    """

    y_true = np.asarray(y_true)
    n = len(y_true)

    y_pred = y_true.copy()

    # Number of intentionally incorrect predictions.
    n_errors = int(round(n * (1 - target_accuracy)))

    error_indices = rng.choice(
        n,
        size=n_errors,
        replace=False,
    )

    for idx in error_indices:
        incorrect_classes = [
            c for c in range(len(CLASSES))
            if c != y_true[idx]
        ]
        y_pred[idx] = rng.choice(incorrect_classes)

    # Generate class probabilities.
    logits = rng.normal(
        loc=0.0,
        scale=confidence_noise,
        size=(n, len(CLASSES)),
    )

    # Give the predicted class the highest simulated score.
    for i in range(n):
        logits[i, y_pred[i]] += 3.0

    # Softmax.
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    return y_pred, probabilities


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def top_k_accuracy(y_true, probabilities, k=1):
    """
    Calculate Top-k accuracy.
    """

    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    correct = [
        true_label in predictions
        for true_label, predictions in zip(y_true, top_k)
    ]

    return np.mean(correct)


def calculate_metrics(y_true, y_pred, probabilities):
    """
    Calculate classification metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Top-1 Accuracy": top_k_accuracy(
            y_true,
            probabilities,
            k=1,
        ),
        "Top-5 Accuracy": top_k_accuracy(
            y_true,
            probabilities,
            k=min(5, len(CLASSES)),
        ),
    }

    # One-vs-rest mAP.
    y_true_binary = label_binarize(
        y_true,
        classes=np.arange(len(CLASSES)),
    )

    metrics["mAP"] = average_precision_score(
        y_true_binary,
        probabilities,
        average="macro",
    )

    return metrics


# ============================================================
# 6. SIMULATED LATENCY / THROUGHPUT
# ============================================================

def simulate_latency(latency_ms: float, n_items: int = 100):
    """
    Simulate inference timing around a target latency.

    The sleep operation is intentionally short and is only used to
    demonstrate how latency and throughput can be measured.
    """

    samples = rng.normal(
        latency_ms,
        latency_ms * 0.05,
        size=n_items,
    )
    samples = np.maximum(samples, latency_ms * 0.50)

    measured_latency = float(np.mean(samples))
    throughput = 1000.0 / measured_latency

    return measured_latency, throughput


# ============================================================
# 7. MODEL CONFIGURATION
# ============================================================

MODELS = {
    "Text": {
        "TF-IDF + Logistic Regression": {
            "accuracy": 0.846,
            "latency_ms": 0.30,
        },
        "RoBERTa-Base": {
            "accuracy": 0.917,
            "latency_ms": 6.00,
        },
        "RoBERTa-Large": {
            "accuracy": 0.928,
            "latency_ms": 13.50,
        },
    },

    "Image": {
        "ResNet-50": {
            "accuracy": 0.894,
            "latency_ms": 3.50,
        },
        "ViT-B/16": {
            "accuracy": 0.921,
            "latency_ms": 6.80,
        },
    },

    "Video": {
        "I3D": {
            "accuracy": 0.827,
            "latency_ms": 55.00,
        },
        "Video Swin-B": {
            "accuracy": 0.879,
            "latency_ms": 75.00,
        },
    },
}


# ============================================================
# 8. RUN EXPERIMENT
# ============================================================

def run_experiment():
    dataset_sizes = {
        "Text": 1500,
        "Image": 750,
        "Video": 300,
    }

    modality_mapping = {
        "Text": "text",
        "Image": "image",
        "Video": "video",
    }

    all_results = []

    for modality, models in MODELS.items():

        print("\n" + "=" * 70)
        print(f"{modality.upper()} EXPERIMENT")
        print("=" * 70)

        # ----------------------------------------------------
        # Simulated dataset loading
        # ----------------------------------------------------
        df = create_simulated_dataset(
            modality_mapping[modality],
            dataset_sizes[modality],
        )

        train, validation, test = split_dataset(df)

        print(f"Dataset size     : {len(df)}")
        print(f"Training samples : {len(train)}")
        print(f"Validation       : {len(validation)}")
        print(f"Test samples     : {len(test)}")

        y_true = pd.Categorical(
            test["label"],
            categories=CLASSES,
        ).codes

        # ----------------------------------------------------
        # Evaluate each candidate model
        # ----------------------------------------------------
        for model_name, config in models.items():

            y_pred, probabilities = simulate_predictions(
                y_true,
                target_accuracy=config["accuracy"],
            )

            metrics = calculate_metrics(
                y_true,
                y_pred,
                probabilities,
            )

            latency, throughput = simulate_latency(
                config["latency_ms"]
            )

            result = {
                "Modality": modality,
                "Model": model_name,
                **metrics,
                "Latency (ms/item)": latency,
                "Throughput (items/sec)": throughput,
            }

            all_results.append(result)

            # Confusion matrix
            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=np.arange(len(CLASSES)),
            )

            print(f"\nModel: {model_name}")
            print("-" * 50)

            for metric, value in metrics.items():
                print(f"{metric:20s}: {value * 100:6.2f}%")

            print(
                f"{'Latency (ms/item)':20s}: "
                f"{latency:6.2f}"
            )

            print(
                f"{'Throughput (items/s)':20s}: "
                f"{throughput:6.2f}"
            )

            print("\nConfusion Matrix:")
            print(
                pd.DataFrame(
                    cm,
                    index=CLASSES,
                    columns=CLASSES,
                )
            )

            # Save confusion matrix.
            safe_name = (
                f"{modality}_{model_name}"
                .replace(" ", "_")
                .replace("/", "-")
                .replace("+", "plus")
            )

            pd.DataFrame(
                cm,
                index=CLASSES,
                columns=CLASSES,
            ).to_csv(
                OUTPUT_DIR / f"{safe_name}_confusion_matrix.csv"
            )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_results)

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Top-1 Accuracy",
        "Top-5 Accuracy",
        "mAP",
    ]

    display_df = results_df.copy()

    for col in percentage_columns:
        display_df[col] = (
            display_df[col] * 100
        ).round(2)

    display_df["Latency (ms/item)"] = (
        display_df["Latency (ms/item)"].round(2)
    )

    display_df["Throughput (items/sec)"] = (
        display_df["Throughput (items/sec)"].round(2)
    )

    print("\n" + "=" * 90)
    print("FINAL SIMULATED BENCHMARK")
    print("=" * 90)
    print(display_df.to_string(index=False))

    # Save results.
    results_df.to_csv(
        OUTPUT_DIR / "simulated_benchmark_results.csv",
        index=False,
    )

    display_df.to_csv(
        OUTPUT_DIR / "simulated_benchmark_results_percent.csv",
        index=False,
    )

    return results_df


# ============================================================
# 9. PLOT F1 COMPARISON
# ============================================================

def plot_f1_comparison(results_df):
    """
    Generate a simple F1 comparison chart.
    """

    plt.figure(figsize=(11, 6))

    labels = (
        results_df["Modality"]
        + " - "
        + results_df["Model"]
    )

    plt.bar(
        labels,
        results_df["F1"] * 100,
    )

    plt.ylabel("Macro F1 (%)")
    plt.xlabel("Model")
    plt.title(
        "SIMULATED Macro-F1 Comparison Across Media Modalities"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR / "simulated_f1_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print(f"\nSaved chart: {output_file}")


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WEEK 3 AI CONTENT CATEGORIZATION EXPERIMENT SIMULATION")
    print("=" * 70)
    print("WARNING: Results are SIMULATED/HYPOTHETICAL.")
    print("No real deep-learning model is trained by this script.")
    print("=" * 70)

    results = run_experiment()

    plot_f1_comparison(results)

    print("\nExperiment completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

"""Static analytics plots for a completed MARS evaluation run."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


ERROR_LABELS = {
    "wrong_entity_correct_type": "wrong entity / correct type",
    "wrong_entity_type": "wrong entity type",
    "unknown_type": "unknown type",
    "unmatched_entity_in_set": "unmatched in set",
    "ambiguous_alias_correct_type": "ambiguous alias",
    "skipped_entity": "skipped entity",
}
RECOVERABILITY_LABELS = ("TRUE", "FALSE", "GUESSABLE")


def _plotting_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib and numpy. Install them in "
            "the environment used to run FLAN-T5."
        ) from error
    return plt, np


def validate_plot_dependencies() -> None:
    """Fail before model inference when static plotting dependencies are absent."""
    _plotting_modules()


def _save(fig: Any, destination: Path, plt: Any) -> Path:
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _target_records(
    results: list[dict[str, Any]], errors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from evaluation import evaluation_metrics as evaluator

    errors_by_position = {(error["row"], error["index"]): error for error in errors}
    records: list[dict[str, Any]] = []
    for result in results:
        flags = result.get("entity_recoverable")
        for entity in evaluator.collect_entity_information(result)["entities"]:
            index = entity["mask_index"]
            error = errors_by_position.get((result["row"], index))
            records.append(
                {
                    "target_type": entity["target_type"],
                    "prediction_type": (
                        error["prediction_type"]
                        if error
                        else entity["matched_prediction_type"]
                    ),
                    "recovered": bool(entity["recovered"]),
                    "error_type": error["error_type"] if error else None,
                    "entity_recoverable": (
                        str(flags[index]).upper() if flags is not None else None
                    ),
                }
            )
    return records


def _entity_type_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    recovered: Counter[str] = Counter()
    for record in records:
        entity_type = record["target_type"]
        totals[entity_type] += 1
        recovered[entity_type] += record["recovered"]
    return [
        {
            "type": entity_type,
            "targets": total,
            "recovered": recovered[entity_type],
            "rate": 100 * recovered[entity_type] / total,
        }
        for entity_type, total in totals.items()
    ]


def _recovery_rate_plot(summary: list[dict[str, Any]], destination: Path) -> Path:
    plt, _ = _plotting_modules()
    values = sorted(summary, key=lambda row: row["rate"])
    height = max(4.5, 0.42 * len(values))
    fig, ax = plt.subplots(figsize=(9, height))
    bars = ax.barh([row["type"] for row in values], [row["rate"] for row in values])
    ax.bar_label(bars, labels=[f"{row['rate']:.1f}%" for row in values], padding=3)
    ax.set(
        xlim=(0, 100),
        xlabel="Recovery rate (%)",
        title="Recovery rate by target entity type",
    )
    ax.grid(False)
    return _save(fig, destination, plt)


def _entity_count_plot(summary: list[dict[str, Any]], destination: Path) -> Path:
    plt, _ = _plotting_modules()
    values = sorted(summary, key=lambda row: row["targets"])
    height = max(4.8, 0.42 * len(values))
    fig, ax = plt.subplots(figsize=(10, height))
    labels = [row["type"] for row in values]
    total = ax.barh(
        labels,
        [row["targets"] for row in values],
        color="tab:blue",
        alpha=0.25,
        label="Total targets",
    )
    recovered = ax.barh(
        labels,
        [row["recovered"] for row in values],
        color="tab:blue",
        label="Recovered targets",
    )
    ax.bar_label(total, labels=[str(row["targets"]) for row in values], padding=3)
    ax.bar_label(
        recovered, labels=[str(row["recovered"]) for row in values], padding=3
    )
    ax.set(xlabel="Count", title="Recovered targets within all targets, by entity type")
    ax.legend()
    ax.grid(False)
    return _save(fig, destination, plt)


def _error_category_plot(records: list[dict[str, Any]], destination: Path) -> Path:
    plt, _ = _plotting_modules()
    counts = Counter(
        record["error_type"] for record in records if not record["recovered"]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not counts:
        ax.text(0.5, 0.5, "No unrecovered targets", ha="center", va="center")
        ax.set_axis_off()
    else:
        total = sum(counts.values())
        values = sorted(counts.items(), key=lambda item: item[1])
        bars = ax.barh(
            [ERROR_LABELS.get(name, name) for name, _ in values],
            [count for _, count in values],
        )
        ax.bar_label(
            bars,
            labels=[f"{count} ({100 * count / total:.1f}%)" for _, count in values],
            padding=3,
        )
        ax.set(xlabel="Unrecovered targets")
        ax.grid(False)
    ax.set_title("Error-category distribution")
    return _save(fig, destination, plt)


def _recoverability_plot(
    records: list[dict[str, Any]], summary: list[dict[str, Any]], destination: Path
) -> Path | None:
    annotated = [record for record in records if record["entity_recoverable"] is not None]
    if not annotated:
        return None
    unknown_labels = sorted(
        {record["entity_recoverable"] for record in annotated}
        - set(RECOVERABILITY_LABELS)
    )
    if unknown_labels:
        raise ValueError(
            "Unknown entity_recoverable labels: " + ", ".join(unknown_labels)
        )
    plt, np = _plotting_modules()
    entity_types = [row["type"] for row in sorted(summary, key=lambda row: row["rate"], reverse=True)]
    totals: Counter[tuple[str, str]] = Counter()
    recovered: Counter[tuple[str, str]] = Counter()
    for record in annotated:
        key = (record["target_type"], record["entity_recoverable"])
        totals[key] += 1
        recovered[key] += record["recovered"]

    positions = np.arange(len(entity_types))
    width = 0.25
    colors = ("tab:blue", "tab:orange", "tab:green")
    fig, ax = plt.subplots(figsize=(max(11, 0.75 * len(entity_types)), 5))
    for offset, label, color in zip((-width, 0, width), RECOVERABILITY_LABELS, colors):
        x = positions + offset
        ax.bar(
            x,
            [totals[(entity_type, label)] for entity_type in entity_types],
            width=width,
            color=color,
            alpha=0.25,
        )
        ax.bar(
            x,
            [recovered[(entity_type, label)] for entity_type in entity_types],
            width=width,
            color=color,
            label=label,
        )
    ax.set(
        xticks=positions,
        xticklabels=entity_types,
        xlabel="Target entity type",
        ylabel="Count",
        title="Recovered targets within all targets, by recoverability label",
    )
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="entity_recoverable")
    fig.text(
        0.5,
        0.01,
        "Pale bar = all targets; solid bar = recovered targets.",
        ha="center",
    )
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, destination, plt)


def _type_confusion_plot(records: list[dict[str, Any]], destination: Path) -> Path:
    plt, _ = _plotting_modules()
    counts = Counter(
        (record["target_type"], record["prediction_type"])
        for record in records
        if not record["recovered"]
    )
    values = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    values.reverse()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.34 * len(values))))
    if not values:
        ax.text(0.5, 0.5, "No type confusions", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [f"{target} → {prediction}" for (target, prediction), _ in values]
        bars = ax.barh(labels, [count for _, count in values])
        ax.bar_label(bars, labels=[str(count) for _, count in values], padding=3)
        ax.set(xlabel="Unrecovered targets")
        ax.grid(False)
    ax.set_title("Most frequent target → prediction type confusions")
    return _save(fig, destination, plt)


def _cosine_plot(metrics: dict[str, Any], destination: Path) -> Path | None:
    cosine = metrics.get("modernbert_swap_similarity")
    if not cosine:
        return None
    by_type = cosine.get("entity_type_mean_cosine_similarity", {})
    if not by_type:
        return None
    plt, _ = _plotting_modules()
    values = sorted(by_type.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.4 * len(values))))
    bars = ax.barh([name for name, _ in values], [value for _, value in values])
    ax.bar_label(bars, labels=[f"{value:.3f}" for _, value in values], padding=3)
    ax.set(
        xlim=(0, 1),
        xlabel="Mean cosine similarity",
        title="Target–prediction cosine similarity by entity type",
    )
    ax.grid(False)
    return _save(fig, destination, plt)


def write_evaluation_plots(
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Create the report's numbered PNG analytics in ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = _target_records(results, errors)
    if not records:
        return []
    summary = _entity_type_summary(records)
    plots = [
        _recovery_rate_plot(summary, destination / "1_recovery_rate_by_ent_type.png"),
        _entity_count_plot(summary, destination / "2_recovered_counts_ent_type.png"),
        _error_category_plot(records, destination / "3_error_categories.png"),
    ]
    recoverability = _recoverability_plot(
        records, summary, destination / "4_recovered_counts_recoverability.png"
    )
    if recoverability is not None:
        plots.append(recoverability)
    plots.append(_type_confusion_plot(records, destination / "5_type_confusions.png"))
    cosine = _cosine_plot(metrics, destination / "6_cosine_sim.png")
    if cosine is not None:
        plots.append(cosine)
    return plots

"""Run FLAN-T5 prediction, MARS evaluation, and HTML reporting.

Example (from any working directory)::

    python run_flant5.py --model models/flan-t5-base-aligned-2k/final \
        --dataset data/xsum_with_mask

The command creates a timestamped run directory under ``output/`` containing
``config.json``, ``predictions.json``, ``metrics.json``, ``errors.json``,
numbered analytics PNGs, and ``prediction_report.html``.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTOR_NAME = "flan_t5_continuous_generation"
DEFAULT_METRICS = (
    "exact_match",
    "entity_type_errors",
    # "recoverability_errors",  # requires entity_recoverable in the dataset
)
MASK_RE = re.compile(r"<mask>")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a FLAN-T5 checkpoint on a masked dataset, evaluate the "
            "predictions, and create a self-contained HTML report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/flan-t5-base-aligned-2k/final",
        help="Local checkpoint directory or Hugging Face model identifier.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory previously saved with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "DatasetDict split to use. When omitted, all splits are combined; "
            "for a plain Dataset this option is unnecessary."
        ),
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=None,
        help="Optional eligible-row limit for a smaller test run.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--output",
        help=(
            "Run directory. By default a timestamped *_evaluation directory is "
            "created under the project's output/ directory."
        ),
    )
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--ner-batch-size", type=positive_int, default=8)
    parser.add_argument("--ner-model", default="en_core_web_trf")
    parser.add_argument(
        "--max-input-length",
        type=optional_positive_int,
        default=2048,
        help="Maximum tokens per generated FLAN-T5 chunk; use 'none' to disable.",
    )
    parser.add_argument("--max-sentinels", type=positive_int, default=100)
    parser.add_argument(
        "--chunk-overlap-sentences",
        type=positive_int,
        default=3,
        help=(
            "Sentence count per half of the overlap; adjacent chunks share "
            "twice this many sentences."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--no-guided-decoding",
        action="store_true",
        help="Disable ordered-sentinel generation constraints.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not include the summary in the model input.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Evaluation metrics to calculate.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def optional_positive_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "null"}:
        return None
    return positive_int(value)


def resolve_existing_path(value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


def resolve_model(value: str) -> tuple[str, Path | None]:
    """Resolve local checkpoints while still accepting Hugging Face IDs."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.exists():
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise NotADirectoryError(f"Model checkpoint is not a directory: {resolved}")
        return str(resolved), resolved
    if Path(value).is_absolute() or value.startswith((".", "~")):
        raise FileNotFoundError(f"Model checkpoint does not exist: {candidate.resolve()}")
    return value, None


def choose_device(torch_module: Any, requested: str):
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    name = "cuda" if requested == "auto" and torch_module.cuda.is_available() else requested
    if name == "auto":
        name = "cpu"
    return torch_module.device(name)


def create_run_dir(output: str | None, dataset: Path, split_label: str | None) -> Path:
    if output:
        destination = Path(output).expanduser()
        if not destination.is_absolute():
            destination = PROJECT_ROOT / destination
        destination = destination.resolve()
    else:
        parent = PROJECT_ROOT / "output"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", dataset.name).strip("-") or "dataset"
        split_suffix = ""
        if split_label:
            split_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", split_label).strip("-")
            if split_slug:
                split_suffix = f"_{split_slug}"
        destination = parent / (
            f"{datetime.now():%Y%m%d_%H%M%S}_{slug}{split_suffix}_evaluation"
        )
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def validate_dataset_columns(dataset: Any, *, require_recoverability: bool = False) -> None:
    required = {"summary", "masked_text", "demasked_words", "original_text"}
    if require_recoverability:
        required.add("entity_recoverable")
    columns = set(getattr(dataset, "column_names", ()))
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def eligible_indices(
    dataset: Any,
    *,
    samples: int | None,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    candidates = list(range(len(dataset)))
    if samples is not None:
        random.Random(seed).shuffle(candidates)

    selected: list[int] = []
    rejected: list[dict[str, Any]] = []
    for row_index in candidates:
        row = dataset[row_index]
        try:
            mask_count = len(MASK_RE.findall(row["masked_text"]))
            entity_count = len(row["demasked_words"])
            if mask_count == 0:
                raise ValueError("Unsupported mask count: 0")
            if mask_count != entity_count:
                raise ValueError(
                    f"Found {mask_count} masks but {entity_count} gold entities"
                )
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"row": row_index, "reason": str(error)})
            continue
        selected.append(row_index)
        if samples is not None and len(selected) == samples:
            break

    selected.sort()
    return selected, rejected


def persist_predictions(
    destination: Path,
    prediction_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    predictions = [
        {key: value for key, value in prediction.items() if key != "id"}
        for prediction in prediction_rows
    ]
    path = destination / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def build_evaluation_results(
    dataset: Any, prediction_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        row_index = prediction["row"]
        row = dataset[row_index]
        if len(prediction["entities"]) != len(row["demasked_words"]):
            raise ValueError(f"Mask alignment failed at dataset row {row_index}.")
        if prediction["status"] != "ok":
            continue
        results.append(
            {
                "row": row_index,
                "prediction_entities": prediction["entities"],
                "target_entities": row["demasked_words"],
                "entity_recoverable": row.get("entity_recoverable"),
                "summary": row["summary"],
                "original_text": row["original_text"],
                "masked_text": row["masked_text"],
            }
        )
    if not results:
        raise RuntimeError("No successful predictions are available for evaluation.")
    return results


def json_for_html(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def write_html_report(
    destination: Path,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    plot_paths: list[Path],
) -> Path:
    from evaluation import evaluation_metrics as evaluator

    errors_by_position = {(error["row"], error["index"]): error for error in errors}

    def hover_detail(result: dict[str, Any], index: int, fallback: str) -> str:
        error = errors_by_position.get((result["row"], index))
        return json.dumps(error, ensure_ascii=False, indent=2) if error else fallback

    def render_text(result: dict[str, Any], *, show_predictions: bool) -> str:
        collected = evaluator.collect_entity_information(result)
        parts: list[str] = []
        cursor = 0
        for index, match in enumerate(MASK_RE.finditer(result["masked_text"])):
            parts.append(html.escape(result["masked_text"][cursor : match.start()]))
            entity = collected["entities"][index]
            detail = evaluator.classify_mask_error(
                result, index, collected["alignment"], entity
            )
            prediction = entity["matched_prediction"] or entity["prediction"] or "[missing]"
            shown = prediction if show_predictions else entity["target"]
            css = "correct" if entity["recovered"] else "incorrect"
            fallback = (
                f"mask {index}: target: {entity['target']} | prediction: {prediction} | "
                f"target type: {entity['target_type']} | prediction type: "
                f"{detail['prediction_type']} | error type: {detail['category']} | "
                f"evidence: {evaluator.error_evidence(detail) or '[none]'}"
            )
            escaped_detail = html.escape(
                hover_detail(result, index, fallback), quote=True
            )
            parts.append(
                f'<mark class="{css}" data-mask="{index}" '
                f'data-detail="{escaped_detail}">{html.escape(shown)}</mark>'
            )
            cursor = match.end()
        parts.append(html.escape(result["masked_text"][cursor:]))
        return "".join(parts)

    cards = []
    for result in results:
        cards.append(
            f'''<details class="example" open>
  <summary>Row {result['row']}</summary>
  <section class="summary-pane"><h3>Summary</h3><div class="text">{html.escape(result['summary'])}</div></section>
  <div class="comparison">
    <section><h3>Original text</h3><div class="text">{render_text(result, show_predictions=False)}</div></section>
    <section class="right-pane"><h3>Model predictions</h3><div class="text">{render_text(result, show_predictions=True)}</div></section>
  </div>
</details>'''
        )

    plot_cards = []
    for plot_path in plot_paths:
        encoded = base64.b64encode(plot_path.read_bytes()).decode("ascii")
        title = plot_path.stem.removeprefix("1_").removeprefix("2_")
        title = title.removeprefix("3_").removeprefix("4_")
        title = title.removeprefix("5_").removeprefix("6_").replace("_", " ")
        plot_cards.append(
            f'<figure><img src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(title)}"></figure>'
        )

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FLAN-T5 evaluation report</title>
<style>
:root {{ color-scheme:light; font-family:Inter,system-ui,sans-serif; }}
body {{ margin:0; background:#f4f6f8; color:#17202a; }}
main {{ max-width:1200px; margin:auto; padding:28px; }}
.overview,.example {{ background:#fff; border:1px solid #dfe6e9; border-radius:10px; }}
.overview {{ padding:18px; margin-bottom:20px; }}
pre {{ overflow:auto; padding:12px; background:#f7f8fa; border-radius:7px; }}
.plots {{ display:grid; grid-template-columns:1fr; gap:16px; margin:20px 0; }}
figure {{ margin:0; padding:12px; background:#fff; border:1px solid #dfe6e9; border-radius:10px; }}
figure img {{ display:block; width:100%; height:auto; }}
.example {{ margin:14px 0; }}
.example>summary {{ cursor:pointer; padding:14px 18px; font-weight:700; }}
.summary-pane {{ border-bottom:1px solid #dfe6e9; }}
.comparison {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); }}
.right-pane {{ border-left:1px solid #dfe6e9; }}
section {{ min-width:0; padding:0 18px 18px; }}
h3 {{ margin-bottom:7px; }}
.text {{ white-space:pre-wrap; line-height:1.6; font-family:Georgia,serif; }}
mark {{ padding:1px 4px; border-radius:4px; }}
mark.correct {{ background:#c8f7d2; color:#145a32; }}
mark.incorrect {{ background:#ffd1d1; color:#8b1a1a; }}
mark[data-mask] {{ cursor:crosshair; transition:box-shadow 120ms ease,filter 120ms ease; }}
mark.paired-hover {{ box-shadow:0 0 0 3px #2d6cdf,0 0 0 6px rgba(45,108,223,.22); filter:saturate(1.3) brightness(1.04); }}
#hover-detail {{ display:none; position:fixed; z-index:10; max-width:min(560px,90vw); max-height:50vh; overflow:auto; padding:12px; border-radius:7px; background:#17202a; color:#fff; box-shadow:0 5px 18px rgba(0,0,0,.3); white-space:pre-wrap; font:12px/1.45 ui-monospace,monospace; pointer-events:none; }}
@media(max-width:850px) {{ .comparison {{ grid-template-columns:1fr; }} .right-pane {{ border-left:0; border-top:1px solid #dfe6e9; padding-top:12px; }} }}
</style></head><body><main>
<h1>FLAN-T5 evaluation report</h1>
<div class="overview"><h2>Metrics</h2><pre>{json_for_html(metrics)}</pre>
<details><summary>Run metadata</summary><pre>{json_for_html(metadata)}</pre></details></div>
<h2>Evaluation analytics</h2><div class="plots">{''.join(plot_cards)}</div>
<p><mark class="correct">green = recovered</mark> &nbsp; <mark class="incorrect">red = unrecovered</mark></p>
{''.join(cards)}
</main><aside id="hover-detail" role="tooltip"></aside>
<script>(()=>{{const popup=document.querySelector('#hover-detail');const pair=(mark,on)=>{{const example=mark.closest('.example');if(example)example.querySelectorAll(`mark[data-mask="${{mark.dataset.mask}}"]`).forEach(item=>item.classList.toggle('paired-hover',on));}};const hide=()=>popup.style.display='none';document.addEventListener('pointerover',event=>{{const mark=event.target.closest('mark[data-mask]');if(!mark)return;pair(mark,true);popup.textContent=mark.dataset.detail;popup.style.display='block';}});document.addEventListener('pointermove',event=>{{if(popup.style.display==='block'){{popup.style.left=Math.min(event.clientX+14,innerWidth-popup.offsetWidth-8)+'px';popup.style.top=Math.min(event.clientY+14,innerHeight-popup.offsetHeight-8)+'px';}}}});document.addEventListener('pointerout',event=>{{const mark=event.target.closest('mark[data-mask]');if(mark){{pair(mark,false);hide();}}}});}})();</script>
</body></html>'''
    report_path = destination / "prediction_report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> Path:
    import spacy
    import torch
    from datasets import DatasetDict, concatenate_datasets, load_from_disk
    from evaluation import evaluation_metrics as evaluator
    from evaluation.evaluation_plots import (
        validate_plot_dependencies,
        write_evaluation_plots,
    )
    from evaluation.evaluation_predictors import load_flan_t5, predict_dataset

    validate_plot_dependencies()

    unsupported_metrics = sorted(set(args.metrics) - set(evaluator.METRICS))
    if unsupported_metrics:
        supported = ", ".join(sorted(evaluator.METRICS))
        raise ValueError(
            f"Unsupported metrics: {', '.join(unsupported_metrics)}. "
            f"Supported metrics: {supported}"
        )
    ner_path = Path(args.ner_model).expanduser()
    if not ner_path.exists() and not spacy.util.is_package(args.ner_model):
        raise RuntimeError(
            f"spaCy model {args.ner_model!r} is not installed. Install it with: "
            f"python -m spacy download {args.ner_model}"
        )

    model_source, _ = resolve_model(args.model)
    dataset_path = resolve_existing_path(args.dataset, "Dataset")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset is not a directory: {dataset_path}")

    device = choose_device(torch, args.device)
    settings = {
        "max_input_length": args.max_input_length,
        "max_chunk_length": args.max_input_length,
        "max_sentinels": args.max_sentinels,
        "chunk_overlap_sentences": args.chunk_overlap_sentences,
        "guided_decoding": not args.no_guided_decoding,
        "batch_size": args.batch_size,
        "use_summary": not args.no_summary,
        "device": device,
        "model_dtype": "float16" if device.type == "cuda" else "float32",
    }

    print(f"Loading dataset: {dataset_path}", flush=True)
    loaded = load_from_disk(str(dataset_path))
    if isinstance(loaded, DatasetDict):
        if args.split is not None and args.split not in loaded:
            available = ", ".join(loaded.keys())
            raise ValueError(
                f"Dataset split {args.split!r} does not exist. Available: {available}"
            )
        if args.split is None:
            selected_splits = list(loaded.keys())
            if not selected_splits:
                raise ValueError("The DatasetDict contains no splits.")
            dataset = concatenate_datasets([loaded[name] for name in selected_splits])
            print(f"Using all splits: {', '.join(selected_splits)}", flush=True)
        else:
            selected_splits = [args.split]
            dataset = loaded[args.split]
    else:
        if args.split is not None:
            print(
                f"Note: --split {args.split!r} is ignored because the saved "
                "object is a plain Dataset.",
                file=sys.stderr,
            )
        selected_splits = None
        dataset = loaded
    validate_dataset_columns(
        dataset,
        require_recoverability="recoverability_errors" in args.metrics,
    )

    print(f"Loading model on {device}: {model_source}", flush=True)
    model, tokenizer = load_flan_t5(model_source, settings)
    requested_samples = args.samples
    if settings["max_input_length"] is None:
        print(
            f"Chunk token limit disabled; validating all {len(dataset)} rows...",
            flush=True,
        )
    else:
        print(
            f"Validating {len(dataset)} rows for automatic chunking at "
            f"max_chunk_length={settings['max_chunk_length']}...",
            flush=True,
        )
    selected_indices, rejected = eligible_indices(
        dataset,
        samples=requested_samples,
        seed=args.seed,
    )
    if not selected_indices:
        raise RuntimeError("The dataset contains no rows eligible for this configuration.")
    if requested_samples is not None and len(selected_indices) < requested_samples:
        print(
            f"Warning: requested {requested_samples} rows, but only "
            f"{len(selected_indices)} are eligible.",
            file=sys.stderr,
        )
    if requested_samples is None:
        print(
            f"Selected {len(selected_indices)} rows; filtered out {len(rejected)}.",
            flush=True,
        )
    else:
        examined = len(selected_indices) + len(rejected)
        print(
            f"Selected {len(selected_indices)} eligible rows after examining "
            f"{examined} rows ({len(rejected)} rejected).",
            flush=True,
        )

    if selected_splits is None:
        output_split_label = None
    elif args.split is None:
        output_split_label = "all-splits"
    else:
        output_split_label = args.split
    run_dir = create_run_dir(args.output, dataset_path, output_split_label)
    created_at = datetime.now().isoformat(timespec="seconds")
    metadata = {
        "model_checkpoint": model_source,
        "dataset_path": str(dataset_path),
        "dataset_split": args.split,
        "dataset_splits": selected_splits,
        "predictor": PREDICTOR_NAME,
        "created_at": created_at,
        "device": str(device),
        "requested_samples": "all" if args.samples is None else args.samples,
        "dataset_rows_before_filtering": len(dataset),
        "selected_samples": len(selected_indices),
        "rows_rejected_during_selection": len(rejected),
        "sampling_seed": args.seed,
        "settings": {
            key: str(value) if key == "device" else value
            for key, value in settings.items()
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Generating predictions for {len(selected_indices)} rows...", flush=True)
    prediction_rows, _predictor_skipped = predict_dataset(
        model,
        tokenizer,
        dataset,
        PREDICTOR_NAME,
        settings,
        indices=selected_indices,
    )
    prefiltered_rows = []
    for rejected_row in rejected:
        row_index = rejected_row["row"]
        row = dataset[row_index]
        prefiltered_rows.append(
            {
                "row": row_index,
                "status": "skipped",
                "reason": rejected_row["reason"],
                "entities": [None] * len(row.get("demasked_words", [])),
            }
        )
    prediction_rows.extend(prefiltered_rows)
    prediction_rows.sort(key=lambda prediction: prediction["row"])
    skipped_count = sum(
        prediction["status"] == "skipped" for prediction in prediction_rows
    )
    predictions_path = persist_predictions(
        run_dir, prediction_rows, metadata
    )
    print(
        f"Saved predictions: {predictions_path} "
        f"({len(prediction_rows) - skipped_count} ok, {skipped_count} skipped)",
        flush=True,
    )

    results = build_evaluation_results(dataset, prediction_rows)
    print(
        f"Evaluating {len(results)} rows with spaCy model {args.ner_model}...",
        flush=True,
    )
    evaluator.prepare_results(
        results, model_name=args.ner_model, batch_size=args.ner_batch_size
    )
    evaluation_metadata = {
        **metadata,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_metrics": args.metrics,
        "prediction_file": str(predictions_path),
        "evaluator_backend": "spacy",
        "ner_model": args.ner_model,
        "ner_batch_size": args.ner_batch_size,
    }
    metrics, errors = evaluator.write_evaluation_artifacts(
        results,
        args.metrics,
        run_dir,
        metadata=evaluation_metadata,
        examples_skipped_during_prediction=skipped_count,
    )
    print("Creating evaluation analytics plots...", flush=True)
    plot_paths = write_evaluation_plots(
        results, errors, metrics["metrics"], run_dir
    )
    report_path = write_html_report(
        run_dir,
        results,
        errors,
        metrics["metrics"],
        evaluation_metadata,
        plot_paths,
    )

    print(json.dumps(metrics["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved {len(plot_paths)} analytics plots to: {run_dir}")
    print(f"HTML report: {report_path}")
    return report_path


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        run(args)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped by user.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

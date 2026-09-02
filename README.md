# MARS FLAN-T5 evaluation runner

`run_flant5.py` runs a FLAN-T5 checkpoint on a saved Hugging Face dataset, evaluates its entity-recovery predictions, creates analytics plots, and writes a self-contained HTML report.

## Run

Run from the repository root, in the same Python environment used for the notebooks:

```powershell
python run_flant5.py --dataset data/mars_test_200
```

The default model is `models/flan-t5-base-aligned-2k/final`. To use another local checkpoint or Hugging Face model ID:

```powershell
python run_flant5.py `
  --model models/flan-t5-base-aligned-2k/final `
  --dataset data/mars_test_200 `
  --split [None | str] `
  --max-input-length 2048 `
  --batch-size 2 `
  --max-sentinels 50 `
  --chunk-overlap-sentences 2 

## Dataset requirements

The dataset must have been written with `datasets.save_to_disk()` and must contain these columns:

| Column | Meaning |
| --- | --- |
| `original_text` | Unmasked source document. |
| `summary` | Summary supplied to FLAN-T5 as context. |
| `masked_text` | Source document containing `<mask>` placeholders. |
| `demasked_words` | Gold entity text, in the same order as `<mask>` placeholders. |

`masked_text` and `demasked_words` must align exactly. Rows with zero masks or a mismatch are recorded as `status: "skipped"`; they are not sent to the model.

For `--metrics recoverability_errors`, the dataset must also include `entity_recoverable`, with one label per masked entity.

### Plain Dataset versus DatasetDict

- A plain `Dataset` needs only `--dataset`; `--split` is ignored.
- With a `DatasetDict`, use `--split test` to run one split.
- Omit `--split` to concatenate and run every split. The generated directory name includes `all-splits` in this case.

Example:

```powershell
python run_flant5.py --dataset data/all_data_1000_2k --split multi_news
```

## Long documents and many masks

The FLAN-T5 predictor chunks rows automatically. It keeps every mask prediction in original order and merges chunk predictions back into one prediction per dataset row.

```powershell
python run_flant5.py `
  --dataset data/agent_inputs_with_mask `
  --max-input-length 2048 `
  --max-sentinels 30 `
  --chunk-overlap-sentences 3
```

- `--max-input-length` is the maximum input tokens per generated chunk. The default is `2048`; pass `none` to remove the token limit.
- `--max-sentinels` is the maximum **effective** masks predicted in one chunk. It must be from 1 to 100 because T5 exposes 100 sentinel tokens.
- `--chunk-overlap-sentences` is the context on each side of a chunk boundary. The default is `3`, so adjacent chunks share six sentences. Context masks remain literal `<mask>` and are not predicted again.

Rows are skipped only when their masks are malformed, a single sentence cannot fit, or the configured overlap plus one effective sentence cannot fit within the chunk limits.

## Useful options

```text
--batch-size N                  FLAN-T5 generation batch size (default: 2)
--device auto|cuda|cpu          Inference device (default: auto)
--no-summary                    Do not include the summary in model input
--no-guided-decoding            Disable ordered sentinel decoding constraints
--ner-model MODEL               spaCy model for evaluation (default: en_core_web_trf)
--ner-batch-size N              spaCy evaluation batch size (default: 8)
--metrics NAME [NAME ...]       Metrics to calculate
--output PATH                   Explicit new output directory
```

Supported metrics are `exact_match`, `entity_type_errors`, `recoverability_errors`, and `modernbert_swap_similarity`. The defaults are `exact_match` and `entity_type_errors`.

On CUDA, FLAN-T5 loads in FP16. CPU runs use FP32 and are substantially slower.

## Output

Without `--output`, each run creates a new directory under `output/`:

```text
output/YYYYMMDD_HHMMSS_<dataset>[_<split>]_evaluation/
```

It contains:

| File | Contents |
| --- | --- |
| `config.json` | Command configuration and run metadata. |
| `predictions.json` | One prediction record per input row, including skipped rows. |
| `metrics.json` | Calculated metric values. |
| `errors.json` | Per-entity error details. |
| `1_*.png` through `6_*.png` | Evaluation analytics plots. |
| `prediction_report.html` | Self-contained report with embedded plots and side-by-side gold/predicted text. |

Open `prediction_report.html` in a browser. All rows with successful predictions are expanded by default; hover a highlighted entity to see its evaluation detail.

## Dependencies

The runner needs the project inference/evaluation dependencies, plus the selected spaCy model. If spaCy reports that `en_core_web_trf` is missing, install it in the active environment:

```powershell
python -m spacy download en_core_web_trf
```

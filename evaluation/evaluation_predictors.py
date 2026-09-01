"""Model-specific, dataset-level entity prediction functions."""

import re
from bisect import bisect_right

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessor


MASK_PATTERN = re.compile(r"<mask>")
SENTINEL_PATTERN = re.compile(r"<extra_id_(\d+)>")
MAX_GENERATED_TOKENS = 1024
DEFAULT_CHUNK_OVERLAP_SENTENCES = 3
_SENTENCE_NLP = None


class OrderedSentinelLogitsProcessor(LogitsProcessor):
    """Force non-empty, ordered T5 sentinel spans during generation."""

    def __init__(self, tokenizer, entity_counts: list[int]):
        self.entity_counts = entity_counts
        self.decoder_start_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sentinel_ids = {
            int(match.group(1)): token_id
            for token, token_id in tokenizer.get_vocab().items()
            if (match := SENTINEL_PATTERN.fullmatch(token))
        }
        self.all_sentinel_ids = set(self.sentinel_ids.values())
        self.special_ids = set(getattr(tokenizer, "all_special_ids", ())) | self.all_sentinel_ids

    def __call__(self, input_ids, scores):
        for batch_index, sequence in enumerate(input_ids.tolist()):
            if sequence and sequence[0] == self.decoder_start_id:
                sequence = sequence[1:]
            last_sentinel, has_content = -1, False
            for token_id in sequence:
                if token_id in self.all_sentinel_ids:
                    last_sentinel = next(index for index, sentinel_id in self.sentinel_ids.items() if sentinel_id == token_id)
                    has_content = False
                elif token_id not in self.special_ids:
                    has_content = True

            if last_sentinel < 0:
                scores[batch_index].fill_(float("-inf"))
                scores[batch_index, self.sentinel_ids[0]] = 0
                continue

            blocked = self.all_sentinel_ids | (self.special_ids - {self.eos_id})
            if last_sentinel + 1 < self.entity_counts[batch_index]:
                blocked.add(self.eos_id)
            if not has_content:
                blocked |= self.special_ids
            else:
                next_sentinel = self.sentinel_ids.get(last_sentinel + 1)
                if next_sentinel is not None and last_sentinel + 1 < self.entity_counts[batch_index]:
                    blocked.discard(next_sentinel)
            scores[batch_index, list(blocked)] = float("-inf")
        return scores


def normalize_prediction(text: str) -> str:
    """Preserve generated content while normalizing whitespace only."""
    return " ".join(text.split())


def number_masks(masked_text: str, entity_count: int, max_sentinels: int) -> str:
    mask_count = len(MASK_PATTERN.findall(masked_text))
    if mask_count != entity_count:
        raise ValueError(f"Found {mask_count} masks but {entity_count} gold entities")
    if not 0 < mask_count <= max_sentinels:
        raise ValueError(f"Unsupported mask count: {mask_count}")
    index = 0

    def replacement(_match):
        nonlocal index
        token = f"<extra_id_{index}>"
        index += 1
        return token

    return MASK_PATTERN.sub(replacement, masked_text)


def make_flan_t5_source(row: dict, settings: dict) -> str:
    document = number_masks(row["masked_text"], len(row["demasked_words"]), settings["max_sentinels"])
    if settings["use_summary"]:
        return f"summary: {row['summary']}\nmasked document: {document}"
    return f"masked document: {document}"


def make_flan_t5_target(row: dict) -> str:
    return " ".join(f"<extra_id_{index}> {entity}" for index, entity in enumerate(row["demasked_words"]))


def _flan_t5_prefix(row: dict, use_summary: bool) -> str:
    if use_summary:
        return f"summary: {row['summary']}\nmasked document: "
    return "masked document: "


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return sentencizer spans that partition ``text`` without losing whitespace."""
    global _SENTENCE_NLP
    import spacy

    if _SENTENCE_NLP is None:
        _SENTENCE_NLP = spacy.blank("en")
        _SENTENCE_NLP.add_pipe("sentencizer")
    _SENTENCE_NLP.max_length = max(_SENTENCE_NLP.max_length, len(text) + 1)
    starts = [sentence.start_char for sentence in _SENTENCE_NLP(text).sents] or [0]
    starts[0] = 0
    return list(zip(starts, starts[1:] + [len(text)]))


def chunk_flan_t5_row(
    row: dict,
    tokenizer,
    *,
    max_sentinels: int,
    max_chunk_length: int | None,
    overlap_sentences: int = DEFAULT_CHUNK_OVERLAP_SENTENCES,
    use_summary: bool = True,
) -> list[dict]:
    """Build FLAN-T5 chunks with rolling sentence ownership overlap.

    Adjacent chunks share ``2 * overlap_sentences`` sentences. In the earlier
    chunk the first half is effective and the second half is context; those
    roles reverse in the next chunk. Every global mask is therefore effective
    exactly once.
    """
    if not 0 < max_sentinels <= 100:
        raise ValueError("max_sentinels must be in [1, 100]")
    if max_chunk_length is not None and max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive or None")
    if overlap_sentences <= 0:
        raise ValueError("overlap_sentences must be positive")

    masked_text = row["masked_text"]
    matches = list(MASK_PATTERN.finditer(masked_text))
    if not matches:
        raise ValueError("Unsupported mask count: 0")
    if len(row["demasked_words"]) != len(matches):
        raise ValueError(
            f"Found {len(matches)} masks but {len(row['demasked_words'])} gold entities"
        )
    prefix = _flan_t5_prefix(row, use_summary)

    def build_chunk_source(
        left: int,
        right: int,
        owned_indices: list[int],
        *,
        tokenize_source: bool = True,
    ) -> tuple[str, str, int | None, list[int]]:
        owned_local = {
            global_index: local_index
            for local_index, global_index in enumerate(owned_indices)
        }
        window_indices = [
            index
            for index, match in enumerate(matches)
            if left <= match.start() and match.end() <= right
        ]
        pieces: list[str] = []
        cursor = left
        for global_index in window_indices:
            match = matches[global_index]
            pieces.append(masked_text[cursor : match.start()])
            if global_index in owned_local:
                pieces.append(f"<extra_id_{owned_local[global_index]}>")
            else:
                pieces.append("<mask>")
            cursor = match.end()
        pieces.append(masked_text[cursor:right])
        chunk_text = "".join(pieces)
        source = prefix + chunk_text
        input_tokens = None
        if tokenize_source:
            input_tokens = len(
                tokenizer(
                    source,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
        context_indices = [
            index for index in window_indices if index not in owned_local
        ]
        return chunk_text, source, input_tokens, context_indices

    all_indices = list(range(len(matches)))
    if len(matches) <= max_sentinels:
        chunk_text, source, input_tokens, _ = build_chunk_source(
            0, len(masked_text), all_indices
        )
        if max_chunk_length is None or input_tokens <= max_chunk_length:
            return [
                {
                    "chunk_index": 0,
                    "source": source,
                    "masked_text": chunk_text,
                    "input_tokens": input_tokens,
                    "effective_global_mask_indices": all_indices,
                    "context_global_mask_indices": [],
                    "window_char_start": 0,
                    "window_char_end": len(masked_text),
                }
            ]

    sentence_spans = _sentence_spans(masked_text)
    sentence_starts = [left for left, _ in sentence_spans]
    sentence_masks: list[list[int]] = [[] for _ in sentence_spans]
    for mask_index, match in enumerate(matches):
        sentence_index = max(0, bisect_right(sentence_starts, match.start()) - 1)
        sentence_masks[sentence_index].append(mask_index)

    prefix_tokens = len(
        tokenizer(prefix, add_special_tokens=False, verbose=False)["input_ids"]
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    effective_sentence_tokens: list[int] = []
    context_sentence_tokens: list[int] = []
    for sentence_index, (left, right) in enumerate(sentence_spans):
        effective_text, _, _, _ = build_chunk_source(
            left,
            right,
            sentence_masks[sentence_index],
            tokenize_source=False,
        )
        context_text, _, _, _ = build_chunk_source(
            left, right, [], tokenize_source=False
        )
        effective_sentence_tokens.append(
            len(
                tokenizer(
                    effective_text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
        )
        context_sentence_tokens.append(
            len(
                tokenizer(
                    context_text,
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
        )

    chunks: list[dict] = []
    group_start = 0
    sentence_count = len(sentence_spans)
    shared_sentences = overlap_sentences * 2
    while group_start < sentence_count:
        is_first = not chunks
        best_end = None
        for candidate_end in range(group_start + 1, sentence_count + 1):
            is_final = candidate_end == sentence_count
            left_context_end = (
                group_start
                if is_first
                else min(candidate_end, group_start + overlap_sentences)
            )
            right_context_start = (
                candidate_end
                if is_final
                else max(left_context_end, candidate_end - overlap_sentences)
            )
            candidate_owned = [
                index
                for sentence_index in range(left_context_end, right_context_start)
                for index in sentence_masks[sentence_index]
            ]
            if len(candidate_owned) > max_sentinels:
                break
            estimated_tokens = (
                prefix_tokens
                + special_tokens
                + sum(context_sentence_tokens[group_start:left_context_end])
                + sum(
                    effective_sentence_tokens[left_context_end:right_context_start]
                )
                + sum(context_sentence_tokens[right_context_start:candidate_end])
            )
            if (
                max_chunk_length is not None
                and estimated_tokens > max_chunk_length
            ):
                break
            best_end = candidate_end

        if best_end is None:
            raise ValueError(
                f"Sentence {group_start} cannot fit within the chunk limits"
            )

        group_end = best_end
        while True:
            is_final = group_end == sentence_count
            left_context_end = (
                group_start if is_first else group_start + overlap_sentences
            )
            right_context_start = (
                group_end if is_final else group_end - overlap_sentences
            )
            if not is_final and group_end - group_start <= shared_sentences:
                raise ValueError(
                    f"Chunk beginning at sentence {group_start} cannot fit "
                    f"{shared_sentences} shared sentences plus an effective sentence"
                )
            owned_indices = [
                index
                for sentence_index in range(left_context_end, right_context_start)
                for index in sentence_masks[sentence_index]
            ]
            left = sentence_spans[group_start][0]
            right = sentence_spans[group_end - 1][1]
            chunk_text, source, input_tokens, context_indices = build_chunk_source(
                left, right, owned_indices
            )
            if max_chunk_length is None or input_tokens <= max_chunk_length:
                break
            group_end -= 1
            if group_end <= group_start:
                raise ValueError(
                    f"Sentence {group_start} cannot fit within {max_chunk_length} tokens"
                )

        chunks.append(
            {
                "chunk_index": len(chunks),
                "source": source,
                "masked_text": chunk_text,
                "input_tokens": input_tokens,
                "effective_global_mask_indices": owned_indices,
                "context_global_mask_indices": context_indices,
                "window_char_start": left,
                "window_char_end": right,
            }
        )
        group_start = group_end if is_final else group_end - shared_sentences

    owned = [
        index
        for chunk in chunks
        for index in chunk["effective_global_mask_indices"]
    ]
    if owned != all_indices:
        raise RuntimeError("FLAN-T5 chunk ownership does not cover every mask once")
    return chunks


def load_flan_t5(checkpoint, settings: dict):
    """Load and validate a FLAN-T5 checkpoint for continuous sentinel generation."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    device = settings["device"]
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, torch_dtype=model_dtype
    ).to(device)
    model.config.use_cache = True
    model.eval()
    sentinels = [f"<extra_id_{index}>" for index in range(settings["max_sentinels"])]
    ids = tokenizer.convert_tokens_to_ids(sentinels)
    if any(token_id in (None, tokenizer.unk_token_id) for token_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Tokenizer does not contain distinct <extra_id_N> T5 sentinel tokens.")
    return model, tokenizer


def _parse_flan_t5_entities(decoded_text: str, tokenizer) -> dict[int, str]:
    matches = list(SENTINEL_PATTERN.finditer(decoded_text))
    entities = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(decoded_text)
        span = decoded_text[match.end():end]
        span = span.replace(tokenizer.eos_token or "", "").replace(tokenizer.pad_token or "", "")
        prediction = normalize_prediction(span)
        index = int(match.group(1))
        if index not in entities and prediction:
            entities[index] = prediction
    return entities


def predict_flan_t5_continuous_generation(model, tokenizer, batch: list[dict], settings: dict) -> list[dict]:
    """Generate all entity spans for a prepared FLAN-T5 batch."""
    max_input_length = settings.get(
        "max_chunk_length", settings.get("max_input_length")
    )
    tokenization = {
        "padding": True,
        "truncation": max_input_length is not None,
        "return_tensors": "pt",
    }
    if max_input_length is not None:
        tokenization["max_length"] = max_input_length
    encoded = tokenizer(
        [item["source"] for item in batch], **tokenization
    ).to(settings["device"])
    device = settings["device"]
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16
    generation_kwargs = {
        # Avoid Transformers' short default generation limit. This is a safety
        # ceiling, not a dataset filter or target truncation setting.
        "max_new_tokens": MAX_GENERATED_TOKENS,
        "num_beams": 1,
        "do_sample": False,
    }
    if settings.get("guided_decoding", False):
        generation_kwargs["logits_processor"] = [
            OrderedSentinelLogitsProcessor(tokenizer, [item["entity_count"] for item in batch])
        ]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        generated = model.generate(**encoded, **generation_kwargs)
    decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=False)
    results = []
    for item, decoded in zip(batch, decoded_batch):
        parsed = _parse_flan_t5_entities(decoded, tokenizer)
        results.append({"row": item["row"], "id": item["id"], "decoded": decoded, "status": "ok", "entities": [parsed.get(index) for index in range(item["entity_count"])]})
    return results


PREDICTORS = {"flan_t5_continuous_generation": predict_flan_t5_continuous_generation}


def get_predictor(name: str):
    try:
        return PREDICTORS[name]
    except KeyError as error:
        raise ValueError(f"Unknown predictor {name!r}. Supported: {', '.join(sorted(PREDICTORS))}") from error


def _selected_indices(dataset, indices: list[int] | None) -> list[int]:
    """Validate selected dataset rows, defaulting to the full dataset."""
    selected = list(range(len(dataset))) if indices is None else list(indices)
    if len(set(selected)) != len(selected):
        raise ValueError("indices must not contain duplicate dataset rows")
    for row_index in selected:
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise TypeError("indices must contain integer dataset rows")
        if not 0 <= row_index < len(dataset):
            raise IndexError(f"Dataset row {row_index} is out of range")
    return selected


def _predict_flan_t5_chunked_dataset(
    model,
    tokenizer,
    dataset,
    settings: dict,
    indices: list[int] | None,
):
    """Chunk FLAN-T5 rows, generate effective spans, and merge globally."""
    selected_indices = _selected_indices(dataset, indices)
    max_chunk_length = settings.get(
        "max_chunk_length", settings.get("max_input_length")
    )
    overlap_sentences = settings.get(
        "chunk_overlap_sentences", DEFAULT_CHUNK_OVERLAP_SENTENCES
    )
    prediction_rows: list[dict | None] = [None] * len(selected_indices)
    row_positions = {
        row_index: position for position, row_index in enumerate(selected_indices)
    }
    row_states: dict[int, dict] = {}
    prepared_jobs: list[dict] = []
    job_locations: dict[int, tuple[int, int]] = {}
    job_ownership: dict[int, list[int]] = {}
    skipped: list[dict] = []

    for row_index in tqdm(selected_indices, desc="Chunking FLAN-T5 rows"):
        row = dataset[row_index]
        try:
            chunks = chunk_flan_t5_row(
                row,
                tokenizer,
                max_sentinels=settings["max_sentinels"],
                max_chunk_length=max_chunk_length,
                overlap_sentences=overlap_sentences,
                use_summary=settings.get("use_summary", True),
            )
            entity_count = len(row["demasked_words"])
            state = {
                "id": row.get("id", str(row_index)),
                "entities": [None] * entity_count,
                "assigned": set(),
                "decoded": [],
                "chunk_count": len(chunks),
            }
            row_states[row_index] = state
            for chunk in chunks:
                owned = chunk["effective_global_mask_indices"]
                if not owned:
                    continue
                job_index = len(prepared_jobs)
                prepared_jobs.append(
                    {
                        "row": job_index,
                        "id": f"{state['id']}:chunk-{chunk['chunk_index']}",
                        "source": chunk["source"],
                        "entity_count": len(owned),
                    }
                )
                job_locations[job_index] = (row_index, chunk["chunk_index"])
                job_ownership[job_index] = owned
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
            skipped.append({"row": row_index, "reason": reason})
            prediction_rows[row_positions[row_index]] = {
                "row": row_index,
                "id": row.get("id", str(row_index)),
                "status": "skipped",
                "reason": reason,
                "entities": [None] * len(row.get("demasked_words", [])),
            }

    for start in tqdm(
        range(0, len(prepared_jobs), settings["batch_size"]),
        desc="Generating FLAN-T5 chunks",
    ):
        batch = prepared_jobs[start : start + settings["batch_size"]]
        generated = predict_flan_t5_continuous_generation(
            model, tokenizer, batch, settings
        )
        for result in generated:
            row_index, _chunk_index = job_locations[result["row"]]
            state = row_states[row_index]
            chunk_owned = job_ownership[result["row"]]
            if len(chunk_owned) != len(result["entities"]):
                raise RuntimeError("Chunk prediction count does not match ownership")
            for global_index, entity in zip(chunk_owned, result["entities"]):
                if global_index in state["assigned"]:
                    raise RuntimeError(
                        f"Row {row_index} mask {global_index} was predicted twice"
                    )
                state["entities"][global_index] = entity
                state["assigned"].add(global_index)
            state["decoded"].append(result["decoded"])

    for row_index, state in row_states.items():
        expected = set(range(len(state["entities"])))
        if state["assigned"] != expected:
            missing = sorted(expected - state["assigned"])
            raise RuntimeError(f"Row {row_index} has unassigned masks: {missing}")
        prediction_rows[row_positions[row_index]] = {
            "row": row_index,
            "id": state["id"],
            "decoded": state["decoded"],
            "status": "ok",
            "entities": state["entities"],
            "chunk_count": state["chunk_count"],
        }

    if any(result is None for result in prediction_rows):
        raise RuntimeError("Prediction output does not cover every selected dataset row")
    return prediction_rows, skipped


def predict_dataset(
    model,
    tokenizer,
    dataset,
    predictor_name: str,
    settings: dict,
    indices: list[int] | None = None,
):
    """Generate predictions for selected original dataset rows.

    ``indices`` controls both the selected rows and their output order. When
    omitted, every dataset row is predicted in its natural order.
    """
    if predictor_name == "flan_t5_continuous_generation":
        return _predict_flan_t5_chunked_dataset(
            model, tokenizer, dataset, settings, indices
        )

    predict_batch = get_predictor(predictor_name)
    prepared, skipped = [], []
    selected_indices = _selected_indices(dataset, indices)
    row_positions = {row_index: position for position, row_index in enumerate(selected_indices)}
    prediction_rows = [None] * len(selected_indices)
    for row_index in selected_indices:
        row = dataset[row_index]
        try:
            source = make_flan_t5_source(row, settings)
            max_input_length = settings["max_input_length"]
            if max_input_length is not None:
                source_length = len(
                    tokenizer(
                        source,
                        truncation=True,
                        max_length=max_input_length + 1,
                    )["input_ids"]
                )
                if source_length > max_input_length:
                    raise ValueError(f"input has {source_length} tokens")
            prepared.append({"row": row_index, "id": row.get("id", str(row_index)), "source": source, "entity_count": len(row["demasked_words"])})
        except (KeyError, TypeError, ValueError) as error:
            reason = str(error)
            skipped.append({"row": row_index, "reason": reason})
            prediction_rows[row_positions[row_index]] = {
                "row": row_index,
                "id": row.get("id", str(row_index)),
                "status": "skipped",
                "reason": reason,
                "entities": [None] * len(row.get("demasked_words", [])),
            }
    for start in tqdm(range(0, len(prepared), settings["batch_size"]), desc="Generating"):
        for result in predict_batch(model, tokenizer, prepared[start:start + settings["batch_size"]], settings):
            prediction_rows[row_positions[result["row"]]] = result
    if any(result is None for result in prediction_rows):
        raise RuntimeError("Prediction output does not cover every selected dataset row.")
    return prediction_rows, skipped

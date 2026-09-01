"""Metrics and comparison helpers for MARS entity predictions."""

import re
import string
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path


ARTICLE_PATTERN = re.compile(r"\b(?:a|an|the)\b", flags=re.IGNORECASE)
CNN_ATTRIBUTION_PATTERN = re.compile(r"^(?:\(CNN\)|CNN\))(?=\S)", flags=re.IGNORECASE)
RECOVERABILITY_LABELS = ("TRUE", "FALSE", "GUESSABLE")
ENTITY_TYPE_ERROR_CATEGORIES = (
    "wrong_entity_correct_type",
    "ambiguous_alias_correct_type",
    "wrong_entity_type",
    "unknown_type",
)
UNKNOWN_TYPE = "UNKNOWN"
_TARGET_TYPES = "_evaluation_target_types"
_PREDICTION_TYPES = "_evaluation_prediction_types"
_EVIDENCE = "_evaluation_evidence"
_ENTITY_MENTIONS = "_evaluation_entity_mentions"



def _result_rows(results: list[dict]) -> list[int]:
    """Return unique original dataset rows represented by evaluation results."""
    rows = []
    for result in results:
        row = result.get("row")
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise ValueError("Each evaluation result requires a non-negative integer row")
        rows.append(row)
    if len(set(rows)) != len(rows):
        raise ValueError("Evaluation results contain duplicate dataset rows")
    return rows


def _fill_masks(
    masked_text: str, replacements: list[str | None]
) -> tuple[str, list[tuple[int, int] | None]]:
    """Fill a masked template and return each replacement's character span."""
    parts = masked_text.split("<mask>")
    if len(parts) != len(replacements) + 1:
        raise ValueError("Mask/entity count differs")

    text_parts, spans, cursor = [], [], 0
    for prefix, replacement in zip(parts[:-1], replacements):
        value = "" if replacement is None else str(replacement)
        text_parts.extend((prefix, value))
        cursor += len(prefix)
        spans.append((cursor, cursor + len(value)) if value else None)
        cursor += len(value)
    text_parts.append(parts[-1])
    return "".join(text_parts), spans


def _entity_type_at_span(doc, span: tuple[int, int] | None) -> str:
    """Return the best spaCy label for a filled mask span."""
    if span is None:
        return UNKNOWN_TYPE
    start, end = span
    exact = [entity for entity in doc.ents if (entity.start_char, entity.end_char) == span]
    if exact:
        return exact[0].label_

    # spaCy can include neighbouring name tokens in one entity. Prefer the
    # smallest entity that fully contains the mask, then any overlapping one.
    containing = [
        entity for entity in doc.ents
        if entity.start_char <= start and end <= entity.end_char
    ]
    if containing:
        return min(containing, key=lambda entity: entity.end_char - entity.start_char).label_
    overlapping = [
        entity for entity in doc.ents
        if max(start, entity.start_char) < min(end, entity.end_char)
    ]
    if not overlapping:
        return UNKNOWN_TYPE
    return max(
        overlapping,
        key=lambda entity: min(end, entity.end_char) - max(start, entity.start_char),
    ).label_


def _entity_types_at_spans(doc, spans: list[tuple[int, int] | None]) -> list[str]:
    return [_entity_type_at_span(doc, span) for span in spans]


def _spacy_entities(doc, *, deduplicate: bool) -> list[dict]:
    """Convert spaCy entities to evaluator records."""
    records, seen = [], set()
    for entity in doc.ents:
        normalized = complete_normalize(entity.text)
        key = (normalized, entity.label_)
        if not normalized or (deduplicate and key in seen):
            continue
        seen.add(key)
        records.append({
            "entity": entity.text,
            "entity_type": entity.label_,
            "source": "spacy",
            "start_char": entity.start_char,
            "end_char": entity.end_char,
        })
    return records


def _evidence_entities(doc) -> list[dict]:
    """Return unique typed entities, retaining their first mention."""
    return _spacy_entities(doc, deduplicate=True)


def _entity_mentions(doc) -> list[dict]:
    """Return every spaCy mention, including repeated entities."""
    return _spacy_entities(doc, deduplicate=False)


def prepare_results(
    results: list[dict],
    model_name: str = "en_core_web_trf",
    batch_size: int = 8,
) -> list[dict]:
    """Annotate targets, predictions, and source-text evidence with spaCy.

    Dataset ``entity_types`` are deliberately ignored: all types used by this
    evaluator come from the selected spaCy pipeline.
    """
    if batch_size < 1:
        raise ValueError("spaCy batch_size must be at least 1")
    _result_rows(results)
    if not results:
        return results
    prepared = []
    for result in results:
        for field in (
            "original_text", "masked_text", "target_entities", "prediction_entities"
        ):
            if field not in result:
                raise ValueError(f"evaluation_metrics requires the {field} field")
        if len(result["target_entities"]) != len(result["prediction_entities"]):
            raise ValueError("prediction_entities must contain one value per target entity")

        target_text, target_spans = _fill_masks(
            result["masked_text"], result["target_entities"]
        )
        if target_text != result["original_text"]:
            raise ValueError(
                f"original_text does not match masked_text and targets at row "
                f"{result.get('row', '[unknown]')}"
            )
        prediction_text, prediction_spans = _fill_masks(
            result["masked_text"], result["prediction_entities"]
        )
        prepared.append((target_spans, prediction_text, prediction_spans))

    import spacy
    spacy.prefer_gpu()
    nlp = spacy.load(model_name)
    original_docs = nlp.pipe(
        (result["original_text"] for result in results), batch_size=batch_size
    )
    prediction_docs = nlp.pipe(
        (item[1] for item in prepared), batch_size=batch_size
    )
    for result, item, original_doc, prediction_doc in zip(
        results, prepared, original_docs, prediction_docs, strict=True
    ):
        target_spans, _, prediction_spans = item
        result[_TARGET_TYPES] = _entity_types_at_spans(original_doc, target_spans)
        result[_PREDICTION_TYPES] = _entity_types_at_spans(
            prediction_doc, prediction_spans
        )
        result[_EVIDENCE] = _evidence_entities(original_doc)
        result[_ENTITY_MENTIONS] = _entity_mentions(original_doc)
    return results


def _prepared_value(result: dict, key: str):
    try:
        return result[key]
    except KeyError as error:
        raise ValueError("Call prepare_results before evaluating") from error


def _result_entity_types(result: dict) -> list[str]:
    """Return spaCy-derived target types prepared for evaluation."""
    return _prepared_value(result, _TARGET_TYPES)


def _result_prediction_types(result: dict) -> list[str]:
    """Return spaCy-derived prediction types prepared for evaluation."""
    return _prepared_value(result, _PREDICTION_TYPES)


def normalize_target_entity(text: str | None) -> str:
    """Normalize a target entity, dropping the dataset's trailing CNN attribution."""
    if text is None:
        return ""
    return complete_normalize(CNN_ATTRIBUTION_PATTERN.sub("", text))


def complete_normalize(text: str | None) -> str:
    """Canonical comparison form: lowercase, punctuation/article-free text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?:['’]s|['’])\s*$", "", text)
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = ARTICLE_PATTERN.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str | None) -> list[str]:
    return complete_normalize(text).split()


def _acronym(tokens: list[str]) -> str:
    return "".join(token[0] for token in tokens if token)


def _organization_shortened_name(entity_1: str, entity_2: str) -> bool:
    """Whether prediction first is a short form of entity entity_2."""
    entity_1_tokens, entity_2_tokens = _tokens(entity_1), _tokens(entity_2)
    added = len(entity_2_tokens) - len(entity_1_tokens)
    return (
        0 < added <= 2
        and (
            entity_2_tokens[:len(entity_1_tokens)] == entity_1_tokens
            or entity_2_tokens[-len(entity_1_tokens):] == entity_1_tokens
        )
    )


def _alias_kind(prediction: str, entity: str, entity_type: str | None) -> str | None:
    """Return why prediction can refer to entity, otherwise None."""
    prediction_tokens, entity_tokens = _tokens(prediction), _tokens(entity)
    if not prediction_tokens or not entity_tokens:
        return None
    if prediction_tokens == entity_tokens:
        return "equivalent"

    # abbreviations
    prediction_compact = "".join(prediction_tokens)
    entity_compact = "".join(entity_tokens)
    if (len(entity_tokens) > 1 and prediction_compact == _acronym(entity_tokens)) or \
       (len(prediction_tokens) > 1 and entity_compact == _acronym(prediction_tokens)):
        return "abbreviation"

    # organization short forms
    if entity_type == "ORG" and _organization_shortened_name(prediction, entity):
        return "organization_shortened_name"

    # human names
    if entity_type == "PERSON" and len(prediction_tokens) < len(entity_tokens):
        contains_name_part = any(
            entity_tokens[start:start + len(prediction_tokens)] == prediction_tokens
            for start in range(len(entity_tokens) - len(prediction_tokens) + 1)
        )
        if contains_name_part:
            return "name_part"
    if entity_type == "PERSON" and len(prediction_tokens) == len(entity_tokens):
        if all(
            predicted_word == target_word
            or (len(predicted_word) == 1 and predicted_word == target_word[0])
            or (len(target_word) == 1 and target_word == predicted_word[0])
            for predicted_word, target_word in zip(prediction_tokens, entity_tokens)):
            return "name_initial"
    return None


def find_aliases(prediction: str | None, candidates: list[dict]) -> list[dict]:
    """Find candidate entities to which a prediction can be an alias.

    Candidates contain ``entity`` and may contain ``entity_type`` and ``index``.
    """
    if not prediction:
        return []
    matches = []
    for candidate in candidates:
        kind = _alias_kind(
            prediction,
            candidate["entity"],
            candidate.get("match_entity_type", candidate.get("entity_type")),
        )
        if kind:
            match = {
                "index": candidate.get("index"),
                "entity": candidate["entity"],
                "entity_type": candidate.get("entity_type"),
                "kind": kind,
            }
            for key in ("source", "start_char", "end_char"):
                if key in candidate:
                    match[key] = candidate[key]
            matches.append(match)
    return matches


def target_aliases(
    prediction: str,
    target: str,
    target_type: str,
    candidates: list[dict],
) -> tuple[list[dict], bool]:
    """Match normally, with reverse matching reserved for person-name parts."""
    reverse = target_type == "PERSON" and _alias_kind(target, prediction, target_type) is not None
    return find_aliases(target if reverse else prediction, candidates), reverse


def _distinct_alias_entities(
    prediction: str, alias_matches: list[dict], entity_type: str
) -> list[str]:
    """Return distinct likely referents behind alias matches, not mask mentions.

    For a person name part, longer matched names are the candidate referents.
    """
    prediction_size = len(_tokens(prediction))
    entity_forms = [complete_normalize(match["entity"]) for match in alias_matches]
    if entity_type == "PERSON":
        expanded_forms = [form for form in entity_forms if len(form.split()) > prediction_size]
        if expanded_forms:
            return sorted(set(expanded_forms))
    if (entity_type or "").upper() == "GPE":
        forms = sorted(set(entity_forms))
        initialisms = [
            form for form in forms
            if len(form.split()) >= 2
            and all(len(token) == 1 for token in form.split())
        ]
        expanded = [form for form in forms if form not in initialisms]
        unresolved_initialisms = [
            initialism for initialism in initialisms
            if not any(
                "".join(initialism.split()) == _acronym(expansion.split())
                for expansion in expanded
            )
        ]
        return sorted(set(expanded + unresolved_initialisms))
    if any(match["kind"] == "organization_shortened_name" for match in alias_matches):
        expanded_forms = [
            form for form in entity_forms
            if len(form.split()) > len(_tokens(prediction))
        ]
        if expanded_forms:
            return sorted(set(expanded_forms))
    return sorted(set(entity_forms))


MONTH_PATTERN = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
RELATIVE_DATE_PATTERN = re.compile(
    r"\b(?:today|tomorrow|yesterday|tonight|this|last|next|later|earlier|previous|"
    r"following)\s+(?:day|week|month|year|morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(r"(?:[$€£¥]|\b(?:usd|eur|gbp|dollars?|euros?|pounds?)\b)", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"(?:%|\bpercent\b)", re.IGNORECASE)
ORDINAL_PATTERN = re.compile(
    r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[,.]\d+)?\b")
TIME_PATTERN = re.compile(r"\b(?:\d{1,2}:\d{2}|a\.m\.|p\.m\.|am|pm|noon|midnight)\b", re.IGNORECASE)


def infer_rule_entity_type(text: str | None) -> str:
    """Return only high-confidence entity types; named entities remain UNKNOWN."""
    if not text:
        return UNKNOWN_TYPE
    if PERCENT_PATTERN.search(text):
        return "PERCENT"
    if MONEY_PATTERN.search(text):
        return "MONEY"
    if TIME_PATTERN.search(text):
        return "TIME"
    if (
        MONTH_PATTERN.search(text)
        or DATE_PATTERN.search(text)
        or RELATIVE_DATE_PATTERN.search(text)
    ):
        return "DATE"
    if ORDINAL_PATTERN.fullmatch(text.strip()):
        return "ORDINAL"
    if NUMBER_PATTERN.fullmatch(text.strip().replace(",", "")):
        return "CARDINAL"
    return UNKNOWN_TYPE


def _target_entity_candidates(result: dict) -> list[dict]:
    return [
        {"index": index, "entity": entity, "entity_type": entity_type}
        for index, (entity, entity_type) in enumerate(
            zip(result["target_entities"], _result_entity_types(result))
        )
    ]


def _all_entity_candidates(result: dict) -> list[dict]:
    """Return every available entity mention that may serve as evidence."""
    candidates = _target_entity_candidates(result)
    candidates.extend(result.get(_EVIDENCE, []))
    return candidates


def _same_type_candidates(result: dict, target_type: str) -> list[dict]:
    return [
        candidate for candidate in _all_entity_candidates(result)
        if candidate.get("entity_type") == target_type
    ]


LIST_SEPARATOR_PATTERN = re.compile(r"^\s*(?:[,;/&]|\band\b|\bor\b|\s)+\s*$", re.IGNORECASE)


def _same_type_list_gap(
    result: dict,
    left_span: tuple[int, int],
    right_span: tuple[int, int],
    entity_type: str,
) -> bool:
    """Check that a gap contains only list syntax and same-type entities."""
    gap_start, gap_end = left_span[1], right_span[0]
    remaining = list(result["original_text"][gap_start:gap_end])

    for mention in _prepared_value(result, _ENTITY_MENTIONS):
        start, end = mention["start_char"], mention["end_char"]
        if not (gap_start <= start and end <= gap_end):
            continue
        if mention["entity_type"] != entity_type:
            return False
        remaining[start - gap_start:end - gap_start] = " " * (end - start)

    return bool(LIST_SEPARATOR_PATTERN.fullmatch("".join(remaining)))


def find_entity_groups(result: dict) -> list[list[int]]:
    """Group same-type masks connected by a spaCy-recognized entity list.

    Unmasked entities may occur between masks, but they must have the same
    spaCy type. After removing those mentions, only list punctuation and
    conjunctions may remain.
    """
    count = len(result["target_entities"])
    original_text, spans = _fill_masks(
        result["masked_text"], result["target_entities"]
    )
    if original_text != result["original_text"] or any(span is None for span in spans):
        raise ValueError("Cannot align target entities with original_text")

    entity_types = _result_entity_types(result)
    groups, index = [], 0
    while index < count:
        group = [index]
        while index + 1 < count:
            same_type = entity_types[index] == entity_types[index + 1]
            list_gap = same_type and _same_type_list_gap(
                result, spans[index], spans[index + 1], entity_types[index]
            )
            if not list_gap:
                break
            index += 1
            group.append(index)
        groups.append(group)
        index += 1
    return groups


def mask_context_snippet(
    masked_text: str,
    mask_index: int,
    left_words: int = 5,
    right_words: int = 2,
) -> str:
    """Return a compact masked-text snippet around one entity position."""
    masks = list(re.finditer(r"<mask>", masked_text))
    if not 0 <= mask_index < len(masks):
        raise IndexError(f"Mask index {mask_index} is outside the masked text.")
    mask = masks[mask_index]
    left = masked_text[:mask.start()].split()[-left_words:]
    right = masked_text[mask.end():].split()[:right_words]
    return " ".join(left + ["<mask>"] + right)


def _matches_target_in_row(result: dict, prediction: str, target_index: int) -> bool:
    target = result["target_entities"][target_index]
    target_type = _result_entity_types(result)[target_index]
    if complete_normalize(prediction) == normalize_target_entity(target):
        return True
    aliases, reverse = target_aliases(
        prediction,
        target,
        target_type,
        _same_type_candidates(result, target_type),
    )
    return (
        target_index in {match["index"] for match in aliases}
        and len(
            _distinct_alias_entities(
                target if reverse else prediction, aliases, target_type
            )
        ) == 1
    )


def entity_set_alignment(result: dict) -> dict:
    """Return maximum one-to-one matching for each list group and singleton."""
    by_target, groups = {}, []
    predictions = result["prediction_entities"]
    for indices in find_entity_groups(result):
        prediction_indices = [index for index in indices if predictions[index]]
        compatible = {
            prediction_index: [
                target_index for target_index in indices
                if _matches_target_in_row(
                    result, predictions[prediction_index], target_index
                )
            ]
            for prediction_index in prediction_indices
        }
        assigned = {}

        def assign(prediction_index: int, seen: set[int]) -> bool:
            for target_index in compatible[prediction_index]:
                if target_index in seen:
                    continue
                seen.add(target_index)
                if target_index not in assigned or assign(assigned[target_index], seen):
                    assigned[target_index] = prediction_index
                    return True
            return False

        for prediction_index in prediction_indices:
            assign(prediction_index, set())
        detail = {
            "mask_indices": indices,
            "predictions": [predictions[index] for index in indices],
            "target_entities": [result["target_entities"][index] for index in indices],
            "matches": [
                {"prediction_mask": prediction_index, "target_mask": target_index}
                for target_index, prediction_index in sorted(assigned.items())
            ],
            "correct_entities": len(assigned),
            "incorrect_entities": len(indices) - len(assigned),
        }
        groups.append(detail)
        for target_index in indices:
            by_target[target_index] = {**detail, "prediction_mask": assigned.get(target_index)}
    return {"by_target": by_target, "groups": groups}


def collect_entity_information(result: dict) -> dict:
    """Collect all target-level evidence before assigning an error category."""
    alignment = entity_set_alignment(result)
    targets = result["target_entities"]
    target_types = _result_entity_types(result)
    prediction_types = _result_prediction_types(result)
    masked_text = result.get("masked_text", "")
    contexts_available = len(list(re.finditer(r"<mask>", masked_text))) == len(targets)
    entities = []
    for index, (target, target_type) in enumerate(zip(targets, target_types)):
        set_detail = alignment["by_target"][index]
        matched_mask = set_detail["prediction_mask"]
        direct_prediction = result["prediction_entities"][index]
        matched_prediction = (
            result["prediction_entities"][matched_mask] if matched_mask is not None else None
        )
        matched_prediction_type = (
            prediction_types[matched_mask] if matched_mask is not None else UNKNOWN_TYPE
        )
        comparison_prediction = matched_prediction or direct_prediction
        alias_matches, alias_reverse = ([], False)
        if comparison_prediction:
            alias_matches, alias_reverse = target_aliases(
                comparison_prediction, target, target_type,
                _same_type_candidates(result, target_type),
            )
        entities.append({
            "mask_index": index,
            "target": target,
            "target_type": target_type,
            "prediction": direct_prediction,
            "prediction_type": prediction_types[index],
            "matched_prediction": matched_prediction,
            "matched_prediction_mask": matched_mask,
            "matched_prediction_type": matched_prediction_type,
            "strict_exact": bool(direct_prediction)
            and complete_normalize(direct_prediction) == normalize_target_entity(target),
            "matched_exact": bool(matched_prediction)
            and complete_normalize(matched_prediction) == normalize_target_entity(target),
            "recovered": matched_mask is not None,
            "target_presence": target_presence_kind(
                target,
                target_type,
                result.get("summary", ""),
                result.get("masked_text", ""),
            ),
            "context": mask_context_snippet(masked_text, index) if contexts_available else "",
            "entity_set": set_detail if len(set_detail["mask_indices"]) > 1 else None,
            "alias_matches": alias_matches,
            "alias_reverse": alias_reverse,
        })
    return {"alignment": alignment, "entities": entities}


def _contains_token_sequence(tokens: list[str], candidate: list[str]) -> bool:
    width = len(candidate)
    return bool(candidate) and any(
        tokens[index:index + width] == candidate
        for index in range(len(tokens) - width + 1)
    )


def target_alias_forms(target: str, target_type: str) -> list[list[str]]:
    """Generate deterministic forms that make a target observable in context."""
    tokens = _tokens(target)
    forms = [tokens]
    if target_type == "PERSON" and len(tokens) > 1:
        # Full names can be observable through first names, surnames, or other
        # contiguous name parts in the summary/document.
        forms.extend(
            tokens[start:end]
            for start in range(len(tokens))
            for end in range(start + 1, len(tokens) + 1)
            if end - start < len(tokens)
        )
    if len(tokens) > 1:
        forms.append([token[0] for token in tokens if token])
    unique = []
    for form in forms:
        if form and form not in unique:
            unique.append(form)
    return unique


def target_presence_kind(target: str, target_type: str, summary: str, masked_text: str) -> str:
    """Return whether context contains the target verbatim, via alias, or not at all."""
    context_tokens = _tokens(f"{summary} {masked_text}")
    forms = target_alias_forms(target, target_type)
    if forms and _contains_token_sequence(context_tokens, forms[0]):
        return "verbatim"
    if any(_contains_token_sequence(context_tokens, form) for form in forms[1:]):
        return "alias"
    return "missing"


def recoverability_errors(results: list[dict]) -> dict:
    """Report accuracy by the dataset's per-target recoverability annotation."""
    totals = Counter()
    recovered = Counter()
    annotated = False
    for result in results:
        flags = result.get("entity_recoverable")
        if flags is None:
            continue
        annotated = True
        if len(flags) != len(result["target_entities"]):
            raise ValueError("entity_recoverable must have one label per target entity")
        for flag, entity in zip(flags, collect_entity_information(result)["entities"]):
            label = str(flag).upper()
            if label not in RECOVERABILITY_LABELS:
                raise ValueError(f"Unknown entity_recoverable label: {flag!r}")
            totals[label] += 1
            recovered[label] += entity["recovered"]
    if not annotated:
        return {}
    return {
        label.lower(): {
            "targets": totals[label],
            "recovered": recovered[label],
            "recovered_rate": recovered[label] / max(totals[label], 1),
        }
        for label in RECOVERABILITY_LABELS
    }


def _classify_alias(
    result: dict,
    mask_index: int,
    prediction: str,
    prediction_type: str,
    target: str,
    target_type: str,
) -> tuple[dict | None, list[dict], list[dict]]:
    """Classify a target alias and return reusable same-type evidence."""
    candidates = _same_type_candidates(result, target_type)
    matches, reverse = target_aliases(prediction, target, target_type, candidates)
    if mask_index not in {match.get("index") for match in matches}:
        return None, candidates, matches

    referent = target if reverse else prediction
    distinct = _distinct_alias_entities(referent, matches, target_type)
    if len(distinct) == 1:
        return {
            "category": "resolved_alias",
            "prediction_type": prediction_type,
            "alias_matches": matches,
            "resolved_entity": distinct[0],
        }, candidates, matches
    if len(distinct) > 1:
        return {
            "category": "ambiguous_alias_correct_type",
            "prediction_type": prediction_type,
            "alias_matches": matches,
            "distinct_alias_entities": distinct,
        }, candidates, matches
    return None, candidates, matches


def _classify_unknown_target_type(
    result: dict, prediction: str, prediction_type: str
) -> dict:
    candidates = [
        candidate for candidate in _all_entity_candidates(result)
        if candidate.get("entity_type") not in {None, UNKNOWN_TYPE}
    ]
    matches = find_aliases(prediction, candidates)
    matched_types = {match["entity_type"] for match in matches}
    if prediction_type == UNKNOWN_TYPE:
        prediction_type = (
            next(iter(matched_types)) if len(matched_types) == 1
            else infer_rule_entity_type(prediction)
        )
    return {
        "category": "unknown_type",
        "prediction_type": prediction_type,
        "target_type_unknown": True,
        "matched_entities": matches,
    }


def _classify_typed_mismatch(
    result: dict,
    mask_index: int,
    prediction: str,
    target_type: str,
    prediction_type: str,
    same_type_candidates: list[dict],
    same_type_matches: list[dict],
) -> dict:
    same_type_other = [
        match for match in same_type_matches if match.get("index") != mask_index
    ]
    if not same_type_other:
        normalized = complete_normalize(prediction)
        same_type_other = [
            candidate for candidate in same_type_candidates
            if candidate.get("index") != mask_index
            and complete_normalize(candidate["entity"]) == normalized
        ]

    if prediction_type != UNKNOWN_TYPE:
        detail = {
            "category": (
                "wrong_entity_correct_type"
                if prediction_type == target_type
                else "wrong_entity_type"
            ),
            "prediction_type": prediction_type,
        }
        if same_type_other:
            detail["matched_entities"] = same_type_other
        return detail

    if same_type_other:
        return {
            "category": "wrong_entity_correct_type",
            "prediction_type": target_type,
            "matched_entities": same_type_other,
        }

    other_type_candidates = [
        candidate for candidate in _all_entity_candidates(result)
        if candidate.get("entity_type") not in {None, target_type}
    ]
    other_type_matches = find_aliases(prediction, other_type_candidates)
    rule_type = infer_rule_entity_type(prediction)
    if rule_type == target_type and rule_type != UNKNOWN_TYPE:
        return {"category": "wrong_entity_correct_type", "prediction_type": rule_type}
    if other_type_matches or (rule_type != UNKNOWN_TYPE and rule_type != target_type):
        matched_types = {match["entity_type"] for match in other_type_matches}
        prediction_type = next(iter(matched_types)) if len(matched_types) == 1 else rule_type
        return {
            "category": "wrong_entity_type",
            "prediction_type": prediction_type,
            "matched_entities": other_type_matches,
        }
    return {"category": "unknown_type", "prediction_type": rule_type}


def classify_mask_error(
    result: dict,
    mask_index: int,
    alignment: dict | None = None,
    entity_info: dict | None = None,
) -> dict:
    """Assign one deterministic category to a target prediction."""
    collected = collect_entity_information(result) if entity_info is None else None
    entity = collected["entities"][mask_index] if entity_info is None else entity_info
    alignment = (collected["alignment"] if collected else alignment) or entity_set_alignment(result)
    set_detail = alignment["by_target"][mask_index]
    target = entity["target"]
    target_type = entity["target_type"]
    prediction = entity["prediction"]
    prediction_type = entity["prediction_type"]

    if len(set_detail["mask_indices"]) > 1:
        category = (
            "resolved_entity_set"
            if set_detail["prediction_mask"] is not None
            else "unmatched_entity_in_set"
        )
        return {
            "category": category,
            "prediction_type": (
                entity["matched_prediction_type"]
                if set_detail["prediction_mask"] is not None
                else prediction_type
            ),
            "entity_set": set_detail,
        }
    if not complete_normalize(prediction):
        return {"category": "skipped_entity", "prediction_type": UNKNOWN_TYPE}
    if complete_normalize(prediction) == normalize_target_entity(target):
        return {"category": "exact_match", "prediction_type": prediction_type}

    alias_detail, same_type_candidates, same_type_matches = _classify_alias(
        result, mask_index, prediction, prediction_type, target, target_type
    )
    if alias_detail:
        return alias_detail
    if target_type == UNKNOWN_TYPE:
        return _classify_unknown_target_type(result, prediction, prediction_type)
    return _classify_typed_mismatch(
        result,
        mask_index,
        prediction,
        target_type,
        prediction_type,
        same_type_candidates,
        same_type_matches,
    )


def error_evidence(detail: dict) -> str:
    """Return compact category-specific evidence for manual error inspection."""
    if detail.get("distinct_alias_entities"):
        return "alias candidates: " + ", ".join(detail["distinct_alias_entities"])
    if detail.get("matched_entities"):
        matches = detail["matched_entities"]
        return "matched entities: " + ", ".join(
            (
                f"{match.get('entity', '[unknown]')} ({match['entity_type']})"
                if match.get("entity_type") else match.get("entity", "[unknown]")
            )
            for match in matches
        )
    if detail.get("entity_set"):
        entity_set = detail["entity_set"]
        return f"set matches: {entity_set['correct_entities']}/{len(entity_set['mask_indices'])}"
    if detail.get("target_type_unknown"):
        return "target type is UNKNOWN"
    if detail.get("prediction_type") and detail["prediction_type"] != UNKNOWN_TYPE:
        return f"prediction type: {detail['prediction_type']}"
    return ""


def collect_errors(results: list[dict]) -> list[dict]:
    """Return one standardized record for every unrecovered target."""
    errors = []
    for result in results:
        flags = result.get("entity_recoverable")
        targets = result["target_entities"]
        if flags is not None and len(flags) != len(targets):
            raise ValueError("entity_recoverable must have one label per target entity")
        collected = collect_entity_information(result)
        summary_word_count = len(re.findall(r"\b\w+\b", result.get("summary", "")))
        for entity in collected["entities"]:
            if entity["recovered"]:
                continue
            index = entity["mask_index"]
            detail = classify_mask_error(result, index, collected["alignment"], entity)
            error = {
                "row": result["row"],
                "index": index,
                "target": entity["target"],
                "target_type": entity["target_type"],
                "predicted": entity["prediction"],
                "prediction_type": detail["prediction_type"],
                "error_type": detail["category"],
                "target_presence": entity["target_presence"],
                "entity_recoverable": str(flags[index]).upper() if flags is not None else None,
                "context": entity["context"],
                "target_count": len(targets),
                "summary_word_count": summary_word_count,
                "targets_per_summary_word": len(targets) / max(summary_word_count, 1),
                "evidence": error_evidence(detail),
            }
            if flags is None:
                del error["entity_recoverable"]
            errors.append(error)
    return errors


def entity_type_errors(results: list[dict]) -> dict:
    """Classify non-recovered targets after recovery scoring has been applied."""
    labels = []
    for result in results:
        collected = collect_entity_information(result)
        alignment = collected["alignment"]
        for entity in collected["entities"]:
            detail = classify_mask_error(result, entity["mask_index"], alignment, entity)
            if detail["category"] in ENTITY_TYPE_ERROR_CATEGORIES:
                labels.append(detail["category"])
    counts = Counter(labels)
    total = sum(counts.values())
    return {
        "classified_entity_errors": total,
        "counts": {name: counts[name] for name in ENTITY_TYPE_ERROR_CATEGORIES},
        "rates": {
            name: counts[name] / max(total, 1) for name in ENTITY_TYPE_ERROR_CATEGORIES
        },
    }


def exact_match(results: list[dict]) -> dict:
    """Recovery metric: strict exact matches and set/alias-aware total accuracy."""
    target_total = predicted_total = strict_exact = recovered = 0
    aliases = group_matches = group_count = 0
    missing_predictions = unrecovered_targets = sequences_recovered = 0
    for result in results:
        collected = collect_entity_information(result)
        alignment = collected["alignment"]
        predicted_total += sum(
            prediction is not None for prediction in result["prediction_entities"]
        )
        target_total += len(result["target_entities"])
        missing_predictions += sum(not prediction for prediction in result["prediction_entities"])
        recovered += sum(group["correct_entities"] for group in alignment["groups"])
        unrecovered_targets += sum(group["incorrect_entities"] for group in alignment["groups"])
        for group in alignment["groups"]:
            if len(group["mask_indices"]) > 1:
                group_count += 1
                group_matches += group["correct_entities"]
        strict_exact += sum(entity["strict_exact"] for entity in collected["entities"])
        for entity in collected["entities"]:
            if not entity["recovered"] or entity["matched_exact"]:
                continue
            alias_matches = entity["alias_matches"]
            referent = entity["target"] if entity["alias_reverse"] else entity["matched_prediction"]
            if (
                entity["mask_index"] in {match["index"] for match in alias_matches}
                and len(_distinct_alias_entities(
                    referent, alias_matches, entity["target_type"]
                )) == 1
            ):
                aliases += 1
        sequences_recovered += all(
            group["incorrect_entities"] == 0 for group in alignment["groups"]
        )

    return {
        "target_entities": target_total,
        "exact_matches": strict_exact,
        "total_correct": recovered,
        "missing_predictions": missing_predictions,
        "unrecovered_targets": unrecovered_targets,
        "recovered_aliases": aliases,
        "matches_in_groups": group_matches,
        "entity_groups": group_count,
        "predicted_entities": predicted_total,
        "exact_match_accuracy": strict_exact / max(target_total, 1),
        "total_accuracy": recovered / max(target_total, 1),
        "sequence_accuracy": sequences_recovered / max(len(results), 1),
    }


@lru_cache(maxsize=1)
def _load_modernbert_embedder():
    """Load the encoder only when the semantic-similarity metric is selected."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "modernbert_swap_similarity requires torch and transformers."
        ) from error

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _mean_pool_embeddings(last_hidden_state, attention_mask):
    """Mean-pool non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def modernbert_swap_similarity(results: list[dict]) -> dict:
    """Measure semantic similarity between a gold entity and its slot prediction.

    "Before swap" is the source-side ``target_entities`` string and "after
    swap" is the corresponding generated ``prediction_entities`` string.
    Missing predictions are reported separately rather than treated as an
    arbitrary embedding. Cosine similarity is computed from mean-pooled
    ModernBERT-base final-layer token embeddings.
    """
    pairs: list[tuple[str, str, str]] = []
    missing_predictions = 0
    for result in results:
        targets = result["target_entities"]
        predictions = result["prediction_entities"]
        if len(targets) != len(predictions):
            raise ValueError(
                "prediction_entities must contain one entry per target entity"
            )
        types = _result_entity_types(result)
        for target, prediction, entity_type in zip(targets, predictions, types):
            if not complete_normalize(prediction):
                missing_predictions += 1
                continue
            pairs.append((str(target), str(prediction), entity_type))

    if not pairs:
        return {
            "compared_entities": 0,
            "missing_predictions": missing_predictions,
            "mean_cosine_similarity": 0.0,
            "median_cosine_similarity": 0.0,
            "entity_type_mean_cosine_similarity": {},
        }

    import torch

    model, tokenizer, device = _load_modernbert_embedder()
    similarities: list[float] = []
    by_type: dict[str, list[float]] = {}
    batch_size = 32
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        texts = [text for pair in batch for text in pair[:2]]
        encoded = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            embeddings = _mean_pool_embeddings(
                model(**encoded).last_hidden_state, encoded["attention_mask"]
            )
        values = torch.nn.functional.cosine_similarity(
            embeddings[0::2], embeddings[1::2], dim=1
        ).cpu().tolist()
        for (_, _, entity_type), similarity in zip(batch, values):
            similarity = float(similarity)
            similarities.append(similarity)
            by_type.setdefault(entity_type, []).append(similarity)

    ordered = sorted(similarities)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "compared_entities": len(similarities),
        "missing_predictions": missing_predictions,
        "mean_cosine_similarity": sum(similarities) / len(similarities),
        "median_cosine_similarity": median,
        "entity_type_mean_cosine_similarity": {
            entity_type: sum(values) / len(values)
            for entity_type, values in sorted(by_type.items())
        },
    }


def write_evaluation_artifacts(
    results: list[dict],
    metric_names: list[str],
    output_dir: str | Path,
    *,
    metadata: dict | None = None,
    examples_skipped_during_prediction: int = 0,
) -> tuple[dict, list[dict]]:
    """Evaluate results and persist the standard ``metrics.json`` and ``errors.json``.

    The output schema is the one used by ``ipynbs/evaluate_predictions.ipynb``.
    ``errors.json`` contains the complete per-slot records produced by
    :func:`collect_errors`, so a report can use it as its single source of
    hover-detail data.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _result_rows(results)
    errors = collect_errors(results)
    metrics_document = {
        "metadata": metadata or {},
        "examples_evaluated": len(results),
        "examples_skipped_during_prediction": examples_skipped_during_prediction,
        "metrics": evaluate(results, metric_names),
    }
    import json

    (destination / "metrics.json").write_text(
        json.dumps(metrics_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics_document, errors


METRICS = {
    "exact_match": exact_match,
    "entity_type_errors": entity_type_errors,
    "recoverability_errors": recoverability_errors,
    "modernbert_swap_similarity": modernbert_swap_similarity,
}


def evaluate(results: list[dict], metric_names: list[str]) -> dict:
    metrics = {}
    for name in metric_names:
        try:
            metric = METRICS[name]
        except KeyError as error:
            supported = ", ".join(sorted(METRICS))
            raise ValueError(f"Unknown metric {name!r}. Supported: {supported}") from error
        metrics[name] = metric(results)
    return metrics

from __future__ import annotations

import csv
import hashlib
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sg_legal_rag.ingestion.splits import normalize_case_family
from sg_legal_rag.ingestion.validation import EXPECTED_FIELDS, REQUIRED_TEXT_FIELDS

from .benchmark import QueryRecord, add_query, load_test_urls, percentile


@dataclass(frozen=True)
class HistoricalContext:
    case_key: str
    raw_case: str
    source_url: str
    source_reference: str
    source_year: int
    text: str
    original_chars: int
    identifier_matched: bool
    digest: str


@dataclass(frozen=True)
class CorpusRepairDataset:
    case_keys: tuple[str, ...]
    case_texts: tuple[str, ...]
    contexts: tuple[HistoricalContext, ...]
    context_case_ids: np.ndarray
    profiles: tuple[str, ...]
    historical_case_ids: frozenset[int]
    queries_by_mode: dict[str, list[QueryRecord]]
    audit: dict[str, Any]
    test_urls: frozenset[str]


def normalize_space(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def canonical_case_key(text: str) -> str:
    return normalize_space(text).casefold()


def extract_context_window(paragraph: str, cited_case: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 1:
        raise ValueError("maximum passage length must be positive")
    text = normalize_space(paragraph)
    target = normalize_space(cited_case)
    if not text:
        raise ValueError("historical context cannot be empty")

    position = text.casefold().find(target.casefold()) if target else -1
    matched = position >= 0
    if len(text) <= max_chars:
        return text, matched
    if not matched:
        return text[:max_chars].rstrip(), False

    target_midpoint = position + len(target) // 2
    start = max(0, target_midpoint - max_chars // 2)
    start = min(start, len(text) - max_chars)
    return text[start : start + max_chars].strip(), True


def context_digest(text: str) -> str:
    return hashlib.sha256(normalize_space(text).encode("utf-8")).hexdigest()


def build_case_profile(
    case_text: str,
    contexts: list[HistoricalContext],
    *,
    max_passages: int,
    max_identifier_chars: int,
    max_context_chars: int,
    max_total_chars: int,
) -> tuple[str, int]:
    if min(max_passages, max_identifier_chars, max_context_chars, max_total_chars) < 1:
        raise ValueError("profile limits must be positive")

    bounded_identifier = normalize_space(case_text)[:max_identifier_chars].rstrip()
    profile = f"CASE IDENTIFIER\n{bounded_identifier}"[:max_total_chars]
    included = 0
    for context in contexts[:max_passages]:
        label = f"HISTORICAL CITATION CONTEXT {included + 1}"
        separator = f"\n\n{label}\n"
        available = max_total_chars - len(profile) - len(separator)
        if available < 1:
            break
        excerpt, _ = extract_context_window(
            context.text,
            context.raw_case,
            min(max_context_chars, available),
        )
        if not excerpt:
            continue
        profile += separator + excerpt
        included += 1
    return profile, included


def _representative(values: set[str]) -> str:
    return min(values, key=lambda value: (value.casefold(), value))


def _context_sort_key(context: HistoricalContext) -> tuple[int, str, str, str]:
    return (-context.source_year, context.source_url, context.source_reference, context.digest)


def _potential_alias_audit(
    case_keys: tuple[str, ...], case_texts: tuple[str, ...]
) -> tuple[int, list[list[str]]]:
    families: dict[str, set[str]] = defaultdict(set)
    for case_key, case_text in zip(case_keys, case_texts, strict=True):
        family = normalize_case_family(case_text)
        if family:
            families[family].add(case_key)
    groups = [sorted(values) for values in families.values() if len(values) > 1]
    groups.sort(key=lambda values: values[0])
    samples = [
        [case_texts[case_keys.index(case_key)] for case_key in group[:5]] for group in groups[:10]
    ]
    return len(groups), samples


def _coverage(
    queries_by_mode: dict[str, list[QueryRecord]],
    case_to_id: dict[str, int],
    historical_case_ids: frozenset[int],
) -> dict[str, Any]:
    unique_targets = {
        case_to_id[target]
        for queries in queries_by_mode.values()
        for query in queries
        for target in query.relevant_texts
    }
    unique_warm = unique_targets & historical_case_ids
    modes: dict[str, Any] = {}
    for mode, queries in queries_by_mode.items():
        target_labels = [case_to_id[target] for query in queries for target in query.relevant_texts]
        warm_labels = sum(target in historical_case_ids for target in target_labels)
        warm_queries = sum(
            any(case_to_id[target] in historical_case_ids for target in query.relevant_texts)
            for query in queries
        )
        modes[mode] = {
            "queries": len(queries),
            "warm_start_queries": warm_queries,
            "cold_start_only_queries": len(queries) - warm_queries,
            "warm_start_query_percent": (100.0 * warm_queries / len(queries) if queries else 0.0),
            "target_labels": len(target_labels),
            "warm_start_target_labels": warm_labels,
            "cold_start_target_labels": len(target_labels) - warm_labels,
            "warm_start_target_label_percent": (
                100.0 * warm_labels / len(target_labels) if target_labels else 0.0
            ),
        }
    return {
        "unique_test_targets": len(unique_targets),
        "warm_start_unique_targets": len(unique_warm),
        "cold_start_unique_targets": len(unique_targets) - len(unique_warm),
        "warm_start_unique_target_percent": (
            100.0 * len(unique_warm) / len(unique_targets) if unique_targets else 0.0
        ),
        "modes": modes,
    }


def load_corpus_repair_dataset(
    csv_path: Path,
    split_path: Path,
    *,
    evidence_cutoff_year: int,
    max_passage_chars: int,
    max_profile_passages: int,
    max_profile_identifier_chars: int,
    max_profile_context_chars: int,
    max_profile_chars: int,
) -> CorpusRepairDataset:
    test_urls = load_test_urls(split_path)
    aliases: dict[str, set[str]] = defaultdict(set)
    contexts_by_case: dict[str, list[HistoricalContext]] = defaultdict(list)
    seen_contexts: set[tuple[str, str]] = set()
    builders: dict[str, dict[str, QueryRecord]] = {
        "facts_only": {},
        "principle_only": {},
        "facts_principle": {},
    }
    historical_rows = 0
    usable_historical_rows = 0
    duplicate_contexts = 0
    matched_identifiers = 0
    paragraph_lengths: list[int] = []

    csv.field_size_limit(sys.maxsize)
    with csv_path.open("r", encoding="latin-1", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError("core CSV schema mismatch")
        for row in reader:
            eligible = not any(not (row.get(field) or "").strip() for field in REQUIRED_TEXT_FIELDS)
            if not eligible:
                continue
            raw_case = row["Cited Case"].strip()
            case_key = canonical_case_key(raw_case)
            aliases[case_key].add(raw_case)
            year = int(row["Year"].strip())
            url = row["Judgment_URL"].strip()

            if year <= evidence_cutoff_year:
                historical_rows += 1
                if url in test_urls:
                    raise ValueError(
                        f"test judgment {url} is not later than the historical evidence cutoff"
                    )
                paragraph = row["Paragraph"].strip()
                if paragraph:
                    usable_historical_rows += 1
                    normalized_paragraph = normalize_space(paragraph)
                    digest = context_digest(normalized_paragraph)
                    deduplication_key = (case_key, digest)
                    if deduplication_key in seen_contexts:
                        duplicate_contexts += 1
                    else:
                        seen_contexts.add(deduplication_key)
                        text, matched = extract_context_window(
                            normalized_paragraph, raw_case, max_passage_chars
                        )
                        matched_identifiers += int(matched)
                        paragraph_lengths.append(len(normalized_paragraph))
                        contexts_by_case[case_key].append(
                            HistoricalContext(
                                case_key=case_key,
                                raw_case=raw_case,
                                source_url=url,
                                source_reference=row["Judgment_Reference"].strip(),
                                source_year=year,
                                text=text,
                                original_chars=len(normalized_paragraph),
                                identifier_matched=matched,
                                digest=digest,
                            )
                        )

            if url not in test_urls:
                continue
            fact = row["Fact_Query"].strip()
            principle = row["Key Principles Illustrated"].strip()
            court = row["Court_Type"].strip()
            query_year = row["Year"].strip()
            add_query(builders["facts_only"], fact, fact, case_key, court, query_year)
            add_query(
                builders["principle_only"],
                principle,
                principle,
                case_key,
                court,
                query_year,
            )
            add_query(
                builders["facts_principle"],
                f"{fact}\0{principle}",
                f"{fact} {principle}",
                case_key,
                court,
                query_year,
            )

    case_keys = tuple(sorted(aliases))
    case_texts = tuple(_representative(aliases[key]) for key in case_keys)
    case_to_id = {key: index for index, key in enumerate(case_keys)}
    ordered_contexts: list[HistoricalContext] = []
    profiles: list[str] = []
    profile_context_counts: list[int] = []
    for case_key, case_text in zip(case_keys, case_texts, strict=True):
        case_contexts = sorted(contexts_by_case.get(case_key, ()), key=_context_sort_key)
        ordered_contexts.extend(case_contexts)
        profile, included = build_case_profile(
            case_text,
            case_contexts,
            max_passages=max_profile_passages,
            max_identifier_chars=max_profile_identifier_chars,
            max_context_chars=max_profile_context_chars,
            max_total_chars=max_profile_chars,
        )
        profiles.append(profile)
        profile_context_counts.append(included)

    context_case_ids = np.asarray(
        [case_to_id[context.case_key] for context in ordered_contexts], dtype=np.int64
    )
    historical_case_ids = frozenset(int(value) for value in np.unique(context_case_ids))
    queries_by_mode = {
        mode: [records[key] for key in sorted(records)] for mode, records in builders.items()
    }
    unresolved_groups, unresolved_samples = _potential_alias_audit(case_keys, case_texts)
    paragraph_lengths.sort()
    audit = {
        "evidence_cutoff_year": evidence_cutoff_year,
        "historical_rows": historical_rows,
        "usable_historical_rows_before_deduplication": usable_historical_rows,
        "historical_contexts_after_deduplication": len(ordered_contexts),
        "duplicate_historical_contexts_removed": duplicate_contexts,
        "historical_cases": len(historical_case_ids),
        "identifier_matched_contexts": matched_identifiers,
        "identifier_matched_context_percent": 100.0 * matched_identifiers / len(ordered_contexts),
        "paragraph_character_lengths": {
            "p50": percentile(paragraph_lengths, 0.50),
            "p95": percentile(paragraph_lengths, 0.95),
            "max": max(paragraph_lengths),
        },
        "raw_candidate_identifiers": sum(len(values) for values in aliases.values()),
        "canonical_candidate_identifiers": len(case_keys),
        "conservative_variants_merged": sum(len(values) - 1 for values in aliases.values()),
        "unresolved_case_family_groups": unresolved_groups,
        "unresolved_case_family_samples": unresolved_samples,
        "profiles_with_context": sum(value > 0 for value in profile_context_counts),
        "profiles_without_context": sum(value == 0 for value in profile_context_counts),
        "profile_identifiers_truncated": sum(
            len(normalize_space(case_text)) > max_profile_identifier_chars
            for case_text in case_texts
        ),
        "profile_contexts_included": {
            str(count): profile_context_counts.count(count)
            for count in sorted(set(profile_context_counts))
        },
        "coverage": _coverage(queries_by_mode, case_to_id, historical_case_ids),
    }
    return CorpusRepairDataset(
        case_keys=case_keys,
        case_texts=case_texts,
        contexts=tuple(ordered_contexts),
        context_case_ids=context_case_ids,
        profiles=tuple(profiles),
        historical_case_ids=historical_case_ids,
        queries_by_mode=queries_by_mode,
        audit=audit,
        test_urls=frozenset(test_urls),
    )


def max_aggregate_passage_scores(
    passage_scores: np.ndarray, passage_case_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if passage_scores.ndim != 1 or passage_case_ids.ndim != 1:
        raise ValueError("passage scores and case IDs must be one-dimensional")
    if len(passage_scores) != len(passage_case_ids):
        raise ValueError("passage scores and case IDs must be aligned")
    if not len(passage_scores):
        raise ValueError("at least one passage score is required")
    if not np.isfinite(passage_scores).all():
        raise ValueError("passage scores must be finite")

    case_ids, inverse = np.unique(passage_case_ids, return_inverse=True)
    case_scores = np.full(len(case_ids), -np.inf, dtype=np.float32)
    np.maximum.at(case_scores, inverse, passage_scores)
    return case_ids, case_scores


def max_aggregate_sparse_passage_scores(
    passage_scores: dict[int, float],
    passage_case_positions: np.ndarray,
    candidate_case_count: int,
) -> np.ndarray:
    if candidate_case_count < 1:
        raise ValueError("candidate case count must be positive")
    if passage_case_positions.ndim != 1:
        raise ValueError("passage case positions must be one-dimensional")
    if len(passage_case_positions) and (
        passage_case_positions.min() < 0 or passage_case_positions.max() >= candidate_case_count
    ):
        raise IndexError("passage case position out of range")
    case_scores = np.zeros(candidate_case_count, dtype=np.float32)
    for passage_id, score in passage_scores.items():
        if not 0 <= passage_id < len(passage_case_positions):
            raise IndexError(f"passage index out of range: {passage_id}")
        position = int(passage_case_positions[passage_id])
        case_scores[position] = max(case_scores[position], score)
    return case_scores

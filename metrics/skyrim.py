# phrase_error.py
#
# A metric for “Did the candidate transcription contain the phrases
# that were actually spoken?”  Phrases are read once from a plain-text
# file (one phrase per line, default: original_skyrim_data/words.txt).
#
# Result is a single float in the range [0, 1]:
#   0.0  → all required phrases were present in every prediction
#   1.0  → every required phrase was missed
#
# If none of the reference lines contain any monitored phrase, the
# metric returns None.

import re
from pathlib import Path
from typing import List, Optional

import datasets
import evaluate


_CITATION = "N/A - bespoke metric derived from Skyrim voice-line evaluation script"

_DESCRIPTION = """
Phrase Error Rate (PER) measures how often a transcription **omits** short,
pre-defined target phrases that *are* present in the reference.  
A phrase contributes **one error** if it appears in the reference but not in
the prediction; otherwise it contributes **zero**.  
The metric averages those 0/1 errors across all testable phrase occurrences.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (List[str]): model transcriptions.
    references  (List[str]): ground-truth transcriptions.
    phrases_path (str, optional):
        Path to a UTF-8 text file containing the phrases to
        monitor, **one phrase per line**.
        Defaults to "original_skyrim_data/words.txt".
Returns:
    float | None: average phrase-error rate, or None if no reference line
        contained a monitored phrase (“untestable” dataset).
"""


class _PhraseMatcher:
    """Pre-processes the phrase list once for fast look-ups."""

    _NON_WORD_CHARS = re.compile(r"[^0-9a-z\s]")

    @staticmethod
    def _norm(txt: str) -> str:
        """Lower-case, replace hyphens by spaces, strip punctuation."""
        txt = txt.lower().replace("-", " ")
        return _PhraseMatcher._NON_WORD_CHARS.sub("", txt)

    def __init__(self, phrases: List[str]) -> None:
        self.entries = []
        for phrase in phrases:
            phrase = phrase.strip()
            if not phrase:
                continue
            norm = self._norm(phrase)
            # For very short phrases we demand word-boundaries to avoid
            # accidental matches inside other words.
            pattern = (
                re.compile(rf"(?:^|[ \?\.\!,]){re.escape(norm)}(?=$|[ \?\.\!,])")
                if len(norm) < 5
                else None
            )
            self.entries.append({"norm": norm, "pattern": pattern})

    def match_in_reference(self, ref_norm: str) -> List[dict]:
        """Return list of phrases that occur in the *reference* line."""
        hits = []
        for entry in self.entries:
            if entry["pattern"]:
                if entry["pattern"].search(ref_norm):
                    hits.append(entry)
            elif entry["norm"] in ref_norm:
                hits.append(entry)
        return hits

    def present_in_prediction(self, entry: dict, pred_norm: str) -> bool:
        """Return True iff a specific phrase occurs in the *prediction*."""
        if entry["pattern"]:
            return entry["pattern"].search(pred_norm) is not None
        return entry["norm"] in pred_norm


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PhraseErrorRate(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _compute(
        self,
        predictions: List[str],
        references: List[str],
        phrases_path: str = "original_skyrim_data/words.txt",
    ) -> Optional[float]:
        phrase_file = Path(phrases_path)
        if not phrase_file.exists():
            raise FileNotFoundError(f"Phrase list not found: {phrase_file.resolve()}")
        phrases = phrase_file.read_text(encoding="utf-8").splitlines()
        matcher = _PhraseMatcher(phrases)

        total, missed = 0, 0
        for pred, ref in zip(predictions, references):
            ref_norm = _PhraseMatcher._norm(ref)
            candidates = matcher.match_in_reference(ref_norm)
            if not candidates:
                continue  # untestable line
            pred_norm = _PhraseMatcher._norm(pred)
            for entry in candidates:
                total += 1
                if not matcher.present_in_prediction(entry, pred_norm):
                    missed += 1

        return (missed / total) if total > 0 else None

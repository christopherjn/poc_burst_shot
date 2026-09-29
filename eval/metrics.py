"""Clustering metrics against hand labels that allow "optional" links.

Labels hold atomic groups (images that must end up together) plus optional links between
atomic groups whose merging is a judgement call. Pairs across an optional link are ignored
by every metric: merging or splitting them is neither rewarded nor penalised.
"""
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


@dataclass
class Labels:
    group_of: dict[str, str]              # image stem -> atomic group name
    optional: set[frozenset[str]]         # pairs of atomic group names

    def relation(self, a: str, b: str) -> str:
        ga, gb = self.group_of[a], self.group_of[b]
        if ga == gb:
            return "same"
        if frozenset((ga, gb)) in self.optional:
            return "ignore"
        return "different"


def load_labels(path: Path) -> Labels:
    data = json.loads(Path(path).read_text())
    group_of = {stem: name for name, stems in data["groups"].items() for stem in stems}
    optional = {frozenset(pair) for pair in data.get("optional_links", [])}
    return Labels(group_of, optional)


@dataclass
class Scores:
    pair_precision: float
    pair_recall: float
    pair_f1: float
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    false_merges: int      # predicted-together pairs that should be apart
    missed_links: int      # should-be-together pairs that were split
    n_pred_groups: int
    n_true_groups: int

    def row(self) -> str:
        return (f"pairF1 {self.pair_f1:.3f} (P {self.pair_precision:.3f} R {self.pair_recall:.3f})  "
                f"B3F1 {self.bcubed_f1:.3f} (P {self.bcubed_precision:.3f} R {self.bcubed_recall:.3f})  "
                f"merges {self.false_merges:4d} splits {self.missed_links:4d}  "
                f"groups {self.n_pred_groups}/{self.n_true_groups}")


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def score(pred_groups: list[list[str]], labels: Labels) -> Scores:
    """`pred_groups` is a list of groups of image stems; every labelled image must appear once."""
    pred_of = {stem: i for i, group in enumerate(pred_groups) for stem in group}
    stems = sorted(labels.group_of)
    missing = set(stems) - set(pred_of)
    if missing:
        raise ValueError(f"{len(missing)} labelled images missing from prediction, e.g. {sorted(missing)[:3]}")

    tp = fp = fn = 0
    for a, b in combinations(stems, 2):
        rel = labels.relation(a, b)
        together = pred_of[a] == pred_of[b]
        if rel == "same":
            tp += together
            fn += not together
        elif rel == "different":
            fp += together
    pp = tp / (tp + fp) if tp + fp else 1.0
    pr = tp / (tp + fn) if tp + fn else 1.0

    bp = br = 0.0
    for a in stems:
        pred_mates = [b for b in stems if pred_of[b] == pred_of[a]]
        scored = [b for b in pred_mates if labels.relation(a, b) != "ignore"]
        correct = sum(labels.relation(a, b) == "same" for b in scored)
        true_size = sum(labels.group_of[b] == labels.group_of[a] for b in stems)
        bp += correct / len(scored)
        br += correct / true_size
    bp /= len(stems)
    br /= len(stems)

    return Scores(pp, pr, _f1(pp, pr), bp, br, _f1(bp, br), fp, fn,
                  len({pred_of[s] for s in stems}), len(set(labels.group_of.values())))

import csv
import math

import torch

from utils.per_token_metrics import PerTokenMetrics


def test_per_token_metrics_exports_counts_losses_summaries_and_plot(tmp_path):
    tracker = PerTokenMetrics(
        tmp_path, {"tiny": 3}, {"tiny": {0: "\\n", 1: "a", 2: "\\t"}}
    )
    tracker.count_training_batch("tiny", torch.tensor([[0, 1, 1, 2]]))
    tracker.begin_evaluation()
    targets = torch.tensor([[0, 1, 1]])
    logits = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]])
    tracker.add_evaluation_batch("tiny", "train", logits, targets)
    tracker.add_evaluation_batch("tiny", "val", logits, targets)
    tracker.export(10)

    with open(tracker.detail_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["training_seen_count"]) for row in rows] == [1, 2, 1]
    assert [row["token_text_escaped"] for row in rows] == ["\\n", "a", "\\t"]
    assert math.isclose(float(rows[0]["val_loss"]), 0.0949229, rel_tol=1e-5)
    assert int(rows[1]["val_eval_count"]) == 2

    with open(tracker.summary_path, newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert {row["metric"] for row in summaries} == {
        "train_loss", "val_loss", "training_seen_count"
    }
    assert "skew" in summaries[0] and "excess_kurtosis" in summaries[0]
    html = tracker.plot_path.read_text(encoding="utf-8")
    assert "validation loss" in html
    assert "times seen in training" in html
    assert "Summary statistics" in html
    assert "ordered lowest to highest training occurrence" in html
    assert "token_text_escaped" in html

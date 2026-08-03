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
    tracker.begin_evaluation()
    tracker.add_evaluation_batch("tiny", "train", logits, targets)
    tracker.add_evaluation_batch("tiny", "val", logits, targets)
    tracker.export(20)

    with open(tracker.detail_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [int(row["training_seen_count"]) for row in rows[:3]] == [1, 2, 1]
    assert [row["token_text_escaped"] for row in rows[:3]] == ["\\n", "a", "\\t"]
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
    assert "ordered highest to lowest sampled training loss" in html
    assert "trainingLossPlot" in html
    assert "token_text_escaped" in html
    assert "Selected-token loss and cumulative appearances vs iteration" in html
    assert "Selected-token loss vs cumulative appearances" in html
    assert "appearanceByIterationTraces" in html
    assert "cumulative training appearances" in html
    assert "Primary Plotly CDN unavailable" in html
    assert "cdn.jsdelivr.net" in html
    assert "Unable to render" in html
    assert "multiple" in html
    assert "Y-axis scale:" in html
    assert "left logarithmic" in html
    assert "right logarithmic" in html
    assert "yaxis2.type" in html


def test_per_token_metrics_migrates_legacy_detail_csv(tmp_path):
    detail_path = tmp_path / "per_token_metrics.csv"
    detail_path.write_text(
        "iteration,dataset,token_id,train_loss,train_eval_count,val_loss,val_eval_count,training_seen_count\n"
        "10,tiny,0,1.5,2,2.5,3,4\n"
        "20,tiny,0,\\n,1.25,2,2.25,3,8\n",
        encoding="utf-8",
    )

    tracker = PerTokenMetrics(tmp_path, {"tiny": 1}, {"tiny": {0: "\\n"}})

    with detail_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["token_text_escaped"] == "\\n"
    assert rows[0]["train_loss"] == "1.5"
    assert rows[0]["training_seen_count"] == "4"
    assert rows[1]["token_text_escaped"] == "\\n"
    assert rows[1]["train_loss"] == "1.25"
    assert rows[1]["training_seen_count"] == "8"

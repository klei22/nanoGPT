import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.per_token_metrics import PerTokenMetrics


def test_per_token_metrics_exports_counts_losses_summaries_and_plot(tmp_path):
    tracker = PerTokenMetrics(tmp_path, {"tiny": 3})
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
    assert math.isclose(float(rows[0]["val_loss"]), 0.0949229, rel_tol=1e-5)
    assert int(rows[1]["val_eval_count"]) == 2

    with open(tracker.summary_path, newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert {row["metric"] for row in summaries} == {
        "train_loss", "val_loss", "training_seen_count"
    }
    assert "skew" in summaries[0] and "excess_kurtosis" in summaries[0]
    html = Path(tracker.plot_path).read_text(encoding="utf-8")
    assert "validation loss" in html
    assert "times seen in training" in html

    coverage = next(row for row in summaries if row["metric"] == "training_seen_count")
    assert int(coverage["populated_tokens"]) == 3


def test_training_coverage_counts_only_observed_tokens(tmp_path):
    tracker = PerTokenMetrics(tmp_path, {"tiny": 5})
    tracker.count_training_batch("tiny", torch.tensor([0, 0, 3]))
    tracker.export(1)

    with open(tracker.summary_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coverage = next(row for row in rows if row["metric"] == "training_seen_count")
    assert int(coverage["populated_tokens"]) == 2


def test_resume_restores_and_extends_cumulative_counts(tmp_path):
    first = PerTokenMetrics(tmp_path / "first", {"tiny": 3})
    first.count_training_batch("tiny", torch.tensor([0, 1, 1]))
    first.synchronize_training_counts()

    resumed = PerTokenMetrics(
        tmp_path / "resumed", {"tiny": 3}, initial_seen=first.state_dict()
    )
    resumed.count_training_batch("tiny", torch.tensor([1, 2]))
    resumed.synchronize_training_counts()
    np.testing.assert_array_equal(resumed.seen["tiny"], [1, 3, 1])


def _distributed_count_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}"
    )
    tracker = PerTokenMetrics(Path(output_dir) / f"rank-{rank}", {"tiny": 4})
    tracker.count_training_batch("tiny", torch.tensor([rank, rank, 3]))
    tracker.synchronize_training_counts(distributed=True)
    Path(output_dir, f"counts-{rank}.json").write_text(
        json.dumps(tracker.seen["tiny"].tolist()), encoding="utf-8"
    )
    dist.destroy_process_group()


def test_distributed_counts_are_aggregated_on_every_rank(tmp_path):
    init_file = tmp_path / "dist-init"
    mp.spawn(
        _distributed_count_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        counts = json.loads((tmp_path / f"counts-{rank}.json").read_text(encoding="utf-8"))
        assert counts == [2, 2, 0, 2]

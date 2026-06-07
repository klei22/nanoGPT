import sys
import textwrap
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization_and_search" / "nsga_search"))
from remote_trainer import RemoteTrainer


def test_submit_trial_shards_supports_local_host_without_ssh(tmp_path):
    runner = tmp_path / "runner.py"
    runner.write_text(
        textwrap.dedent(
            """
            import argparse
            import yaml

            parser = argparse.ArgumentParser()
            parser.add_argument("--trials_yaml")
            parser.add_argument("--results_yaml")
            args = parser.parse_args()

            trials = yaml.safe_load(open(args.trials_yaml)) or []
            yaml.safe_dump(
                [{"status": "completed", "trial_id": trial["trial_id"]} for trial in trials],
                open(args.results_yaml, "w"),
            )
            """
        )
    )
    trials_yaml = tmp_path / "trials.yaml"
    yaml.safe_dump([{"trial_id": "a"}, {"trial_id": "b"}], trials_yaml.open("w"))

    trainer = RemoteTrainer(hosts=["local"], user="unused")

    assert trainer.submit_trial_shards(
        path_to_yaml=str(trials_yaml),
        remote_work_dir=str(tmp_path),
        dir_name="demo",
        runner_script=str(runner),
        runner_yaml_arg="--trials_yaml",
        runner_results_arg="--results_yaml",
        conda_env="definitely_missing_env",
    )
    assert trainer.wait_for_all(poll_interval=0.1, timeout=5)

    fetched = trainer.fetch_job_file("results.yaml", str(tmp_path / "fetched"))
    assert len(fetched) == 1
    results = yaml.safe_load(fetched[0].read_text())
    assert [result["trial_id"] for result in results] == ["a", "b"]

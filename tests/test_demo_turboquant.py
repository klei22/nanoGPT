from pathlib import Path
import os
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).parents[1]
DEMO = REPO_ROOT / "analysis" / "vector_distribution" / "demo_turboquant.sh"


class TurboQuantDemoTest(unittest.TestCase):
    def test_demo_invokes_all_visualization_and_metric_analyses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls.txt"
            python = bin_dir / "python3"
            python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$MOCK_CALLS\"\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            environment = os.environ.copy()
            environment.update({
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "MOCK_CALLS": str(calls),
                "NSIDE": "1",
                "ANGULAR_DIM": "8",
                "ANGULAR_TRIALS": "1",
                "ANGLE_STEP": "90",
                "EVENNESS_SAMPLES": "8",
                "EVENNESS_CAPS": "1",
                "DIM_SWEEP_TRIALS": "1",
                "DIM_SWEEP_ANGLE_STEP": "90",
                "HIGH_DIM_EVENNESS_SAMPLES": "8",
                "HIGH_DIM_EVENNESS_BATCH_SIZE": "4",
                "HIGH_DIM_DISTORTION_TRIALS": "1",
                "HIGH_DIM_ANGLE_STEP": "90",
                "LOW_BIT_EM_TRIALS": "1",
                "LOW_BIT_EM_ANGLE_STEP": "90",
                "GROUPED_QUANT_TRIALS": "1",
                "GROUPED_QUANT_ANGLE_STEP": "90",
            })

            subprocess.run(["bash", str(DEMO), str(root / "output")],
                           check=True, cwd=REPO_ROOT, env=environment,
                           capture_output=True, text=True)
            invocations = calls.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(invocations), 17)
        self.assertEqual(sum("vector_distribution_analysis.py" in call
                             for call in invocations), 6)
        self.assertTrue(any("turboquant_angular_distortion.py" in call and
                            "--pair-mode sparse" in call for call in invocations))
        self.assertTrue(any("turboquant_angular_distortion.py" in call and
                            "--pair-mode isotropic" in call and
                            "--no-transformed-tq" in call for call in invocations))
        self.assertTrue(any("angle_space_evenness.py" in call and
                            "--samples 8" in call for call in invocations))
        self.assertTrue(any("isotropic_dimension_sweep.py" in call and
                            "--min-dim 256 --max-dim 2048" in call and
                            "--trials 1" in call for call in invocations))
        self.assertTrue(any("high_dim_angle_space_evenness.py" in call and
                            "--samples 8 --batch-size 4" in call
                            for call in invocations))
        high_dimensional = [call for call in invocations
                            if "isotropic_distortion_d" in call]
        self.assertEqual(len(high_dimensional), 4)
        self.assertTrue(any("low_bit_em_comparison.py" in call and
                            "--trials 1 --angles-step 90" in call
                            for call in invocations))
        self.assertTrue(any("grouped_quantization_sweep.py" in call and
                            "--dimension 2048 --trials 1" in call
                            for call in invocations))


if __name__ == "__main__":
    unittest.main()

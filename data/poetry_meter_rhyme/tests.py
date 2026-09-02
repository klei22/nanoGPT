import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("poetry_builder", HERE / "build_dataset.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def test_rhyme_normalization_and_shorthand(self):
        self.assertEqual(builder.canonicalize_rhyme(["c", "d", "c", "d"]), "ABAB")
        self.assertEqual(builder.expand_rhyme("a a *", 6), ["a", "a", "a", "a", "a", "a"])

    def test_haider_nine_plus_columns_and_supplied_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "English" / "eng_meter_3fold"
            path.mkdir(parents=True)
            (path / "sample.meter.test.3fold.txt").write_text(
                "1\tShall\t-\t.\tNN\t0\tiambic\tiambic.penta\t-+\n"
                "2\tI\t+\t.\tNN\t0\tiambic\tiambic.penta\t-+\n\n")
            rows = builder.parse_haider(Path(tmp))
            self.assertEqual((rows[0]["text"], rows[0]["split"], rows[0]["syllable_stress"]), ("Shall I", "test", "01"))
            self.assertEqual(rows[0]["scansion"], "-+")

    def test_chicago_is_grouped_by_author(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("AUTHOR Ada\nTITLE Hidden\nRHYME c d c d\none\ntwo\nthree\nfour\nRHYME-POEM\n")
            rows = builder.parse_chicago(Path(tmp))
            self.assertEqual(rows[0]["rhyme_scheme"], "ABAB")
            self.assertEqual(rows[0]["group_id"], "poet:ada")

    def test_benchmark_has_a_real_minimal_pair(self):
        record = {"id": "x", "group_id": "p", "source": "haider", "task": "meter",
                  "text": "a line", "meter": "iambic", "measure": "iambic.penta",
                  "syllable_stress": "0101", "scansion": "-+-+", "split": "test", "content_hash": "0f"}
        row = builder.benchmark_rows(record)[0]
        self.assertEqual(row["task"], "meter_minimal_pair")
        self.assertNotEqual(row["choices"][0], row["choices"][1])


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from wireless_rtc.analysis.run_selection import load_run_selection, selection_audit


class RunSelectionTests(unittest.TestCase):
    def fixture(self, root):
        configs = root/"configs"
        configs.mkdir()
        session = root/"archived"/"any-name"
        session.mkdir(parents=True)
        (session/"manifest.json").write_text(json.dumps(dict(setting="walking",split="development",status="complete")))
        path = configs/"network.json"
        path.write_text(json.dumps(dict(schema_version=1,base_dir="..",runs={"weak":["archived/any-name"]})))
        return path, session

    def test_relative_paths_are_config_relative_and_references_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, session = self.fixture(root)
            prediction = path.parent/"prediction.json"
            prediction.write_text(json.dumps(dict(schema_version=1,runs_config="network.json")))
            selected = load_run_selection(prediction)
            self.assertEqual(selected["entries"][0]["path"],str(session.resolve()))
            self.assertEqual(selected["settings"],["weak"])
            audit = selection_audit(selected)
            self.assertEqual(audit["entries"][0]["saved_split"],"development")
            self.assertEqual(audit["entries"][0]["setting"],"weak")
            self.assertEqual(audit["procedure_config"],str(prediction.resolve()))
            self.assertTrue(audit["config_sha256"])

    def test_duplicate_aliases_and_missing_selected_paths_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path, session = self.fixture(Path(temp))
            document = json.loads(path.read_text())
            document["runs"]["strong"] = [str(session.resolve())]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,"Duplicate run"):
                load_run_selection(path)
            document["runs"] = {"weak":["missing"]}
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError,"missing"):
                load_run_selection(path)

    def test_setting_filter_does_not_load_unselected_group(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.fixture(Path(temp))
            document = json.loads(path.read_text())
            document["runs"]["strong"] = ["not-collected-yet"]
            path.write_text(json.dumps(document))
            self.assertEqual(len(load_run_selection(path,["weak"])["entries"]),1)
            with self.assertRaisesRegex(ValueError,"missing"):
                load_run_selection(path)

    def test_self_reference_and_empty_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.fixture(Path(temp))
            path.write_text(json.dumps(dict(schema_version=1,runs_config=path.name)))
            with self.assertRaisesRegex(ValueError,"itself"):
                load_run_selection(path)
            path.write_text(json.dumps(dict(schema_version=1,runs={"weak":[]})))
            with self.assertRaisesRegex(ValueError,"at least one"):
                load_run_selection(path)

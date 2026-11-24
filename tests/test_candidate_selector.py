"""
test_candidate_selector.py

Tests voor src/candidate/model/train_candidate_selector.py.

De tests maken eigen data met seed 7 in een tijdelijke map, die aan het eind
wordt verwijderd. De echte data in outputs/candidate worden dus niet aangeraakt.
Omdat dit andere data zijn, blijven de echte testfamilies van seed 42 ongezien.

Importeren doet niets: het werk begint pas in setUpClass. De tests laten geen
testscores zien, ook niet als een test mislukt.

Gebruik (vanuit de hoofdmap van de repo):
    python tests/test_candidate_selector.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# De hoofdmap in sys.path, zodat src.candidate ook werkt als je dit bestand direct draait.
HOOFDMAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOFDMAP))

from src.candidate.data import features

from src.candidate.model import train_candidate_selector as selector


# Andere data dan de echte (seed 42), en minder scenario's zodat de tests sneller zijn.
SEED = "7"
N = "600"


def draai_selector(*argumenten):
    """Draai het script vanuit de hoofdmap. De uitvoer blijft binnen deze test."""
    return subprocess.run(
        [sys.executable, "-m", "src.candidate.model.train_candidate_selector", *argumenten],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
    )


def lees_run(data_map, test_geopend):
    """Geef de runmap en run.json van de run met deze waarde van test_opened."""
    for run_json in Path(data_map, "runs").glob("*/DUMMY_run.json"):
        run = json.loads(run_json.read_text(encoding="utf-8"))
        if run["test_opened"] == test_geopend:
            return run_json.parent, run
    raise AssertionError("Geen run gevonden.")


class TestSelectorRuns(unittest.TestCase):
    """Eén run zonder en één run met --open-test, op de data van seed 7."""

    @classmethod
    def setUpClass(cls):
        cls.map = tempfile.mkdtemp(prefix="test_selector_")
        cls.addClassCleanup(shutil.rmtree, cls.map, ignore_errors=True)

        cls.dicht = draai_selector("--regenerate", "--seed", SEED, "--n", N, "--output-dir", cls.map)
        cls.open = draai_selector("--open-test", "--output-dir", cls.map)

    def test_dicht_laat_niets_van_de_test_zien(self):
        self.assertEqual(self.dicht.returncode, 0, self.dicht.stderr)
        run_map, run = lees_run(self.map, False)

        self.assertIs(run["test_opened"], False)
        self.assertIsNone(run["test_opened_at"])
        self.assertNotIn("test_scores", run)

        # Geen testfamilie in cv_results.csv.
        TEST_FAMILIES = json.loads(Path(self.map, "split_manifest.json").read_text(encoding="utf-8"))["test_families"]
        cv = pd.read_csv(run_map / "DUMMY_cv_results.csv")
        families = {f for rij in cv["families"] for f in rij.split()}
        self.assertTrue(families)
        self.assertFalse(families & set(TEST_FAMILIES))
        self.assertFalse({int(f.split("__")[1]) for f in TEST_FAMILIES} & set(cv["input_length_tokens"]))

        # In geen enkel uitvoerbestand, en ook niet in de uitvoer van het script.
        for bestand in run_map.iterdir():
            inhoud = bestand.read_bytes()
            for familie in TEST_FAMILIES:
                self.assertNotIn(familie.encode(), inhoud, bestand.name)
            self.assertNotIn(b"test_scores", inhoud, bestand.name)
        for familie in TEST_FAMILIES:
            self.assertNotIn(familie, self.dicht.stdout)
        self.assertNotIn("Testfamilies (", self.dicht.stdout)

    def test_open_heeft_testscores(self):
        # Alleen de returncode en de sleutels: de scores zelf komen niet in de testuitvoer.
        self.assertEqual(self.open.returncode, 0, "--open-test mislukte")
        _, run = lees_run(self.map, True)

        self.assertIs(run["test_opened"], True)
        self.assertIsNotNone(run["test_opened_at"])
        self.assertIn(run["gekozen_kandidaat"], run["test_scores"])
        for naam in [selector.BASELINE, selector.OPZOEK, selector.CONSERVATIEF]:
            self.assertIn(naam, run["test_scores"])
            self.assertEqual(sorted(run["test_scores"][naam]), sorted(run["test_scores"][run["gekozen_kandidaat"]]))

class TestZonderRun(unittest.TestCase):
    """Tests die geen pipeline nodig hebben."""

    def test_afgeleide_features_gebruiken_alleen_gebruikersinvoer(self):
        x = pd.DataFrame({
            "input_length_tokens": [128, 256],
            "min_macro_f1": [0.85, 0.80],
            "max_latency_ms_per_doc": [20.0, 50.0],
            "monthly_volume": [10_000, 1_000],
            "max_energy_kwh_per_month": [0.01, float("nan")],
            "hardware": ["rtx3080", "ryzen5_3600x"],
            "has_energy_budget": [True, False],
        })
        self.assertEqual(sorted(x.columns), sorted(features.FEATURE_KOLOMMEN))

        # Gemeten kolommen met onzinwaarden: de uitkomst mag niet veranderen.
        met_meting = x.assign(**{kolom: 999.0 for kolom in features.GEMETEN_KOLOMMEN})

        opties = {"budget_groot": True, "log10": True, "budget_per_doc": True,
                  "latency_per_token": True, "cpu_interacties": True}
        # Lukt dit met alleen FEATURE_KOLOMMEN, dan is geen andere kolom nodig.
        uit = features.maak_features(x, **opties)
        pd.testing.assert_frame_equal(uit, features.maak_features(met_meting, **opties))

        for kolom in uit.columns:
            self.assertNotIn(kolom, features.GEMETEN_KOLOMMEN)

    def test_selectieregel(self):
        def kandidaat(naam, complexiteit, f1, violation):
            return {"naam": naam, "complexiteit": complexiteit, "macro_f1": f1, "violation_rate": violation}

        # Hoogste macro-F1 binnen de grens. Een beter model boven de grens telt niet.
        groep = [kandidaat("regel", 1, 0.90, 0.01), kandidaat("boom", 3, 0.94, 0.04),
                 kandidaat("hgb", 4, 0.99, 0.06)]
        self.assertEqual(selector.kies_kandidaat(groep, 0.05)[0], "boom")

        # Hoogstens 0,01 lager en eenvoudiger: de eenvoudigere wint, de eenvoudigste eerst.
        groep = [kandidaat("regel", 1, 0.935, 0.01), kandidaat("lr", 2, 0.938, 0.02),
                 kandidaat("hgb", 4, 0.94, 0.04)]
        self.assertEqual(selector.kies_kandidaat(groep, 0.05)[0], "regel")

        # Geen kandidaat haalt de grens: de laagste violation rate, met een waarschuwing.
        groep = [kandidaat("boom", 3, 0.94, 0.08), kandidaat("hgb", 4, 0.99, 0.06)]
        naam, _, waarschuwing = selector.kies_kandidaat(groep, 0.05)
        self.assertEqual(naam, "hgb")
        self.assertIsNotNone(waarschuwing)


if __name__ == "__main__":
    unittest.main()

"""
test_candidate_predict.py

Tests voor src/candidate/model/predict_candidate.py.

De tests maken eigen data en runs met seed 7 in een tijdelijke map, die aan het
eind wordt verwijderd. De echte data in outputs/candidate worden niet aangeraakt,
en --open-test wordt niet gebruikt.

Importeren doet niets: het werk begint pas in setUpClass.

Gebruik (vanuit de hoofdmap van de repo):
    python tests/test_candidate_predict.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

# De hoofdmap in sys.path, zodat src.candidate ook werkt als je dit bestand direct draait.
HOOFDMAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOFDMAP))

from src.candidate.model import train_candidate_selector as selector
from src.candidate.model.candidates import REGELS, kans_ja, kies_categorie, maak_kandidaten
from src.candidate.data.labelling_rule import laad_benchmark, maak_configuratietabel, voldoende_categorieen


# Andere data dan de echte (seed 42), en minder scenario's zodat de tests sneller zijn.
SEED = "7"
N = "600"

# Eén zelfbedacht scenario, op een lengte uit de training.
SCENARIO = ["--hardware", "rtx3080", "--input-length", "128", "--min-f1", "0.85",
            "--max-latency-ms", "20", "--volume", "10000", "--budget-kwh", "0.05"]


def draai_selector(*argumenten):
    """Draai de selector vanuit de hoofdmap. De uitvoer blijft binnen deze test."""
    uit = subprocess.run(
        [sys.executable, "-m", "src.candidate.model.train_candidate_selector", *argumenten],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
    )
    if uit.returncode != 0:
        raise AssertionError(uit.stderr)


def draai_predict(*argumenten):
    """Draai predict_candidate vanuit de hoofdmap."""
    return subprocess.run(
        [sys.executable, "-m", "src.candidate.model.predict_candidate", *SCENARIO, *argumenten],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
    )


def gekozen_categorie(uitvoer):
    """Haal de gekozen categorie uit de uitvoer."""
    return re.search(r"gekozen categorie:\s+(\S+)", uitvoer).group(1)


class TestPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.map = Path(tempfile.mkdtemp(prefix="test_predict_"))
        cls.addClassCleanup(shutil.rmtree, cls.map, ignore_errors=True)
        cls.data = ["--benchmark", str(cls.map / "dummy_benchmark.csv"),
                    "--manifest", str(cls.map / "split_manifest.json")]

        # Twee runs op dezelfde data. Bij seed 7 kiest de standaardgrens een regel. De tweede
        # run (grens 1) is de nieuwste map in runs/. De multiclass-run wordt in zijn eigen test gemaakt.
        draai_selector("--regenerate", "--seed", SEED, "--n", N, "--output-dir", str(cls.map))
        cls.run_regel = sorted((cls.map / "runs").iterdir())[-1]
        draai_selector("--output-dir", str(cls.map), "--max-violation-rate", "1")
        cls.run_model = sorted((cls.map / "runs").iterdir())[-1]

        # De selector kiest bij seed 7 geen ja/nee-model, dus die run maken we hier zelf:
        # met dezelfde code, alleen op de trainfamilies, en buiten runs/.
        cls.run_ja_nee = cls.map / "extra" / "ja_nee"
        cls.run_ja_nee.mkdir(parents=True)
        shutil.copy(cls.run_regel / "DUMMY_run.json", cls.run_ja_nee / "DUMMY_run.json")
        joblib.dump(cls.maak_ja_nee_eindmodel(), cls.run_ja_nee / "DUMMY_model.joblib")

    @classmethod
    def maak_ja_nee_eindmodel(cls):
        manifest = json.loads((cls.map / "split_manifest.json").read_text(encoding="utf-8"))
        scenarios = pd.read_csv(cls.map / "scenarios.csv")
        train = scenarios[~scenarios["family_id"].isin(manifest["test_families"])].reset_index(drop=True)

        configuraties = maak_configuratietabel(laad_benchmark(cls.map / "dummy_benchmark.csv"))
        configuraties = configuraties[~selector.familie_van(configuraties).isin(manifest["test_families"])]
        voldoende = pd.Series(
            [voldoende_categorieen(configuraties, s) for s in train.to_dict("records")], index=train.index
        )

        kandidaat = next(k for k in maak_kandidaten() if k["soort"] == "ja_nee" and k["instellingen"]["t"] == 0.9)
        return selector.train_eindmodel(kandidaat, train, configuraties, voldoende)

    def test_regel_als_gekozen_kandidaat(self):
        eind = joblib.load(self.run_regel / "DUMMY_model.joblib")
        self.assertEqual(eind["soort"], "regel")

        resultaat = draai_predict("--run", str(self.run_regel), *self.data)
        self.assertEqual(resultaat.returncode, 0, resultaat.stderr)
        self.assertIn("geen kansen", resultaat.stdout)
        self.assertNotIn("Drempel t:", resultaat.stdout)

        # Dezelfde uitkomst als de regel zelf, met de configuraties uit de run.
        scenario = pd.DataFrame([{
            "hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.85,
            "max_latency_ms_per_doc": 20.0, "monthly_volume": 10_000.0,
            "max_energy_kwh_per_month": 0.05, "has_energy_budget": True,
        }])
        verwacht = REGELS[eind["naam"]](scenario, pd.DataFrame(eind["configuraties"]))[0]
        self.assertEqual(gekozen_categorie(resultaat.stdout), verwacht)

    def test_ja_nee_model_klopt_met_kies_categorie(self):
        resultaat = draai_predict("--run", str(self.run_ja_nee), *self.data)
        self.assertEqual(resultaat.returncode, 0, resultaat.stderr)

        geprint = {c: float(k) for c, k in re.findall(r"P\((\w+) voldoet\):\s+([\d.]+)", resultaat.stdout)}
        t = float(re.search(r"drempel t:\s+([\d.]+)", resultaat.stdout).group(1))

        # De geprinte kansen moeten de kansen van het opgeslagen model zijn.
        eind = joblib.load(self.run_ja_nee / "DUMMY_model.joblib")
        self.assertEqual(t, eind["t"])
        x = pd.DataFrame([{
            "input_length_tokens": 128, "min_macro_f1": 0.85, "max_latency_ms_per_doc": 20.0,
            "monthly_volume": 10_000.0, "max_energy_kwh_per_month": 0.05,
            "hardware": "rtx3080", "has_energy_budget": True,
        }])[eind["features"]]
        # Eén ja/nee-model per categorie, en van elk wordt de kans geprint.
        self.assertEqual(sorted(eind["pipeline"]), ["compact", "groot", "medium"])
        echt = {c: float(kans_ja(model, x)[0]) for c, model in eind["pipeline"].items()}
        self.assertEqual(sorted(geprint), sorted(echt))
        for categorie in echt:
            self.assertAlmostEqual(geprint[categorie], echt[categorie], places=3)

        self.assertEqual(gekozen_categorie(resultaat.stdout), kies_categorie(echt, t))

    def test_kies_categorie_met_vaste_waarden(self):
        # Compact gaat voor, dan medium, dan groot, anders geen. Precies op de drempel telt mee.
        def kansen(compact, medium, groot):
            return {"compact": compact, "medium": medium, "groot": groot}

        self.assertEqual(kies_categorie(kansen(0.95, 0.95, 0.95), 0.9), "compact")
        self.assertEqual(kies_categorie(kansen(0.9, 0.1, 0.1), 0.9), "compact")
        self.assertEqual(kies_categorie(kansen(0.1, 0.95, 0.95), 0.9), "medium")
        self.assertEqual(kies_categorie(kansen(0.1, 0.1, 0.95), 0.9), "groot")
        self.assertEqual(kies_categorie(kansen(0.85, 0.85, 0.85), 0.9), "geen")
        self.assertEqual(kies_categorie(kansen(0.85, 0.85, 0.85), 0.6), "compact")
        # Een oud model zonder groot: die categorie telt als "voldoet niet".
        self.assertEqual(kies_categorie({"compact": 0.1, "medium": 0.1}, 0.9), "geen")

    def test_multiclass_model_kiest_de_hoogste_kans(self):
        # Welke kandidaat de selector kiest, hangt af van de testlengte die de split kiest.
        # Daarom maken we de multiclass-run hier zelf, net als de ja/nee-run: alleen op de
        # trainfamilies, en buiten runs/.
        manifest = json.loads((self.map / "split_manifest.json").read_text(encoding="utf-8"))
        scenarios = pd.read_csv(self.map / "scenarios.csv")
        train = scenarios[~scenarios["family_id"].isin(manifest["test_families"])].reset_index(drop=True)
        kandidaat = next(k for k in maak_kandidaten() if k["soort"] == "model")
        # Een multiclass-model gebruikt alleen de trainscenario's, geen configuraties.
        eind = selector.train_eindmodel(kandidaat, train, None, None)

        run_model = self.map / "extra" / "model"
        run_model.mkdir(parents=True)
        shutil.copy(self.run_regel / "DUMMY_run.json", run_model / "DUMMY_run.json")
        joblib.dump(eind, run_model / "DUMMY_model.joblib")

        eind = joblib.load(run_model / "DUMMY_model.joblib")
        self.assertEqual(eind["soort"], "model")

        resultaat = draai_predict("--run", str(run_model), *self.data)
        self.assertEqual(resultaat.returncode, 0, resultaat.stderr)

        kansen = {k: float(v) for k, v in re.findall(r"P\((\w+)\):\s+([\d.]+)", resultaat.stdout)}
        self.assertEqual(sorted(kansen), sorted(eind["klassen"]))
        self.assertEqual(gekozen_categorie(resultaat.stdout), max(kansen, key=kansen.get))


if __name__ == "__main__":
    unittest.main()

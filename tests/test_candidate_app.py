"""
test_candidate_app.py

Tests voor src/candidate/app/koppeling.py en de losse app.py.

De tests maken eigen data en runs met seed 7 in een tijdelijke map, die aan het
eind wordt verwijderd. De echte data in outputs/candidate worden niet aangeraakt,
en --open-test wordt niet gebruikt.

Importeren doet niets: het werk begint pas in setUpClass.

Gebruik (vanuit de hoofdmap van de repo):
    python tests/test_candidate_app.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

# De hoofdmap in sys.path, zodat src.candidate ook werkt als je dit bestand direct draait.
HOOFDMAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOFDMAP))

from src.candidate.app import koppeling
from src.candidate.model import predict_candidate as predict
from src.candidate.model import train_candidate_selector as selector
from src.candidate.model.candidates import maak_kandidaten
from src.candidate.data.labelling_rule import laad_benchmark, maak_configuratietabel, voldoende_categorieen


# Andere data dan de echte (seed 42), en minder scenario's zodat de tests sneller zijn.
SEED = "7"
N = "600"

CPU = "CPU · standaard werkstation"
GPU = "GPU · lokaal werkstation"
ONBEKEND = "Onbekend / nog te bepalen"

# Eén zelfbedacht scenario: in de waarden van de app en in de opties van predict_candidate.
APP_SCENARIO = (GPU, 128, 85, 20.0, 10_000, 0.05)
CLI_SCENARIO = ["--hardware", "rtx3080", "--input-length", "128", "--min-f1", "0.85",
                "--max-latency-ms", "20", "--volume", "10000", "--budget-kwh", "0.05"]


def draai_selector(*argumenten):
    """Draai de selector vanuit de hoofdmap. De uitvoer blijft binnen deze test."""
    uit = subprocess.run(
        [sys.executable, "-m", "src.candidate.model.train_candidate_selector", *argumenten],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
    )
    if uit.returncode != 0:
        raise AssertionError(uit.stderr)


def html_van(app):
    """Alle tekst uit de st.html-blokken van de app."""
    return "\n".join(blok.proto.body for blok in app.get("html"))


class TestKoppeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.map = Path(tempfile.mkdtemp(prefix="test_app_"))
        cls.addClassCleanup(shutil.rmtree, cls.map, ignore_errors=True)
        cls.benchmark = cls.map / "dummy_benchmark.csv"
        cls.manifest = cls.map / "split_manifest.json"

        # Twee runs op dezelfde data, net als in test_candidate_predict.py: bij seed 7 kiest
        # de standaardgrens een regel, en een grens van 1 laat een multiclass-model winnen.
        draai_selector("--regenerate", "--seed", SEED, "--n", N, "--output-dir", str(cls.map))
        cls.run_regel = sorted((cls.map / "runs").iterdir())[-1]
        draai_selector("--output-dir", str(cls.map), "--max-violation-rate", "1")
        cls.run_model = sorted((cls.map / "runs").iterdir())[-1]

        # De selector kiest bij seed 7 geen ja/nee-model, dus die run maken we hier zelf.
        cls.run_ja_nee = cls.map / "extra" / "ja_nee"
        cls.run_ja_nee.mkdir(parents=True)
        shutil.copy(cls.run_regel / "DUMMY_run.json", cls.run_ja_nee / "DUMMY_run.json")
        joblib.dump(cls.maak_ja_nee_eindmodel(), cls.run_ja_nee / "DUMMY_model.joblib")

    @classmethod
    def maak_ja_nee_eindmodel(cls):
        manifest = json.loads(cls.manifest.read_text(encoding="utf-8"))
        scenarios = pd.read_csv(cls.map / "scenarios.csv")
        train = scenarios[~scenarios["family_id"].isin(manifest["test_families"])].reset_index(drop=True)

        configuraties = maak_configuratietabel(laad_benchmark(cls.benchmark))
        configuraties = configuraties[~selector.familie_van(configuraties).isin(manifest["test_families"])]
        voldoende = pd.Series(
            [voldoende_categorieen(configuraties, s) for s in train.to_dict("records")], index=train.index
        )

        kandidaat = next(k for k in maak_kandidaten() if k["soort"] == "ja_nee" and k["instellingen"]["t"] == 0.9)
        return selector.train_eindmodel(kandidaat, train, configuraties, voldoende)

    def laad(self, run):
        return koppeling.laad_voor_app(str(run), self.benchmark, self.manifest)

    def lege_uitvoermap(self):
        """Een lege map die de plaats inneemt van outputs/candidate."""
        leeg = Path(tempfile.mkdtemp(prefix="test_app_leeg_"))
        self.addCleanup(shutil.rmtree, leeg, ignore_errors=True)
        patch = mock.patch.object(predict, "UITVOER_MAP", leeg)
        patch.start()
        self.addCleanup(patch.stop)
        return leeg

    def draai_predict(self, run):
        uit = subprocess.run(
            [sys.executable, "-m", "src.candidate.model.predict_candidate", *CLI_SCENARIO, "--run", str(run),
             "--benchmark", str(self.benchmark), "--manifest", str(self.manifest)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
        )
        self.assertEqual(uit.returncode, 0, uit.stderr)
        return uit.stdout

    def assert_geen_categorie(self, resultaat, *in_reden):
        self.assertIsNone(resultaat["categorie"])
        for tekst in in_reden:
            self.assertIn(tekst, resultaat["reden"])

    def test_kwaliteit_in_procent_wordt_fractie(self):
        for procent, fractie in [(85, 0.85), (0, 0.0), (100, 1.0), (7, 0.07)]:
            scenario = koppeling.naar_scenario(GPU, 128, procent, 20, 10_000, None)
            self.assertEqual(scenario["min_macro_f1"], fractie)

    def test_hardware_koppeling(self):
        self.assertEqual(koppeling.naar_scenario(CPU, 128, 85, 20, 10_000, None)["hardware"], "ryzen5_3600x")
        self.assertEqual(koppeling.naar_scenario(GPU, 128, 85, 20, 10_000, None)["hardware"], "rtx3080")
        self.assertIsNone(koppeling.naar_scenario(ONBEKEND, 128, 85, 20, 10_000, None)["hardware"])

    def test_onbekende_hardware_geeft_reden(self):
        geladen = self.laad(self.run_ja_nee)
        resultaat = koppeling.voorspel_voor_app(ONBEKEND, 128, 85, 20.0, 10_000, None, geladen)
        self.assert_geen_categorie(resultaat, "hardware")

    def test_inputlengte_buiten_bereik_geeft_reden(self):
        geladen = self.laad(self.run_ja_nee)
        for tokens in [31, 513, 1_000]:
            resultaat = koppeling.voorspel_voor_app(GPU, tokens, 85, 20.0, 10_000, None, geladen)
            self.assert_geen_categorie(resultaat, str(tokens), "32 tot en met 512")

        # De randen van het bereik zijn wel gemeten.
        for tokens in [32, 512]:
            resultaat = koppeling.voorspel_voor_app(GPU, tokens, 85, 20.0, 10_000, None, geladen)
            self.assertIsNotNone(resultaat["categorie"])

    def test_geen_run_geeft_reden(self):
        self.lege_uitvoermap()
        self.assertIsNone(koppeling.nieuwste_run())
        self.assert_geen_categorie(koppeling.voorspel_voor_app(*APP_SCENARIO), "Geen run gevonden")

    def test_ja_nee_run_zelfde_als_predict_candidate(self):
        uitvoer = self.draai_predict(self.run_ja_nee)
        resultaat = koppeling.voorspel_voor_app(*APP_SCENARIO, self.laad(self.run_ja_nee))

        self.assertEqual(resultaat["categorie"], re.search(r"gekozen categorie:\s+(\S+)", uitvoer).group(1))
        self.assertEqual(sorted(resultaat["kansen"]), ["compact", "groot", "medium"])
        for categorie, kans in resultaat["kansen"].items():
            self.assertIn(f"{f'P({categorie} voldoet):':<19} {kans:.3f}", uitvoer)
        self.assertIn(f"drempel t:          {resultaat['t']:g}", uitvoer)

        self.assertEqual(resultaat["t"], 0.9)
        self.assertIn(resultaat["kandidaat"], uitvoer)
        self.assertEqual(resultaat["run_map"], str(self.run_ja_nee))
        self.assertTrue(resultaat["is_dummy"])
        self.assertIsNone(resultaat["reden"])

    def start_app(self):
        # De cache is voor het hele proces, dus elke test begint leeg.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        return AppTest.from_file(str(HOOFDMAP / "app.py"), default_timeout=60).run()

    def test_app_start_zonder_run(self):
        self.lege_uitvoermap()
        app = self.start_app()

        self.assertEqual(len(app.exception), 0, app.exception)
        html = html_van(app)
        self.assertIn("Geen categorie", html)
        self.assertIn("Geen run gevonden", html)
        self.assertNotIn("VOORSPELD", html)

    def test_app_toont_de_voorspelling_met_run(self):
        uitvoer = self.lege_uitvoermap()
        app = self.start_app()
        self.assertIn("Geen categorie", html_van(app))

        # Een run die later komt, wordt opgepikt zonder de app te herstarten.
        shutil.copytree(self.run_ja_nee, uitvoer / "runs" / "20000101T000000_000000Z")
        shutil.copy(self.benchmark, uitvoer)
        shutil.copy(self.manifest, uitvoer)

        app.selectbox(key="hardware").select(GPU)
        app.number_input(key="input_tokens").set_value(128)
        app.number_input(key="documents").set_value(10_000)
        app.checkbox(key="has_energy_budget").check().run()
        self.assertEqual(app.number_input(key="energy_budget_kwh").value, 1.0)
        app.number_input(key="energy_budget_kwh").set_value(0.05).run()
        self.assertEqual(len(app.exception), 0, app.exception)

        verwacht = koppeling.voorspel_voor_app(*APP_SCENARIO, self.laad(self.run_ja_nee))
        html = html_van(app)
        self.assertIn(f'<div class="big-value">{verwacht["categorie"]}</div>', html)
        self.assertIn(f"voorspeld door {verwacht['kandidaat']}", html)
        for categorie, kans in verwacht["kansen"].items():
            self.assertIn(f"Kans dat {categorie} voldoet", html)
            self.assertIn(f"{kans:.3f}".replace(".", ","), html)
        self.assertIn("DUMMY benchmarkdata", html)
        # De test is dicht, dus het label uit de benchmark staat er niet bij.
        self.assertNotIn("Volgens de benchmark", html)
        self.assertIn("batch size 1", html)
        self.assertNotIn("Geen categorie", html)

        # Geen advies: het besluit blijft bij de gebruiker (QR-03).
        self.assertNotIn("gebruik dit model", html.lower())
        self.assertNotIn("beste keuze", html.lower())


if __name__ == "__main__":
    unittest.main()

"""
test_candidate_pipeline.py

Tests voor de hele pipeline:
    dummy benchmark -> labelregel -> scenario's -> features -> split

Het script draait de vier scripts zoals je ze zelf ook draait (via de
command line), maar schrijft alles naar een tijdelijke map, die aan het eind
wordt verwijderd. De bestanden in outputs/candidate worden dus niet aangeraakt.
Daarna controleert het per onderdeel of de uitkomst klopt.

Elke test is een gewone functie die een AssertionError (of een andere fout)
geeft als er iets mis is. Aan het eind staat per test OK of FOUT, en het
script eindigt met exitcode 1 als er een test mislukt is.

Gebruik (vanuit de hoofdmap van de repo):
    python tests/test_candidate_pipeline.py
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# De hoofdmap in sys.path, zodat src.candidate ook werkt als je dit bestand direct draait.
HOOFDMAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOOFDMAP))

from src.candidate.data import features
from src.candidate.data import generate_scenarios as generator
from src.candidate.data import labelling_rule as regel
from src.candidate.data import split_scenarios as split


# ---------------------------------------------------------------------------
# Instellingen van de tests
# ---------------------------------------------------------------------------

# Dezelfde instellingen als in het verslag.
SEED = 42
AANTAL_SCENARIOS = 2000

# Een andere seed, om te zien dat die ook echt andere data geeft.
ANDERE_SEED = 7

# Aantal families in de dummy benchmark: 2 hardware x 8 lengtes.
VERWACHT_AANTAL_FAMILIES = 2 * 8

# Aantal scenario's waarop de eigenschapstests draaien (alle 2000 duurt lang).
AANTAL_EIGENSCHAP_SCENARIOS = 300


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def draai_script(naam, *argumenten):
    """Draai een script uit src.candidate vanuit de hoofdmap en geef de output terug."""
    resultaat = subprocess.run(
        [sys.executable, "-m", f"src.candidate.data.{naam}", *argumenten],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HOOFDMAP,
    )
    if resultaat.returncode != 0:
        raise AssertionError(
            f"{naam} stopte met exitcode {resultaat.returncode}:\n{resultaat.stderr}"
        )
    return resultaat.stdout


def sha256(pad):
    """Geef de SHA-256 van een bestand."""
    return hashlib.sha256(Path(pad).read_bytes()).hexdigest()


def maak_pipeline(map_, seed=SEED, n=AANTAL_SCENARIOS):
    """Draai de vier scripts na elkaar in een map. Geeft de paden terug."""
    paden = {
        "benchmark": map_ / "dummy_benchmark.csv",
        "scenarios": map_ / "scenarios.csv",
        "manifest": map_ / "split_manifest.json",
    }
    draai_script("create_dummy_benchmark", "--seed", str(seed), "--out", str(paden["benchmark"]))
    paden["labelregel_output"] = draai_script(
        "labelling_rule", "--benchmark", str(paden["benchmark"])
    )
    draai_script(
        "generate_scenarios", "--seed", str(seed), "--n", str(n),
        "--benchmark", str(paden["benchmark"]), "--out", str(paden["scenarios"]),
    )
    draai_script(
        "split_scenarios", "--scenarios", str(paden["scenarios"]),
        "--benchmark", str(paden["benchmark"]), "--out", str(paden["manifest"]),
    )
    return paden


def label_onafhankelijk(benchmark, scenario):
    """Bereken het label opnieuw, los van labelling_rule.py.

    Bewust anders geschreven (gewone Python-lussen in plaats van pandas), zodat
    een fout in labelling_rule.py hier niet ook in zit. Volgt de regel uit het
    verslag letterlijk.
    """
    # Herhalingen middelen per configuratie in deze context.
    metingen = {}
    for rij in benchmark.itertuples(index=False):
        if rij.hardware != scenario["hardware"]:
            continue
        if rij.input_length_tokens != scenario["input_length_tokens"]:
            continue
        sleutel = (rij.model, rij.precision, rij.category)
        metingen.setdefault(sleutel, []).append(
            (rij.macro_f1, rij.latency_ms_per_doc, rij.energy_j_per_doc)
        )
    if not metingen:
        raise ValueError("Context niet gemeten.")

    budget = scenario["max_energy_kwh_per_month"]
    heeft_budget = budget is not None and not (isinstance(budget, float) and np.isnan(budget))

    voldoende = set()
    for (_, _, categorie), runs in metingen.items():
        f1 = round(sum(r[0] for r in runs) / len(runs), 6)
        latency = round(sum(r[1] for r in runs) / len(runs), 6)
        energie_j = round(sum(r[2] for r in runs) / len(runs), 6)

        if f1 < scenario["min_macro_f1"]:
            continue
        if latency > scenario["max_latency_ms_per_doc"]:
            continue
        if heeft_budget and energie_j * scenario["monthly_volume"] / 3_600_000 > budget:
            continue
        voldoende.add(categorie)

    # De kleinste categorie wint.
    for categorie in ["compact", "medium", "groot"]:
        if categorie in voldoende:
            return categorie
    return "geen"


def verwacht_fout(functie, *argumenten):
    """Controleer dat een functie een ValueError geeft."""
    try:
        functie(*argumenten)
    except ValueError:
        return
    raise AssertionError(f"{functie.__name__} gaf geen ValueError.")


# ---------------------------------------------------------------------------
# De gedeelde data voor alle tests (één keer gemaakt, in een tijdelijke map)
# ---------------------------------------------------------------------------

TIJDELIJKE_MAP = None
PADEN = None
BENCHMARK = None
CONFIGURATIES = None
SCENARIOS = None
MANIFEST = None


def setup_module():
    """Maak de gedeelde data. Dit gebeurt pas als de tests echt draaien, niet bij importeren."""
    global TIJDELIJKE_MAP, PADEN, BENCHMARK, CONFIGURATIES, SCENARIOS, MANIFEST

    TIJDELIJKE_MAP = Path(tempfile.mkdtemp(prefix="test_pipeline_"))
    PADEN = maak_pipeline(TIJDELIJKE_MAP / "run1")
    BENCHMARK = pd.read_csv(PADEN["benchmark"])
    CONFIGURATIES = regel.maak_configuratietabel(BENCHMARK)
    SCENARIOS = pd.read_csv(PADEN["scenarios"])
    MANIFEST = json.loads(Path(PADEN["manifest"]).read_text(encoding="utf-8"))


def teardown_module():
    """Verwijder de tijdelijke map met de gedeelde data."""
    global TIJDELIJKE_MAP

    if TIJDELIJKE_MAP is not None:
        shutil.rmtree(TIJDELIJKE_MAP, ignore_errors=True)
        TIJDELIJKE_MAP = None


# ---------------------------------------------------------------------------
# Tests: labelregel
# ---------------------------------------------------------------------------

def test_omrekening_kwh():
    """kWh/maand = J/doc x volume / 3.600.000."""
    assert regel.energie_kwh_per_maand(3_600_000, 1) == 1
    assert abs(regel.energie_kwh_per_maand(0.36, 100_000) - 0.01) < 1e-12
    assert abs(regel.energie_kwh_per_maand(3.6, 10_000) - 0.01) < 1e-12


def test_voorbeelden_kloppen():
    """Alle handgekozen voorbeelden krijgen het verwachte label."""
    for scenario in regel.VOORBEELD_SCENARIOS:
        label = regel.bepaal_label(CONFIGURATIES, scenario)
        assert label == scenario["verwacht"], (scenario, label)
    assert f"{len(regel.VOORBEELD_SCENARIOS)} van {len(regel.VOORBEELD_SCENARIOS)} voorbeelden kloppen" in PADEN["labelregel_output"]


def test_niet_gemeten_context_geeft_fout():
    """Een context zonder metingen geeft een fout, geen "geen"."""
    basis = {"min_macro_f1": 0.8, "max_latency_ms_per_doc": 100,
             "monthly_volume": 1000, "max_energy_kwh_per_month": None}
    verwacht_fout(regel.bepaal_label, CONFIGURATIES, {**basis, "hardware": "rtx3080", "input_length_tokens": 300})
    verwacht_fout(regel.bepaal_label, CONFIGURATIES, {**basis, "hardware": "tpu", "input_length_tokens": 128})


def test_grens_telt_als_voldoende():
    """Een configuratie die precies op de grens zit, voldoet (>= en <=)."""
    rij = CONFIGURATIES[
        (CONFIGURATIES["hardware"] == "rtx3080") & (CONFIGURATIES["input_length_tokens"] == 128)
        & (CONFIGURATIES["model"] == "BERT-base") & (CONFIGURATIES["precision"] == "fp32")
    ].iloc[0]
    volume = 10_000
    scenario = {
        "hardware": "rtx3080", "input_length_tokens": 128,
        "min_macro_f1": rij["macro_f1"], "max_latency_ms_per_doc": rij["latency_ms_per_doc"],
        "monthly_volume": volume,
        "max_energy_kwh_per_month": regel.energie_kwh_per_maand(rij["energy_j_per_doc"], volume),
    }
    in_context = regel.configuraties_voor_context(CONFIGURATIES, scenario)
    voldoet = regel.voldoet_aan_eisen(in_context, scenario)
    assert voldoet[in_context.index == rij.name].all()


def test_none_en_nan_betekenen_geen_budget():
    """Een leeg budget (None of NaN) geeft hetzelfde label als helemaal geen eis op energie."""
    for _, rij in SCENARIOS.head(AANTAL_EIGENSCHAP_SCENARIOS).iterrows():
        scenario = rij.to_dict()
        met_none = regel.bepaal_label(CONFIGURATIES, {**scenario, "max_energy_kwh_per_month": None})
        met_nan = regel.bepaal_label(CONFIGURATIES, {**scenario, "max_energy_kwh_per_month": float("nan")})
        zonder_sleutel = {k: v for k, v in scenario.items() if k != "max_energy_kwh_per_month"}
        assert met_none == met_nan == regel.bepaal_label(CONFIGURATIES, zonder_sleutel)


def test_kleinste_categorie_wint():
    """Als meer categorieën voldoen, wint de kleinste: compact, dan medium, dan groot."""
    scenario = {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.0,
                "max_latency_ms_per_doc": 1e9, "monthly_volume": 1, "max_energy_kwh_per_month": None}
    assert regel.voldoende_categorieen(CONFIGURATIES, scenario) == {"compact", "medium", "groot"}
    assert regel.bepaal_label(CONFIGURATIES, scenario) == "compact"
    # Zonder compact (te hoge kwaliteit voor DistilBERT) wint medium boven groot.
    zonder_compact = {**scenario, "min_macro_f1": 0.86}
    assert regel.voldoende_categorieen(CONFIGURATIES, zonder_compact) == {"medium", "groot"}
    assert regel.bepaal_label(CONFIGURATIES, zonder_compact) == "medium"


def test_labelregel_gelijk_aan_onafhankelijke_versie():
    """Alle labels in de CSV zijn gelijk aan een los geschreven versie van de regel."""
    verschillen = []
    for _, rij in SCENARIOS.iterrows():
        scenario = rij.to_dict()
        if label_onafhankelijk(BENCHMARK, scenario) != scenario["label"]:
            verschillen.append(scenario["scenario_id"])
    assert not verschillen, f"{len(verschillen)} verschillen, bijv. {verschillen[:5]}"


def test_ruimere_eis_geeft_nooit_minder_categorieen():
    """Ruimere latency, lagere F1-eis, groter budget of lager volume: de voldoende categorieën worden nooit minder."""
    for _, rij in SCENARIOS.head(AANTAL_EIGENSCHAP_SCENARIOS).iterrows():
        scenario = rij.to_dict()
        basis = regel.voldoende_categorieen(CONFIGURATIES, scenario)

        ruimer = [
            {**scenario, "max_latency_ms_per_doc": scenario["max_latency_ms_per_doc"] * 2},
            {**scenario, "min_macro_f1": scenario["min_macro_f1"] - 0.02},
        ]
        if scenario["has_energy_budget"]:
            ruimer.append({**scenario, "max_energy_kwh_per_month": scenario["max_energy_kwh_per_month"] * 2})
            ruimer.append({**scenario, "monthly_volume": scenario["monthly_volume"] / 2})
        for ander in ruimer:
            assert basis <= regel.voldoende_categorieen(CONFIGURATIES, ander), (scenario, ander)


# ---------------------------------------------------------------------------
# Tests: scenariogenerator
# ---------------------------------------------------------------------------

def test_scenarios_reproduceerbaar():
    """Zelfde seed geeft hetzelfde (ook bij een kleinere n), een andere seed iets anders."""
    klein = generator.maak_scenarios(BENCHMARK, 100, SEED)
    pd.testing.assert_frame_equal(klein, SCENARIOS.head(100), check_dtype=False)
    ander = generator.maak_scenarios(BENCHMARK, 100, ANDERE_SEED)
    assert not klein.drop(columns="scenario_id").equals(ander.drop(columns="scenario_id"))


def test_labels_uit_csv_opnieuw_berekenen():
    """De labels uit de opgeslagen CSV zijn gelijk aan de labelregel (lege kWh-cel = NaN)."""
    opnieuw = SCENARIOS.apply(lambda rij: regel.bepaal_label(CONFIGURATIES, rij.to_dict()), axis=1)
    assert (opnieuw == SCENARIOS["label"]).all()


# ---------------------------------------------------------------------------
# Tests: features
# ---------------------------------------------------------------------------

def test_features_alleen_gebruikersinvoer():
    """X bevat precies de featurekolommen, zonder gemeten uitkomsten of id's."""
    x, y = features.selecteer_features(SCENARIOS)
    assert list(x.columns) == features.FEATURE_KOLOMMEN
    assert len(set(features.FEATURE_KOLOMMEN)) == len(features.FEATURE_KOLOMMEN)
    assert not set(x.columns) & set(features.GEEN_FEATURES + features.GEMETEN_KOLOMMEN)
    assert y.name == features.LABEL_KOLOM and len(y) == len(SCENARIOS.index)
    # input_length_tokens blijft numeriek.
    assert pd.api.types.is_numeric_dtype(x["input_length_tokens"])


def test_features_vangen_lekken():
    """Gemeten kolommen, id's als feature en ontbrekende kolommen geven een fout."""
    verwacht_fout(features.selecteer_features, SCENARIOS.assign(macro_f1=0.85))
    verwacht_fout(features.selecteer_features, SCENARIOS.assign(category="compact"))
    verwacht_fout(features.selecteer_features, SCENARIOS.drop(columns="monthly_volume"))

    origineel = features.FEATURE_KOLOMMEN[:]
    try:
        for verboden in ["family_id", "label", "energy_j_per_doc", "config_id"]:
            features.FEATURE_KOLOMMEN[:] = origineel + [verboden]
            verwacht_fout(features.controleer_alleen_gebruikersinvoer, SCENARIOS)
    finally:
        features.FEATURE_KOLOMMEN[:] = origineel


# ---------------------------------------------------------------------------
# Tests: groepsgewijze split
# ---------------------------------------------------------------------------

def test_split_families_gescheiden():
    """Elke familie zit in precies één deel, en testscenario's hebben geen fold."""
    toegewezen = split.pas_manifest_toe(SCENARIOS, MANIFEST)
    # Elke familie heeft één split en één fold.
    assert (toegewezen.groupby("family_id")["split"].nunique() == 1).all()
    assert (toegewezen.groupby("family_id")["cv_fold"].nunique(dropna=False) == 1).all()
    assert toegewezen["family_id"].nunique() == VERWACHT_AANTAL_FAMILIES

    test = toegewezen[toegewezen["split"] == split.SPLIT_TEST]
    train = toegewezen[toegewezen["split"] == split.SPLIT_TRAIN]
    assert test["cv_fold"].isna().all()
    assert train["cv_fold"].notna().all()
    assert test["input_length_tokens"].nunique() == split.AANTAL_TESTLENGTES
    assert not set(test["input_length_tokens"]) & set(train["input_length_tokens"])


def test_split_datalekcontroles_slagen():
    """Op de echte split slagen alle datalekcontroles."""
    manifest, toegewezen = split.maak_manifest(SCENARIOS, PADEN["scenarios"], PADEN["benchmark"])
    controles = split.controleer_datalek(SCENARIOS, toegewezen, manifest)
    mislukt = [naam for naam, ok in controles if not ok]
    assert not mislukt, mislukt
    assert len(controles) == 9


def test_split_datalekcontroles_vangen_fouten():
    """Expres ingebouwde fouten worden door de controles gevangen."""
    # Randlengte en niet-bestaande familie: foutmelding.
    for families_ in (["rtx3080__512"], ["rtx3080__300"]):
        with mock.patch.object(split, "kies_test_families", return_value=families_):
            verwacht_fout(split.maak_manifest, SCENARIOS, PADEN["scenarios"], PADEN["benchmark"])

    # Alleen de GPU-familie als test: de testlengte zit dan via de CPU in train.
    gpu = [f for f in MANIFEST["test_families"] if f.startswith("rtx3080__")]
    with mock.patch.object(split, "kies_test_families", return_value=gpu):
        manifest, toegewezen = split.maak_manifest(SCENARIOS, PADEN["scenarios"], PADEN["benchmark"])
    mislukt = [n for n, ok in split.controleer_datalek(SCENARIOS, toegewezen, manifest) if not ok]
    assert mislukt == ["Testinputlengte komt niet voor in train"], mislukt

    # Testfamilie ook in een fold, en een trainfamilie in twee folds.
    for extra, verwachte_fout in (
        (MANIFEST["test_families"][0], "Geen testscenario in een fold"),
        (MANIFEST["cv_folds"][1]["families"][0], "Elke trainfamilie zit in precies 1 fold"),
    ):
        manifest = json.loads(json.dumps(MANIFEST))
        manifest["cv_folds"][0]["families"].append(extra)
        toegewezen = split.pas_manifest_toe(SCENARIOS, manifest)
        mislukt = [n for n, ok in split.controleer_datalek(SCENARIOS, toegewezen, manifest) if not ok]
        assert verwachte_fout in mislukt, mislukt


# ---------------------------------------------------------------------------
# Tests: de hele pipeline
# ---------------------------------------------------------------------------

def test_pipeline_twee_keer_identiek():
    """De hele pipeline nog een keer draaien geeft byte-voor-byte dezelfde bestanden."""
    tweede = maak_pipeline(TIJDELIJKE_MAP / "run2")
    for soort in ["benchmark", "scenarios"]:
        assert sha256(tweede[soort]) == sha256(PADEN[soort]), soort
    assert sha256(tweede["manifest"]) == sha256(PADEN["manifest"])


# ---------------------------------------------------------------------------
# Tests draaien
# ---------------------------------------------------------------------------

def main():
    setup_module()

    # Met finally wordt de tijdelijke map ook opgeruimd als er een test mislukt.
    try:
        tests = [(naam, functie) for naam, functie in globals().items()
                 if naam.startswith("test_") and callable(functie)]

        mislukt = []
        for naam, functie in tests:
            try:
                functie()
                print(f"[OK]   {naam}")
            except Exception:
                mislukt.append(naam)
                print(f"[FOUT] {naam}")
                print("       " + traceback.format_exc().strip().replace("\n", "\n       "))

        print()
        print(f"{len(tests) - len(mislukt)} van {len(tests)} tests geslaagd.")
        if mislukt:
            sys.exit(1)
    finally:
        teardown_module()


if __name__ == "__main__":
    main()

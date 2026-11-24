"""
train_candidate_selector.py

Voert de hele kandidaatselectie uit: data controleren,
alle kandidaten evalueren op de CV-folds, één kandidaat kiezen, die opnieuw
trainen op alle trainfamilies, en alles vastleggen in een runmap.

De kandidaten staan in candidates.py. Hun instellingen komen uit 04_modelverbeteringen.ipynb:
    regels:  baseline, opzoekregel (stap 1), conservatieve opzoekregel variant b (stap 11)
    modellen: logistische regressie en beslisboom (stap 7),
              HistGradientBoosting (stap 9), ja/nee-HGB monotoon (stap 10, per t apart)

Selectieregel (vast in de code, gemeten op de middelste folds, die worden uit de
folds in het manifest afgeleid: alle folds behalve de kortste en de langste lengte):
    1. Houd de kandidaten over met violation rate <= --max-violation-rate.
    2. Kies daaruit de hoogste macro-F1.
    3. Zit een eenvoudiger kandidaat (lagere complexiteit) daar hoogstens 0,01
       macro-F1 onder, dan wint die. Zijn er meer, dan de eenvoudigste.
    4. Haalt geen kandidaat de grens, kies dan de laagste violation rate en
       geef een duidelijke waarschuwing.
    Complexiteit, van eenvoudig naar ingewikkeld: regels, logistische regressie,
    beslisboom, HGB, ja/nee-HGB.

De testfamilies blijven standaard dicht. De datalekcontroles uit
split_scenarios kijken wel naar de testrijen, maar daarvan wordt alleen
geslaagd of mislukt gebruikt. Daarna worden die rijen weggegooid: er wordt niets
op getraind of geëvalueerd, en er wordt niets over geprint of opgeslagen. Met
--open-test wordt de gekozen kandidaat één keer op de testfamilies geëvalueerd,
naast de baseline en beide opzoekregels.

--regenerate met een andere seed of n dan de standaard stopt met een foutmelding
als de data in outputs/candidate zou komen, want dat overschrijft de echte data.
Gebruik dan --output-dir, of --force om toch te overschrijven.

Gebruik (vanuit de hoofdmap van de repo):
    python -m src.candidate.model.train_candidate_selector
    python -m src.candidate.model.train_candidate_selector --regenerate --seed 7 --n 500 --output-dir outputs/candidate/proef
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score

from src.candidate.model.candidates import (
    BASELINE,
    CONSERVATIEF,
    OPZOEK,
    REGELS,
    kans_ja,
    kies_categorie,
    kolommen_ja_nee,
    maak_kandidaten,
    maak_model,
    maak_pipeline,
)
from src.candidate.data.create_dummy_benchmark import maak_dummy_benchmark
from src.candidate.data.features import FEATURE_KOLOMMEN, selecteer_features
from src.candidate.data.generate_scenarios import maak_scenarios
from src.candidate.data.labelling_rule import (
    CATEGORIE_VOORKEUR,
    LABEL_GEEN,
    laad_benchmark,
    maak_configuratietabel,
    voldoende_categorieen,
)
from src.candidate.data.split_scenarios import (
    controleer_datalek,
    maak_manifest,
    pas_manifest_toe,
    sha256_van_bestand,
)


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# De hoofdmap van de repo, en de map met de gegenereerde data.
HOOFDMAP = Path(__file__).resolve().parents[3]
UITVOER_MAP = HOOFDMAP / "outputs" / "candidate"

BENCHMARK_BESTAND = "dummy_benchmark.csv"
SCENARIOS_BESTAND = "scenarios.csv"
MANIFEST_BESTAND = "split_manifest.json"


# ---------------------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------------------

# Standaardwaarden voor --regenerate.
STANDAARD_SEED = 42
STANDAARD_N = 2000

# De klassen, in de volgorde van de confusion matrix.
LABELS = CATEGORIE_VOORKEUR + [LABEL_GEEN]

# Selectieregel: de standaardgrens op de violation rate, en hoeveel macro-F1 een
# eenvoudiger kandidaat mag inleveren.
STANDAARD_MAX_VIOLATION = 0.05
MARGE_EENVOUDIGER = 0.01


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def herbouw_data(map_, seed, n):
    """Maak benchmark, scenario's en manifest opnieuw met de bestaande functies.

    De print_samenvatting-functies worden hier bewust niet gebruikt: die tonen
    ook cijfers van de testfamilies.
    """
    map_.mkdir(parents=True, exist_ok=True)
    benchmark_pad = map_ / BENCHMARK_BESTAND
    scenarios_pad = map_ / SCENARIOS_BESTAND

    maak_dummy_benchmark(seed).to_csv(benchmark_pad, index=False, lineterminator="\n")
    maak_scenarios(laad_benchmark(benchmark_pad), n, seed).to_csv(scenarios_pad, index=False, lineterminator="\n")

    scenarios = pd.read_csv(scenarios_pad)
    manifest, toegewezen = maak_manifest(scenarios, scenarios_pad, benchmark_pad)
    controleer_datalekken(scenarios, toegewezen, manifest)
    (map_ / MANIFEST_BESTAND).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def controleer_regenerate(data_map, seed, n, force):
    """Stop als --regenerate de echte data zou overschrijven met andere data.

    De opgeloste paden worden vergeleken, zodat ook --output-dir outputs/candidate
    als de standaardmap telt. Seed en n van de standaard geven dezelfde bestanden.
    """
    is_standaardmap = Path(data_map).resolve() == UITVOER_MAP.resolve()
    andere_data = seed != STANDAARD_SEED or n != STANDAARD_N

    if is_standaardmap and andere_data and not force:
        raise SystemExit(
            f"--regenerate met seed {seed} en n {n} zou de echte data in {UITVOER_MAP} overschrijven, "
            f"en dan klopt die niet meer met eerdere runs. Kies een andere map met --output-dir, "
            f"of gebruik --force om toch te overschrijven."
        )


def controleer_sha(manifest, benchmark_pad, scenarios_pad):
    """Stop als de CSV's niet de versie zijn waar het manifest bij hoort."""
    for naam, pad, sleutel in [("benchmark", benchmark_pad, "benchmark_csv"),
                               ("scenario's", scenarios_pad, "scenarios_csv")]:
        if sha256_van_bestand(pad) != manifest[sleutel]["sha256"]:
            raise SystemExit(f"SHA-256 van de {naam} ({pad}) klopt niet met het splitmanifest.")


def controleer_datalekken(scenarios, toegewezen, manifest):
    """Stop als een datalekcontrole mislukt. Geeft het aantal controles terug."""
    controles = controleer_datalek(scenarios, toegewezen, manifest)
    mislukt = [naam for naam, ok in controles if not ok]
    if mislukt:
        raise SystemExit(f"Datalekcontrole mislukt: {mislukt}")
    return len(controles)


def familie_van(configuraties):
    """Geef per configuratie de family_id."""
    return configuraties["hardware"] + "__" + configuraties["input_length_tokens"].astype(str)


# ---------------------------------------------------------------------------
# Cross-validatie
# ---------------------------------------------------------------------------

def voorspel_model(model, x):
    """De klasse met de hoogste kans, zoals in het notebook."""
    kansen = pd.DataFrame(model.predict_proba(x), columns=model.classes_)
    return kansen.idxmax(axis=1).tolist()


def maak_ja_nee_doelen(voldoende):
    """Per categorie het target "deze categorie voldoet". Dit zijn labels, geen features."""
    return {c: voldoende.map(lambda v, c=c: c in v) for c in CATEGORIE_VOORKEUR}


def train_ja_nee(x, doelen, kolommen):
    """Train één ja/nee-model per categorie."""
    return {c: maak_model(kolommen).fit(x, doel) for c, doel in doelen.items()}


def kies_met_ja_nee(modellen, x, t):
    """Voorspel met de ja/nee-modellen: per rij de kleinste categorie met kans >= t."""
    kansen = {c: kans_ja(model, x) for c, model in modellen.items()}
    return [kies_categorie({c: kansen[c][i] for c in kansen}, t) for i in range(len(x))]


def cv_voorspellingen(kandidaten, train, train_configuraties, manifest, voldoende):
    """Voorspel elk trainscenario met een kandidaat die die fold niet heeft gezien."""
    x, y = selecteer_features(train)
    doelen = maak_ja_nee_doelen(voldoende)
    kolommen = kolommen_ja_nee(x)
    familie = familie_van(train_configuraties)

    uit = {k["naam"]: pd.Series(index=train.index, dtype=object) for k in kandidaten}
    for fold in manifest["cv_folds"]:
        in_fold = train["family_id"].isin(fold["families"])
        fold_configuraties = train_configuraties[~familie.isin(fold["families"])]

        for kandidaat in kandidaten:
            if kandidaat["soort"] == "regel":
                uit[kandidaat["naam"]][in_fold] = REGELS[kandidaat["naam"]](train[in_fold], fold_configuraties)
            elif kandidaat["soort"] == "model":
                model = maak_pipeline(kandidaat).fit(x[~in_fold], y[~in_fold])
                uit[kandidaat["naam"]][in_fold] = voorspel_model(model, x[in_fold])

        # De ja/nee-modellen één keer per fold. Alleen de drempel verschilt per kandidaat.
        modellen = train_ja_nee(x[~in_fold], {c: d[~in_fold] for c, d in doelen.items()}, kolommen)
        for kandidaat in kandidaten:
            if kandidaat["soort"] == "ja_nee":
                t = kandidaat["instellingen"]["t"]
                uit[kandidaat["naam"]][in_fold] = kies_met_ja_nee(modellen, x[in_fold], t)

    return uit


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def scores(label, voorspeld, voldoende):
    """Dezelfde maten en definities als 01_baseline.ipynb. Alle drie zijn lijsten of Series."""
    echt = pd.Series(list(label))
    keuze = pd.Series(list(voorspeld))
    passend = list(voldoende)

    matrix = confusion_matrix(echt, keuze, labels=LABELS)

    # Violations: haalt de gekozen categorie de eisen niet?
    gekozen = keuze != LABEL_GEEN
    violations = sum(k not in p for k, p in zip(keuze, passend) if k != LABEL_GEEN)

    # Gemiste kansen: "geen" voorspeld terwijl er wel een categorie voldoet.
    kansen = echt != LABEL_GEEN
    gemist = int((keuze[kansen] == LABEL_GEEN).sum())

    uit = {
        "macro_f1": float(f1_score(echt, keuze, labels=LABELS, average="macro", zero_division=0)),
        "violations": int(violations),
        "keuzes": int(gekozen.sum()),
        "violation_rate": float(violations / gekozen.sum()) if gekozen.sum() > 0 else 0.0,
        "gemiste_kansen": gemist,
        "kansen": int(kansen.sum()),
        "gemiste_kans_rate": float(gemist / kansen.sum()) if kansen.sum() > 0 else 0.0,
    }
    # De confusion matrix als losse kolommen: cm_<echt>_<voorspeld>.
    for i, echt_label in enumerate(LABELS):
        for j, voorspeld_label in enumerate(LABELS):
            uit[f"cm_{echt_label}_{voorspeld_label}"] = int(matrix[i, j])
    return uit


def evalueer(kandidaten, voorspellingen, train, voldoende, manifest):
    """Maak de tabellen: per fold (cv_results) en totaal/midden/rand (summary)."""
    per_fold, samenvatting = [], []
    lengte = train["input_length_tokens"]

    # De middelste folds en de randfolds. Op de randfolds moet een model extrapoleren.
    foldlengtes = sorted(fold["input_length_tokens"] for fold in manifest["cv_folds"])
    MIDDEN, RAND = foldlengtes[1:-1], [foldlengtes[0], foldlengtes[-1]]

    for kandidaat in kandidaten:
        naam = kandidaat["naam"]
        voorspeld = voorspellingen[naam]

        def score_van(masker):
            return scores(train.loc[masker, "label"], voorspeld[masker], voldoende[masker])

        for fold in manifest["cv_folds"]:
            in_fold = train["family_id"].isin(fold["families"])
            per_fold.append({"kandidaat": naam, "fold": fold["fold"],
                             "input_length_tokens": fold["input_length_tokens"],
                             "families": " ".join(fold["families"]), **score_van(in_fold)})

        delen = {"totaal": pd.Series(True, index=train.index), "midden": lengte.isin(MIDDEN)}
        for rand in RAND:
            delen[f"rand_{rand}"] = lengte == rand
        for deel, masker in delen.items():
            samenvatting.append({"kandidaat": naam, "complexiteit": kandidaat["complexiteit"],
                                 "deel": deel, **score_van(masker)})

    return pd.DataFrame(per_fold), pd.DataFrame(samenvatting)


# ---------------------------------------------------------------------------
# Selectie
# ---------------------------------------------------------------------------

def getal(waarde, cijfers=3):
    """Schrijf een getal met een komma, zoals in de documenten."""
    return f"{waarde:.{cijfers}f}".replace(".", ",")


def procent(waarde):
    return getal(100 * waarde, 1) + "%"


def kies_kandidaat(midden, max_violation):
    """Pas de selectieregel toe. midden: dicts met naam, complexiteit, macro_f1 en violation_rate.

    Geeft (naam, reden, waarschuwing). De waarschuwing is None als de grens gehaald is.
    """
    toegelaten = [k for k in midden if k["violation_rate"] <= max_violation]

    if not toegelaten:
        keuze = min(midden, key=lambda k: (k["violation_rate"], -k["macro_f1"], k["complexiteit"]))
        waarschuwing = (
            f"Geen enkele kandidaat haalt de grens van {procent(max_violation)} violation rate "
            f"op de middelste folds. De laagste is {procent(keuze['violation_rate'])}."
        )
        reden = (
            f"Geen kandidaat haalt de grens, dus de laagste violation rate op de middelste folds: "
            f"{procent(keuze['violation_rate'])} (macro-F1 {getal(keuze['macro_f1'])})."
        )
        return keuze["naam"], reden, waarschuwing

    beste = max(toegelaten, key=lambda k: (k["macro_f1"], -k["violation_rate"], -k["complexiteit"]))
    reden = (
        f"{len(toegelaten)} van {len(midden)} kandidaten halen de grens van {procent(max_violation)} "
        f"violation rate op de middelste folds. Daarvan heeft {beste['naam']} de hoogste macro-F1 "
        f"({getal(beste['macro_f1'])}, violation rate {procent(beste['violation_rate'])})."
    )

    eenvoudiger = [k for k in toegelaten if k["complexiteit"] < beste["complexiteit"]
                   and beste["macro_f1"] - k["macro_f1"] <= MARGE_EENVOUDIGER]
    if eenvoudiger:
        keuze = min(eenvoudiger, key=lambda k: (k["complexiteit"], -k["macro_f1"]))
        reden += (
            f" {keuze['naam']} is eenvoudiger en zit daar hoogstens {getal(MARGE_EENVOUDIGER, 2)} onder "
            f"(macro-F1 {getal(keuze['macro_f1'])}, violation rate {procent(keuze['violation_rate'])}), "
            f"dus die wint."
        )
        return keuze["naam"], reden, None

    reden += f" Geen eenvoudiger kandidaat zit hoogstens {getal(MARGE_EENVOUDIGER, 2)} macro-F1 lager."
    return beste["naam"], reden, None


def maak_ranglijst(samenvatting, max_violation):
    """De ranglijst: eerst wie de grens haalt, daarbinnen op macro-F1 op de middelste folds."""
    midden = samenvatting[samenvatting["deel"] == "midden"]
    rijen = pd.DataFrame({
        "kandidaat": midden["kandidaat"],
        "complexiteit": midden["complexiteit"],
        "macro_f1_midden": midden["macro_f1"].round(3),
        "violations_midden": midden["violations"].astype(str) + " van " + midden["keuzes"].astype(str),
        "violation_rate_midden": (100 * midden["violation_rate"]).round(1),
        "gemiste_kans_rate_midden": (100 * midden["gemiste_kans_rate"]).round(1),
        "haalt_grens": midden["violation_rate"] <= max_violation,
    })
    rijen = rijen.sort_values(["haalt_grens", "macro_f1_midden"], ascending=[False, False])
    rijen.insert(0, "rang", range(1, len(rijen) + 1))
    return rijen


# ---------------------------------------------------------------------------
# Eindmodel
# ---------------------------------------------------------------------------

def train_eindmodel(kandidaat, train, train_configuraties, voldoende):
    """Train de gekozen kandidaat op alle trainfamilies. Bij een regel: de instellingen."""
    x, y = selecteer_features(train)
    eind = {"naam": kandidaat["naam"], "soort": kandidaat["soort"], "instellingen": kandidaat["instellingen"],
            "klassen": LABELS, "t": None, "features": FEATURE_KOLOMMEN}

    if kandidaat["soort"] == "regel":
        # Een regel heeft geen model: de drempels en metingen zijn zijn instellingen.
        eind["gemeten_lengtes"] = sorted(int(l) for l in train_configuraties["input_length_tokens"].unique())
        eind["configuraties"] = train_configuraties.to_dict("records")
    elif kandidaat["soort"] == "model":
        eind["pipeline"] = maak_pipeline(kandidaat).fit(x, y)
        eind["klassen"] = list(eind["pipeline"].classes_)
    else:
        eind["pipeline"] = train_ja_nee(x, maak_ja_nee_doelen(voldoende), kolommen_ja_nee(x))
        eind["t"] = kandidaat["instellingen"]["t"]

    return eind


def voorspel_eindmodel(eind, scenarios, train_configuraties):
    """Voorspel nieuwe scenario's met het eindmodel."""
    if eind["soort"] == "regel":
        return REGELS[eind["naam"]](scenarios, train_configuraties)

    x, _ = selecteer_features(scenarios)
    if eind["soort"] == "model":
        return voorspel_model(eind["pipeline"], x)

    return kies_met_ja_nee(eind["pipeline"], x, eind["t"])


def evalueer_test(eind, test, configuraties, train_configuraties):
    """Evalueer één keer op de testfamilies: de keuze, de baseline en beide opzoekregels."""
    voldoende = [voldoende_categorieen(configuraties, s) for s in test.to_dict("records")]

    resultaat = {eind["naam"]: scores(test["label"], voorspel_eindmodel(eind, test, train_configuraties), voldoende)}
    for naam in [BASELINE, OPZOEK, CONSERVATIEF]:
        if naam not in resultaat:
            resultaat[naam] = scores(test["label"], REGELS[naam](test, train_configuraties), voldoende)
    return resultaat


# ---------------------------------------------------------------------------
# Runmap
# ---------------------------------------------------------------------------

def git_commit():
    """Geef de huidige git-commit, of None als dat niet lukt."""
    try:
        uitvoer = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HOOFDMAP, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return uitvoer.stdout.strip() if uitvoer.returncode == 0 else None


def lees_argumenten():
    """Lees de command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="Voer de kandidaatselectie uit."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(UITVOER_MAP),
        help="Map met de data en de runs (standaard: outputs/candidate).",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Maak benchmark, scenario's en manifest eerst opnieuw in --output-dir.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sla de controle over die voorkomt dat --regenerate met een andere seed of n "
             "de data in outputs/candidate overschrijft.",
    )
    parser.add_argument(
        "--seed", type=int, default=STANDAARD_SEED,
        help=f"Seed voor --regenerate (standaard: {STANDAARD_SEED}).",
    )
    parser.add_argument(
        "--n", type=int, default=STANDAARD_N,
        help=f"Aantal scenario's voor --regenerate (standaard: {STANDAARD_N}).",
    )
    parser.add_argument(
        "--max-violation-rate", type=float, default=STANDAARD_MAX_VIOLATION,
        help=f"Hoogste violation rate op de middelste folds (standaard: {STANDAARD_MAX_VIOLATION}).",
    )
    parser.add_argument(
        "--open-test", action="store_true",
        help="Evalueer de gekozen kandidaat op de testfamilies. Doe dit pas aan het eind, één keer.",
    )
    return parser.parse_args()


def main():
    args = lees_argumenten()
    data_map = Path(args.output_dir)
    benchmark_pad = data_map / BENCHMARK_BESTAND
    scenarios_pad = data_map / SCENARIOS_BESTAND
    manifest_pad = data_map / MANIFEST_BESTAND

    if args.regenerate:
        controleer_regenerate(data_map, args.seed, args.n, args.force)
        herbouw_data(data_map, args.seed, args.n)

    for pad in [benchmark_pad, scenarios_pad, manifest_pad]:
        if not pad.exists():
            raise SystemExit(f"{pad} bestaat niet. Maak de data eerst, of gebruik --regenerate.")

    manifest = json.loads(manifest_pad.read_text(encoding="utf-8"))
    controleer_sha(manifest, benchmark_pad, scenarios_pad)

    benchmark = laad_benchmark(benchmark_pad)
    is_dummy = "is_dummy" in benchmark.columns and bool(benchmark["is_dummy"].all())
    if is_dummy:
        print("LET OP: DUMMY data")
        print()

    # De datalekcontroles kijken naar alle rijen, maar geven alleen goed of fout terug.
    scenarios = pd.read_csv(scenarios_pad)
    toegewezen = pas_manifest_toe(scenarios, manifest)
    aantal_controles = controleer_datalekken(scenarios, toegewezen, manifest)
    print(f"SHA-256 klopt met het manifest, {aantal_controles} datalekcontroles geslaagd.")

    # Vanaf hier zijn de testfamilies weg, tenzij --open-test ze aan het eind nodig heeft.
    test_families = manifest["test_families"]
    configuraties = maak_configuratietabel(benchmark)
    is_test_scenario = scenarios["family_id"].isin(test_families)
    train = scenarios[~is_test_scenario].reset_index(drop=True)
    train_configuraties = configuraties[~familie_van(configuraties).isin(test_families)]
    test = scenarios[is_test_scenario].reset_index(drop=True) if args.open_test else None
    if not args.open_test:
        del configuraties
    del scenarios, toegewezen, benchmark

    print(f"{len(train.index)} scenario's uit de trainfamilies, {len(train_configuraties.index)} configuraties.")
    print()

    voldoende = pd.Series(
        [voldoende_categorieen(train_configuraties, s) for s in train.to_dict("records")], index=train.index
    )

    kandidaten = maak_kandidaten()
    voorspellingen = cv_voorspellingen(kandidaten, train, train_configuraties, manifest, voldoende)
    per_fold, samenvatting = evalueer(kandidaten, voorspellingen, train, voldoende, manifest)

    midden = samenvatting[samenvatting["deel"] == "midden"].rename(columns={"kandidaat": "naam"}).to_dict("records")
    gekozen_naam, reden, waarschuwing = kies_kandidaat(midden, args.max_violation_rate)
    gekozen = next(k for k in kandidaten if k["naam"] == gekozen_naam)

    print(f"Ranglijst (middelste folds, grens violation rate {procent(args.max_violation_rate)}):")
    print(maak_ranglijst(samenvatting, args.max_violation_rate).to_string(index=False))
    print()
    if waarschuwing:
        print(f"WAARSCHUWING: {waarschuwing}")
        print()
    print(f"Gekozen kandidaat: {gekozen_naam}")
    print(f"Reden: {reden}")
    print()

    eind = train_eindmodel(gekozen, train, train_configuraties, voldoende)

    tijdstip = datetime.now(timezone.utc)
    run_map = data_map / "runs" / tijdstip.strftime("%Y%m%dT%H%M%S_%fZ")
    run_map.mkdir(parents=True)
    voorvoegsel = "DUMMY_" if is_dummy else ""

    per_fold.to_csv(run_map / f"{voorvoegsel}cv_results.csv", index=False, lineterminator="\n")
    samenvatting.to_csv(run_map / f"{voorvoegsel}summary.csv", index=False, lineterminator="\n")
    joblib.dump(eind, run_map / f"{voorvoegsel}model.joblib")

    run = {
        "timestamp": tijdstip.isoformat(),
        "seed": args.seed,
        "n": manifest["scenarios_csv"]["aantal_scenarios"],
        "cli_argumenten": vars(args),
        "sha256": {
            "benchmark": sha256_van_bestand(benchmark_pad),
            "scenarios": sha256_van_bestand(scenarios_pad),
            "manifest": sha256_van_bestand(manifest_pad),
        },
        "git_commit": git_commit(),
        "features": FEATURE_KOLOMMEN,
        "instellingen_per_kandidaat": {k["naam"]: k["instellingen"] for k in kandidaten},
        "max_violation_rate": args.max_violation_rate,
        "gekozen_kandidaat": gekozen_naam,
        "reden": reden,
        "waarschuwing": waarschuwing,
        "test_opened": bool(args.open_test),
        "test_opened_at": None,
        "versies": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "is_dummy": is_dummy,
    }

    if args.open_test:
        test_scores = evalueer_test(eind, test, configuraties, train_configuraties)
        run["test_opened_at"] = datetime.now(timezone.utc).isoformat()
        run["test_scores"] = test_scores

        print("Testfamilies (één keer geopend):")
        tabel = pd.DataFrame(test_scores).T[
            ["macro_f1", "violations", "keuzes", "violation_rate", "gemiste_kansen", "kansen", "gemiste_kans_rate"]
        ]
        print(tabel.to_string())
        print()

    (run_map / f"{voorvoegsel}run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Run opgeslagen in: {run_map}")
    if not args.open_test:
        print("De testfamilies zijn niet gebruikt (open ze aan het eind met --open-test).")


if __name__ == "__main__":
    main()

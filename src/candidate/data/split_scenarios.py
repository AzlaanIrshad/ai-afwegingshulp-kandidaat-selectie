"""
split_scenarios.py

Groepsgewijze split voor de kandidaat-modelcategorie.

Scenario's uit dezelfde scenariofamilie (hardware x inputlengte) lijken sterk
op elkaar: ze worden met dezelfde benchmarkconfiguraties gelabeld. Bij een
willekeurige split zitten er bijna dezelfde scenario's in train en test, en
is de score te optimistisch. Daarom blijven hele families bij elkaar:

1. Testset: een paar hele families gaan apart voor de eindtest. Die worden
   pas aan het eind één keer gebruikt.
2. Cross-validatie: de overige families worden verdeeld over folds. Elke
   fold bevat alle families met één inputlengte (dus beide hardware-
   opstellingen). Zo vraagt elke validatiefold hetzelfde als de testset:
   voorspellen bij een inputlengte die in de training niet voorkwam.

De split wordt vastgelegd in een splitmanifest (JSON): welke families in de
test zitten, welke families in welke fold, de SHA-256 van de scenario- en
benchmark-CSV, en de labelverdeling. Het trainingsscript leest dit manifest
en rekent de split niet opnieuw uit.

Voor het wegschrijven controleert het script op datalekken. Gaat er een
controle mis, dan stopt het met een foutmelding.

Dit script traint GEEN model en bouwt GEEN baseline.

Gebruik:
    python -m src.candidate.data.split_scenarios
    python -m src.candidate.data.split_scenarios --scenarios outputs/candidate/scenarios.csv --out outputs/candidate/split_manifest.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.candidate.data.features import (
    FEATURE_KOLOMMEN,
    GROEP_KOLOM,
    LABEL_KOLOM,
    controleer_alleen_gebruikersinvoer,
)
from src.candidate.data.labelling_rule import CATEGORIE_VOORKEUR, LABEL_GEEN


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# Map voor de gegenereerde data, berekend vanaf de hoofdmap van de repo. Zo hangen
# de standaardpaden niet af van de map waar je staat.
UITVOER_MAP = Path(__file__).resolve().parents[3] / "outputs" / "candidate"


# ---------------------------------------------------------------------------
# Instellingen van de split
# ---------------------------------------------------------------------------

# De families voor de eindtest worden willekeurig gekozen met GroupShuffleSplit,
# met de inputlengte als groep. Zo gaat een hele lengte naar de test, met beide
# hardware-families, en komt die lengte in de training nergens voor. Met een
# vaste seed is de keuze altijd hetzelfde, maar ik weet niet welke lengte het is.
# De kortste en de langste lengte doen niet mee: die blijven randfolds, zodat
# het model op de test moet interpoleren en niet extrapoleren.
# Twee testlengtes: met de 8 lengtes van de echte benchmark blijven er 6 folds
# over (4 middelste), en hangt de eindscore niet af van één lengte.
# Dit ligt vast VOOR er een model getraind is, en wordt niet meer aangepast
# op basis van de resultaten.
SPLIT_SEED = 42
AANTAL_TESTLENGTES = 2

# Labels die in elke set moeten voorkomen. Ontbreekt er een, dan kan ik de
# macro-F1 voor die klasse niet uitrekenen.
VERWACHTE_LABELS = CATEGORIE_VOORKEUR + [LABEL_GEEN]

# Namen van de splits in de output.
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"


# ---------------------------------------------------------------------------
# Functies
# ---------------------------------------------------------------------------

def sha256_van_bestand(pad):
    """Geef de SHA-256 van een bestand, zodat vastligt op welke versie de split hoort."""
    return hashlib.sha256(Path(pad).read_bytes()).hexdigest()


def families_met_lengte(scenarios):
    """Geef één rij per familie met de hardware en inputlengte."""
    return (
        scenarios[[GROEP_KOLOM, "hardware", "input_length_tokens"]]
        .drop_duplicates()
        .sort_values(["input_length_tokens", "hardware"])
        .reset_index(drop=True)
    )


def kies_test_families(families):
    """Kies de testfamilies: hele inputlengtes, zonder de kortste en de langste."""
    lengtes = families["input_length_tokens"]
    binnen = families[~lengtes.isin([lengtes.min(), lengtes.max()])]
    splitser = GroupShuffleSplit(n_splits=1, test_size=AANTAL_TESTLENGTES, random_state=SPLIT_SEED)
    _, test_rijen = next(splitser.split(binnen, groups=binnen["input_length_tokens"]))
    return sorted(binnen.iloc[test_rijen][GROEP_KOLOM])


def controleer_test_families(families, test_families):
    """Stop als een testfamilie niet bestaat of niet tussen de trainlengtes ligt."""
    onbekend = [f for f in test_families if f not in set(families[GROEP_KOLOM])]
    if onbekend:
        raise ValueError(f"Testfamilies komen niet voor in de scenario's: {onbekend}")

    is_test = families[GROEP_KOLOM].isin(test_families)
    train = families[~is_test]

    # Elke testlengte moet per hardware tussen de kleinste en grootste
    # trainlengte liggen. Anders moet het model buiten het bereik voorspellen.
    for _, familie in families[is_test].iterrows():
        lengtes = train.loc[train["hardware"] == familie["hardware"], "input_length_tokens"]
        if lengtes.empty or not (lengtes.min() < familie["input_length_tokens"] < lengtes.max()):
            raise ValueError(
                f"Testfamilie {familie[GROEP_KOLOM]} ligt niet tussen de trainlengtes "
                f"voor {familie['hardware']}."
            )


def maak_folds(families, test_families):
    """Verdeel de trainfamilies over folds: één fold per inputlengte."""
    train = families[~families[GROEP_KOLOM].isin(test_families)]

    folds = []
    # Gesorteerd, zodat de foldnummers altijd hetzelfde zijn.
    for fold, lengte in enumerate(sorted(train["input_length_tokens"].unique())):
        in_fold = train[train["input_length_tokens"] == lengte]
        folds.append({
            "fold": fold,
            "input_length_tokens": int(lengte),
            "families": sorted(in_fold[GROEP_KOLOM]),
        })
    return folds


def pas_manifest_toe(scenarios, manifest):
    """Geef de scenario's terug met een kolom "split" en een kolom "cv_fold".

    cv_fold is het foldnummer voor trainscenario's en leeg voor de testset.
    Het trainingsscript kan deze functie gebruiken om de split uit het
    manifest over te nemen.
    """
    fold_per_familie = {
        familie: fold["fold"]
        for fold in manifest["cv_folds"]
        for familie in fold["families"]
    }

    resultaat = scenarios.copy()
    resultaat["split"] = resultaat[GROEP_KOLOM].map(
        lambda f: SPLIT_TEST if f in manifest["test_families"] else SPLIT_TRAIN
    )
    # Int64 (met hoofdletter) kan een lege waarde bevatten, gewone int niet.
    resultaat["cv_fold"] = resultaat[GROEP_KOLOM].map(fold_per_familie).astype("Int64")
    return resultaat


def labelverdeling(scenarios):
    """Geef het aantal scenario's per label, in een vaste volgorde."""
    telling = scenarios[LABEL_KOLOM].value_counts()
    return {label: int(telling.get(label, 0)) for label in VERWACHTE_LABELS}


def controleer_datalek(scenarios, toegewezen, manifest):
    """Voer de datalekcontroles uit. Geeft een lijst met (controle, ok)."""
    train = toegewezen[toegewezen["split"] == SPLIT_TRAIN]
    test = toegewezen[toegewezen["split"] == SPLIT_TEST]
    controles = []

    # 1. Alleen gebruikersinvoer als feature. Deze functie stopt zelf met een
    #    foutmelding als het misgaat. Komt hij erdoor, dan is het goed.
    controleer_alleen_gebruikersinvoer(scenarios)
    controles.append(("Features zijn alleen gebruikersinvoer", True))

    # 2. Geen familie in zowel train als test.
    overlap = set(train[GROEP_KOLOM]) & set(test[GROEP_KOLOM])
    controles.append(("Geen familie in train en test tegelijk", not overlap))

    # 3. Elk scenario hoort bij precies één deel: test, of één fold.
    zonder_plek = toegewezen[(toegewezen["split"] == SPLIT_TRAIN) & toegewezen["cv_fold"].isna()]
    controles.append(("Elk trainscenario zit in een fold", zonder_plek.empty))

    # Een testfamilie mag in geen enkele fold staan. Anders komt de test via
    # de cross-validatie toch in de training terecht.
    controles.append(("Geen testscenario in een fold", bool(test["cv_fold"].isna().all())))

    families_in_folds = [f for fold in manifest["cv_folds"] for f in fold["families"]]
    controles.append((
        "Elke trainfamilie zit in precies 1 fold",
        len(families_in_folds) == len(set(families_in_folds)),
    ))

    # 4. De inputlengte van de test komt in de training niet voor.
    gedeelde_lengtes = set(train["input_length_tokens"]) & set(test["input_length_tokens"])
    controles.append(("Testinputlengte komt niet voor in train", not gedeelde_lengtes))

    # 5. Hetzelfde voor elke fold: de validatielengte zit niet in de rest.
    folds_ok = True
    for fold in manifest["cv_folds"]:
        rest = train[train["cv_fold"] != fold["fold"]]
        if fold["input_length_tokens"] in set(rest["input_length_tokens"]):
            folds_ok = False
    controles.append(("Validatielengte komt niet voor in de rest van de fold", folds_ok))

    # 6. Geen scenario dubbel (zelfde scenario_id in train en test).
    controles.append((
        "Geen scenario_id in train en test tegelijk",
        not (set(train["scenario_id"]) & set(test["scenario_id"])),
    ))

    # 7. Elke klasse komt in de test en in elke validatiefold voor, anders kan
    #    ik de macro-F1 niet goed uitrekenen. Geen lek, wel een voorwaarde.
    alle_klassen = all(n > 0 for n in labelverdeling(test).values()) and all(
        all(n > 0 for n in labelverdeling(train[train["cv_fold"] == fold["fold"]]).values())
        for fold in manifest["cv_folds"]
    )
    controles.append(("Elke klasse in test en in elke validatiefold", alle_klassen))

    return controles


def maak_manifest(scenarios, scenarios_pad, benchmark_pad):
    """Maak het splitmanifest als dictionary."""
    families = families_met_lengte(scenarios)
    test_families = kies_test_families(families)
    controleer_test_families(families, test_families)
    folds = maak_folds(families, test_families)

    # Alleen de bestandsnaam, niet het hele pad: het pad hangt af van de map
    # waaruit je het script draait. De SHA-256 legt de versie vast.
    manifest = {
        # Neem de dummy-markering over: een split op nepdata is ook nep.
        "is_dummy": bool(scenarios["is_dummy"].all()),
        "scenarios_csv": {
            "bestand": Path(scenarios_pad).name,
            "sha256": sha256_van_bestand(scenarios_pad),
            "aantal_scenarios": len(scenarios.index),
        },
        "benchmark_csv": {
            "bestand": Path(benchmark_pad).name,
            "sha256": sha256_van_bestand(benchmark_pad),
        },
        "features": FEATURE_KOLOMMEN,
        "label_kolom": LABEL_KOLOM,
        "groep_kolom": GROEP_KOLOM,
        "test_families": test_families,
        "split": {
            "methode": "GroupShuffleSplit, met input_length_tokens als groep",
            "seed": SPLIT_SEED,
            "aantal_testlengtes": AANTAL_TESTLENGTES,
            "regel": "randen uitgesloten: de kortste en de langste inputlengte gaan nooit naar de test",
        },
        "cv_folds": folds,
    }

    # Aantallen en labelverdeling per deel, ter controle bij het inlezen. De
    # labelverdeling van de test staat er bewust niet in.
    toegewezen = pas_manifest_toe(scenarios, manifest)
    manifest["aantallen"] = {
        "train": int((toegewezen["split"] == SPLIT_TRAIN).sum()),
        "test": int((toegewezen["split"] == SPLIT_TEST).sum()),
    }
    manifest["labelverdeling"] = {
        "train": labelverdeling(toegewezen[toegewezen["split"] == SPLIT_TRAIN]),
    }
    for fold in manifest["cv_folds"]:
        in_fold = toegewezen[toegewezen["cv_fold"] == fold["fold"]]
        fold["aantal_scenarios"] = len(in_fold.index)
        fold["labelverdeling"] = labelverdeling(in_fold)

    return manifest, toegewezen


def print_samenvatting(manifest, controles):
    """Print de split en de uitkomst van de datalekcontroles.

    Welke families in de test zitten, wordt niet geprint. Ook de lengtes van de
    folds niet: daaruit zou je de testlengte kunnen afleiden.
    """
    print(f"Aantal train: {manifest['aantallen']['train']}, "
          f"aantal test: {manifest['aantallen']['test']}")
    print()

    rijen = []
    for fold in manifest["cv_folds"]:
        rijen.append({
            "deel": f"fold {fold['fold']}",
            **fold["labelverdeling"],
        })
    print("Labelverdeling per deel:")
    print(pd.DataFrame(rijen).to_string(index=False))
    print()

    print("Datalekcontroles:")
    for naam, ok in controles:
        print(f"  [{'OK' if ok else 'FOUT'}] {naam}")


def lees_argumenten():
    """Lees de command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="Maak een groepsgewijze split op scenariofamilie en schrijf een splitmanifest."
    )
    parser.add_argument(
        "--scenarios", type=str, default=str(UITVOER_MAP / "scenarios.csv"),
        help="Pad naar de scenario-CSV (standaard: outputs/candidate/scenarios.csv).",
    )
    parser.add_argument(
        "--benchmark", type=str, default=str(UITVOER_MAP / "dummy_benchmark.csv"),
        help="Pad naar de benchmark-CSV waarmee de scenario's gelabeld zijn "
             "(standaard: outputs/candidate/dummy_benchmark.csv). Alleen voor de SHA-256 in het manifest.",
    )
    parser.add_argument(
        "--out", type=str, default=str(UITVOER_MAP / "split_manifest.json"),
        help="Pad naar het splitmanifest (standaard: outputs/candidate/split_manifest.json).",
    )
    return parser.parse_args()


def main():
    args = lees_argumenten()

    scenarios = pd.read_csv(args.scenarios)
    manifest, toegewezen = maak_manifest(scenarios, args.scenarios, args.benchmark)
    controles = controleer_datalek(scenarios, toegewezen, manifest)

    if manifest["is_dummy"]:
        print("LET OP: gebaseerd op DUMMY data, de split zegt niets over echte modellen.")
        print()

    print_samenvatting(manifest, controles)

    # Bij een mislukte controle schrijven we geen manifest weg.
    mislukt = [naam for naam, ok in controles if not ok]
    if mislukt:
        raise SystemExit(f"Datalekcontrole mislukt, geen manifest geschreven: {mislukt}")

    # Maak de outputmap aan als die nog niet bestaat. Geen tijdstempel in het
    # manifest, zodat dezelfde input altijd hetzelfde bestand geeft.
    out_pad = Path(args.out)
    out_pad.parent.mkdir(parents=True, exist_ok=True)
    out_pad.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print()
    print(f"Splitmanifest opgeslagen in: {out_pad}")


if __name__ == "__main__":
    main()

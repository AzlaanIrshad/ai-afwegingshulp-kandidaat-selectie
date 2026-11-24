"""
generate_scenarios.py

Scenariogenerator voor de kandidaat-modelcategorie.

Een scenario is wat een gebruiker invult in de afwegingshulp: de eisen
(minimale kwaliteit, maximale latency, eventueel een energiebudget per maand)
en de context (hardware, inputlengte, aantal documenten per maand). Dit
script maakt een groot aantal willekeurige scenario's en geeft elk scenario
een label met de labelregel uit labelling_rule.py ("compact", "medium",
"groot" of "geen").

Elk scenario krijgt ook een family_id (hardware + inputlengte). Die is alleen
bedoeld voor de groepsgewijze split later, niet als feature (zie features.py).

De grenswaarden hieronder zijn afgestemd op het bereik van de dummy
benchmark. Met dezelfde seed komt er altijd precies dezelfde CSV uit.

Gebruik:
    python -m src.candidate.data.generate_scenarios
    python -m src.candidate.data.generate_scenarios --seed 7 --n 500 --out outputs/candidate/scenarios.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidate.data.features import FEATURE_KOLOMMEN, GROEP_KOLOM
from src.candidate.data.labelling_rule import (
    J_PER_KWH,
    bepaal_label,
    laad_benchmark,
    maak_configuratietabel,
)


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# Map voor de gegenereerde data, berekend vanaf de hoofdmap van de repo. Zo hangen
# de standaardpaden niet af van de map waar je staat.
UITVOER_MAP = Path(__file__).resolve().parents[3] / "outputs" / "candidate"


# ---------------------------------------------------------------------------
# Grenswaarden van de scenario's (afgestemd op de dummy benchmark)
# ---------------------------------------------------------------------------

# Alle grenzen liggen net buiten het bereik van de dummy benchmark, zodat er
# scenario's zijn die elk model haalt en scenario's die geen model haalt. De
# grenzen gelden voor beide hardware-opstellingen tegelijk: een gebruiker
# kiest zijn eisen niet op basis van de hardware.
#
# De grenzen zijn afgestemd met seed 42 en n = 2000, zodat geen klasse onder
# ongeveer 15% komt en geen hardware-opstelling (bijna) alleen één label
# krijgt. Met de eerste keuze (latency 1-300 ms, energie 0,1-20 J, F1
# 0,80-0,90) kreeg 73% van de CPU-scenario's het label geen.

# Minimale kwaliteit (macro-F1): gelijkmatig tussen deze grenzen. De modellen
# in de benchmark halen ongeveer 0,84-0,92 (op GPU en CPU hetzelfde). De
# bovengrens ligt net boven het beste model, zodat een klein deel van de
# scenario's een kwaliteit vraagt die niemand haalt.
MIN_F1_ONDER = 0.80
MIN_F1_BOVEN = 0.925

# Maximale latency in ms per document: log-uniform tussen deze grenzen.
# Log-uniform omdat latency over meerdere ordes van grootte loopt (ongeveer
# 0,4 ms op de GPU tot 285 ms op de CPU), zo komen strenge en ruime eisen
# even vaak voor. Ondergrens 2 ms: daaronder haalt de CPU nooit iets en de
# GPU bijna niets, dus die scenario's worden bijna allemaal geen. Bovengrens
# 1000 ms (1 seconde per document, bijvoorbeeld een nachtelijke batch): dan
# heeft de CPU ook genoeg scenario's waarin de latency geen probleem is.
MAX_LATENCY_ONDER_MS = 2.0
MAX_LATENCY_BOVEN_MS = 1000.0

# Maandvolume: aantal documenten per maand, log-uniform tussen deze grenzen
# (verzonnen). Van een klein team (duizend documenten) tot een grote
# organisatie (een miljoen documenten).
MAANDVOLUME_ONDER = 1_000
MAANDVOLUME_BOVEN = 1_000_000

# Kans dat een scenario een energiebudget heeft. Niet elke gebruiker heeft er
# een, dus de helft van de scenario's krijgt een budget.
KANS_ENERGIEBUDGET = 0.5

# Verborgen doel voor het energiebudget in joule per document: log-uniform
# tussen deze grenzen. Het budget in de CSV is in kWh per maand en wordt
# hieruit berekend met het maandvolume. Door in J per document te trekken
# blijft het budget in hetzelfde bereik als de benchmark (ongeveer 0,1 J op
# de GPU tot 18 J op de CPU), welk maandvolume er ook getrokken wordt. Dit
# doel zelf komt NIET in de CSV. Ondergrens 0,2 J: daaronder past alleen
# DistilBERT fp16 bij 64 tokens. Bovengrens 30 J: ruim boven het duurste
# CPU-model, zodat er ook budgetten zijn die niets uitsluiten.
DOEL_ENERGIE_ONDER_J = 0.2
DOEL_ENERGIE_BOVEN_J = 30.0

# Het energiebudget in kWh per maand wordt afgerond op zoveel significante
# cijfers. Een vast aantal decimalen werkt niet: bij een klein volume is het
# budget soms maar een paar honderdste Wh.
BUDGET_SIGNIFICANTE_CIJFERS = 4

# Kolommen van de output, in deze volgorde.
KOLOMMEN = [
    "scenario_id",
    "family_id",
    "hardware",
    "input_length_tokens",
    "min_macro_f1",
    "max_latency_ms_per_doc",
    "monthly_volume",
    "has_energy_budget",
    "max_energy_kwh_per_month",
    "label",
    "is_dummy",
]


# ---------------------------------------------------------------------------
# Functies
# ---------------------------------------------------------------------------

def log_uniform(rng, onder, boven):
    """Trek één getal log-uniform tussen onder en boven."""
    return float(np.exp(rng.uniform(np.log(onder), np.log(boven))))


def rond_significant(waarde, cijfers):
    """Rond een getal af op een aantal significante cijfers (0.000123456 -> 0.0001235)."""
    return float(f"{waarde:.{cijfers}g}")


def maak_family_id(hardware, input_lengte):
    """Geef de scenariofamilie: hardware + "__" + inputlengte (bijv. "rtx3080__256")."""
    return f"{hardware}__{input_lengte}"


def trek_scenario(rng, hardware_opties, inputlengte_opties):
    """Trek één willekeurig scenario (alleen eisen en context, nog geen label)."""
    # Alle trekkingen staan in een vaste volgorde en gebeuren altijd, ook als
    # het scenario geen budget krijgt. Zo blijft het aantal random trekkingen
    # per scenario gelijk en geeft dezelfde seed dezelfde CSV.

    # Context: alleen waarden die in de benchmark gemeten zijn, anders kan de
    # labelregel er niets over zeggen.
    hardware = str(rng.choice(hardware_opties))
    input_lengte = int(rng.choice(inputlengte_opties))

    min_f1 = rng.uniform(MIN_F1_ONDER, MIN_F1_BOVEN)
    max_latency = log_uniform(rng, MAX_LATENCY_ONDER_MS, MAX_LATENCY_BOVEN_MS)

    # Maandvolume als heel aantal documenten.
    maandvolume = int(round(log_uniform(rng, MAANDVOLUME_ONDER, MAANDVOLUME_BOVEN)))

    # Energiebudget is optioneel. Eerst trekken we een verborgen doel in J per
    # document, en rekenen dat om naar kWh per maand:
    #   kWh/maand = J/doc x documenten/maand / 3.600.000
    heeft_budget = bool(rng.random() < KANS_ENERGIEBUDGET)
    doel_energie_j = log_uniform(rng, DOEL_ENERGIE_ONDER_J, DOEL_ENERGIE_BOVEN_J)
    max_energie_kwh = rond_significant(
        doel_energie_j * maandvolume / J_PER_KWH, BUDGET_SIGNIFICANTE_CIJFERS
    )

    return {
        "family_id": maak_family_id(hardware, input_lengte),
        "hardware": hardware,
        "input_length_tokens": input_lengte,
        "min_macro_f1": round(min_f1, 3),
        "max_latency_ms_per_doc": round(max_latency, 2),
        "monthly_volume": maandvolume,
        "has_energy_budget": heeft_budget,
        "max_energy_kwh_per_month": max_energie_kwh if heeft_budget else None,
    }


def controleer_feature_kolommen():
    """Controleer dat elke feature uit features.py ook echt een kolom van de CSV is."""
    ontbreekt = [kolom for kolom in FEATURE_KOLOMMEN + [GROEP_KOLOM] if kolom not in KOLOMMEN]
    if ontbreekt:
        raise ValueError(f"Kolommen uit features.py ontbreken in de CSV: {ontbreekt}")


def maak_scenarios(benchmark, n, seed):
    """Maak n gelabelde scenario's als DataFrame (één rij per scenario)."""
    controleer_feature_kolommen()

    # Eén random generator met vaste seed: zelfde seed = zelfde scenario's.
    rng = np.random.default_rng(seed)

    configuraties = maak_configuratietabel(benchmark)

    # Gesorteerd, zodat de volgorde van de rijen in de benchmark-CSV de
    # trekkingen niet beïnvloedt.
    hardware_opties = sorted(configuraties["hardware"].unique())
    inputlengte_opties = sorted(configuraties["input_length_tokens"].unique())

    # Neem de dummy-markering van de benchmark over: scenario's op nepdata
    # zijn zelf ook nep.
    is_dummy = "is_dummy" in benchmark.columns and bool(benchmark["is_dummy"].all())

    rijen = []
    for scenario_id in range(n):
        scenario = trek_scenario(rng, hardware_opties, inputlengte_opties)

        # Het label komt uit de labelregel op de benchmark, de scenario's
        # zelf bevatten geen gemeten uitkomsten.
        scenario["label"] = bepaal_label(configuraties, scenario)
        scenario["scenario_id"] = scenario_id
        scenario["is_dummy"] = is_dummy
        rijen.append(scenario)

    return pd.DataFrame(rijen, columns=KOLOMMEN)


def print_samenvatting(df):
    """Print het aantal scenario's en de verdeling van de labels."""
    print(f"Aantal scenario's: {len(df.index)}")
    print(f"Aantal scenariofamilies: {df['family_id'].nunique()}")
    print()

    print("Labelverdeling:")
    verdeling = df["label"].value_counts()
    print(pd.DataFrame({
        "aantal": verdeling,
        "aandeel": (verdeling / len(df.index)).round(3),
    }).to_string())
    print()

    # Per hardware ook als aandeel, om te zien of een opstelling (bijna)
    # helemaal één label krijgt.
    print("Labels per hardware (aantal):")
    print(pd.crosstab(df["hardware"], df["label"], margins=True).to_string())
    print()
    print("Labels per hardware (aandeel per rij):")
    print(pd.crosstab(df["hardware"], df["label"], normalize="index").round(3).to_string())
    print()

    print("Labels per inputlengte:")
    print(pd.crosstab(df["input_length_tokens"], df["label"]).to_string())
    print()

    met_budget = int(df["has_energy_budget"].sum())
    print(f"Scenario's met energiebudget: {met_budget} van {len(df.index)}")


def lees_argumenten():
    """Lees de command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="Maak gelabelde scenario's op basis van een benchmark-CSV."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed voor de random generator (standaard: 42).",
    )
    parser.add_argument(
        "--n", type=int, default=2000,
        help="Aantal scenario's (standaard: 2000, zoals in het verslag).",
    )
    parser.add_argument(
        "--benchmark", type=str, default=str(UITVOER_MAP / "dummy_benchmark.csv"),
        help="Pad naar de benchmark-CSV (standaard: outputs/candidate/dummy_benchmark.csv).",
    )
    parser.add_argument(
        "--out", type=str, default=str(UITVOER_MAP / "scenarios.csv"),
        help="Pad naar de output-CSV (standaard: outputs/candidate/scenarios.csv).",
    )
    return parser.parse_args()


def main():
    args = lees_argumenten()

    benchmark = laad_benchmark(args.benchmark)
    df = maak_scenarios(benchmark, args.n, args.seed)

    # Maak de outputmap aan als die nog niet bestaat.
    out_pad = Path(args.out)
    out_pad.parent.mkdir(parents=True, exist_ok=True)
    # Altijd "\n" als regeleinde: anders schrijft Windows "\r\n" en Mac/Linux
    # "\n", en verschilt de SHA-256 in het splitmanifest per besturingssysteem.
    df.to_csv(out_pad, index=False, lineterminator="\n")

    if bool(df["is_dummy"].all()):
        print("LET OP: gebaseerd op DUMMY data, de labels zeggen niets over echte modellen.")
        print()

    print(f"Scenario's opgeslagen in: {out_pad}")
    print_samenvatting(df)


if __name__ == "__main__":
    main()

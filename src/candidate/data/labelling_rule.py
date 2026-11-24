"""
labelling_rule.py

Labelregel voor de kandidaat-modelcategorie.

Dit script bepaalt voor één scenario (de eisen van een gebruiker plus de
context) welk label hoort bij de benchmarkresultaten:
    "compact", "medium", "groot" of "geen" (geen onderzochte categorie voldoet).

De regel:
    Een categorie voldoet aan een scenario als minstens één configuratie uit
    die categorie tegelijk de minimale kwaliteit, de maximale latency en (als
    het scenario er een heeft) het energiebudget haalt. Voldoet geen enkele
    categorie, dan is het label "geen".

Het energiebudget is in kWh per maand. De benchmark meet energie per
document, dus die wordt eerst omgerekend met het maandvolume (aantal
documenten per maand) van het scenario.

Een configuratie is één combinatie van model x hardware x precisie x
inputlengte x batchgrootte, herkenbaar aan de config_id. De herhalingen uit
de benchmark worden eerst per config_id gemiddeld.

Dit script bouwt GEEN scenariogenerator en traint GEEN model. Het levert
alleen de functies die dat later gebruiken. Draai je het direct, dan labelt
het een paar handgekozen voorbeeldscenario's als controle.

Gebruik:
    python -m src.candidate.data.labelling_rule
    python -m src.candidate.data.labelling_rule --benchmark outputs/candidate/dummy_benchmark.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.candidate.data.categorieen import CATEGORIEEN


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# Map voor de gegenereerde data, berekend vanaf de hoofdmap van de repo. Zo hangen
# de standaardpaden niet af van de map waar je staat.
UITVOER_MAP = Path(__file__).resolve().parents[3] / "outputs" / "candidate"


# ---------------------------------------------------------------------------
# Instellingen van de labelregel
# ---------------------------------------------------------------------------

# Label als geen enkele onderzochte categorie aan de eisen voldoet.
LABEL_GEEN = "geen"

# Volgorde waarin categorieën winnen als er meerdere voldoen: de eerste die
# voldoet wordt het label. Nu wint de kleinste categorie, omdat de
# afwegingshulp de kleinste categorie moet laten zien die goed genoeg is.
# LET OP: dit is nog een open vraag en kan later
# veranderen.
CATEGORIE_VOORKEUR = CATEGORIEEN

# Kolom met de id van één configuratie. Alle herhalingen van dezelfde
# configuratie hebben dezelfde config_id. Daarop worden ze samen genomen.
CONFIG_ID_KOLOM = "config_id"

# Kolommen die samen één configuratie bepalen (de herhaling hoort er niet bij).
# Ze blijven in de configuratietabel staan, omdat de labelregel ze gebruikt
# (hardware, inputlengte, categorie).
CONFIGURATIE_KOLOMMEN = [
    "model",
    "n_params_m",
    "category",
    "hardware",
    "precision",
    "input_length_tokens",
    "batch_size",
]

# Gemeten waarden die per configuratie over de herhalingen worden gemiddeld.
MEETWAARDE_KOLOMMEN = ["macro_f1", "latency_ms_per_doc", "energy_j_per_doc"]

# Aantal joule in één kWh (1000 W x 3600 s). Nodig om energie per document
# om te rekenen naar een energiebudget per maand.
J_PER_KWH = 3_600_000


# ---------------------------------------------------------------------------
# Functies
# ---------------------------------------------------------------------------

def laad_benchmark(pad):
    """Lees de benchmark-CSV (één rij per gemeten run)."""
    return pd.read_csv(pad)


def controleer_config_ids(benchmark):
    """Stop als een config_id bij meer dan één combinatie van configuratiekolommen hoort.

    Dan zouden runs van verschillende configuraties samen gemiddeld worden.
    """
    if CONFIG_ID_KOLOM not in benchmark.columns:
        raise ValueError(f"De benchmark heeft geen kolom {CONFIG_ID_KOLOM}.")

    combinaties = benchmark.groupby(CONFIG_ID_KOLOM)[CONFIGURATIE_KOLOMMEN].nunique()
    fout = combinaties[(combinaties > 1).any(axis=1)].index.tolist()
    if fout:
        raise ValueError(
            f"Deze config_ids horen bij meer dan één configuratie: {fout}"
        )


def maak_configuratietabel(benchmark):
    """Middel de herhalingen per config_id, zodat er één rij per configuratie overblijft."""
    controleer_config_ids(benchmark)

    # Groeperen op config_id. De configuratiekolommen gaan mee in de groupby,
    # zodat ze in de tabel blijven staan. Door de controle hierboven geven ze
    # geen extra groepen.
    return (
        benchmark
        .groupby([CONFIG_ID_KOLOM] + CONFIGURATIE_KOLOMMEN, as_index=False)[MEETWAARDE_KOLOMMEN]
        .mean()
        .round(6)
    )


def configuraties_voor_context(configuraties, scenario):
    """Houd alleen configuraties over die gemeten zijn in de context van het scenario."""
    in_context = configuraties[
        (configuraties["hardware"] == scenario["hardware"])
        & (configuraties["input_length_tokens"] == scenario["input_length_tokens"])
    ]

    # Zonder metingen in deze context kan de regel niets zeggen. Dan is "geen"
    # misleidend (het is niet gemeten, niet afgekeurd), dus stoppen we hier.
    if in_context.empty:
        raise ValueError(
            f"Geen configuraties gemeten voor hardware={scenario['hardware']} "
            f"en input_length_tokens={scenario['input_length_tokens']}."
        )

    return in_context


def energie_kwh_per_maand(energie_j_per_doc, maandvolume):
    """Reken energie per document om naar kWh per maand bij een maandvolume.

    kWh/maand = J/doc x documenten/maand / 3.600.000
    (1 kWh = 1000 W x 3600 s = 3.600.000 J)
    """
    return energie_j_per_doc * maandvolume / J_PER_KWH


def voldoet_aan_eisen(configuraties, scenario):
    """Geef per configuratie True/False: haalt die alle eisen van het scenario?"""
    voldoet = (
        (configuraties["macro_f1"] >= scenario["min_macro_f1"])
        & (configuraties["latency_ms_per_doc"] <= scenario["max_latency_ms_per_doc"])
    )

    # Het energiebudget is optioneel: None (of een lege cel uit een CSV, die
    # pandas als NaN inleest) betekent "geen eis op energie".
    max_energie_kwh = scenario.get("max_energy_kwh_per_month")
    if pd.notna(max_energie_kwh):
        # Het budget geldt per maand, de benchmark meet per document. Reken
        # daarom elke configuratie om met het maandvolume van het scenario.
        energie_kwh = energie_kwh_per_maand(
            configuraties["energy_j_per_doc"], scenario["monthly_volume"]
        )
        voldoet &= energie_kwh <= max_energie_kwh

    return voldoet


def voldoende_categorieen(configuraties, scenario):
    """Geef de categorieën waarin minstens één configuratie aan alle eisen voldoet."""
    in_context = configuraties_voor_context(configuraties, scenario)
    geschikt = in_context[voldoet_aan_eisen(in_context, scenario)]
    return set(geschikt["category"])


def bepaal_label(configuraties, scenario):
    """Bepaal het label van één scenario: de eerste voldoende categorie, anders "geen"."""
    voldoende = voldoende_categorieen(configuraties, scenario)

    for categorie in CATEGORIE_VOORKEUR:
        if categorie in voldoende:
            return categorie

    return LABEL_GEEN


# ---------------------------------------------------------------------------
# Controle met handgekozen voorbeeldscenario's
# ---------------------------------------------------------------------------

# Een paar scenario's met het label dat we op de dummy benchmark (seed 42)
# verwachten. Dit is geen scenariogenerator, alleen een snelle controle dat
# de regel doet wat hij moet doen. Het verwachte label is met de hand bepaald
# uit de gemiddelde configuraties (die print het script hieronder ook), niet
# overgenomen uit de output.
VOORBEELD_SCENARIOS = [
    # Lage kwaliteitseis: DistilBERT fp32 haalt F1 0,837 in 1,6 ms, dus
    # compact voldoet en wint.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.80,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "compact"},
    # Basisvoorbeeld: F1 >= 0,85 en latency <= 20 ms. DistilBERT
    # haalt de kwaliteit niet (hoogstens 0,838), BERT-base fp32 wel (0,869 in
    # 3,0 ms), dus medium.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "medium"},
    # Kwaliteitseis hoger dan elk model haalt (hoogstens 0,890): geen.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.95,
     "max_latency_ms_per_doc": 50, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "geen"},
    # Medium-kwaliteit nodig, maar bij 512 tokens is het snelste medium-model
    # (BERT-base fp16, 7,4 ms) te traag voor 5 ms: geen.
    {"hardware": "rtx3080", "input_length_tokens": 512, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 5, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "geen"},
    # Hoog volume: 0,01 kWh per maand bij 100.000 documenten is 0,36 J per
    # document. Het zuinigste medium-model (BERT-base fp16, 0,42 J = 0,0117
    # kWh per maand) past er niet in, DistilBERT haalt de F1 niet: geen.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 20, "monthly_volume": 100_000,
     "max_energy_kwh_per_month": 0.01, "verwacht": "geen"},
    # Zelfde eisen en zelfde budget, maar laag volume: 10.000 documenten is
    # 3,6 J per document. BERT-base fp16 kost dan 0,0012 kWh per maand en
    # past ruim: medium. Het volume maakt hier het verschil.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": 0.01, "verwacht": "medium"},
    # Dezelfde eisen als het basisvoorbeeld, maar op de CPU: het
    # snelste medium-model (BERT-base, 24,5 ms) is te traag voor 20 ms en
    # DistilBERT haalt de F1 niet: geen. De hardware maakt hier het verschil.
    {"hardware": "ryzen5_3600x", "input_length_tokens": 128, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "geen"},
    # CPU met lage kwaliteitseis: DistilBERT haalt F1 0,841 in 12,5 ms, dus
    # compact.
    {"hardware": "ryzen5_3600x", "input_length_tokens": 128, "min_macro_f1": 0.80,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "compact"},
    # CPU met ruimere latency: BERT-base haalt F1 0,871 in 24,5 ms en kost
    # 0,0043 kWh per maand bij 10.000 documenten, binnen 0,01 kWh: medium.
    {"hardware": "ryzen5_3600x", "input_length_tokens": 128, "min_macro_f1": 0.85,
     "max_latency_ms_per_doc": 50, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": 0.01, "verwacht": "medium"},
    # Hoge kwaliteit: alleen BERT-large haalt F1 0,90 (0,922, fp16 in 5,5 ms).
    # De medium-modellen halen hoogstens 0,877: groot.
    {"hardware": "rtx3080", "input_length_tokens": 128, "min_macro_f1": 0.90,
     "max_latency_ms_per_doc": 20, "monthly_volume": 10_000,
     "max_energy_kwh_per_month": None, "verwacht": "groot"},
]


def print_configuraties_van_voorbeelden(configuraties):
    """Print de gemiddelde configuraties in elke context die een voorbeeld gebruikt."""
    contexten = sorted({
        (scenario["hardware"], scenario["input_length_tokens"])
        for scenario in VOORBEELD_SCENARIOS
    })

    for hardware, input_lengte in contexten:
        in_context = configuraties_voor_context(
            configuraties,
            {"hardware": hardware, "input_length_tokens": input_lengte},
        )
        print(f"Configuraties voor {hardware}, {input_lengte} tokens:")
        print(
            in_context[["category", "model", "precision"] + MEETWAARDE_KOLOMMEN]
            .sort_values(["category", "model", "precision"])
            .round(4)
            .to_string(index=False)
        )
        print()


def controleer_voorbeelden(configuraties):
    """Label de voorbeeldscenario's en print of het label klopt met de verwachting."""
    rijen = []
    for scenario in VOORBEELD_SCENARIOS:
        label = bepaal_label(configuraties, scenario)
        rijen.append({
            "hardware": scenario["hardware"],
            "input_len": scenario["input_length_tokens"],
            "min_f1": scenario["min_macro_f1"],
            "max_lat_ms": scenario["max_latency_ms_per_doc"],
            "volume": scenario["monthly_volume"],
            "max_kwh_maand": scenario["max_energy_kwh_per_month"],
            "voldoet": ", ".join(sorted(voldoende_categorieen(configuraties, scenario))) or "-",
            "label": label,
            "verwacht": scenario["verwacht"],
            "ok": label == scenario["verwacht"],
        })

    resultaat = pd.DataFrame(rijen)
    print(resultaat.to_string(index=False))
    print()
    print(f"{resultaat['ok'].sum()} van {len(resultaat)} voorbeelden kloppen.")


def lees_argumenten():
    """Lees de command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="Controleer de labelregel op een paar voorbeeldscenario's."
    )
    parser.add_argument(
        "--benchmark", type=str, default=str(UITVOER_MAP / "dummy_benchmark.csv"),
        help="Pad naar de benchmark-CSV (standaard: outputs/candidate/dummy_benchmark.csv).",
    )
    return parser.parse_args()


def main():
    args = lees_argumenten()

    benchmark = laad_benchmark(args.benchmark)
    configuraties = maak_configuratietabel(benchmark)

    if "is_dummy" in benchmark.columns and bool(benchmark["is_dummy"].all()):
        print("LET OP: dit is DUMMY data, de labels zeggen niets over echte modellen.")
        print()

    print(f"Aantal configuraties (herhalingen gemiddeld): {len(configuraties.index)}")
    print()
    print_configuraties_van_voorbeelden(configuraties)
    controleer_voorbeelden(configuraties)


if __name__ == "__main__":
    main()

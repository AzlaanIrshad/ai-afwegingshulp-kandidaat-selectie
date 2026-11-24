"""
create_dummy_benchmark.py

LET OP: DIT IS DUMMY DATA.

Dit script maakt een NEPPE benchmarkdataset. Alle getallen zijn verzonnen met
simpele formules plus wat willekeurige ruis. De data is alleen bedoeld om de
pipeline (scenario's -> labels -> classifier) te kunnen
bouwen en testen zolang de echte benchmark van het team nog niet klaar is.

De getallen zeggen NIETS over echte modellen, echte hardware of echt
energieverbruik. Gebruik ze nooit voor conclusies in een verslag.

Eén rij = één gemeten run van één configuratie:
    model x hardware x precisie x inputlengte x herhaling

Elke configuratie heeft een config_id. Alle herhalingen van dezelfde
configuratie krijgen dezelfde config_id.

Gebruik:
    python -m src.candidate.data.create_dummy_benchmark
    python -m src.candidate.data.create_dummy_benchmark --seed 7 --out outputs/candidate/dummy_benchmark.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidate.data.categorieen import categoriseer


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# Map voor de gegenereerde data, berekend vanaf de hoofdmap van de repo. Zo hangen
# de standaardpaden niet af van de map waar je staat.
UITVOER_MAP = Path(__file__).resolve().parents[3] / "outputs" / "candidate"


# ---------------------------------------------------------------------------
# Configuraties (pas deze aan als het team andere opstellingen meet)
# ---------------------------------------------------------------------------

# Modellen met hun aantal parameters in miljoenen.
# De parameteraantallen kloppen ongeveer met de echte modellen. De categorie
# wordt hieronder afgeleid uit het aantal parameters (niet hard-coded).
MODELLEN = {
    "DistilBERT": 66,
    "BERT-base": 110,
    "RoBERTa-base": 125,
    "BERT-large": 340,
}

# De categorie per model (compact, medium of groot). De grenzen worden berekend
# met de standaardafwijking van log10(parameters) over deze modellen. Zie
# categorieen.py. Met deze vier modellen: DistilBERT compact, BERT-base en
# RoBERTa-base medium, BERT-large groot.
CATEGORIE_PER_MODEL = categoriseer(MODELLEN)

# Hardware-opstellingen: een GPU en een CPU. Twee opstellingen geven meer
# scenariofamilies (hardware x inputlengte) voor de groepsgewijze split.
HARDWARE = ["rtx3080", "ryzen5_3600x"]

# Precisies per hardware-opstelling. Op de CPU draaien we alleen fp32:
# fp16 is op een CPU meestal niet sneller (vaak zelfs trager), omdat de
# rekenkernen daar niet voor gemaakt zijn. Dat meten heeft dus weinig zin.
PRECISIES_PER_HARDWARE = {
    "rtx3080": ["fp32", "fp16"],
    "ryzen5_3600x": ["fp32"],
}

# Inputlengtes in tokens. BERT-achtige modellen accepteren maximaal 512
# tokens, dus langer kan niet. Acht lengtes geven met twee hardware-opstellingen
# 16 scenariofamilies.
INPUT_LENGTES = [32, 64, 96, 128, 192, 256, 384, 512]

# Aantal herhalingen per configuratie.
AANTAL_HERHALINGEN = 3

# Batchgrootte staat vast op 1 (wordt wel als kolom opgeslagen).
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Verzonnen getallen voor de formules (VERZONNEN, niet gemeten!)
# ---------------------------------------------------------------------------

# Basiskwaliteit (macro-F1) per model. Verzonnen, met als enige idee:
# een groter model scoort een beetje hoger. BERT-large (groot) staat iets verder
# boven de rest (0,92), zodat er genoeg scenario's zijn waarin alleen groot
# voldoet. Met 0,89 kreeg maar 5% van de scenario's het label groot, met 0,92
# ongeveer 15% (de eis voor elke klasse).
BASIS_MACRO_F1 = {
    "DistilBERT": 0.84,
    "BERT-base": 0.87,
    "RoBERTa-base": 0.88,
    "BERT-large": 0.92,
}

# fp16 kost een heel klein beetje kwaliteit (verzonnen).
FP16_F1_VERLIES = 0.002

# Standaardafwijking van de ruis op macro-F1 per herhaling (verzonnen).
F1_RUIS_SD = 0.005

# Basislatency in ms per document, bij 128 tokens en fp32 (verzonnen).
# Grotere modellen zijn trager. BERT-large is duidelijk het traagst.
BASIS_LATENCY_MS = {
    "DistilBERT": 1.5,
    "BERT-base": 3.0,
    "RoBERTa-base": 3.2,
    "BERT-large": 9.0,
}

# De inputlengte waarbij de basislatency hoort.
BASIS_INPUT_LENGTE = 128

# fp16 is sneller: latency x 0.6 (verzonnen).
FP16_LATENCY_FACTOR = 0.6

# Latency op de CPU ten opzichte van de rtx3080 in fp32 (verzonnen factor).
# De basislatency hierboven hoort bij de rtx3080. De CPU is 8x zo traag.
CPU_LATENCY_FACTOR = 8.0

# Hardware-opstellingen die op de CPU draaien (krijgen de CPU-factor).
CPU_HARDWARE = ["ryzen5_3600x"]

# Multiplicatieve ruis op latency: ongeveer +-5% (sd 0.05 rond factor 1).
LATENCY_RUIS_SD = 0.05

# Gemiddeld vermogen in watt per hardware-opstelling tijdens inferentie
# (verzonnen). Voor de rtx3080 is dat het GPU-vermogen, voor de
# ryzen5_3600x het CPU-vermogen.
GEMIDDELD_VERMOGEN_W = {
    "rtx3080": 220,
    "ryzen5_3600x": 65,
}

# Extra multiplicatieve ruis op energie: ongeveer +-3% (sd 0.03 rond factor 1).
ENERGIE_RUIS_SD = 0.03

# Kolommen van de output, in deze volgorde.
KOLOMMEN = [
    "config_id",
    "model",
    "n_params_m",
    "category",
    "hardware",
    "precision",
    "input_length_tokens",
    "batch_size",
    "repeat",
    "macro_f1",
    "latency_ms_per_doc",
    "energy_j_per_doc",
    "is_dummy",
]


# ---------------------------------------------------------------------------
# Functies
# ---------------------------------------------------------------------------

def maak_config_id(model, hardware, precisie, input_lengte, batch_size):
    """Geef de config_id van één configuratie, bijv. "distilbert__rtx3080__fp32__64__bs1".

    Opbouw: model (kleine letters), hardware, precisie, inputlengte en "bs" +
    batchgrootte, gescheiden door "__". De herhaling hoort er bewust niet in:
    zo hebben alle herhalingen van dezelfde configuratie dezelfde config_id,
    en kun je ze daarop samen nemen.
    """
    return f"{model.lower()}__{hardware}__{precisie}__{input_lengte}__bs{batch_size}"


def genereer_macro_f1(rng, model, precisie):
    """Verzin een macro-F1: basiswaarde - fp16-verlies + ruis, geclipt op [0, 1]."""
    f1 = BASIS_MACRO_F1[model]

    # fp16 kost een heel klein beetje kwaliteit.
    if precisie == "fp16":
        f1 -= FP16_F1_VERLIES

    # Kleine normale ruis per herhaling.
    f1 += rng.normal(0.0, F1_RUIS_SD)

    # Een F1-score kan nooit buiten [0, 1] liggen.
    return float(np.clip(f1, 0.0, 1.0))


def genereer_latency_ms(rng, model, hardware, precisie, input_lengte):
    """Verzin een latency in ms per document."""
    latency = BASIS_LATENCY_MS[model]

    # Latency schaalt lineair mee met de inputlengte (256 tokens = 2x zo lang).
    latency *= input_lengte / BASIS_INPUT_LENGTE

    # De CPU is een vaste factor trager dan de rtx3080 in fp32.
    if hardware in CPU_HARDWARE:
        latency *= CPU_LATENCY_FACTOR

    # fp16 is sneller dan fp32 (komt alleen op de GPU voor).
    if precisie == "fp16":
        latency *= FP16_LATENCY_FACTOR

    # Multiplicatieve ruis van ongeveer +-5%.
    latency *= rng.normal(1.0, LATENCY_RUIS_SD)

    return latency


def genereer_energie_j(rng, latency_ms, hardware):
    """Verzin energie in joule per document: latency (s) x vermogen (W) + ruis."""
    latency_s = latency_ms / 1000.0

    # Energie (J) = tijd (s) x gemiddeld vermogen (W).
    energie = latency_s * GEMIDDELD_VERMOGEN_W[hardware]

    # Kleine extra multiplicatieve ruis van ongeveer +-3%, zodat energie niet
    # exact evenredig is met latency.
    energie *= rng.normal(1.0, ENERGIE_RUIS_SD)

    return energie


def maak_dummy_benchmark(seed):
    """Maak de volledige dummy benchmark als DataFrame (één rij per run)."""
    # Eén random generator met vaste seed: zelfde seed = zelfde CSV.
    rng = np.random.default_rng(seed)

    rijen = []

    # Loop over alle combinaties: model x hardware x precisie x inputlengte x herhaling.
    # Welke precisies er zijn, hangt af van de hardware (op de CPU alleen fp32).
    for model, n_params_m in MODELLEN.items():
        categorie = CATEGORIE_PER_MODEL[model]

        for hardware in HARDWARE:
            for precisie in PRECISIES_PER_HARDWARE[hardware]:
                for input_lengte in INPUT_LENGTES:
                    for herhaling in range(1, AANTAL_HERHALINGEN + 1):
                        macro_f1 = genereer_macro_f1(rng, model, precisie)
                        latency_ms = genereer_latency_ms(
                            rng, model, hardware, precisie, input_lengte
                        )
                        energie_j = genereer_energie_j(rng, latency_ms, hardware)

                        rijen.append({
                            "config_id": maak_config_id(
                                model, hardware, precisie, input_lengte, BATCH_SIZE
                            ),
                            "model": model,
                            "n_params_m": n_params_m,
                            "category": categorie,
                            "hardware": hardware,
                            "precision": precisie,
                            "input_length_tokens": input_lengte,
                            "batch_size": BATCH_SIZE,
                            "repeat": herhaling,
                            "macro_f1": round(macro_f1, 4),
                            "latency_ms_per_doc": round(latency_ms, 3),
                            "energy_j_per_doc": round(energie_j, 3),
                            # Elke rij is expliciet gemarkeerd als nepdata.
                            "is_dummy": True,
                        })

    return pd.DataFrame(rijen, columns=KOLOMMEN)


def print_samenvatting(df):
    """Print het aantal rijen en families en de gemiddelden per configuratie."""
    # Een scenariofamilie is één combinatie van hardware x inputlengte.
    aantal_families = len(
        df[["hardware", "input_length_tokens"]].drop_duplicates().index
    )

    print(f"Aantal rijen: {len(df.index)}")
    print(f"Aantal unieke config_ids (configuraties): {df['config_id'].nunique()}")
    print(f"Aantal scenariofamilies (hardware x inputlengte): {aantal_families}")
    print()

    # sort=False houdt de volgorde van de constanten aan (klein -> groot model).
    gemiddelden = (
        df.groupby(
            ["category", "model", "hardware", "precision", "input_length_tokens"],
            sort=False,
        )
        [["macro_f1", "latency_ms_per_doc", "energy_j_per_doc"]]
        .mean()
        .round({"macro_f1": 4, "latency_ms_per_doc": 3, "energy_j_per_doc": 3})
    )

    print("Gemiddelden per model x hardware x precisie x inputlengte (DUMMY DATA):")
    print(gemiddelden.to_string())


def lees_argumenten():
    """Lees de command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="Maak een DUMMY benchmark-CSV om de pipeline te testen."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed voor de random generator (standaard: 42).",
    )
    parser.add_argument(
        "--out", type=str, default=str(UITVOER_MAP / "dummy_benchmark.csv"),
        help="Pad naar de output-CSV (standaard: outputs/candidate/dummy_benchmark.csv).",
    )
    return parser.parse_args()


def main():
    args = lees_argumenten()

    df = maak_dummy_benchmark(args.seed)

    # Maak de outputmap aan als die nog niet bestaat.
    out_pad = Path(args.out)
    out_pad.parent.mkdir(parents=True, exist_ok=True)
    # Altijd "\n" als regeleinde: anders schrijft Windows "\r\n" en Mac/Linux
    # "\n", en verschilt de SHA-256 in het splitmanifest per besturingssysteem.
    df.to_csv(out_pad, index=False, lineterminator="\n")

    print(f"Dummy benchmark opgeslagen in: {out_pad}")
    print_samenvatting(df)


if __name__ == "__main__":
    main()

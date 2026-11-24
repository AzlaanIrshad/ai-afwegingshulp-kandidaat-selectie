"""
predict_candidate.py

Voorspelt de kandidaat-modelcategorie voor één scenario dat je zelf invult.

Dit script traint zelf niets. Het laadt de kandidaat die train_candidate_selector.py
heeft gekozen uit een runmap: standaard de nieuwste map in outputs/candidate/runs,
of de map die je met --run opgeeft. De kandidaat, de drempel t en de feature-lijst
komen precies uit die run. Dat kan een regel, een model of een paar ja/nee-modellen
zijn. Draai dus eerst train_candidate_selector.py.

De benchmark en het manifest zijn alleen nodig voor de modelnamen, de waarschuwingen
en de vergelijking met de labelregel. Ze moeten bij de run horen: de SHA-256 wordt
vergeleken met run.json, en bij een verschil stopt het script.

voorspel_scenario() is de enige voorspelfunctie. Dit script en de app
(app/koppeling.py) gebruiken hem allebei, zodat ze altijd hetzelfde antwoord geven.

Zolang de test dicht is, wordt het label uit de benchmark voor geen enkel scenario
getoond. Alleen de testfamilies verbergen zou juist laten zien welke dat zijn.

Gebruik:
    python -m src.candidate.model.predict_candidate --hardware rtx3080 --input-length 128 --min-f1 0.85 --max-latency-ms 20 --volume 100000 --budget-kwh 0.01
    python -m src.candidate.model.predict_candidate --hardware ryzen5_3600x --input-length 192 --min-f1 0.80 --max-latency-ms 50 --volume 10000 --run outputs/candidate/runs/<map>
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.candidate.model.candidates import REGELS, kans_ja, kies_categorie
from src.candidate.data.generate_scenarios import (
    DOEL_ENERGIE_BOVEN_J,
    DOEL_ENERGIE_ONDER_J,
    MAANDVOLUME_BOVEN,
    MAANDVOLUME_ONDER,
    MAX_LATENCY_BOVEN_MS,
    MAX_LATENCY_ONDER_MS,
    MIN_F1_BOVEN,
    MIN_F1_ONDER,
)
from src.candidate.data.labelling_rule import (
    CATEGORIE_VOORKEUR,
    J_PER_KWH,
    bepaal_label,
    laad_benchmark,
    maak_configuratietabel,
)
from src.candidate.data.split_scenarios import sha256_van_bestand


# ---------------------------------------------------------------------------
# Paden
# ---------------------------------------------------------------------------

# Map voor de gegenereerde data, berekend vanaf de hoofdmap van de repo. Zo hangen
# de standaardpaden niet af van de map waar je staat.
UITVOER_MAP = Path(__file__).resolve().parents[3] / "outputs" / "candidate"


# ---------------------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------------------

# De hardware waarvoor het model features heeft.
HARDWARE_OPTIES = ["rtx3080", "ryzen5_3600x"]


# ---------------------------------------------------------------------------
# De run laden en voorspellen
# ---------------------------------------------------------------------------

def zoek_run(pad):
    """Geef de runmap: de opgegeven map, of anders de nieuwste op mapnaam."""
    if pad is not None:
        run_map = Path(pad)
        if not run_map.is_dir():
            raise SystemExit(f"De runmap {run_map} bestaat niet.")
        return run_map

    runs_map = UITVOER_MAP / "runs"
    mappen = sorted(m for m in runs_map.iterdir() if m.is_dir()) if runs_map.is_dir() else []
    if not mappen:
        raise SystemExit(
            f"Geen run gevonden in {runs_map}. "
            f"Draai eerst: python -m src.candidate.model.train_candidate_selector"
        )
    return mappen[-1]


def vind_bestand(run_map, naam):
    """Zoek een bestand in de runmap. Bij dummy data begint de naam met DUMMY_."""
    for voorvoegsel in ["DUMMY_", ""]:
        pad = run_map / f"{voorvoegsel}{naam}"
        if pad.exists():
            return pad
    raise SystemExit(f"{naam} ontbreekt in de runmap {run_map}.")


def laad_run(run_map):
    """Laad de gekozen kandidaat en run.json uit de runmap."""
    eind = joblib.load(vind_bestand(run_map, "model.joblib"))
    run = json.loads(vind_bestand(run_map, "run.json").read_text(encoding="utf-8"))
    return eind, run


def controleer_data(run, run_map, benchmark_pad, manifest_pad):
    """Stop als de benchmark of het manifest niet de versie is waar de run bij hoort."""
    klopt = all(
        Path(pad).exists() and sha256_van_bestand(pad) == run["sha256"][sleutel]
        for pad, sleutel in [(benchmark_pad, "benchmark"), (manifest_pad, "manifest")]
    )
    if not klopt:
        raise SystemExit(
            f"De data hoort niet bij de run {run_map}: de SHA-256 klopt niet met run.json.\n"
            f"  benchmark: {benchmark_pad}\n"
            f"  manifest:  {manifest_pad}\n"
            f"Geef de juiste --benchmark en --manifest op, of kies een andere --run."
        )


def controleer_sklearn_versie(run):
    """Stop als de run met een andere scikit-learn-versie is gemaakt.

    Een model uit een andere versie laadt soms zonder foutmelding, maar is dan niet betrouwbaar.
    """
    versie = run["versies"]["sklearn"]
    if versie != sklearn.__version__:
        raise SystemExit(
            f"Het model kon niet worden geladen: de run is gemaakt met scikit-learn {versie}, "
            f"deze omgeving heeft scikit-learn {sklearn.__version__}."
        )


def laad_context(benchmark_pad):
    """Laad de benchmark, de configuraties en de gemeten inputlengtes.

    Het manifest wordt hier niet gelezen: welke families in de test zitten, hoort
    niet in de voorspelling of in de waarschuwingen terecht te komen.
    """
    benchmark = laad_benchmark(benchmark_pad)
    configuraties = maak_configuratietabel(benchmark)
    gemeten_lengtes = sorted(int(l) for l in configuraties["input_length_tokens"].unique())
    return benchmark, configuraties, gemeten_lengtes


def laad(run_pad=None, benchmark_pad=None, manifest_pad=None):
    """Laad alles wat nodig is om te voorspellen, als dict. Stopt met SystemExit als iets niet klopt.

    Zonder run_pad wordt de nieuwste run gebruikt, zonder paden de bestanden in outputs/candidate.
    """
    benchmark_pad = benchmark_pad or UITVOER_MAP / "dummy_benchmark.csv"
    manifest_pad = manifest_pad or UITVOER_MAP / "split_manifest.json"

    run_map = zoek_run(run_pad)
    # Eerst de versie controleren, pas daarna het model laden.
    controleer_sklearn_versie(json.loads(vind_bestand(run_map, "run.json").read_text(encoding="utf-8")))
    eind, run = laad_run(run_map)
    controleer_data(run, run_map, benchmark_pad, manifest_pad)
    benchmark, configuraties, gemeten_lengtes = laad_context(benchmark_pad)

    return {
        "run_map": str(run_map),
        "eind": eind,
        "is_dummy": run["is_dummy"],
        "test_geopend": run["test_opened"],
        "benchmark": benchmark,
        "configuraties": configuraties,
        "gemeten_lengtes": gemeten_lengtes,
    }


def voorspel_regel(eind, scenario):
    """Pas een opgeslagen regel toe met de configuraties uit de run."""
    return REGELS[eind["naam"]](pd.DataFrame([scenario]), pd.DataFrame(eind["configuraties"]))[0]


def voorspel(eind, scenario):
    """Voorspel met de gekozen kandidaat. Geef de categorie en de kansen terug (None bij een regel)."""
    x = pd.DataFrame([scenario])[eind["features"]]

    if eind["soort"] == "ja_nee":
        # Eén ja/nee-model per categorie: de kans dat die categorie voldoet.
        kansen = {c: float(kans_ja(model, x)[0]) for c, model in eind["pipeline"].items()}
        return kies_categorie(kansen, eind["t"]), kansen
    if eind["soort"] == "model":
        kansen = dict(zip(eind["pipeline"].classes_, eind["pipeline"].predict_proba(x)[0]))
        # De klasse met de hoogste kans, zoals bij de evaluatie in de selector.
        return max(kansen, key=kansen.get), kansen
    return voorspel_regel(eind, scenario), None


def voorspel_scenario(geladen, scenario):
    """Voorspel voor één scenario met wat laad() teruggaf. Dit is de enige voorspelfunctie.

    Geeft een dict met de categorie, de kansen (None bij een regel), de kandidaat, de
    waarschuwingen, de voorbeeldmodellen per categorie en het label uit de benchmark.
    Het label is None als de context niet gemeten is, en ook zolang de test dicht is.
    """
    eind = geladen["eind"]
    categorie, kansen = voorspel(eind, scenario)

    # Zolang de test dicht is, voor geen enkel scenario het label
    label = label_als_gemeten(geladen["configuraties"], scenario) if geladen["test_geopend"] else None

    return {
        "categorie": categorie,
        "kansen": kansen,
        "soort": eind["soort"],
        "t": eind["t"],
        "kandidaat": eind["naam"],
        "waarschuwingen": maak_waarschuwingen(scenario, geladen["gemeten_lengtes"]),
        "voorbeeldmodellen": {
            c: modellen_in_categorie(geladen["benchmark"], c) for c in CATEGORIE_VOORKEUR
        },
        "label": label,
    }


def print_voorspelling(resultaat):
    """Print de voorspelling uit voorspel_scenario."""
    kansen = resultaat["kansen"]

    print("Voorspelling:")
    if resultaat["soort"] == "ja_nee":
        for categorie, kans in kansen.items():
            print(f"  {f'P({categorie} voldoet):':<19} {kans:.3f}")
        print(f"  drempel t:          {resultaat['t']:g}")
    elif resultaat["soort"] == "model":
        for klasse, kans in kansen.items():
            print(f"  P({klasse}): {kans:.3f}")
    else:
        print("  geen kansen: de gekozen kandidaat is een regel")

    print(f"  gekozen categorie:  {resultaat['categorie']}")
    print()


# ---------------------------------------------------------------------------
# Scenario en waarschuwingen
# ---------------------------------------------------------------------------

def maak_scenario(args):
    """Zet de ingevulde waarden om naar een scenario met de kolomnamen van de dataset."""
    heeft_budget = args.budget_kwh is not None
    return {
        "hardware": args.hardware,
        "input_length_tokens": args.input_length,
        "min_macro_f1": args.min_f1,
        "max_latency_ms_per_doc": args.max_latency_ms,
        "monthly_volume": args.volume,
        "has_energy_budget": heeft_budget,
        # NaN betekent "geen budget", net als een lege cel in de scenario-CSV.
        "max_energy_kwh_per_month": args.budget_kwh if heeft_budget else float("nan"),
    }


def maak_waarschuwingen(scenario, gemeten_lengtes):
    """Geef de waarschuwingen. Er wordt altijd voorspeld, ook met waarschuwingen.

    Voor de inputlengte wordt alleen gekeken of die gemeten is, niet of die in de
    training zat. Anders zou de waarschuwing laten zien welke lengtes in de test zitten.
    """
    waarschuwingen = []

    lengte = scenario["input_length_tokens"]
    laagste, hoogste = min(gemeten_lengtes), max(gemeten_lengtes)
    if lengte < laagste or lengte > hoogste:
        waarschuwingen.append(
            f"De inputlengte van {lengte} tokens ligt buiten het gemeten bereik ({laagste} tot en met "
            f"{hoogste} tokens). De voorspelling is hier minder betrouwbaar."
        )
    elif lengte not in gemeten_lengtes:
        onder = max(l for l in gemeten_lengtes if l < lengte)
        boven = min(l for l in gemeten_lengtes if l > lengte)
        waarschuwingen.append(
            f"De inputlengte van {lengte} tokens is niet gemeten. De voorspelling is een schatting "
            f"tussen de gemeten lengtes {onder} en {boven} tokens."
        )

    grenzen = [
        ("min-f1", scenario["min_macro_f1"], MIN_F1_ONDER, MIN_F1_BOVEN),
        ("max-latency-ms", scenario["max_latency_ms_per_doc"], MAX_LATENCY_ONDER_MS, MAX_LATENCY_BOVEN_MS),
        ("volume", scenario["monthly_volume"], MAANDVOLUME_ONDER, MAANDVOLUME_BOVEN),
    ]

    # De generator trekt het budget in J per document, dus daar staan de grenzen voor.
    if scenario["has_energy_budget"]:
        budget_j = scenario["max_energy_kwh_per_month"] * J_PER_KWH / scenario["monthly_volume"]
        grenzen.append(("budget (J per document)", budget_j, DOEL_ENERGIE_ONDER_J, DOEL_ENERGIE_BOVEN_J))

    for naam, waarde, onder, boven in grenzen:
        if waarde < onder or waarde > boven:
            waarschuwingen.append(
                f"{naam} = {waarde:g} ligt buiten de grenzen van generate_scenarios.py ({onder:g} tot en met {boven:g})."
            )

    return waarschuwingen


def modellen_in_categorie(benchmark, categorie):
    """Geef de modellen uit de benchmark in deze categorie."""
    return list(benchmark.loc[benchmark["category"] == categorie, "model"].unique())


def label_als_gemeten(configuraties, scenario):
    """Geef het label volgens de labelregel, of None als deze context niet gemeten is."""
    in_context = (
        (configuraties["hardware"] == scenario["hardware"])
        & (configuraties["input_length_tokens"] == scenario["input_length_tokens"])
    )
    return bepaal_label(configuraties, scenario) if in_context.any() else None


def print_scenario(scenario):
    """Print het ingevulde scenario."""
    print("Scenario:")
    print(f"  hardware:          {scenario['hardware']}")
    print(f"  input-length:      {scenario['input_length_tokens']} tokens")
    print(f"  minimale macro-F1: {scenario['min_macro_f1']:g}")
    print(f"  maximale latency:  {scenario['max_latency_ms_per_doc']:g} ms per document")
    print(f"  volume:            {scenario['monthly_volume']:g} documenten per maand")
    if scenario["has_energy_budget"]:
        print(f"  energiebudget:     {scenario['max_energy_kwh_per_month']:g} kWh per maand")
    else:
        print("  energiebudget:     geen")
    print()


def lees_argumenten():
    """Lees de command-line argumenten en controleer wat niet berekend kan worden."""
    parser = argparse.ArgumentParser(
        description="Voorspel de kandidaat-modelcategorie voor één zelf ingevuld scenario."
    )
    parser.add_argument("--hardware", type=str, required=True,
                        help="Hardware: rtx3080 of ryzen5_3600x.")
    parser.add_argument("--input-length", type=int, required=True,
                        help="Inputlengte in tokens.")
    parser.add_argument("--min-f1", type=float, required=True,
                        help="Minimale macro-F1.")
    parser.add_argument("--max-latency-ms", type=float, required=True,
                        help="Maximale latency in ms per document.")
    parser.add_argument("--volume", type=float, required=True,
                        help="Aantal documenten per maand.")
    parser.add_argument("--budget-kwh", type=float, default=None,
                        help="Energiebudget in kWh per maand (optioneel, weglaten = geen budget).")
    parser.add_argument("--run", type=str, default=None,
                        help="Pad naar een runmap van train_candidate_selector.py "
                             "(standaard: de nieuwste map in outputs/candidate/runs).")
    parser.add_argument("--benchmark", type=str, default=str(UITVOER_MAP / "dummy_benchmark.csv"),
                        help="Pad naar de benchmark-CSV (standaard: outputs/candidate/dummy_benchmark.csv).")
    parser.add_argument("--manifest", type=str, default=str(UITVOER_MAP / "split_manifest.json"),
                        help="Pad naar het splitmanifest (standaard: outputs/candidate/split_manifest.json).")
    args = parser.parse_args()

    if args.hardware not in HARDWARE_OPTIES:
        parser.error(f"onbekende hardware '{args.hardware}'. Kies uit: {', '.join(HARDWARE_OPTIES)}.")

    # De features delen hierdoor of nemen er de log10 van, dus deze moeten groter zijn dan 0.
    for naam, waarde in [("--input-length", args.input_length), ("--max-latency-ms", args.max_latency_ms),
                         ("--volume", args.volume), ("--budget-kwh", args.budget_kwh)]:
        if waarde is not None and waarde <= 0:
            parser.error(f"{naam} moet groter zijn dan 0.")

    return args


def main():
    args = lees_argumenten()

    geladen = laad(args.run, args.benchmark, args.manifest)
    eind = geladen["eind"]

    if geladen["is_dummy"]:
        print("LET OP: DUMMY data")
        print()

    print(f"Run:       {geladen['run_map']}")
    print(f"Kandidaat: {eind['naam']}")
    if eind["t"] is not None:
        print(f"Drempel t: {eind['t']:g}")
    print()

    scenario = maak_scenario(args)
    print_scenario(scenario)

    resultaat = voorspel_scenario(geladen, scenario)
    print_voorspelling(resultaat)

    # Ter illustratie: de voorspelling gaat over de categorie, niet over één model.
    print("Modellen in de gekozen categorie (ter illustratie):")
    modellen = resultaat["voorbeeldmodellen"].get(resultaat["categorie"], [])
    print(f"  {', '.join(modellen) if modellen else 'geen: geen onderzochte categorie voldoet'}")
    print()

    print("Waarschuwingen:")
    for waarschuwing in resultaat["waarschuwingen"]:
        print(f"  - {waarschuwing}")
    if not resultaat["waarschuwingen"]:
        print("  geen")
    print()

    # Ter vergelijking het label uit de metingen: pas na het openen van de test,
    # en alleen als de context gemeten is.
    if not geladen["test_geopend"]:
        print("Label volgens de labelregel: niet getoond zolang de test dicht is.")
    elif resultaat["label"] is None:
        print("Label volgens de labelregel: niet beschikbaar, deze context is niet gemeten.")
    else:
        print(f"Label volgens de labelregel (ter vergelijking): {resultaat['label']}")


if __name__ == "__main__":
    main()

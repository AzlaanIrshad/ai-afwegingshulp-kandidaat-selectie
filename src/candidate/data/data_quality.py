"""
data_quality.py: Data Quality Test Suite voor de kandidaat-selector. Zes checks op de benchmark en de
scenario's, een rapport in markdown (template: data_quality_template.md), en exitcode 1 bij
minstens één FAIL.

Gebruik (vanuit de hoofdmap van de repo): python -m src.candidate.data.data_quality
"""

import argparse
from datetime import date
from pathlib import Path
from string import Template

import pandas as pd

from src.candidate.data.labelling_rule import (
    LABEL_GEEN, MEETWAARDE_KOLOMMEN, UITVOER_MAP, bepaal_label, controleer_config_ids, laad_benchmark,
    maak_configuratietabel,
)
from src.candidate.data.split_scenarios import sha256_van_bestand

# Het template van het rapport. De vaste teksten staan daar, niet in de code.
TEMPLATE_PAD = Path(__file__).with_name("data_quality_template.md")

RAPPORT_PAD = UITVOER_MAP.parents[1] / "docs/data-quality-report.md"

# Met één model zegt een categorie alleen iets over dat ene model.
MIN_MODELLEN = 2

# Daarboven ziet een gebruiker met die hardware vaker "niets voldoet" dan een categorie.
MAX_AANDEEL_GEEN = 0.5

# Per check: beschrijving, dimensie, waarom de check telt voor dit project, en wat je doet als hij niet slaagt.
CHECKS = {
    "T01": ("Geen lege waarden in de meetkolommen", "Completeness",
            "Een lege meting valt bij het middelen stil weg en kan een label verschuiven.",
            "Meet de configuraties met lege waarden opnieuw, of haal ze uit de benchmark."),
    "T02": ("Meetkolommen zijn getallen, macro_f1 in [0, 1], latency > 0, energie > 0", "Accuracy",
            "Een onmogelijke waarde laat een configuratie aan elke eis voldoen.",
            "Zoek uit waar de onmogelijke waarden vandaan komen en corrigeer of verwijder die metingen."),
    "T03": ("Elke config_id hoort bij één configuratie", "Uniqueness",
            "De labelregel middelt per config_id. Twee configuraties onder één id worden samen gemiddeld.",
            "Geef elke configuratie een eigen config_id en maak daarna de scenario's opnieuw."),
    "T04": ("Labels gelijk aan de labelregel", "Accuracy", "Het label is de waarheid waarop getraind en geëvalueerd wordt.",
            "Maak de scenario's opnieuw met generate_scenarios.py op deze benchmark."),
    "B01": (f"Elke categorie heeft >= {MIN_MODELLEN} modellen", "Representativeness",
            "De app toont een categorie. Met één model is dat een uitspraak over dat ene model.",
            "Meet minstens 2 modellen per categorie. Dit hangt samen met de categorie-indeling, die nog niet "
            "nog vastlegt."),
    "B02": (f"Aandeel \"geen\" per hardware hoogstens {MAX_AANDEEL_GEEN:.0%}", "Fairness",
            "Gebruikers met hardware die vaak \"geen\" krijgt, zien vaker dat niets voldoet.",
            "Bespreek met het team of dit acceptabel is: de CPU is in de dummy 8x trager, dus veel eisen zijn daar "
            "niet haalbaar. Anders de scenariogrenzen of de gemeten modellen aanpassen."),
}


def voorbeelden(waarden):
    """De eerste vijf unieke waarden als tekst, zodat je de fout terugvindt."""
    return ", ".join(list(dict.fromkeys(map(str, waarden)))[:5])


def resultaat(check_id, fouten, goed, bij_fout="FAIL"):
    """(id, beschrijving, dimensie, waarom, uitkomst, toelichting) van één check."""
    return (check_id, *CHECKS[check_id][:3], bij_fout if fouten else "PASS", ", ".join(fouten) + "." if fouten else goed)


def t01(benchmark, scenarios):
    leeg = {k: benchmark.loc[benchmark[k].isna(), "config_id"] for k in MEETWAARDE_KOLOMMEN}
    return resultaat("T01", [f"{k} leeg in {len(ids)} rijen (bijv. {voorbeelden(ids)})" for k, ids in leeg.items() if len(ids)],
                     f"Geen lege meetwaarden in {len(benchmark)} rijen.")


def t02(benchmark, scenarios):
    geldig = {"macro_f1": lambda w: w.between(0, 1), "latency_ms_per_doc": lambda w: w > 0, "energy_j_per_doc": lambda w: w > 0}
    # Tekst in een getalkolom wordt NaN en telt dus als ongeldig.
    fout = {k: benchmark[benchmark[k].notna() & ~f(pd.to_numeric(benchmark[k], errors="coerce"))] for k, f in geldig.items()}
    return resultaat("T02", [f"{k} ongeldig in {len(r)} rijen (bijv. {voorbeelden(r['config_id'])}, waarden {voorbeelden(r[k])})"
                             for k, r in fout.items() if len(r)], "Alle meetwaarden zijn geldige getallen.")


def t03(benchmark, scenarios):
    try:
        controleer_config_ids(benchmark)
        fouten = []
    except ValueError as fout:
        fouten = [str(fout).rstrip(".")]
    return resultaat("T03", fouten, f"{benchmark['config_id'].nunique()} config_id's, elk bij één configuratie.")


def t04(benchmark, scenarios):
    configuraties = maak_configuratietabel(benchmark)
    opnieuw = pd.Series([bepaal_label(configuraties, s) for s in scenarios.to_dict("records")], index=scenarios.index)
    anders = scenarios[scenarios["label"] != opnieuw]
    voorbeeld = voorbeelden(f"scenario_id {s['scenario_id']}: {s['label']}, labelregel {opnieuw[i]}" for i, s in anders.iterrows())
    return resultaat("T04", [f"{len(anders)} van {len(scenarios)} labels wijken af (bijv. {voorbeeld})"] if len(anders) else [],
                     f"Alle {len(scenarios)} labels gelijk aan de labelregel.")


def b01(benchmark, scenarios):
    modellen = benchmark.groupby("category")["model"].unique()
    return resultaat("B01", [f"{c} heeft {len(m)} {'model' if len(m) == 1 else 'modellen'} ({', '.join(sorted(m))})"
                             for c, m in modellen.items() if len(m) < MIN_MODELLEN],
                     ", ".join(f"{c}: {len(m)} modellen" for c, m in modellen.items()) + ".")


def b02(benchmark, scenarios):
    aandeel = (scenarios["label"] == LABEL_GEEN).groupby(scenarios["hardware"]).mean()
    tekst = ", ".join(f"{h} {a * 100:.1f}%".replace(".", ",") for h, a in aandeel.items())
    te_hoog = ", ".join(aandeel[aandeel > MAX_AANDEEL_GEEN].index)
    return resultaat("B02", [f"{tekst}, te hoog: {te_hoog}"] if te_hoog else [], tekst + ".", bij_fout="WARNING")


def tabel(resultaten, letter):
    """De resultatentabel voor de checks die met deze letter beginnen (T of B)."""
    regels = ["| Test ID | Check | Dimension | Why | Result | Notes |", "|---|---|---|---|---|---|"]
    regels += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | **{r[4]}** | {r[5]} |" for r in resultaten if r[0].startswith(letter)]
    return "\n".join(regels)


def rapport(resultaten, status, bestanden):
    """Vul het template (data_quality_template.md) met de resultaten van deze run."""
    ids = lambda uitkomst: ", ".join(r[0] for r in resultaten if r[4] == uitkomst) or "geen"
    return Template(TEMPLATE_PAD.read_text(encoding="utf-8")).substitute(
        status=status,
        datum=date.today().isoformat(),
        dataset_versie=", ".join(f"`{naam}` (SHA-256 `{sha}`)" for naam, sha in bestanden),
        tabel_technisch=tabel(resultaten, "T"),
        tabel_bias=tabel(resultaten, "B"),
        besluit={"FAILED": "Afwijzen", "WARNING": "Accepteren met aandachtspunten", "PASSED": "Accepteren"}[status],
        fails=ids("FAIL"),
        warnings=ids("WARNING"),
        acties="".join(f"\n- {r[0]}: {CHECKS[r[0]][3]}" for r in resultaten if r[4] != "PASS"),
    )


def main():
    parser = argparse.ArgumentParser(description="Data Quality Test Suite voor de kandidaat-selector.")
    parser.add_argument("--benchmark", default=str(UITVOER_MAP / "dummy_benchmark.csv"), help="Benchmark-CSV.")
    parser.add_argument("--scenarios", default=str(UITVOER_MAP / "scenarios.csv"), help="Scenario-CSV.")
    parser.add_argument("--report", default=str(RAPPORT_PAD), help="Pad voor het rapport.")
    args = parser.parse_args()

    benchmark, scenarios = laad_benchmark(args.benchmark), pd.read_csv(args.scenarios)
    resultaten = [check(benchmark, scenarios) for check in [t01, t02, t03, t04, b01, b02]]
    uitkomsten = {r[4] for r in resultaten}
    status = "FAILED" if "FAIL" in uitkomsten else "WARNING" if "WARNING" in uitkomsten else "PASSED"
    bestanden = [(Path(p).name, sha256_van_bestand(p)) for p in [args.benchmark, args.scenarios]]
    Path(args.report).write_text(rapport(resultaten, status, bestanden), encoding="utf-8", newline="\n")

    for r in resultaten:
        print(f"{r[0]}  {r[4]:<7}  {r[1]}" + (f"\n               {r[5]}" if r[4] == "FAIL" else ""))
    print(f"\nStatus: {status}\nRapport geschreven naar: {args.report}")
    if status == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
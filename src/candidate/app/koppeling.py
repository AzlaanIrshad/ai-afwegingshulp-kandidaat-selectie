"""
koppeling.py

Koppelt de kandidaat-categorie aan app.py.

Dit bestand vertaalt alleen de invoer van de app naar een scenario, en het resultaat
naar wat de app toont. Laden en voorspellen gebeurt met laad() en voorspel_scenario()
uit predict_candidate.py, dezelfde functies als op de command line.

Geen enkele fout gaat door naar de app: als er niet voorspeld kan worden, is de
categorie None en staat in "reden" waarom niet.
"""

from types import SimpleNamespace

from src.candidate.model import predict_candidate


# ---------------------------------------------------------------------------
# Vertaling van app-invoer naar modelinvoer
# ---------------------------------------------------------------------------

# Voorlopige koppeling voor de DUMMY benchmark: de hardware in de app is niet de
# hardware die gemeten is. Bij None wordt er niet voorspeld.
HARDWARE_KOPPELING = {
    "CPU · standaard werkstation": "ryzen5_3600x",
    "GPU · lokaal werkstation": "rtx3080",
    "Onbekend / nog te bepalen": None,
}

# Voorlopige koppeling voor de DUMMY benchmark: de app vraagt kwaliteit in %, het
# model verwacht min_macro_f1 als fractie.
PROCENT = 100

# De benchmark is alleen op batch size 1 gemeten.
MELDING_BATCH = "De categorie gaat uit van batch size 1, ook als je hierboven een andere batch size kiest."


def naar_scenario(hardware, tokens, kwaliteit, max_latency_ms, documenten, budget_kwh):
    """Zet de waarden uit de app om naar een scenario. Onbekende hardware wordt None."""
    return predict_candidate.maak_scenario(SimpleNamespace(
        hardware=HARDWARE_KOPPELING.get(hardware),
        input_length=tokens,
        min_f1=kwaliteit / PROCENT,
        max_latency_ms=float(max_latency_ms),
        volume=float(documenten),
        budget_kwh=budget_kwh,
    ))


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------

def nieuwste_run():
    """Geef de nieuwste runmap als tekst, of None als er nog geen run is."""
    try:
        return str(predict_candidate.zoek_run(None))
    except SystemExit:
        return None


def laad_voor_app(run_map=None, benchmark_pad=None, manifest_pad=None):
    """Laad de run met predict_candidate.laad(). Bij een fout een dict met de sleutel "fout".

    Dit is de functie die app.py cachet.
    """
    try:
        return predict_candidate.laad(run_map, benchmark_pad, manifest_pad)
    except SystemExit as fout:
        return {"fout": str(fout)}
    except Exception as fout:
        return {"fout": f"Het model kon niet worden geladen: {fout}"}


# ---------------------------------------------------------------------------
# Voorspellen
# ---------------------------------------------------------------------------

def geen_categorie(reden, geladen):
    """Het resultaat als er niet voorspeld kan worden."""
    return {
        "categorie": None,
        "reden": reden,
        "kansen": None,
        "t": None,
        "kandidaat": None,
        "run_map": geladen.get("run_map"),
        "is_dummy": geladen.get("is_dummy", False),
        "voorbeeldmodellen": {},
        "waarschuwingen": [],
        "label": None,
    }


def voorspel_voor_app(hardware, tokens, kwaliteit, max_latency_ms, documenten, budget_kwh, geladen=None):
    """Voorspel de categorie voor de waarden uit de app.

    Geef het resultaat van laad_voor_app mee als geladen, zodat het model gecachet kan
    worden. Zonder geladen wordt de nieuwste run geladen.
    """
    if geladen is None:
        geladen = laad_voor_app()
    if "fout" in geladen:
        return geen_categorie(geladen["fout"], geladen)

    scenario = naar_scenario(hardware, tokens, kwaliteit, max_latency_ms, documenten, budget_kwh)
    if scenario["hardware"] is None:
        return geen_categorie("De hardware is onbekend: daarvoor zijn geen benchmarkmetingen.", geladen)

    # De features delen hierdoor of nemen er de log10 van, dus deze moeten groter zijn dan 0.
    if min(tokens, max_latency_ms, documenten, 1 if budget_kwh is None else budget_kwh) <= 0:
        return geen_categorie("Documenten, latency en energiebudget moeten groter zijn dan 0.", geladen)

    # Buiten het gemeten bereik voorspelt de app niet. De command line doet dat wel, met een waarschuwing.
    laagste, hoogste = min(geladen["gemeten_lengtes"]), max(geladen["gemeten_lengtes"])
    if tokens < laagste or tokens > hoogste:
        return geen_categorie(f"De inputlengte {tokens} ligt buiten het gemeten bereik van de benchmark "
                              f"({laagste} tot en met {hoogste} tokens).", geladen)

    try:
        resultaat = predict_candidate.voorspel_scenario(geladen, scenario)
    except Exception as fout:
        return geen_categorie(f"De voorspelling is mislukt: {fout}", geladen)

    # Alleen de ja/nee-modellen geven per categorie de kans dat die voldoet. De kansen van
    # een multiclass-model betekenen iets anders, die horen hier dus niet.
    ja_nee = resultaat["soort"] == "ja_nee"
    return {
        "categorie": resultaat["categorie"],
        "reden": None,
        "kansen": resultaat["kansen"] if ja_nee else None,
        "t": resultaat["t"],
        "kandidaat": resultaat["kandidaat"],
        "run_map": geladen["run_map"],
        "is_dummy": geladen["is_dummy"],
        "voorbeeldmodellen": resultaat["voorbeeldmodellen"],
        "waarschuwingen": resultaat["waarschuwingen"],
        "label": resultaat["label"],
    }

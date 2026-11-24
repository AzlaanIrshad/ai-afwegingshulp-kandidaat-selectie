"""
features.py

Welke kolommen van de scenariodataset features zijn voor de classifier van
voor de kandidaat-modelcategorie, en welke niet.

Dit staat in een eigen bestand, zodat de scenariogenerator, de groepsgewijze
split en later train_candidate_selector.py dezelfde lijst gebruiken. Zo hangt
het trainingsscript niet af van de generator: als de scenario's later op de
echte benchmark worden gemaakt, blijft deze lijst gewoon gelden.

Dit bestand bouwt GEEN split en traint GEEN model. Het kiest alleen de
kolommen, controleert dat er geen gemeten uitkomst tussen zit en maakt de
afgeleide features (maak_features).

De regel: alleen wat de gebruiker invult in de afwegingshulp is een feature.
Gemeten uitkomsten uit de benchmark (macro-F1, latency, energie van een model)
zijn NOOIT een feature. Het label wordt daaruit berekend, dus dan zou het
model het antwoord al meekrijgen (datalek).
"""

import numpy as np

from src.candidate.data.labelling_rule import J_PER_KWH


# ---------------------------------------------------------------------------
# Features (alleen invoer van de gebruiker)
# ---------------------------------------------------------------------------

# Numerieke features.
# * input_length_tokens blijft NUMERIEK (geen one-hot). Bij de groepsgewijze
#   split heeft de testfamilie een inputlengte die het model nooit gezien
#   heeft. Als getal kan het model daartussen interpoleren. Als one-hot zou
#   die lengte een kolom zijn die in de training altijd 0 was.
# * max_energy_kwh_per_month kan leeg (NaN) zijn: niet elke gebruiker heeft
#   een energiebudget. Dat moet later nog worden afgehandeld, bijvoorbeeld door
#   de lege cel in te vullen en has_energy_budget (hieronder) mee te geven.
NUMERIEKE_FEATURES = [
    "input_length_tokens",
    "min_macro_f1",
    "max_latency_ms_per_doc",
    "monthly_volume",
    "max_energy_kwh_per_month",
]

# Categorische features (later one-hot). De hardware wordt door de gebruiker
# gekozen en komt in train en test allebei voor.
CATEGORISCHE_FEATURES = [
    "hardware",
]

# Hulpkolom voor het lege energiebudget: True als het scenario een budget heeft.
# Dit is ook gebruikersinvoer (wel of geen budget opgegeven), dus een feature.
# Samen met deze kolom kan het model het verschil zien tussen "geen budget" en
# een budget, ook als de lege cel later wordt ingevuld.
BUDGET_INDICATOR = "has_energy_budget"

# Alle features samen, in deze volgorde. Dit is de enige lijst die het
# trainingsscript gebruikt om X te maken.
FEATURE_KOLOMMEN = NUMERIEKE_FEATURES + CATEGORISCHE_FEATURES + [BUDGET_INDICATOR]


# ---------------------------------------------------------------------------
# Kolommen die GEEN feature zijn
# ---------------------------------------------------------------------------

# * scenario_id: alleen een volgnummer, zegt niets over het scenario.
# * family_id:   hardware + inputlengte. Alleen voor de groepsgewijze split.
#                Als feature zou het model families uit het hoofd leren.
# * label:       de uitkomst die het model moet voorspellen.
# * is_dummy:    markering van nepdata, geen eigenschap van het scenario.
GEEN_FEATURES = ["scenario_id", "family_id", "label", "is_dummy"]

# Kolommen uit de benchmark die NOOIT een feature mogen zijn. Dit zijn de
# gemeten uitkomsten (macro_f1, latency, energie) en de eigenschappen van één
# configuratie (welk model, welke precisie, ...). De gebruiker vult die niet
# in. Het model moet juist voorspellen welke categorie past. Vooral
# "category" zou een direct lek zijn: dat is bijna het label zelf.
# hardware en input_length_tokens staan ook in de benchmark, maar die vult de
# gebruiker zelf in, dus die mogen wel.
GEMETEN_KOLOMMEN = [
    # config_id is de id van één benchmarkconfiguratie, geen gebruikersinvoer.
    "config_id",
    "model",
    "n_params_m",
    "category",
    "precision",
    "batch_size",
    "repeat",
    "macro_f1",
    "latency_ms_per_doc",
    "energy_j_per_doc",
]

# De kolom waarop groepsgewijs gesplitst wordt (hele families apart).
GROEP_KOLOM = "family_id"

# De kolom met het label.
LABEL_KOLOM = "label"

# Getal voor "geen budget": ruimer dan elk budget in de data.
GROOT = 1e12


# ---------------------------------------------------------------------------
# Functies
# ---------------------------------------------------------------------------

def controleer_alleen_gebruikersinvoer(scenarios):
    """Stop met een foutmelding als een feature geen gebruikersinvoer is.

    Controleert vier dingen:
    1. geen feature staat in GEEN_FEATURES (id, familie, label, dummy-markering)
    2. geen feature is een gemeten uitkomst of configuratiekolom uit de benchmark
    3. de scenario's zelf bevatten geen gemeten kolommen (dan is er iets mis
       met de generator, ook als die kolommen geen feature zijn)
    4. alle features en het label staan ook echt in de scenario's.
    """
    fout_in_lijst = [k for k in FEATURE_KOLOMMEN if k in GEEN_FEATURES + GEMETEN_KOLOMMEN]
    if fout_in_lijst:
        raise ValueError(f"Deze features zijn geen gebruikersinvoer: {fout_in_lijst}")

    gemeten_in_csv = [k for k in GEMETEN_KOLOMMEN if k in scenarios.columns]
    if gemeten_in_csv:
        raise ValueError(f"De scenario's bevatten gemeten kolommen: {gemeten_in_csv}")

    ontbreekt = [k for k in FEATURE_KOLOMMEN + [LABEL_KOLOM] if k not in scenarios.columns]
    if ontbreekt:
        raise ValueError(f"Deze kolommen ontbreken in de scenario's: {ontbreekt}")


def selecteer_features(scenarios):
    """Geef X (alleen de features) en y (het label) voor een set scenario's."""
    controleer_alleen_gebruikersinvoer(scenarios)

    # .copy(), zodat aanpassingen aan X (bijv. lege cellen invullen) de
    # originele scenario's niet veranderen.
    x = scenarios[FEATURE_KOLOMMEN].copy()
    y = scenarios[LABEL_KOLOM].copy()
    return x, y


def maak_features(x, budget_groot=False, log10=False, budget_per_doc=False,
                  latency_per_token=False, cpu_interacties=False):
    """Maak de afgeleide features. Gebruikt alleen kolommen uit FEATURE_KOLOMMEN.

    Overgenomen uit 04_modelverbeteringen.ipynb, zodat er één versie is. Staat
    geen enkele optie aan, dan zijn dit de ruwe features met hardware als one-hot.
    """
    uit = x[["input_length_tokens", "min_macro_f1", "max_latency_ms_per_doc",
             "monthly_volume", "max_energy_kwh_per_month"]].astype(float)

    # Een lege budgetcel: 0 (zoals in de eerste notebooks) of een groot getal.
    uit["max_energy_kwh_per_month"] = uit["max_energy_kwh_per_month"].fillna(GROOT if budget_groot else 0)

    if budget_per_doc:
        budget_j = x["max_energy_kwh_per_month"] * J_PER_KWH / x["monthly_volume"]
        uit["budget_j_per_doc"] = budget_j.fillna(GROOT)
        uit["log10_budget_j_per_doc"] = np.log10(uit["budget_j_per_doc"])

    if latency_per_token:
        uit["latency_per_token"] = x["max_latency_ms_per_doc"] / x["input_length_tokens"]
        uit["log10_latency_per_token"] = np.log10(uit["latency_per_token"])

    if log10:
        for kolom in ["max_latency_ms_per_doc", "monthly_volume", "max_energy_kwh_per_month"]:
            uit[f"log10_{kolom}"] = np.log10(uit.pop(kolom))

    numeriek = list(uit.columns)

    uit["has_energy_budget"] = x["has_energy_budget"].astype(float)
    uit["hardware_rtx3080"] = (x["hardware"] == "rtx3080").astype(float)
    uit["hardware_ryzen5_3600x"] = (x["hardware"] == "ryzen5_3600x").astype(float)

    if cpu_interacties:
        for kolom in numeriek:
            uit[f"{kolom}_x_cpu"] = uit[kolom] * uit["hardware_ryzen5_3600x"]

    return uit

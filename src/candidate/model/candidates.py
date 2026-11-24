"""
candidates.py

De kandidaten voor de kandidaat-modelcategorie: de regels, de
modellen en hun instellingen. train_candidate_selector.py en predict_candidate.py
gebruiken allebei dit bestand, zodat die scripts niet van elkaar afhangen.

De instellingen komen uit 04_modelverbeteringen.ipynb. Dit bestand traint zelf
niets en kiest niets: het zegt alleen wat een kandidaat is en hoe die voorspelt.
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.candidate.data.features import maak_features
from src.candidate.data.labelling_rule import (
    CATEGORIE_VOORKEUR,
    LABEL_GEEN,
    bepaal_label,
    voldoende_categorieen,
)


# ---------------------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------------------

# Vaste seed voor de modellen. --seed gaat alleen over het maken van de data.
RANDOM_STATE = 42

# De features van het eindmodel: dezelfde als bij de boommodellen in het notebook.
BOOM_OPTIES = {"budget_per_doc": True, "latency_per_token": True}

# De beste instellingen voor HistGradientBoosting uit stap 9.
HGB_PARAMS = {"max_depth": 8, "min_samples_leaf": 5, "learning_rate": 0.1, "max_iter": 200}

# Namen van de kandidaten.
BASELINE = "Baseline"
OPZOEK = "Opzoekregel"
CONSERVATIEF = "Conservatieve opzoekregel (variant b)"
LR = "Logistische regressie"
BOOM = "Beslisboom"
HGB = "HistGradientBoosting"

# Complexiteit, van eenvoudig naar ingewikkeld.
RANG_REGEL = 1
RANG_LR = 2
RANG_BOOM = 3
RANG_HGB = 4
RANG_JA_NEE = 5

# Drempels voor de ja/nee-modellen (stap 10). Elke t is een eigen kandidaat.
DREMPELS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Logistische regressie (stap 7).
LR_OPTIES = {"budget_groot": True, "log10": True, "budget_per_doc": True,
             "latency_per_token": True, "cpu_interacties": True}
LR_PARAMS = {"C": 10, "max_iter": 1000}

# Beslisboom (stap 7). HGB gebruikt dezelfde features (stap 9).
BOOM_PARAMS = {"max_depth": 10, "min_samples_leaf": 1}


def maak_kandidaten():
    """Geef alle kandidaten met hun soort, complexiteit en instellingen."""
    kandidaten = [
        {"naam": BASELINE, "soort": "regel", "complexiteit": RANG_REGEL,
         "instellingen": {"drempels": "mediaan per hardware en categorie"}},
        {"naam": OPZOEK, "soort": "regel", "complexiteit": RANG_REGEL,
         "instellingen": {"lengte": "dichtstbijzijnde gemeten lengte, bij gelijke afstand de langere"}},
        {"naam": CONSERVATIEF, "soort": "regel", "complexiteit": RANG_REGEL,
         "instellingen": {"lengte": "dichtstbijzijnde kortere en langere lengte moeten allebei voldoen"}},
        {"naam": LR, "soort": "model", "complexiteit": RANG_LR,
         "instellingen": {"features": LR_OPTIES, "params": LR_PARAMS}},
        {"naam": BOOM, "soort": "model", "complexiteit": RANG_BOOM,
         "instellingen": {"features": BOOM_OPTIES, "params": BOOM_PARAMS}},
        {"naam": HGB, "soort": "model", "complexiteit": RANG_HGB,
         "instellingen": {"features": BOOM_OPTIES, "params": HGB_PARAMS}},
    ]
    for t in DREMPELS:
        kandidaten.append({
            "naam": f"Ja/nee HGB monotoon, t={t}", "soort": "ja_nee", "complexiteit": RANG_JA_NEE,
            "instellingen": {"features": BOOM_OPTIES, "params": HGB_PARAMS, "t": t},
        })
    return kandidaten


def maak_pipeline(kandidaat):
    """Maak de pipeline van een model: de features, eventueel schaling, en het model."""
    naam = kandidaat["naam"]
    instellingen = kandidaat["instellingen"]
    stappen = [("features", FunctionTransformer(maak_features, kw_args=instellingen["features"]))]

    if naam == LR:
        # Alleen de logistische regressie heeft schaling nodig.
        stappen.append(("schaal", StandardScaler()))
        schatter = LogisticRegression(random_state=RANDOM_STATE, **instellingen["params"])
    elif naam == BOOM:
        schatter = DecisionTreeClassifier(random_state=RANDOM_STATE, **instellingen["params"])
    else:
        schatter = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **instellingen["params"])

    stappen.append(("model", schatter))
    return Pipeline(stappen)


# ---------------------------------------------------------------------------
# Het ja/nee-model
# ---------------------------------------------------------------------------

def monotone_richting(kolom):
    """Geef de richting voor monotonic_cst: -1 = strenger, +1 = ruimer, 0 = geen richting."""
    if kolom == "has_energy_budget" or kolom.startswith("hardware"):
        return 0
    if "min_macro_f1" in kolom or "input_length_tokens" in kolom or "monthly_volume" in kolom:
        return -1
    if "latency" in kolom or "budget" in kolom or "energy" in kolom:
        return 1
    return 0


def maak_model(kolommen):
    """Maak één ja/nee-model: de features in een pipeline, gevolgd door HistGradientBoosting."""
    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        monotonic_cst={kolom: monotone_richting(kolom) for kolom in kolommen},
        **HGB_PARAMS,
    )
    return Pipeline([("features", FunctionTransformer(maak_features, kw_args=BOOM_OPTIES)), ("model", model)])


def kans_ja(model, x):
    """De kans op "voldoet", per rij."""
    return model.predict_proba(x)[:, list(model.classes_).index(True)]


def kolommen_ja_nee(x):
    """De kolommen na de bewerking. Die zijn nodig voor monotonic_cst."""
    return maak_features(x.head(), **BOOM_OPTIES).columns


def kies_categorie(kansen, t):
    """Dezelfde regel als stap 10: de kleinste categorie met kans >= t, anders geen.

    kansen: {categorie: kans dat die categorie voldoet}, bijvoorbeeld van de ja/nee-modellen.
    """
    for categorie in CATEGORIE_VOORKEUR:
        if kansen.get(categorie, 0.0) >= t:
            return categorie
    return LABEL_GEEN


# ---------------------------------------------------------------------------
# Regels
# ---------------------------------------------------------------------------

def regel_baseline(scenarios, configuraties):
    """De regel uit 01_baseline.ipynb: drempels op de mediaan van de configuraties."""
    drempels = configuraties.groupby(["hardware", "category"])[["macro_f1", "latency_ms_per_doc"]].median()

    voorspeld = []
    for scenario in scenarios.to_dict("records"):
        keuze = LABEL_GEEN
        for categorie in CATEGORIE_VOORKEUR:
            f1, latency = drempels.loc[(scenario["hardware"], categorie)]
            if scenario["min_macro_f1"] <= f1 and scenario["max_latency_ms_per_doc"] >= latency:
                keuze = categorie
                break
        voorspeld.append(keuze)
    return voorspeld


def dichtstbijzijnde(lengte, gemeten):
    """Kleinste afstand. Bij gelijke afstand de langere lengte, want die is strenger."""
    return min(gemeten, key=lambda m: (abs(m - lengte), -m))


def beide_buren(lengte, gemeten):
    """De buren van stap 11, variant b, met dezelfde terugval."""
    korter = [m for m in gemeten if m < lengte]
    langer = [m for m in gemeten if m > lengte]
    if not langer:
        return [dichtstbijzijnde(lengte, gemeten)]
    if not korter:
        return [min(langer)]
    return [max(korter), min(langer)]


def regel_opzoek(scenarios, configuraties):
    """De labelregel op de dichtstbijzijnde gemeten lengte (stap 1)."""
    gemeten = sorted(configuraties["input_length_tokens"].unique())
    return [
        bepaal_label(configuraties, {
            **scenario, "input_length_tokens": dichtstbijzijnde(scenario["input_length_tokens"], gemeten)
        })
        for scenario in scenarios.to_dict("records")
    ]


def regel_conservatief(scenarios, configuraties):
    """Een categorie moet op beide buren voldoen (stap 11, variant b)."""
    gemeten = sorted(configuraties["input_length_tokens"].unique())

    voorspeld = []
    for scenario in scenarios.to_dict("records"):
        per_buur = [
            voldoende_categorieen(configuraties, {**scenario, "input_length_tokens": buur})
            for buur in beide_buren(scenario["input_length_tokens"], gemeten)
        ]
        samen = set.intersection(*per_buur)
        # Dezelfde voorkeur als de labelregel: de kleinste categorie eerst, anders geen.
        voorspeld.append(next((c for c in CATEGORIE_VOORKEUR if c in samen), LABEL_GEEN))
    return voorspeld


REGELS = {BASELINE: regel_baseline, OPZOEK: regel_opzoek, CONSERVATIEF: regel_conservatief}

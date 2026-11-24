"""Klikbaar Streamlit-prototype van de kandidaat-modelcategorie.

Losse versie van de app uit het groepsproject. Alle secties gebruiken de dummy benchmark.
"""

import math
import re
from dataclasses import dataclass
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from src.candidate.app import koppeling as app_koppeling

from src.candidate.data import labelling_rule
from src.candidate.data.categorieen import AANTAL_SIGMA, bepaal_categorie, categoriegrenzen


@dataclass(frozen=True)
class Scenario:
    documents: int = 50_000
    input_tokens: int = 500
    min_quality: int = 85
    hardware: str = "GPU · lokaal werkstation"
    batch_size: int = 1
    max_latency_ms: float = 20.0
    energy_budget_kwh: float | None = None


# De hardware-opties komen uit de koppeling met de benchmark (app_koppeling).
HARDWARE_OPTIONS = list(app_koppeling.HARDWARE_KOPPELING)
# De app start met de GPU geselecteerd.
DEFAULT_HARDWARE = "GPU · lokaal werkstation"

# Modelcategorieën in de volgorde van klein naar groot. De sleutels zijn de waarden uit de
# kolom "category" van de benchmark (zie src/candidate/data/categorieen.py). Een categorie zonder
# metingen wordt als "later" getoond. Zodra er metingen zijn, verschijnt hij vanzelf.
CATEGORIES = [
    dict(key="compact", title="Compact", blurb="Kleine, snelle modellen met laag energiegebruik."),
    dict(key="medium", title="Medium", blurb="Middelgrote modellen: meer kwaliteit, meer energie."),
    dict(key="groot", title="Groot", blurb="Grote modellen met de hoogste kwaliteit en het hoogste verbruik."),
]

MELDING_BATCH = "Alle metingen in de dummy benchmark zijn gedaan met batch size 1."


# ---------------------------------------------------------------------------
# Benchmarkdata
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_benchmark_fallback() -> tuple[pd.DataFrame | None, bool]:
    """Lees de benchmark direct in als er (nog) geen run geladen kan worden."""
    path = labelling_rule.UITVOER_MAP / "dummy_benchmark.csv"
    if not path.exists():
        return None, False
    benchmark = labelling_rule.laad_benchmark(path)
    is_dummy = bool(benchmark["is_dummy"].any()) if "is_dummy" in benchmark else False
    return labelling_rule.maak_configuratietabel(benchmark), is_dummy


def load_configurations() -> tuple[pd.DataFrame | None, bool]:
    """De gemiddelde configuraties uit de benchmark: bij voorkeur die van de geladen run."""
    loaded = load_candidate_run(app_koppeling.nieuwste_run())
    if "configuraties" in loaded:
        return loaded["configuraties"], loaded["is_dummy"]
    return load_benchmark_fallback()


def measured_context(configurations: pd.DataFrame, scenario: "Scenario") -> tuple[pd.DataFrame, int] | None:
    """Configuraties voor de gekozen hardware en de dichtstbijzijnde gemeten inputlengte."""
    hardware = app_koppeling.HARDWARE_KOPPELING.get(scenario.hardware)
    on_hardware = configurations[configurations["hardware"] == hardware]
    if hardware is None or on_hardware.empty:
        return None
    lengths = sorted(int(length) for length in on_hardware["input_length_tokens"].unique())
    used_length = min(lengths, key=lambda length: (abs(length - scenario.input_tokens), length))
    rows = on_hardware[on_hardware["input_length_tokens"] == used_length].copy()

    # Dezelfde regels als labelling_rule.voldoet_aan_eisen, maar per eis apart.
    rows["quality"] = rows["macro_f1"] * 100
    rows["monthly_kwh"] = labelling_rule.energie_kwh_per_maand(rows["energy_j_per_doc"], scenario.documents)
    rows["meets_quality"] = rows["macro_f1"] >= scenario.min_quality / 100
    rows["meets_latency"] = rows["latency_ms_per_doc"] <= scenario.max_latency_ms
    rows["meets_energy"] = (True if scenario.energy_budget_kwh is None
                            else rows["monthly_kwh"] <= scenario.energy_budget_kwh)
    rows["meets_all"] = rows["meets_quality"] & rows["meets_latency"] & rows["meets_energy"]
    return rows, used_length


def nl_number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".translate(str.maketrans({",": ".", ".": ","}))


def nl_auto(value: float) -> str:
    """Nederlandse notatie met ongeveer drie significante cijfers (voor kleine benchmarkwaarden)."""
    if value == 0:
        return "0"
    decimals = min(5, max(0, 2 - math.floor(math.log10(abs(value)))))
    return nl_number(value, decimals)


# ---------------------------------------------------------------------------
# Stijl
# ---------------------------------------------------------------------------

CSS = """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f4f6f8; color: #1d2b33; }
      .stMainBlockContainer { max-width: 1320px; padding: 2rem 2.2rem 3rem; }
      [data-testid="stHeader"] { background: transparent; }

      /* Vaste tekstkleuren, zodat de app er hetzelfde uitziet los van het thema van Streamlit. */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
      .stApp label, .stApp span, .stApp summary,
      [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
      [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
      [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p,
      [data-testid="stExpander"] summary, [data-testid="stExpander"] p {
        color: #1d2b33 !important;
      }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #5b6b75 !important;
      }
      [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {
        background: #ffffff !important; border-color: #d6dee2 !important;
      }
      [data-baseweb="input"] input, [data-baseweb="base-input"] input,
      [data-baseweb="select"] div, [data-baseweb="select"] span {
        color: #1d2b33 !important; -webkit-text-fill-color: #1d2b33 !important;
      }
      [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
        background: #eef2f4 !important; color: #1d2b33 !important;
      }
      [data-baseweb="popover"] li, [data-baseweb="menu"] li {
        background: #ffffff !important; color: #1d2b33 !important;
      }
      [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
      [data-testid="stSliderThumbValue"] { color: #1d2b33 !important; }
      /* Nieuwere Streamlit-versies gebruiken andere invoervelden. Die kleuren ook mee. */
      .stApp, [data-testid="stSidebar"], [data-testid="stSidebarContent"], [data-testid="stMain"] {
        color: #1d2b33 !important;
      }
      [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] [role="group"] {
        background: #ffffff !important; border-color: #d6dee2 !important; color: #1d2b33 !important;
      }
      [data-testid="stNumberInputField"], [data-testid="stSelectbox"] [role="combobox"] {
        color: #1d2b33 !important; -webkit-text-fill-color: #1d2b33 !important;
      }
      [data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg { color: #1d2b33 !important; }
      [role="listbox"], [role="option"] { background: #ffffff !important; color: #1d2b33 !important; }
      [role="option"][data-focused="true"], [role="option"]:hover { background: #eef2f4 !important; }
      [data-testid="stCheckbox"] label:has(input:not(:checked)) > div:not([data-testid]) {
        background-color: #ffffff !important; border: 1px solid #b9c4ca !important;
      }
      [data-testid="stExpander"] details {
        background: #ffffff; border: 1px solid #dfe6e9 !important; border-radius: 12px;
      }

      /* Scenario bovenaan de pagina */
      [data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border-radius: 16px; }
      .input-title { font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 0.5rem; color: #1d2b33; }
      .input-title small { font-size: 0.8rem; font-weight: 400; color: #5b6b75; margin-left: 0.5rem; }
      .input-head { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                    color: #2f6f62; margin: 0.2rem 0 -0.3rem; }
      .input-notes { display: flex; gap: 0.6rem; align-items: flex-start; margin-top: 0.6rem;
                     background: #f7f9fa; border: 1px solid #e7ecee; border-radius: 10px;
                     padding: 0.6rem 0.9rem; font-size: 0.8rem; color: #3f4f59; }
      .input-notes ul { margin: 0; padding-left: 1rem; }
      .stApp .input-notes li { margin: 0.1rem 0; color: #3f4f59 !important; }

      /* Modelgroottes uitgelegd */
      .stat-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 0.4rem; }
      .stat span { display: block; font-size: 0.74rem; color: #5b6b75; }
      .stat strong { display: block; font-size: 1.55rem; font-weight: 500; letter-spacing: -0.03em;
                     font-variant-numeric: tabular-nums; color: #1d2b33; margin-top: 0.15rem; }
      .stat em { display: block; font-style: normal; font-size: 0.7rem; color: #7a8890; }
      .stat-note { font-size: 0.76rem; color: #6b7a83; margin: 0.8rem 0 0.4rem; }

      /* Kop */
      .hero { background: #ffffff; border: 1px solid #e1e7ea; border-radius: 16px;
              padding: 1.6rem 1.8rem; display: flex; justify-content: space-between;
              gap: 1.5rem; align-items: flex-end; flex-wrap: wrap; }
      .eyebrow { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em;
                 color: #2f6f62; text-transform: uppercase; }
      .hero h1 { font-size: clamp(1.7rem, 2.6vw, 2.35rem) !important; letter-spacing: -0.04em;
                 margin: 0.35rem 0 0.4rem; padding: 0; line-height: 1.15; }
      .hero p { margin: 0; color: #4d5d67 !important; max-width: 640px; font-size: 0.98rem; }
      .tags { display: flex; gap: 0.45rem; flex-wrap: wrap; }
      .tag { padding: 0.35rem 0.7rem; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
             background: #eef2f4; color: #3f4f59; white-space: nowrap; }
      .tag.green { background: #e4f1ec; color: #22594d; }
      .tag.amber { background: #fbf1dc; color: #7a5612; }

      /* Secties */
      .section-head { display: flex; align-items: baseline; justify-content: space-between;
                      gap: 1rem; margin: 2.1rem 0 0.8rem; flex-wrap: wrap; }
      .section-head h2 { font-size: 1.25rem !important; margin: 0; padding: 0; letter-spacing: -0.02em; }
      .section-head .num { color: #2f6f62; font-weight: 700; margin-right: 0.5rem; }
      .section-head small { color: #5b6b75; font-size: 0.82rem; }

      /* Advieskaarten */
      .panel { background: #ffffff; border: 1px solid #e1e7ea; border-radius: 16px;
               padding: 1.3rem 1.4rem; height: 100%; }
      .panel-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
                     text-transform: uppercase; color: #6b7a83; }
      .big-value { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1.1;
                   margin: 0.5rem 0 0.2rem; color: #1d2b33; text-transform: capitalize; }
      .muted { font-size: 0.8rem; color: #5b6b75; }
      .kv { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.84rem;
            padding: 0.45rem 0; border-bottom: 1px dashed #e7ecee; }
      .kv strong { font-variant-numeric: tabular-nums; }
      .examples { margin-top: 0.9rem; font-size: 0.84rem; }
      .examples b { display: block; font-size: 0.72rem; color: #6b7a83; font-weight: 600; margin-bottom: 0.2rem; }
      .badge { display: inline-block; padding: 0.25rem 0.55rem; border-radius: 6px; font-size: 0.7rem;
               font-weight: 700; letter-spacing: 0.04em; }
      .badge.dummy { background: #fbf1dc; color: #7a5612; }
      .badge.ok { background: #e4f1ec; color: #22594d; }
      .empty-state { display: flex; gap: 0.8rem; align-items: flex-start; margin-top: 0.8rem;
                     background: #f7f9fa; border: 1px solid #e7ecee; border-radius: 10px;
                     padding: 0.85rem 1rem; font-size: 0.86rem; color: #3f4f59; }
      .note-list { margin: 0.9rem 0 0; padding-left: 1.1rem; font-size: 0.8rem; color: #4d5d67; }
      .note-list li { margin: 0.2rem 0; color: #4d5d67 !important; }
      .measured { margin-top: 0.8rem; background: #eef6f3; border-radius: 10px; padding: 0.7rem 0.9rem;
                  font-size: 0.84rem; color: #22594d; }

      /* Klein waarschuwingsicoon met uitleg bij hover */
      .panel-top { display: flex; justify-content: space-between; gap: 0.5rem; align-items: center; }
      .panel-tools { display: flex; gap: 0.5rem; align-items: center; }
      .tip { position: relative; display: inline-flex; align-items: center; cursor: help; outline: none; }
      .tip-box { visibility: hidden; opacity: 0; transition: opacity 0.15s ease;
                 position: absolute; top: calc(100% + 8px); right: -6px; z-index: 20; width: 290px;
                 background: #fffaf0; border: 1px solid #f0d9a6; border-left: 4px solid #e0a526;
                 border-radius: 10px; padding: 0.7rem 0.85rem; font-size: 0.78rem; line-height: 1.45;
                 color: #5f4a1c; box-shadow: 0 8px 24px rgba(29, 43, 51, 0.14); text-align: left; }
      .tip:hover .tip-box, .tip:focus .tip-box, .tip:focus-within .tip-box { visibility: visible; opacity: 1; }
      .tip-box b { display: block; color: #5c3d00; font-size: 0.82rem; margin-bottom: 0.25rem; }
      .tip-box .tip-row { display: flex; justify-content: space-between; gap: 0.5rem;
                          border-top: 1px dashed #f0d9a6; padding-top: 0.3rem; margin-top: 0.35rem; }
      .tip-box .tip-foot { display: block; margin-top: 0.4rem; color: #8a6a24; font-size: 0.7rem; }

      /* Compacte waarschuwingskaart */
      .warn-panel { margin-top: 12px; background: #fffaf0; border: 1px solid #f0d9a6;
                    border-left: 4px solid #e0a526; border-radius: 12px; padding: 0.75rem 1rem;
                    display: flex; gap: 0.75rem; align-items: flex-start; }
      .warn-panel b { color: #5c3d00; font-size: 0.86rem; }
      .warn-panel p { margin: 0.15rem 0 0; color: #5f4a1c !important; font-size: 0.8rem; }
      .warn-source { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                     color: #946b16 !important; }
      /* Waarschuwingsdriehoek, alleen met CSS (st.html laat geen SVG door) */
      .warn-tri { flex: none; display: inline-flex; align-items: flex-end; justify-content: center;
                  width: 22px; height: 20px; padding-bottom: 2px; box-sizing: border-box;
                  background: #e0a526; clip-path: polygon(50% 0, 100% 100%, 0 100%);
                  color: #3d2800 !important; font-size: 12px; font-weight: 800; line-height: 1; }
      .warn-panel .warn-tri { margin-top: 0.1rem; }

      /* Modelvergelijking */
      .model-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
      .model-card { background: #ffffff; border: 1px solid #e1e7ea; border-top: 4px solid #3f8b7b;
                    border-radius: 14px; padding: 1.1rem 1.15rem; min-width: 0; }
      .model-card.below { border-top-color: #d19a3a; }
      .model-index { font-size: 0.66rem; color: #6b7a83; letter-spacing: 0.12em; font-weight: 600; }
      .model-card h3 { font-size: 1rem !important; line-height: 1.35; margin: 0.35rem 0 0.6rem; padding: 0; }
      .status { display: inline-block; border-radius: 6px; font-size: 0.7rem; font-weight: 700;
                padding: 0.28rem 0.5rem; background: #e4f1ec; color: #22594d !important; }
      .below .status { background: #fbf1dc; color: #7a5612 !important; }
      .quality { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.05em; line-height: 1.1;
                 margin-top: 0.9rem; color: #1d2b33; }
      .quality span { font-size: 1.05rem; color: #6b7a83 !important; margin-left: 2px; }
      .metric-label { font-size: 0.74rem; color: #5b6b75; }
      .quality-track { height: 5px; background: #e8eeeb; border-radius: 8px; margin: 0.7rem 0 0.9rem; }
      .quality-track div { height: 5px; background: #3f8b7b; border-radius: 8px; }
      .quality-track.fail div { background: #d19a3a; }
      .quality.fail { color: #8a5a00 !important; }
      .metric-row.fail { background: #fdf3dd; border-radius: 6px; padding-left: 0.45rem; padding-right: 0.45rem;
                         margin: 0 -0.45rem; border-bottom-color: transparent; }
      .metric-row.fail strong { color: #8a5a00 !important; }
      .metric-row.fail strong::before { content: "! "; font-weight: 800; }
      .metric-row { display: flex; justify-content: space-between; gap: 0.4rem; font-size: 0.82rem;
                    padding: 0.42rem 0; border-bottom: 1px solid #f0f3f4; }
      .metric-row:last-of-type { border-bottom: none; }
      .metric-row strong { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
      .metric-row em { display: block; font-style: normal; font-size: 0.66rem; color: #7a8890; }
      .cat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .context-note { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; margin: -0.2rem 0 0.8rem;
                      font-size: 0.82rem; color: #4d5d67; }
      .model-card .empty-state { font-size: 0.8rem; }
      .card-top { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem;
                  min-height: 1.5rem; }
      .flag { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.22rem 0.5rem;
              border-radius: 999px; white-space: nowrap; }
      .predicted-flag { background: #2f6f62; color: #ffffff !important; }
      .later-flag { background: #eef2f4; color: #5b6b75 !important; }
      .model-card.predicted { border-color: #2f6f62; box-shadow: 0 0 0 2px rgba(47, 111, 98, 0.18); }
      .model-card.later { background: #fafbfc; border-style: dashed; border-top-style: solid;
                          border-top-color: #b9c4ca; }
      .model-card.later h3, .model-card.later .quality { color: #5b6b75 !important; }
      .cat-blurb { font-size: 0.78rem; color: #5b6b75; margin: -0.3rem 0 0.6rem; }
      .cat-models { margin-top: 0.7rem; font-size: 0.78rem; color: #3f4f59; }
      .cat-models b { display: block; font-size: 0.68rem; color: #6b7a83; font-weight: 600; margin-bottom: 0.15rem; }
      .cfg-list { margin-top: 0.6rem; font-size: 0.76rem; }
      .cfg-list b { display: block; font-size: 0.68rem; color: #6b7a83; font-weight: 600; margin-bottom: 0.15rem; }
      .cfg-list ul { list-style: none; margin: 0; padding: 0; }
      .stApp .cfg-list li { margin: 0.12rem 0; }
      .stApp .cfg-list li.ok { color: #22594d !important; }
      .stApp .cfg-list li.no { color: #8a5a00 !important; }
      .summary { margin-top: 14px; background: #ffffff; border: 1px solid #e1e7ea; border-radius: 12px;
                 padding: 0.85rem 1rem; font-size: 0.9rem; display: flex; justify-content: space-between;
                 gap: 1rem; flex-wrap: wrap; }
      .summary small { color: #5b6b75; font-size: 0.8rem; }

      /* Achter de schermen */
      .expander-head { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                       color: #2f6f62; margin: 0.4rem 0 0.6rem; }
      .principle + .expander-head { margin-top: 1.6rem; }
      .cat-table { border-collapse: collapse; font-size: 0.82rem; margin-top: 0.3rem; }

      /* Alle modellen */
      .mtable-wrap { background: #ffffff; border: 1px solid #e1e7ea; border-radius: 14px; overflow-x: auto; }
      .mtable-caption { font-size: 0.78rem; color: #5b6b75; padding: 0.8rem 1.1rem; }
      .mtable { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
      .mtable th { text-align: left; background: #eef6f3; color: #22594d; font-size: 0.72rem; font-weight: 700;
                   padding: 0.7rem 1.1rem; white-space: nowrap; }
      .mtable th.sorted::after { content: " ▼"; font-size: 0.6rem; }
      .mtable th.sorted.asc::after { content: " ▲"; }
      .mtable td { padding: 0.85rem 1.1rem; border-top: 1px solid #f0f3f4; vertical-align: top; color: #1d2b33; }
      .mtable td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
      .mtable td.num.fail { color: #8a5a00 !important; font-weight: 600; }
      .mtable .name { font-weight: 700; }
      .mtable .sub { display: block; font-size: 0.72rem; color: #6b7a83; margin-top: 0.2rem; }
      .mtable .pill { display: inline-block; border-radius: 6px; font-size: 0.72rem; font-weight: 700;
                      padding: 0.3rem 0.55rem; background: #e4f1ec; color: #22594d !important; }
      .mtable .pill.no { background: #fbf1dc; color: #7a5612 !important; }
      .mtable .pill.warn { background: #fff4d6; color: #6b4e00 !important; }
      .mtable .pill.none { background: #eef2f4; color: #5b6b75 !important; }
      .mtable .why-text { display: block; font-size: 0.72rem; color: #5b6b75; margin-top: 0.35rem; max-width: 320px; }
      .cat-table th { text-align: left; font-size: 0.7rem; color: #6b7a83; font-weight: 600;
                      padding: 0.3rem 1.2rem 0.3rem 0; border-bottom: 1px solid #e1e7ea; }
      .cat-table td { padding: 0.3rem 1.2rem 0.3rem 0; border-bottom: 1px solid #f0f3f4; color: #1d2b33; }
      /* Stappen als lichte tijdlijn, omdat ze nu in een wit uitklapblok staan */
      .steps { list-style: none; margin: 0.3rem 0 0; padding: 0; display: grid;
               grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; }
      .stApp .steps li { position: relative; padding-top: 2.1rem; margin: 0; font-size: 0.78rem;
                         color: #5b6b75 !important; }
      .steps li::before { content: ""; position: absolute; top: 0.72rem; left: 1.9rem; right: -0.6rem;
                          height: 2px; background: #dfe6e9; }
      .steps li:last-child::before { display: none; }
      .stApp .steps .dot { position: absolute; top: 0; left: 0; width: 1.5rem; height: 1.5rem; border-radius: 50%;
                           display: flex; align-items: center; justify-content: center; font-size: 0.74rem;
                           font-weight: 700; background: #eef6f3; border: 1px solid #c6ddd5; color: #2f6f62 !important; }
      .stApp .steps b { display: block; font-size: 0.84rem; color: #1d2b33 !important; margin-bottom: 0.15rem; }
      .stApp .steps small { display: block; line-height: 1.5; color: #5b6b75 !important; font-size: 0.76rem; }
      .stApp .steps .human .dot { background: #2f6f62; border-color: #2f6f62; color: #ffffff !important; }
      .stApp .steps .human b { color: #22594d !important; }

      /* Niet / Wel */
      .principle { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1.3rem; }
      .principle > div { display: flex; gap: 0.65rem; align-items: flex-start; border-radius: 10px;
                         padding: 0.7rem 0.85rem; font-size: 0.82rem; color: #3f4f59; line-height: 1.5; }
      .principle .no { background: #fbf1f1; }
      .principle .yes { background: #eef6f3; }
      .stApp .principle .mark { flex: none; width: 1.3rem; height: 1.3rem; border-radius: 50%; display: flex;
                                align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 800;
                                color: #ffffff !important; margin-top: 0.05rem; }
      .principle .no .mark { background: #b55b5b; }
      .principle .yes .mark { background: #2f6f62; }
      .principle b { display: block; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
                     margin-bottom: 0.1rem; }
      .principle .no b { color: #8f3f3f; }
      .principle .yes b { color: #22594d; }
      .footer { margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid #e1e7ea;
                font-size: 0.76rem; color: #6b7a83; display: flex; justify-content: space-between;
                flex-wrap: wrap; gap: 0.5rem; }

      @media (max-width: 1100px) {
        .stMainBlockContainer { padding-left: 1.2rem; padding-right: 1.2rem; }
        .model-grid, .cat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .stat-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      }
      @media (max-width: 650px) {
        .model-grid, .cat-grid, .steps, .principle { grid-template-columns: 1fr; }
        .steps li::before { display: none; }
        .tip-box { width: 230px; }
        .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }

      /* Tooltips (de ? bij invoervelden en de themaknop) altijd leesbaar */
      [role="tooltip"] [data-testid="stTooltipContent"] {
        background: #1d2b33 !important; border-color: #1d2b33 !important;
      }
      [role="tooltip"] [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"],
      [role="tooltip"] [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
      }

      /* Knop om te wisselen tussen licht en donker, rechtsboven */
      .st-key-theme_switch { position: fixed; top: 0.75rem; right: 7.5rem; z-index: 999991; width: auto !important; }
      .st-key-theme_switch button { width: 2.4rem; height: 2.4rem; min-height: 0; padding: 0; border-radius: 999px;
                                    background: #ffffff !important; border: 1px solid #d6dee2 !important;
                                    box-shadow: 0 2px 8px rgba(29, 43, 51, 0.14); font-size: 1.1rem; line-height: 1; }
      .st-key-theme_switch button:hover { border-color: #2f6f62 !important; }
      .st-key-theme_switch button:focus-visible { outline: 2px solid #2f6f62; outline-offset: 2px; }
    </style>
    """

# Elke kleur uit de lichte stijl met zijn tegenhanger in de donkere stijl. Tekst wordt licht,
# vlakken worden donker en de accentkleuren worden lichter, zodat het contrast blijft.
DONKER = {
    # Tekst
    "#1d2b33": "#e6edf0", "#3f4f59": "#c3ced4", "#4d5d67": "#b3c0c7", "#5b6b75": "#9eadb5",
    "#6b7a83": "#8e9da6", "#7a8890": "#83929a",
    # Vlakken en randen
    "#ffffff": "#1b252b", "#f4f6f8": "#11181c", "#f7f9fa": "#222e35", "#fafbfc": "#182126",
    "#eef2f4": "#2a363d", "#e1e7ea": "#2e3b42", "#e7ecee": "#2e3b42", "#f0f3f4": "#263239",
    "#dfe6e9": "#34424a", "#d6dee2": "#3a4952", "#b9c4ca": "#55656e", "#e8eeeb": "#2a3a36",
    # Groen
    "#2f6f62": "#5fb8a3", "#22594d": "#8fd4c1", "#3f8b7b": "#4fa592", "#eef6f3": "#1c3330",
    "#e4f1ec": "#1f3a34", "#c6ddd5": "#2f5a50",
    # Oranje
    "#fbf1dc": "#3a2f17", "#fdf3dd": "#33290f", "#fff4d6": "#3a3015", "#fffaf0": "#2e2614",
    "#7a5612": "#f0c76e", "#8a5a00": "#f2b84b", "#6b4e00": "#f0c76e", "#d19a3a": "#d9a447",
    "#e0a526": "#e0a526", "#f0d9a6": "#5c4a22", "#5f4a1c": "#e8d3a3", "#5c3d00": "#f3d9a0",
    "#946b16": "#e0b45a", "#8a6a24": "#d6b26a", "#3d2800": "#3d2800",
    # Rood
    "#c8102e": "#ff5c6c", "#fbf1f1": "#3a2224", "#b55b5b": "#d98080", "#8f3f3f": "#f0a3a3",
    # Schaduwen
    "rgba(47, 111, 98, 0.18)": "rgba(95, 184, 163, 0.28)", "rgba(29, 43, 51, 0.14)": "rgba(0, 0, 0, 0.45)",
    "color-scheme: light": "color-scheme: dark",
}
KLEUR = re.compile("|".join(re.escape(kleur) for kleur in sorted(DONKER, key=len, reverse=True)))


def donker(tekst: str) -> str:
    """Zet de lichte kleuren in een tekst om naar de donkere kleuren."""
    return KLEUR.sub(lambda gevonden: DONKER[gevonden.group(0)], tekst)


def theme_mode() -> str:
    """Licht of donker: de keuze met de knop, anders het thema van het systeem (via Streamlit)."""
    keuze = st.session_state.get("theme_mode")
    if keuze in ("light", "dark"):
        return keuze
    systeem = getattr(getattr(st.context, "theme", None), "type", None)
    return "dark" if systeem == "dark" else "light"


def toggle_theme() -> None:
    st.session_state["theme_mode"] = "light" if theme_mode() == "dark" else "dark"


def render_theme_switch(mode: str) -> None:
    """Zon of maan rechtsboven. Een klik wisselt tussen licht en donker."""
    with st.container(key="theme_switch"):
        naar = "licht" if mode == "dark" else "donker"
        st.button("☀️" if mode == "dark" else "🌙", key="theme_toggle", on_click=toggle_theme,
                  help=f"Schakel naar {naar} thema")


def render_styles(mode: str) -> None:
    st.html(donker(CSS) if mode == "dark" else CSS)


def section(number: str, title: str, subtitle: str = "") -> None:
    small = f"<small>{escape(subtitle)}</small>" if subtitle else ""
    st.html(f'<div class="section-head"><h2><span class="num">{number}</span>{escape(title)}</h2>{small}</div>')


# ---------------------------------------------------------------------------
# Kop en scenario (horizontaal bovenaan de pagina)
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.html("""
    <div class="hero">
      <div>
        <div class="eyebrow">Inzicht vóór de inzet van AI</div>
        <h1>Duurzame AI Afwegingshulp</h1>
        <p>Vergelijk wat een modelkeuze betekent voor kwaliteit, snelheid en energie. Jij maakt de afweging.</p>
      </div>
      <div class="tags">
        <span class="tag green">Documentclassificatie</span>
        <span class="tag">Datareductie</span>
        <span class="tag amber">Demonstratiedata</span>
      </div>
    </div>
    """)


def render_scenario() -> Scenario:
    """Het scenario: wat de gebruiker invult. Dit staat horizontaal bovenaan de pagina."""
    st.html('<div class="input-title">Jouw scenario <small>Vul je situatie in. De pagina werkt direct bij. '
            'AI-taak: documentclassificatie (datareductie).</small></div>')
    with st.container(border=True):
        volume, requirements, infrastructure, energy = st.columns(4, gap="medium")

        with volume:
            st.html('<div class="input-head">Volume &amp; invoer</div>')
            documents = st.number_input("Documenten per maand", min_value=0, max_value=10_000_000,
                                        value=50_000, step=5_000, key="documents")
            length = st.number_input("Gemiddelde inputlengte (tokens)", min_value=50, max_value=10_000,
                                     value=500, step=50, key="input_tokens",
                                     help="Tokens zijn stukjes tekst. Dit is de gemiddelde lengte per document.")

        with requirements:
            st.html('<div class="input-head">Eisen</div>')
            quality = st.slider("Minimale gewenste kwaliteit (%)", min_value=0, max_value=100,
                                value=85, step=1, key="min_quality", help="Ondergrens voor de fictieve F1-score.")
            max_latency = st.number_input("Maximale latency (ms per document)", min_value=0.1,
                                          max_value=100_000.0, value=20.0, step=1.0, key="max_latency_ms")

        with infrastructure:
            st.html('<div class="input-head">Infrastructuur</div>')
            hardware = st.selectbox("Beschikbare hardware", options=HARDWARE_OPTIONS, key="hardware",
                                    index=HARDWARE_OPTIONS.index(DEFAULT_HARDWARE))

        with energy:
            st.html('<div class="input-head">Energie</div>')
            energy_budget = None
            if st.checkbox("Ik heb een energiebudget", key="has_energy_budget"):
                energy_budget = st.number_input("Energiebudget (kWh per maand)", min_value=0.001,
                                                max_value=1_000_000.0, value=1.0, step=0.1, format="%.3f",
                                                key="energy_budget_kwh")
    return Scenario(documents, length, quality, hardware, 1, max_latency, energy_budget)


def render_input_notes(result: dict) -> None:
    """Meldingen over de invoer, direct onder het scenario (bijvoorbeeld een niet-gemeten inputlengte)."""
    notes = [*result["waarschuwingen"], MELDING_BATCH]
    items = "".join(f"<li>{escape(note)}</li>" for note in notes)
    st.html(f'<div class="input-notes" role="note"><span aria-hidden="true">ⓘ</span><ul>{items}</ul></div>')


# ---------------------------------------------------------------------------
# Advies: kandidaat-categorie
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_candidate_run(run_map: str | None) -> dict:
    """Laadt een run één keer. Een nieuwere run heeft een andere map en wordt dus vanzelf geladen."""
    return app_koppeling.laad_voor_app(run_map)


def predict(scenario: Scenario) -> dict:
    """De voorspelling voor het scenario. Wordt vóór de kaart berekend, zodat de meldingen bij de invoer kunnen."""
    return app_koppeling.voorspel_voor_app(
        scenario.hardware, scenario.input_tokens, scenario.min_quality, scenario.max_latency_ms,
        scenario.documents, scenario.energy_budget_kwh, load_candidate_run(app_koppeling.nieuwste_run()))


def render_candidate_category(result: dict) -> dict:
    badge = ('<span class="badge dummy">DUMMY benchmarkdata</span>' if result["is_dummy"]
             else '<span class="badge ok">Benchmarkdata</span>')
    header = f"""
      <div class="panel-top">
        <div class="panel-label">Voorspelde kandidaat-categorie</div>
        <div class="panel-tools">{badge}</div>
      </div>
    """

    if result["categorie"] is None:
        st.html(f"""
        <div class="panel" role="status">
          {header}
          <div class="empty-state"><span aria-hidden="true">ⓘ</span>
            <div><b>Geen categorie</b><br>{escape(result["reden"])}</div></div>
        </div>
        """)
        return result

    threshold = "" if result["t"] is None else f" · drempel t = {nl_number(result['t'], 1)}"
    if result["kansen"] is None:
        chances = '<div class="muted" style="margin-top:0.6rem">Deze kandidaat geeft geen kansen per categorie.</div>'
    else:
        chances = "".join(
            f'<div class="kv"><span>Kans dat {escape(category)} voldoet</span>'
            f'<strong>{nl_number(chance, 3)}</strong></div>'
            for category, chance in result["kansen"].items()
        )
    measured = ""
    if result["label"] is not None:
        measured = (f'<div class="measured"><b>Volgens de benchmark: {escape(result["label"])}</b><br>'
                    f'<span class="muted">Gemeten waarde uit de benchmark, geen voorspelling.</span></div>')
    st.html(f"""
    <div class="panel" aria-label="Voorspelde categorie: {escape(result['categorie'])}">
      {header}
      <div class="big-value">{escape(result['categorie'])}</div>
      <div class="muted">voorspeld door {escape(result['kandidaat'])}{threshold}</div>
      <div style="margin-top:0.7rem">{chances}</div>
      {measured}
    </div>
    """)
    return result


# ---------------------------------------------------------------------------
# Modelvergelijking
# ---------------------------------------------------------------------------

def span(values, unit: str = "", decimals: int | None = None) -> str:
    """Toon één waarde of een bereik 'laag tot hoog' in Nederlandse notatie."""
    low, high = float(min(values)), float(max(values))
    fmt = nl_auto if decimals is None else (lambda v: nl_number(v, decimals))
    text = fmt(low) if fmt(low) == fmt(high) else f"{fmt(low)} tot {fmt(high)}"
    return f"{text}{unit}"


def configuration_detail(models: pd.DataFrame, labels: dict[str, str]) -> str:
    """Per configuratie (model × precisie) welke eisen die niet haalt.

    Alleen als de configuraties in deze categorie verschillend uitkomen: dan zie je bijvoorbeeld
    dat het ene model de kwaliteit haalt en het andere de latency, maar geen van beide alles.
    """
    results = []
    for _, configuration in models.sort_values(["model", "precision"]).iterrows():
        missing = [labels[aspect] for aspect in labels if not configuration[f"meets_{aspect}"]]
        results.append((f"{configuration['model']} {configuration['precision']}", missing))
    if len({tuple(missing) for _, missing in results}) <= 1:
        return ""
    items = "".join(
        f'<li class="ok">✓ {escape(name)} · haalt alle eisen</li>' if not missing
        else f'<li class="no">✗ {escape(name)} · haalt niet: {escape(", ".join(missing))}</li>'
        for name, missing in results
    )
    return f'<div class="cfg-list"><b>Per configuratie</b><ul>{items}</ul></div>'


def render_comparison(scenario: Scenario, predicted: str | None) -> None:
    configurations, is_dummy = load_configurations()
    if configurations is None:
        st.html('<div class="panel"><div class="empty-state"><span aria-hidden="true">ⓘ</span><div>'
                '<b>Geen metingen gevonden</b><br>De metingen om categorieën te vergelijken zijn nog niet '
                'beschikbaar.</div></div></div>')
        return
    context = measured_context(configurations, scenario)
    if context is None:
        st.html('<div class="panel"><div class="empty-state"><span aria-hidden="true">ⓘ</span><div>'
                '<b>Geen metingen voor deze hardware</b><br>Kies een CPU of GPU om de gemeten '
                'categorieën te vergelijken.</div></div></div>')
        return
    rows, used_length = context

    hardware = app_koppeling.HARDWARE_KOPPELING[scenario.hardware]
    length_note = ("" if used_length == scenario.input_tokens
                   else f" · dichtstbijzijnde gemeten lengte bij {nl_number(scenario.input_tokens)} tokens")
    source = "DUMMY benchmark" if is_dummy else "Benchmark"
    st.html(f'<div class="context-note"><span class="badge {"dummy" if is_dummy else "ok"}">{source}</span>'
            f'Gemeten op <b>{escape(hardware)}</b> · <b>{used_length} tokens</b> · batch 1{escape(length_note)}</div>')

    budget = scenario.energy_budget_kwh
    labels = {"quality": "kwaliteit", "latency": "latency", "energy": "energiebudget"}
    cards = []
    meeting = 0
    measured_categories = 0
    for category in CATEGORIES:
        models = rows[rows["category"] == category["key"]]
        is_predicted = predicted == category["key"]
        if models.empty:
            cards.append(f"""
            <article class="model-card later" aria-label="Categorie {escape(category['title'])}: nog niet gemeten">
              <div class="card-top"><div class="model-index">CATEGORIE</div>
                <span class="flag later-flag">Later · nog niet gemeten</span></div>
              <h3>{escape(category['title'])}</h3>
              <div class="cat-blurb">{escape(category['blurb'])}</div>
              <div class="empty-state">Deze categorie zit nog niet in de benchmark en dus ook niet in de
              voorspelling. Zodra er metingen zijn, verschijnen de waarden hier.</div>
            </article>
            """)
            continue

        measured_categories += 1
        # Zelfde regel als de labelregel: de categorie voldoet als minstens één
        # configuratie alle eisen tegelijk haalt.
        meets = bool(models["meets_all"].any())
        meeting += meets
        fails = {aspect: not models[f"meets_{aspect}"].any() for aspect in labels}
        failed = [labels[a] for a, has_failed in fails.items() if has_failed]
        if meets:
            status = "✓ Voldoet aan je eisen"
        elif failed:
            status = "! Voldoet niet: " + ", ".join(failed)
        else:
            status = "! Geen configuratie haalt alle eisen tegelijk"
        row = lambda aspect: " fail" if fails[aspect] else ""
        classes = " ".join(c for c in ["model-card", "" if meets else "below",
                                       "predicted" if is_predicted else ""] if c)
        flag = '<span class="flag predicted-flag">Voorspeld voor jouw scenario</span>' if is_predicted else ""
        names = ", ".join(sorted(models["model"].unique()))
        config_detail = configuration_detail(models, labels)
        budget_note = "geen budget ingesteld" if budget is None else f"budget {nl_auto(budget)} kWh"
        cards.append(f"""
        <article class="{classes}" aria-label="Categorie {escape(category['title'])}: {escape(status)}">
          <div class="card-top"><div class="model-index">CATEGORIE</div>{flag}</div>
          <h3>{escape(category['title'])}</h3>
          <div class="cat-blurb">{escape(category['blurb'])}</div>
          <span class="status">{escape(status)}</span>
          <div class="quality{row('quality')}">{span(models['quality'], decimals=1)}<span>%</span></div>
          <div class="metric-label">Kwaliteit · macro-F1 · grens {scenario.min_quality}%</div>
          <div class="quality-track{row('quality')}" aria-hidden="true">
            <div style="width: {models['quality'].max():.1f}%"></div></div>
          <div class="metric-row{row('latency')}">
            <span>Latency<em>per document · max {nl_number(scenario.max_latency_ms, 1)} ms</em></span>
            <strong>{span(models['latency_ms_per_doc'], ' ms')}</strong></div>
          <div class="metric-row"><span>Energie<em>per document · gemeten</em></span>
            <strong>{span(models['energy_j_per_doc'], ' J')}</strong></div>
          <div class="metric-row{row('energy')}"><span>Per maand<em>{budget_note}</em></span>
            <strong>{span(models['monthly_kwh'], ' kWh')}</strong></div>
          <div class="metric-row"><span>Modelgrootte<em>parameters</em></span>
            <strong>{span(models['n_params_m'], ' mln', decimals=0)}</strong></div>
          <div class="metric-row"><span>Gemeten configuraties<em>model × precisie</em></span>
            <strong>{len(models)}</strong></div>
          <div class="cat-models"><b>Modellen in deze categorie</b>{escape(names)}</div>
          {config_detail}
        </article>
        """)
    st.html('<div class="model-grid cat-grid">' + "".join(cards) + '</div>')
    st.html(f"""
    <div class="summary" role="status">
      <strong>{meeting} van de {measured_categories} gemeten categorieën voldoen aan al je eisen.</strong>
      <small>Een categorie voldoet als minstens één configuratie kwaliteit, latency en energiebudget tegelijk
      haalt. Bij een bereik zie je de laagste en hoogste gemeten waarde.</small>
    </div>
    """)


# ---------------------------------------------------------------------------
# Alle modellen
# ---------------------------------------------------------------------------

# Sorteeropties: label -> (kolom, oplopend). "category" en "status_key" krijgen een eigen volgorde.
SORT_OPTIONS = {
    "Status (voldoet eerst)": ("status_key", True),
    "Macro-F1 (hoog naar laag)": ("macro_f1", False),
    "Latency (snel naar traag)": ("latency_ms_per_doc", True),
    "Energie (zuinig naar veel)": ("energy_j_per_doc", True),
    "Parameters (klein naar groot)": ("n_params_m", True),
    "Categorie (compact naar groot)": ("category", True),
    "Model (A-Z)": ("model", True),
}
STATUS_ORDER = {"meets": 0, "fails": 1}
STATUS_PILLS = {"meets": ("", "✓ Voldoet aan eisen"), "fails": ("no", "! Voldoet niet")}


def model_status(row, scenario: Scenario) -> tuple[str, str]:
    """Status en uitleg voor één configuratie, met dezelfde eisen als de labelregel."""
    reasons = []
    if not row["meets_quality"]:
        reasons.append(f"Kwaliteit ligt {nl_auto(scenario.min_quality - row['quality'])} procentpunt onder jouw minimum.")
    if not row["meets_latency"]:
        reasons.append(f"Dit model is {nl_auto(row['latency_ms_per_doc'] - scenario.max_latency_ms)} ms langzamer dan toegestaan.")
    if not row["meets_energy"]:
        reasons.append(f"Maandverbruik ligt {nl_auto(row['monthly_kwh'] - scenario.energy_budget_kwh)} kWh boven jouw budget.")
    if not reasons:
        reasons.append("Haalt de ingestelde kwaliteit en latency" +
                       (" en het energiebudget." if scenario.energy_budget_kwh is not None
                        else ". Er is geen energiebudget ingesteld."))
    return ("meets" if row["meets_all"] else "fails"), " ".join(reasons)


def render_all_models(scenario: Scenario) -> None:
    """Alle gemeten configuraties (model × precisie) uit de benchmark naast elkaar, te sorteren."""
    configurations, is_dummy = load_configurations()
    if configurations is None:
        st.html('<div class="empty-state"><span aria-hidden="true">ⓘ</span><div>De benchmark is nog niet '
                'gevonden. Train eerst een run met de selector.</div></div>')
        return
    context = measured_context(configurations, scenario)
    if context is None:
        st.html('<div class="empty-state"><span aria-hidden="true">ⓘ</span><div>Kies een CPU of GPU om de '
                'gemeten modellen te vergelijken.</div></div>')
        return
    rows, used_length = context
    statuses = [model_status(row, scenario) for _, row in rows.iterrows()]
    rows["status_key"] = [status for status, _ in statuses]
    rows["reason"] = [reason for _, reason in statuses]

    sort_column, order_column = st.columns([3, 1])
    with sort_column:
        choice = st.selectbox("Sorteer op", options=list(SORT_OPTIONS), key="sort_models")
    with order_column:
        st.write("")
        reverse = st.checkbox("Omgekeerde volgorde", key="sort_reverse")

    column, ascending = SORT_OPTIONS[choice]
    if reverse:
        ascending = not ascending
    order = {category["key"]: i for i, category in enumerate(CATEGORIES)}
    keys = {"category": order, "status_key": STATUS_ORDER}
    key = (lambda values: values.map(keys[values.name]) if values.name in keys else values)
    rows = rows.sort_values([column, "model", "precision"], ascending=[ascending, True, True], key=key)

    titles = {category["key"]: category["title"] for category in CATEGORIES}
    headers = [
        ("model", "Model &amp; configuratie", "Het model en hoe het gemeten is."),
        ("category", "Categorie", "Compact, medium of groot, met de standaardafwijking (zie 04)."),
        ("macro_f1", "Macro-F1", "Kwaliteit: hoe goed het model documenten indeelt, elke soort even zwaar."),
        ("latency_ms_per_doc", "Latency", "Gemiddelde tijd per document."),
        ("energy_j_per_doc", "Energie", "Energie per document, en per maand bij jouw aantal documenten."),
        ("status_key", "Status &amp; uitleg", "Haalt deze configuratie al je eisen tegelijk?"),
    ]
    head = "".join(
        f'<th title="{tip}" class="{"sorted" + (" asc" if ascending else "") if col == column else ""}">{label} ⓘ</th>'
        for col, label, tip in headers
    )

    body = ""
    for row in rows.to_dict("records"):
        pill_class, pill_text = STATUS_PILLS[row["status_key"]]
        config = f"{row['precision']} · {used_length} tokens · batch 1"
        body += f"""
        <tr>
          <td><span class="name">{escape(str(row['model']))}</span>
            <span class="sub">{escape(config)}</span></td>
          <td>{escape(titles.get(row['category'], str(row['category'])))}
            <span class="sub">{nl_number(row['n_params_m'], 0)} mln parameters</span></td>
          <td class="num{"" if row['meets_quality'] else " fail"}">{nl_number(row['quality'], 2)}%</td>
          <td class="num{"" if row['meets_latency'] else " fail"}">{nl_auto(row['latency_ms_per_doc'])} ms</td>
          <td class="num{"" if row['meets_energy'] else " fail"}">{nl_auto(row['energy_j_per_doc'])} J
            <span class="sub">{nl_auto(row['monthly_kwh'])} kWh/maand</span></td>
          <td><span class="pill {pill_class}">{pill_text}</span>
            <span class="why-text">{escape(row['reason'])}</span></td>
        </tr>"""

    hardware = app_koppeling.HARDWARE_KOPPELING[scenario.hardware]
    length_note = ("" if used_length == scenario.input_tokens
                   else f" (dichtstbijzijnde gemeten lengte bij {nl_number(scenario.input_tokens)} tokens)")
    badge = ('<span class="badge dummy">DUMMY benchmark</span>' if is_dummy
             else '<span class="badge ok">Benchmark</span>')
    st.html(f"""
    <div class="mtable-wrap">
      <div class="mtable-caption">{badge} {rows['model'].nunique()} modellen · {len(rows)} configuraties ·
        {escape(hardware)} · {used_length} tokens{escape(length_note)} · batch 1 · maandverbruik met
        {nl_number(scenario.documents)} documenten</div>
      <table class="mtable" aria-label="Alle gemeten modellen">
        <thead><tr>{head}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """)


# ---------------------------------------------------------------------------
# Modelgroottes uitgelegd
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {"compact": "#9fd3c2", "medium": "#3f8b7b", "groot": "#1f4d43"}
# In het donker is de donkerste kleur bijna niet te zien, dus daar alles een stap lichter.
CATEGORY_COLORS_DARK = {"compact": "#bfe6d9", "medium": "#5fb8a3", "groot": "#2f7d6c"}


def load_model_sizes() -> pd.DataFrame | None:
    """Het aantal parameters per model uit de benchmark."""
    configurations, _ = load_configurations()
    if configurations is None or "n_params_m" not in configurations:
        return None
    sizes = configurations.drop_duplicates("model")
    return pd.DataFrame({"model": sizes["model"], "params_m": sizes["n_params_m"]})


def render_model_sizes(mode: str) -> None:
    """De grenzen van compact, medium en groot, met de spreiding van de gemeten modellen."""
    sizes = load_model_sizes()
    if sizes is None:
        return
    sizes = sizes.sort_values("params_m").reset_index(drop=True)
    lower, upper = categoriegrenzen(sizes["params_m"])
    sizes["categorie"] = [bepaal_categorie(p, (lower, upper)) for p in sizes["params_m"]]

    logs = sizes["params_m"].map(math.log10)
    mean, sigma = logs.mean(), logs.std(ddof=0)

    tiles = [
        ("Gemiddelde", f"{nl_number(10 ** mean, 1)} mln", "op de log-schaal (meetkundig)"),
        ("Standaardafwijking (σ)", nl_number(sigma, 3), f"in log10, een factor {nl_number(10 ** sigma, 2)}"),
        ("Grens compact | medium", f"{nl_number(lower, 1)} mln", f"gemiddelde − {nl_number(AANTAL_SIGMA, 1)} σ"),
        ("Grens medium | groot", f"{nl_number(upper, 1)} mln", f"gemiddelde + {nl_number(AANTAL_SIGMA, 1)} σ"),
        ("Mediaan", f"{nl_number(sizes['params_m'].median(), 1)} mln", "het middelste model"),
        ("Bereik", f"{nl_number(sizes['params_m'].min(), 1)} tot {nl_number(sizes['params_m'].max(), 1)}",
         "kleinste en grootste model (mln)"),
    ]
    items = "".join(f'<div class="stat"><span>{escape(label)}</span><strong>{escape(value)}</strong>'
                    f'<em>{escape(note)}</em></div>' for label, value, note in tiles)
    st.html(f'<div class="stat-grid">{items}</div><div class="stat-note"></div>')

    colors = CATEGORY_COLORS_DARK if mode == "dark" else CATEGORY_COLORS
    kleur = donker if mode == "dark" else (lambda tekst: tekst)
    low, high = sizes["params_m"].min() * 0.7, sizes["params_m"].max() * 1.8
    data = sizes.assign(start=low, log10=logs.round(3),
                        label=[f"{nl_number(p, 0)} mln · {c}" for p, c in zip(sizes["params_m"], sizes["categorie"])])
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("model:N", sort=list(data["model"]), title=None, scale=alt.Scale(paddingInner=0.4)),
        x=alt.X("start:Q", scale=alt.Scale(type="log", domain=[low, high], nice=False),
                title="Aantal parameters (miljoen, logaritmische schaal)"),
        x2="params_m:Q",
        color=alt.Color("categorie:N", title="Categorie",
                        scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("params_m:Q", title="Parameters (mln)", format=".1f"),
                 alt.Tooltip("log10:Q", title="log10"), alt.Tooltip("categorie:N", title="Categorie")],
    )
    lines = pd.DataFrame({"x": [lower, 10 ** mean, upper],
                          "label": [f"{nl_number(lower, 1)} mln", "gemiddelde", f"{nl_number(upper, 1)} mln"],
                          "dash": ["grens", "gemiddelde", "grens"]})
    rules = alt.Chart(lines).mark_rule(color=kleur("#5b6b75")).encode(
        x="x:Q", strokeDash=alt.StrokeDash("dash:N", legend=None,
                                           scale=alt.Scale(domain=["grens", "gemiddelde"], range=[[5, 4], [1, 3]])))
    labels = alt.Chart(lines).mark_text(align="left", dx=4, fontSize=11, color=kleur("#5b6b75")).encode(
        x="x:Q", y=alt.value(-6), text="label:N")
    # Achter elke balk het aantal parameters en de categorie, zodat je niet op de kleur hoeft te vertrouwen.
    bar_labels = alt.Chart(data).mark_text(align="left", dx=5, fontSize=11, color=kleur("#1d2b33")).encode(
        y=alt.Y("model:N", sort=list(data["model"])), x="params_m:Q", text="label:N")
    # Streamlit past de grafiek standaard in de gegeven hoogte, inclusief legenda en as ("fit"). Bij weinig
    # modellen worden de balken dan te smal en vallen ze over elkaar. Met "fit-x" past alleen de breedte zich
    # aan, en krijgt elk model een vaste rij van 34 pixels.
    chart = (bars + rules + labels + bar_labels).properties(
        height=34 * len(data), autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
        background="transparent",
    ).configure_axis(
        labelColor=kleur("#3f4f59"), titleColor=kleur("#5b6b75"), gridColor=kleur("#e7ecee"),
        domainColor=kleur("#b9c4ca"), tickColor=kleur("#b9c4ca"),
    ).configure_legend(
        labelColor=kleur("#3f4f59"), titleColor=kleur("#5b6b75"),
    ).configure_view(stroke=None)
    # theme=None: niet het thema van Streamlit, maar de kleuren van de app zelf.
    st.altair_chart(chart, width="stretch", theme=None)

    st.caption("Elke balk is één model. De gestreepte lijnen zijn de grenzen: een halve standaardafwijking onder en "
               "boven het gemiddelde, op de log-schaal. Op deze schaal liggen de grenzen even ver van het gemiddelde.")



# ---------------------------------------------------------------------------
# Uitleg
# ---------------------------------------------------------------------------

def render_explanation() -> None:
    st.write("")
    with st.expander("Hoe werkt deze app? Achtergrond en aannames"):
        render_background()


def render_background() -> None:
    st.html("""
    <div class="expander-head">Zo komt de uitkomst tot stand</div>
    <ol class="steps" aria-label="Van gebruiksscenario naar menselijke afweging">
      <li><span class="dot">1</span><b>Gebruiksscenario</b><small>Je vult je taak, volume en eisen in.</small></li>
      <li><span class="dot">2</span><b>Benchmarkdata</b><small>Metingen van modellen (nu nog voorbeelddata).</small></li>
      <li><span class="dot">3</span><b>Voorspelling</b>
        <small>Een model voorspelt welke categorie waarschijnlijk aan je eisen voldoet.</small></li>
      <li><span class="dot">4</span><b>Vergelijking</b><small>De gevolgen per categorie naast elkaar.</small></li>
      <li class="human"><span class="dot">5</span><b>Jouw afweging</b><small>Jij maakt de keuze.</small></li>
    </ol>
    <div class="principle">
      <div class="no"><span class="mark">✕</span><div><b>Niet</b>“Modelcategorie X is de beste, gebruik deze.”</div></div>
      <div class="yes"><span class="mark">✓</span><div><b>Wel</b>“Modelcategorie X levert deze kwaliteit, gebruikt ongeveer
        zoveel energie en is zo snel. Modelcategorie Y heeft andere gevolgen.”</div></div>
    </div>
    <div class="expander-head">Aannames</div>
    """)
    st.markdown("""
    - **Voorbeelddata:** de voorspelling en de vergelijking gebruiken dezelfde set metingen. Die metingen
      zijn nu nog verzonnen en zeggen niets over echte modellen.
    - **Welke metingen je ziet:** de kaarten tonen de metingen op de hardware die je kiest, bij de
      inputlengte die het dichtst bij jouw invoer ligt. Alles is gemeten met één document tegelijk
      (batch size 1).
    - **Wanneer voldoet een categorie:** als minstens één model uit die categorie tegelijk de minimale
      kwaliteit, de maximale latency en (als je dat instelt) het energiebudget haalt.
    - **Energie per maand:** het energiegebruik per document maal het aantal documenten per maand.
      Alleen het verwerken van documenten telt mee. Het trainen van modellen, opslag en stroom
      in rust niet.
    - **Kwaliteit:** een score die laat zien hoe goed een model documenten indeelt, waarbij elke soort
      document even zwaar telt. Het is niet hetzelfde als het percentage goed ingedeelde documenten.
    - **Categorieën:** compact, medium en groot volgen uit het aantal parameters van de modellen. Zie
      hieronder hoe de grenzen berekend worden.
    """)
    render_category_definition()


def render_category_definition() -> None:
    """Hoe de modelgroottes compact, medium en groot bepaald worden, met de grenzen van de huidige benchmark."""
    st.html('<div class="expander-head">Modelgroottes: compact, medium en groot</div>')
    st.markdown("""
    Een model krijgt een categorie op basis van zijn **aantal parameters**: hoe groot het model is. De grenzen
    liggen niet vast, maar worden berekend uit de gemeten modellen, met de **standaardafwijking**:

    1. Neem van elk model de log10 van het aantal parameters. Elk model telt één keer.
    2. Bereken daarvan het gemiddelde en de standaardafwijking (σ).
    3. Deel de modellen in:
        - **Compact:** meer dan een halve σ onder het gemiddelde.
        - **Medium:** hoogstens een halve σ van het gemiddelde af.
        - **Groot:** meer dan een halve σ boven het gemiddelde.

    **Waarom log10?** Modelgroottes verschillen vooral in verhouding: van 66 naar 132 miljoen is net zo'n stap
    als van 340 naar 680 miljoen. Zonder log zou één groot model het gemiddelde ver omhoog trekken.
    **Waarom een halve σ?** Zo zijn de drie groepen ongeveer even groot. Komen er modellen bij, dan schuiven
    de grenzen mee. "Groot" betekent dus: groot ten opzichte van de gemeten modellen.
    """)

    st.caption("De grenzen en de indeling van de gemeten modellen zie je in 04 Modelgroottes uitgelegd.")


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Duurzame AI Afwegingshulp", page_icon="🌿", layout="wide",
                       initial_sidebar_state="collapsed")
    mode = theme_mode()
    render_styles(mode)
    render_theme_switch(mode)
    render_header()
    scenario = render_scenario()
    prediction = predict(scenario)
    render_input_notes(prediction)

    section("01", "Advies voor jouw scenario", "Voorspelde kandidaat-categorie")
    render_candidate_category(prediction)

    section("02", "Vergelijk de modelcategorieën", "Gemeten waarden uit de benchmark voor jouw scenario")
    render_comparison(scenario, prediction["categorie"])

    section("03", "Vergelijk alle modellen", "Alle gemeten configuraties uit de benchmark, te sorteren")
    render_all_models(scenario)

    section("04", "Modelgroottes uitgelegd", "Modellen uit de benchmark: parameters en categorieën")
    render_model_sizes(mode)

    render_explanation()

    st.html("""
    <div class="footer">
      <span>Prototype · draait op dummy data</span>
      <span>Geen automatisch modeladvies of besluit over AI-inzet.</span>
    </div>
    """)


if __name__ == "__main__":
    main()
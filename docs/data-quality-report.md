# Data Quality Report: kandidaat-selector

## 1. Executive Summary
- **Status:** FAILED
- **Date of Run:** 2026-11-23
- **Dataset Version:** `dummy_benchmark.csv` (SHA-256 `1437d9045adef258dfceab01b29843c781d8c797093ffd572c2dbbbcbf952452`), `scenarios.csv` (SHA-256 `af839158e89f016ca87cf5a3f9d4a3d9be1d9f0d48ee9c029298cbada4441d3d`)

## 2. Test Suite Overview
Eigen checks in Python met pandas (`src/candidate/data/data_quality.py`), die de labelregel uit `labelling_rule.py` hergebruiken. Draaien vanuit de hoofdmap: `python -m src.candidate.data.data_quality`.

## 3. Detailed Test Results
### 3.1 Technical Integrity (Completeness & Accuracy)
| Test ID | Check | Dimension | Why | Result | Notes |
|---|---|---|---|---|---|
| T01 | Geen lege waarden in de meetkolommen | Completeness | Een lege meting valt bij het middelen stil weg en kan een label verschuiven. | **PASS** | Geen lege meetwaarden in 288 rijen. |
| T02 | Meetkolommen zijn getallen, macro_f1 in [0, 1], latency > 0, energie > 0 | Accuracy | Een onmogelijke waarde laat een configuratie aan elke eis voldoen. | **PASS** | Alle meetwaarden zijn geldige getallen. |
| T03 | Elke config_id hoort bij één configuratie | Uniqueness | De labelregel middelt per config_id. Twee configuraties onder één id worden samen gemiddeld. | **PASS** | 96 config_id's, elk bij één configuratie. |
| T04 | Labels gelijk aan de labelregel | Accuracy | Het label is de waarheid waarop getraind en geëvalueerd wordt. | **PASS** | Alle 2000 labels gelijk aan de labelregel. |

### 3.2 Social Impact & Bias (Representativeness)
| Test ID | Check | Dimension | Why | Result | Notes |
|---|---|---|---|---|---|
| B01 | Elke categorie heeft >= 2 modellen | Representativeness | De app toont een categorie. Met één model is dat een uitspraak over dat ene model. | **FAIL** | compact heeft 1 model (DistilBERT), groot heeft 1 model (BERT-large). |
| B02 | Aandeel "geen" per hardware hoogstens 50% | Fairness | Gebruikers met hardware die vaak "geen" krijgt, zien vaker dat niets voldoet. | **WARNING** | rtx3080 21,3%, ryzen5_3600x 55,7%, te hoog: ryzen5_3600x. |

## 4. Synthetic Data Usage
De benchmark en de scenario's zijn dummy data, gemaakt met `create_dummy_benchmark.py` en `generate_scenarios.py` (seed 42) en gemarkeerd met `is_dummy`. De verdelingen van de scenario's zijn een aanname, niet afgeleid uit echte eisen van gebruikers, en de dummy benchmark bevat alleen Engelstalige modellen. Datalekken tussen train en test worden al door `split_scenarios.py` geblokkeerd.

## 5. Conclusion & Next Steps
**Afwijzen.** FAIL: B01. WARNING: B02.

- B01: Meet minstens 2 modellen per categorie. Dit hangt samen met de categorie-indeling, die nog niet vastligt.
- B02: Bespreek met het team of dit acceptabel is: de CPU is in de dummy 8x trager, dus veel eisen zijn daar niet haalbaar. Anders de scenariogrenzen of de gemeten modellen aanpassen.

Omdat dit dummy data is, betekent 'afwijzen' hier: niet geschikt om conclusies over echte modellen te trekken. Voor het testen van de pipeline is de data wel bruikbaar.

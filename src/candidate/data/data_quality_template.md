# Data Quality Report: kandidaat-selector

## 1. Executive Summary
- **Status:** $status
- **Date of Run:** $datum
- **Dataset Version:** $dataset_versie

## 2. Test Suite Overview
Eigen checks in Python met pandas (`src/candidate/data/data_quality.py`), die de labelregel uit `labelling_rule.py` hergebruiken. Draaien vanuit de hoofdmap: `python -m src.candidate.data.data_quality`.

## 3. Detailed Test Results
### 3.1 Technical Integrity (Completeness & Accuracy)
$tabel_technisch

### 3.2 Social Impact & Bias (Representativeness)
$tabel_bias

## 4. Synthetic Data Usage
De benchmark en de scenario's zijn dummy data, gemaakt met `create_dummy_benchmark.py` en `generate_scenarios.py` (seed 42) en gemarkeerd met `is_dummy`. De verdelingen van de scenario's zijn een aanname, niet afgeleid uit echte eisen van gebruikers, en de dummy benchmark bevat alleen Engelstalige modellen. Datalekken tussen train en test worden al door `split_scenarios.py` geblokkeerd.

## 5. Conclusion & Next Steps
**$besluit.** FAIL: $fails. WARNING: $warnings.
$acties

Omdat dit dummy data is, betekent 'afwijzen' hier: niet geschikt om conclusies over echte modellen te trekken. Voor het testen van de pipeline is de data wel bruikbaar.

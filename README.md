# Kandidaat-modelcategorie

Mijn deel van het groepsproject **Duurzame AI Afwegingshulp**: een classificatiemodel dat op
basis van gebruikerseisen voorspelt welke modelcategorie (*compact*, *medium*, *groot* of *geen*)
waarschijnlijk voldoet aan de minimale kwaliteit, de maximale latency en een optioneel
energiebudget. Het model kiest alleen een technische kandidaat-categorie. Het besluit over wel of
geen AI-inzet blijft bij de gebruiker.

> **Let op:** alles draait op een **dummy benchmark** (verzonnen getallen, seed 42). De getallen
> zeggen niets over echte modellen.

## Wat er in deze map staat

```
kandidaat-selectie/
├── app.py                 Streamlit-app (01 voorspelling, 02 categorieën, 03 alle modellen, 04 modelgroottes)
├── src/candidate/
│   ├── data/              dummy benchmark, labelregel, scenario's, features, split, datakwaliteit
│   ├── model/             kandidaten, selector (trainen en kiezen), voorspellen
│   └── app/               koppeling tussen de app en het model
├── notebooks/             01 baseline, 02 logistische regressie, 03 beslisboom, 04 modelverbeteringen
├── tests/                 de belangrijkste tests
├── docs/                  onderzoeksverslag, model card, datakwaliteitsrapport, modelverbeteringen
└── outputs/               gegenereerde data en runs (niet in git)
```

Zie `src/candidate/README.md` voor alle scripts en opties.

## Installeren

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
pip install -r requirements.txt
```

## Draaien

Alle commando's vanuit deze map:

```
python -m src.candidate.model.train_candidate_selector --regenerate
python -m src.candidate.model.predict_candidate --hardware rtx3080 --input-length 128 --min-f1 0.85 --max-latency-ms 20 --volume 100000 --budget-kwh 0.01
streamlit run app.py
```

1. Maakt de dummy benchmark, de scenario's en de split, evalueert alle kandidaten op de CV-folds,
   kiest er één met de selectieregel en slaat de run op in `outputs/candidate/runs/`.
2. Voorspelt de categorie voor één scenario.
3. Start de app, die de nieuwste run gebruikt.

Datakwaliteitsrapport (na stap 1):

```
python -m src.candidate.data.data_quality
```

## Tests

```
python tests/test_candidate_pipeline.py
python tests/test_candidate_selector.py
python tests/test_candidate_predict.py
python tests/test_candidate_app.py
```

De tests maken hun eigen data in een tijdelijke map en raken `outputs/candidate` niet aan.

## Wat er niet in zit

De onderdelen van teamgenoten (energie-, latency- en kwaliteitsvoorspellers, betrouwbaarheidsdetector
en de gezamenlijke benchmark) zitten er niet in. Daarom gebruiken alle secties van de app de dummy
benchmark, en is er geen keuze voor batch size.

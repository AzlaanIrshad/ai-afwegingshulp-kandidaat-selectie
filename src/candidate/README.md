# Kandidaat-modelcategorie: de pipeline

Deze map bevat de pipeline die voorspelt welke modelcategorie (*compact*, *medium*, *groot*
of *geen*) waarschijnlijk voldoet aan de eisen van een gebruiker: minimale kwaliteit,
maximale latency en eventueel een energiebudget per maand. Het model kiest alleen een
technische kandidaat-categorie. Het neemt geen besluit over wel of geen AI-inzet.

Het onderzoek zelf staat in
`docs/onderzoeksverslag.md`.

> **Let op:** zolang de echte benchmark er niet is, draait alles op een **dummy benchmark**.
> De getallen zeggen niets over echte modellen.

## Installeren

Vanuit de hoofdmap:

```
pip install -r requirements.txt
```

Alle commando's hieronder draai je vanuit de hoofdmap.

## Snel starten

```
python -m src.candidate.model.train_candidate_selector --regenerate
python -m src.candidate.model.predict_candidate --hardware rtx3080 --input-length 128 --min-f1 0.85 --max-latency-ms 20 --volume 100000 --budget-kwh 0.01
streamlit run app.py
```

1. De eerste regel maakt de dummy benchmark, de scenario's en de split opnieuw, evalueert alle kandidaten en slaat de gekozen kandidaat op in een runmap.
2. De tweede regel voorspelt voor één scenario dat je zelf invult.
3. De derde regel start de app, die de nieuwste run gebruikt.

## Structuur

```
src/candidate/
├── data/                         van benchmark naar gesplitste scenario's
│   ├── categorieen.py            compact, medium en groot: gemiddelde ± 0,5σ van log10(aantal parameters)
│   ├── create_dummy_benchmark.py de dummy benchmark (verzonnen getallen)
│   ├── labelling_rule.py         de labelregel: welke categorie voldoet aan een scenario
│   ├── generate_scenarios.py     willekeurige scenario's, gelabeld met de labelregel
│   ├── features.py               welke kolommen features zijn (alleen gebruikersinvoer)
│   ├── split_scenarios.py        groepsgewijze split (GroupShuffleSplit op inputlengte) + manifest
│   ├── data_quality.py           zes kwaliteitscontroles op benchmark en scenario's
│   └── data_quality_template.md  sjabloon voor het rapport
├── model/                        kandidaten trainen, kiezen en gebruiken
│   ├── candidates.py             de kandidaten: regels en modellen met hun instellingen
│   ├── train_candidate_selector.py  evalueert alle kandidaten op de CV-folds en kiest er één
│   └── predict_candidate.py      laadt een run en voorspelt voor één scenario (voorspel_scenario())
└── app/
    └── koppeling.py              vertaalt de invoer van de app naar een scenario
```

De afhankelijkheden lopen één kant op: `data` ← `model` ← `app`.
De notebooks van het onderzoek staan in `notebooks/` in de hoofdmap.

## Opties van de selector

| Optie | Betekenis |
|---|---|
| `--regenerate` | Maak benchmark, scenario's en manifest eerst opnieuw. |
| `--output-dir` | Map voor data en runs (standaard `outputs/candidate`). |
| `--seed`, `--n` | Seed en aantal scenario's bij `--regenerate` (standaard 42 en 2000). Bij andere waarden in `outputs/candidate` stopt het script. Gebruik dan `--output-dir` of `--force`. |
| `--max-violation-rate` | Grens voor de selectieregel (standaard 0,05). |
| `--open-test` | Evalueert de gekozen kandidaat op de testset. **Doe dit één keer, aan het eind, op de echte benchmark.** |

## Uitvoer

Alles komt in `outputs/candidate/`:

| Bestand | Inhoud |
|---|---|
| `dummy_benchmark.csv` | De dummy benchmark. |
| `scenarios.csv` | De gelabelde scenario's. |
| `split_manifest.json` | De split: folds, SHA-256 van de data en de splitinstellingen. |
| `runs/<tijdstempel>/` | Eén map per run: `run.json` (instellingen, gekozen kandidaat, `test_opened`), `model.joblib`, `summary.csv` en `cv_results.csv`. Bij dummy data beginnen de namen met `DUMMY_`. |

`predict_candidate.py` en de app controleren met de SHA-256 in `run.json` of de benchmark en het manifest bij de run horen.

## De testset blijft dicht

De testlengtes worden willekeurig gekozen met een vaste seed. Niemand weet welke het zijn, en zo moet het blijven tot de eindtest:

- open `split_manifest.json` en de testdata niet
- commit het manifest niet (`outputs/` staat in `.gitignore`)
- kijk niet naar uitkomsten per foldlengte, zoals `cv_results.csv`. Met 8 lengtes kun je de testlengtes anders afleiden uit de lengtes die er niet bij staan
- de voorspelling (command line en app) toont geen label uit de benchmark zolang `test_opened` in de run `false` is. Dat geldt voor elk scenario, zodat de testlengtes ook daar niet te herkennen zijn.

## Tests

```
python tests/test_candidate_pipeline.py
python tests/test_candidate_selector.py
python tests/test_candidate_predict.py
python tests/test_candidate_app.py
```

De tests maken hun eigen data in een tijdelijke map. De bestanden in `outputs/candidate` worden niet aangeraakt.

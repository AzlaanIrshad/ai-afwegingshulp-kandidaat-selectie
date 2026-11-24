# Model card: kandidaat-selector

> **Status: sjabloon/WiP** De secties die niet van resultaten afhangen zijn ingevuld. Alles tussen `[ ]` vul ik in na de modelkeuze op de echte benchmark en na het één keer openen van de testset. Opbouw volgens Mitchell et al. (2019).

## 1. Modelgegevens

| | |
|---|---|
| Naam | Kandidaat-selector (kandidaat-modelcategorie) |
| Gemaakt door | Azlaan Irshad, binnen het projectteam Duurzame AI Afwegingshulp |
| Versie | `[commit]`, run `[outputs/candidate/runs/<map>]` |
| Datum | `[datum van de modelkeuze]` |
| Soort model | `[gekozen kandidaat, bijvoorbeeld ja/nee-HGB met drempel t, of de conservatieve opzoekregel]` |
| Code | `src/candidate/` (training: `train_candidate_selector.py`, voorspellen: `predict_candidate.py`) |
| Data | `[naam en SHA-256 van de benchmark]`, scenario's en split met seed 42 |
| Onderzoeksverslag | `onderzoeksverslag-kandidaat-modelcategorie.md` |

## 2. Bedoeld gebruik

- **Taak:** voorspellen welke modelcategorie voor documentclassificatie (compact, medium of groot) waarschijnlijk voldoet aan de eisen van een gebruiker, of dat geen onderzochte categorie voldoet. Het gaat om de **kleinste** categorie die voldoet.
- **Invoer:** minimale kwaliteit (macro-F1), maximale latency per document, aantal documenten per maand, eventueel een energiebudget per maand, inputlengte in tokens en batchgrootte.
- **Gebruikers:** medewerkers die in de afwegingshulp vergelijken wat een modelkeuze betekent voor kwaliteit, snelheid en energie.
- **Rol in de app:** de voorspelling is een **startpunt** voor de vergelijking. De app toont daarnaast de gemeten waarden per categorie. De medewerker maakt zelf de afweging.

## 3. Buiten bereik

- Het model neemt **geen besluit** over wel of geen AI-inzet.
- Het kiest geen specifiek model, alleen een categorie.
- Andere hardware dan de gemeten GPU, andere taken dan documentclassificatie, of andere documenten dan in de benchmark.
- Inputlengtes buiten het gemeten bereik (`[32]` tot en met `[512]` tokens) en batchgroottes die niet gemeten zijn. De app voorspelt daar niet.
- Eisen buiten de grenzen van de scenariogenerator. Daar geeft het model wel een antwoord, maar met een waarschuwing.

## 4. Factoren

De prestaties kunnen verschillen per:
- **inputlengte:** de middelste lengtes (het model interpoleert) tegenover de kortste en langste lengte (het model extrapoleert)
- **batchgrootte:** `[1, 2, 4, 8]`
- **wel of geen energiebudget**
- **hoe streng de eisen zijn:** dicht bij de grens van een categorie zijn fouten waarschijnlijker.

Het model gaat niet over personen. Er zijn dus geen persoonsgroepen om te vergelijken.

## 5. Metrieken

| Metriek | Waarom |
|---|---|
| Macro-F1 over de labels | Algemene kwaliteit, met elke klasse even zwaar. |
| Violation rate | Hoe vaak het model een categorie kiest die **niet** voldoet. Dit is de ernstigste fout: de gebruiker gaat uit van een categorie die de eisen niet haalt. |
| Gemiste-kansrate | Hoe vaak het model een grotere categorie kiest dan nodig. Dat kost onnodig energie, maar is veiliger. |
| Brier-score en reliability diagram | Alleen bij een model met kansen: of de kansen te vertrouwen zijn (LS-A-6). |

**Selectieregel** (vastgelegd vóór de echte resultaten): violation rate ≤ `[5%]` op de middelste CV-folds, daarna de hoogste macro-F1. Een eenvoudiger kandidaat wint als hij hoogstens 0,01 macro-F1 lager scoort.

## 6. Evaluatiedata

- **Modelkeuze:** cross-validatie op de trainlengtes. Elke fold houdt één inputlengte apart, met alle batchgroottes. Er zijn `[6]` folds, waarvan `[4]` in het midden.
- **Eindtest:** `[2]` inputlengtes, willekeurig gekozen met GroupShuffleSplit (seed 42). De kortste en langste lengte zijn uitgesloten. De testlengtes waren tot het openen bij niemand bekend.
- De test is **één keer** geopend: `[datum]`, `[commit]`, `test_opened: true`.

## 7. Trainingsdata

- **Benchmark:** `[naam en versie]`, `[aantal]` modellen, één GPU (`[type]`), fp32, inputlengtes `[32–512]`, batchgroottes `[1–8]`. Alleen rijen met status `complete`.
- **Categorieën:** op basis van het aantal parameters. De grenzen liggen op gemiddelde ± 0,5σ van log10(aantal parameters) over de gemeten modellen (`categorieen.py`): `[grenzen in miljoenen]`.
- **Scenario's:** `[aantal]` scenario's van de scenariogenerator (seed 42), gelabeld met de labelregel: een categorie voldoet als minstens één configuratie tegelijk de kwaliteit, de latency en het energiebudget haalt.
- **Features:** alleen invoer van de gebruiker, nooit gemeten uitkomsten (zie `features.py`).
- Deze versie draait nog op een **dummy benchmark**. `[Verwijderen na de echte benchmark.]`

## 8. Resultaten

**Op de CV-folds (modelkeuze)**

| Kandidaat | Deel | Macro-F1 | Violation rate | Gemiste-kansrate |
|---|---|---|---|---|
| `[gekozen kandidaat]` | midden | | | |
| `[gekozen kandidaat]` | randen | | | |
| Baseline | midden | | | |
| Opzoekregel | midden | | | |
| Conservatieve opzoekregel | midden | | | |

**Op de testset (één keer geopend)**

| Kandidaat | Macro-F1 | Violation rate | Gemiste-kansrate | Onzekerheid |
|---|---|---|---|---|
| `[gekozen kandidaat]` | | | | `[bootstrap-interval of per testlengte]` |
| Baseline | | | | |
| Opzoekregel | | | | |
| Conservatieve opzoekregel | | | | |

`[Korte uitleg: houdt het model stand op onbekende lengtes, en is het beter dan de opzoekregel?]`

## 9. Ethische overwegingen, beperkingen en aanbevelingen

- **Automation bias:** gebruikers volgen een automatische suggestie vaak, ook als die fout is. De app moet daarom altijd de gemeten waarden per categorie tonen, en de voorspelling niet als advies "gebruik deze" presenteren.
- **Violation boven gemiste kans:** bij twijfel is een grotere categorie veiliger, maar die kost meer energie. Dat botst met het duurzaamheidsdoel. `[Beschrijf hoe de gekozen kandidaat deze afweging maakt.]`
- **Geldigheid:** alleen binnen het gemeten bereik (hardware, lengtes, batchgroottes) en de grenzen van de scenariogenerator.
- **Kansen:** `[gekalibreerd of niet, volgens het reliability diagram]`. Toon in de app geen kansen met drie decimalen.
- **Kleine test:** de eindtest bestaat uit `[2]` lengtes. Het testresultaat is daardoor onzeker.
- **Onderhoud:** een nieuwe of gewijzigde benchmark betekent een nieuwe run, een nieuwe test en een nieuwe model card. De SHA-256-controle zorgt dat de app geen model met andere data gebruikt.

## 10. Integratiebesluit

`[Kies één uitkomst en onderbouw die met de punten hierboven.]`
1. **Integreren:** het model haalt de selectieregel op de testset en is beter dan de opzoekregel.
2. **Integreren met voorwaarden:** `[bijvoorbeeld: alleen binnen het gemeten bereik, met een waarschuwing, alleen naast de gemeten waarden]`.
3. **Het ML-model niet integreren:** in plaats daarvan de opzoekregel of alleen de gemeten waarden tonen, omdat `[reden]`.

## Bronnen

- Mitchell, M., et al. (2019). Model cards for model reporting. *Proceedings of FAT\* '19*, 220 tot 229. https://doi.org/10.1145/3287560.3287596

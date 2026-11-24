# Onderzoeksverslag: kandidaat-modelcategorie
 
## Inleiding
 
Dit verslag documenteert mijn onderzoek naar het selecteren van een kandidaat-modelcategorie binnen het project Duurzame AI Afwegingshulp. Doel: een classificatiemodel dat op basis van de eisen van een gebruiker (minimale kwaliteit, maximale latency, eventueel een energiebudget per maand) en de context (hardware, inputlengte, maandvolume in documenten) voorspelt welke modelcategorie daar waarschijnlijk aan voldoet: *compact*, *medium* of *geen onderzochte categorie voldoet*. Zo krijgt de medewerker een concrete categorie om verder te vergelijken, in plaats van alleen losse energiegetallen. Het model neemt géén besluit over wel of geen AI-inzet.
 
## Onderzoeksvraag
 
Welk classificatiemodel voorspelt op basis van gebruikseisen en benchmarkkenmerken welke modelcategorie waarschijnlijk aan de minimale kwaliteit-, latency- en energievoorwaarden voldoet?
 
## Onderzoek

Het onderzoek is in stappen opgebouwd. Per stap staat hieronder waarom hij nodig was, wat ik heb gedaan en wat nog open is.

### Uitgangspunten

**Conclusie.** Het model kiest een technische kandidaatcategorie (*compact*, *medium* of *geen onderzochte categorie voldoet*) binnen de onderzochte benchmarkruimte. De input is wat de gebruiker invoert: minimale kwaliteit, maximale latency, workload, hardware/context en eventueel een energiebudget. Het label volgt met een vaste regel uit de benchmarkresultaten. Daarom is het hele onderzoek afhankelijk van de gezamenlijke benchmarkdataset.

**Kernbegrippen.** Vier begrippen zijn nodig om dit onderzoek te kunnen doen.

| Begrip | Waarom het hier belangrijk is |
|---|---|
| Multiclass-classificatie | Het model kiest één klasse uit drie. |
| Macro-F1 | Elke klasse telt even zwaar, dus het valt op als het model de zeldzame klasse *geen onderzochte categorie voldoet* mist. |
| Constraint violation | Het model kiest een categorie die de eisen volgens de benchmark níet haalt. Dat is een onveilige keuze en de belangrijkste maat in dit onderzoek. |
| Groepsgewijs splitsen | Scenario's uit dezelfde benchmarkconfiguraties lijken sterk op elkaar. Bij een willekeurige split is de score te optimistisch, dus hele scenariofamilies blijven bij elkaar. |

**Gevolgen voor de aanpak.** Een uitkomst die je exact met een formule kunt uitrekenen, is meestal geen goed ML-onderzoek. Mijn labels komen uit een regel op de benchmarktabel. De waarde van ML zit daarom in het generaliseren naar scenario's en contexten die niet precies gemeten zijn. Daarom gebruik ik als features alleen wat de gebruiker invoert (geen gemeten uitkomsten), en test ik aan het eind op scenariofamilies die het model nog niet heeft gezien. Werk dat geen dataset nodig heeft (labelregels, scenariogenerator, opzet van het script) doe ik eerst. Daarom begin ik met een dummy benchmark.

**Open vragen.** (1) Welke modellen vallen in welke categorie? (2) Welke categorie wint als er meerdere voldoen: de zuinigste, de kleinste of de snelste? (3) Hoe definiëren we latency en energie (per document, per batch of per workload), zodat het gelijk is aan de rest van het project?

### Dummy benchmark en gelabelde scenariodataset
 
Status: afgerond (op de dummy benchmark). Klaar: de dummy benchmark, de labelregel, de scenariogenerator, de features (alleen gebruikersinvoer) en de groepsgewijze split met splitmanifest en datalekcontroles.
 
**Waarom dit nodig is.** De labels van mijn classifier komen uit de benchmarkdataset: per configuratie de gemeten kwaliteit, latency en energie. Die benchmark wordt nog door het team gemeten. Als ik daarop wacht, kan ik de scenariogenerator, de labelregel, de baseline en `train_candidate_selector.py` pas heel laat bouwen. Daarom maak ik eerst een dummy benchmark met hetzelfde soort kolommen, en daarop een gelabelde scenariodataset. Zo kan ik de baseline nu al bouwen en testen, en hoef ik later alleen de CSV te vervangen.
 
#### Dummy benchmark
 
**Wat er gemaakt is.** Het script `src/candidate/data/create_dummy_benchmark.py` maakt een nep-benchmark als CSV. Alle getallen zijn verzonnen met simpele formules plus wat willekeurige ruis. Eén rij is één gemeten run van één configuratie: model × hardware × precisie × inputlengte × herhaling. Het script gebruikt alleen numpy en pandas. De versies daarvan staan vast in `requirements.txt` (numpy 2.5.3, pandas 3.0.6). Die installeer je eerst. Met dezelfde seed komt er altijd precies dezelfde CSV uit (`numpy.random.default_rng(seed)`, NumPy Developers, n.d.).
 
```
pip install -r requirements.txt
python -m src.candidate.data.create_dummy_benchmark --seed 42 --out outputs/candidate/dummy_benchmark.csv
```
 
**Configuraties.** Alle configuraties staan als constanten bovenaan het script, zodat ik ze makkelijk kan aanpassen. Met deze instellingen komen er 216 rijen uit: 4 modellen × (2 precisies op de GPU + 1 op de CPU) × 6 inputlengtes × 3 herhalingen. Dat zijn 72 configuraties en 12 scenariofamilies (2 hardware × 6 inputlengtes).
 
| Onderdeel | Waarden | Opmerking |
|---|---|---|
| Modellen | DistilBERT (66M), BERT-base (110M), RoBERTa-base (125M), BERT-large (340M) | Dezelfde BERT-familie als elders in het project. Parameteraantallen uit de papers (Devlin et al., 2018, Liu et al., 2019 en Sanh et al., 2019). |
| Categorie | *compact* < 100M, *medium* 100M tot 400M | Wordt in het script afgeleid uit het aantal parameters, niet hard-coded. Later herzien: drie categorieën met de standaardafwijking (zie *Categorieën compact, medium en groot met de standaardafwijking*). |
| Hardware | `rtx3080` (GPU), `ryzen5_3600x` (CPU) | Twee opstellingen geven meer scenariofamilies voor de groepsgewijze split. |
| Precisie | GPU: fp32, fp16. CPU: alleen fp32 | fp16 is op een CPU meestal niet sneller, dus dat meten heeft weinig zin. |
| Inputlengte | 64, 128, 192, 256, 384, 512 tokens | BERT-achtige modellen accepteren maximaal 512 tokens (Devlin et al., 2018). Met lengtes in het midden kan ik een testfamilie kiezen zonder dat het model buiten het gemeten bereik moet voorspellen. |
| Herhalingen | 3 per configuratie | |
| Batchgrootte | 1 (vast) | Wel opgeslagen als kolom. |
 
**Formules (verzonnen).** Geen van deze getallen is gemeten. Ze zijn alleen zo gekozen dat de verhoudingen logisch zijn: een groter model is iets beter maar trager, langere input is trager, fp16 is sneller en de CPU is veel trager dan de GPU.
 
| Meting | Formule | Verzonnen getallen |
|---|---|---|
| Kwaliteit (macro-F1) | basiswaarde per model − fp16-verlies + ruis, geclipt op [0, 1] | Basis: DistilBERT 0,84, BERT-base 0,87, RoBERTa-base 0,88, BERT-large 0,89. fp16: −0,002. Ruis: normaal, sd 0,005. Op GPU en CPU hetzelfde. |
| Latency (ms per document) | basislatency × (inputlengte / 128) × CPU-factor × fp16-factor × ruis | Basis op de rtx3080 bij 128 tokens en fp32: DistilBERT 1,5, BERT-base 3,0, RoBERTa-base 3,2, BERT-large 9,0. CPU: × 8. fp16: × 0,6. Ruis: ongeveer ±5%. |
| Energie (J per document) | latency in seconden × gemiddeld vermogen × ruis | rtx3080: 220 W. ryzen5_3600x: 65 W. Ruis: ongeveer ±3%. |
 
**Kolommen.** `config_id`, `model`, `n_params_m`, `category`, `hardware`, `precision`, `input_length_tokens`, `batch_size`, `repeat`, `macro_f1`, `latency_ms_per_doc`, `energy_j_per_doc`, `is_dummy`. De kolom `is_dummy` is altijd `True`, zodat nepdata nooit per ongeluk voor echte data wordt aangezien. De structuur van de benchmark (deze kolommen) is met het team besproken en goedgekeurd. De kolomnamen kunnen nog veranderen als de echte benchmark er is.
 
**config_id.** Elke configuratie heeft een `config_id`: model (kleine letters), hardware, precisie, inputlengte en batchgrootte, gescheiden door `__`. Bijvoorbeeld `distilbert__rtx3080__fp32__64__bs1`. De herhaling hoort er niet in, dus alle herhalingen van één configuratie hebben dezelfde `config_id`. Zo blijven ze bij elkaar en kan ik ze per configuratie middelen, op dezelfde manier als in de rest van het project. De dummy benchmark heeft 72 unieke `config_id`'s, elk met 3 herhalingen.
 
**Controle.** Aan het eind print het script het aantal rijen, het aantal scenariofamilies en de gemiddelden per model × hardware × precisie × inputlengte. Een uitsnede (seed 42, dummy data):
 
| Model | Hardware | Precisie | Macro-F1 (128) | Latency ms (128) | Latency ms (512) | Energie J (512) |
|---|---|---|---|---|---|---|
| DistilBERT | rtx3080 | fp32 | 0,837 | 1,56 | 5,95 | 1,31 |
| DistilBERT | rtx3080 | fp16 | 0,838 | 0,88 | 3,56 | 0,78 |
| DistilBERT | ryzen5_3600x | fp32 | 0,841 | 12,52 | 47,97 | 3,14 |
| BERT-base | rtx3080 | fp32 | 0,869 | 2,98 | 11,49 | 2,55 |
| BERT-base | ryzen5_3600x | fp32 | 0,871 | 24,48 | 96,09 | 6,30 |
| RoBERTa-base | rtx3080 | fp32 | 0,878 | 3,10 | 12,94 | 2,84 |
| BERT-large | rtx3080 | fp32 | 0,890 | 8,39 | 34,68 | 7,56 |
| BERT-large | rtx3080 | fp16 | 0,889 | 5,55 | 22,41 | 4,75 |
| BERT-large | ryzen5_3600x | fp32 | 0,891 | 70,53 | 284,61 | 18,22 |
 
De waarden gedragen zich zoals bedoeld: *compact* is sneller en iets slechter dan *medium*, een vier keer zo lange input is ongeveer vier keer zo traag, fp16 kost ongeveer 0,6 keer de tijd van fp32, en de CPU is ongeveer 8 keer zo traag als de GPU in fp32. De CPU gebruikt minder vermogen (65 W tegenover 220 W), maar is zoveel trager dat hij per document ongeveer 2,4 keer zoveel energie kost (8 × 65 / 220). Dat volgt direct uit de verzonnen getallen. Eén ding valt op: het kwaliteitsverlies van fp16 (0,002) is kleiner dan de ruis (sd 0,005). Daardoor scoort fp16 met drie herhalingen soms gemiddeld iets hoger dan fp32 (bijvoorbeeld DistilBERT op de rtx3080 bij 128 tokens: 0,8381 tegenover 0,8372). Ook dat volgt uit de gekozen getallen en is geen fout in het script.
 
**Wat dit wel en niet zegt.** De dummy benchmark is alleen bedoeld om te testen of de pipeline technisch werkt. De getallen zeggen niets over echte modellen, echte hardware of echt energieverbruik, en ik gebruik ze niet voor conclusies. Omdat de data uit vaste formules komt, zullen de labels er later ook heel regelmatig uitzien. Een classifier die op deze data hoog scoort, zegt dus niets over hoe goed hij op echte metingen werkt.
 
**Nog open.** (1) De structuur van de benchmark is met het team besproken en goedgekeurd, maar de kolomnamen kunnen nog veranderen als de echte benchmark er is. Dan pas ik de scripts daarop aan. (2) In *compact* zit nu maar één model (DistilBERT). Of de grens van 100M blijft, of dat er meer modellen per categorie komen, is nog niet afgesproken. (3) Het LLM dat elders in het project is onderzocht, zit er bewust niet in, omdat het niet in *compact* of *medium* past. Mogelijk komt er later een derde categorie. (4) Energie als latency × vast vermogen is een flinke vereenvoudiging. Echte metingen zullen anders uitvallen. (5) De CPU-factor (8×) en het CPU-vermogen (65 W) zijn verzonnen.
 
#### Labelregel
 
Status: geïmplementeerd en getest op de dummy benchmark.
 
Mijn classifier leert van scenario's met een label. Dat label moet met een vaste, controleerbare regel uit de benchmark volgen. Deze regel moet vastliggen vóór de eerste training. Anders kan ik de regel achteraf aanpassen tot de resultaten mooi uitkomen.
 
**De regel.** Een categorie voldoet aan een scenario als minstens één configuratie uit die categorie tegelijk de minimale kwaliteit, de maximale latency en (als het scenario er een heeft) het energiebudget haalt. Voldoet geen enkele categorie, dan is het label *geen*.
 
| Onderdeel | Keuze | Waarom |
|---|---|---|
| Configuratie | Model × hardware × precisie × inputlengte × batchgrootte, herkenbaar aan de `config_id`. De herhalingen worden per `config_id` gemiddeld | Eén meting per configuratie, zodat een toevallig goede of slechte run het label niet bepaalt. Hoort een `config_id` bij meer dan één combinatie van configuratiekolommen, dan stopt de regel met een foutmelding, zodat runs van verschillende configuraties nooit samen gemiddeld worden. |
| Context | Alleen configuraties met dezelfde hardware en inputlengte als het scenario tellen mee | Een model dat op andere hardware snel genoeg is, zegt niets over deze situatie. |
| Precisie | Vrij: fp32 en fp16 tellen allebei | De gebruiker kiest een categorie, niet een precisie. Haalt een model het alleen in fp16, dan voldoet de categorie toch. |
| Energiebudget | Optioneel, in kWh per maand. Zonder budget telt alleen kwaliteit en latency | Niet elke gebruiker heeft een budget. Een budget per maand past bij de workload die de gebruiker invult. |
| Omrekening energie | kWh/maand = J/doc × documenten per maand / 3.600.000 | De benchmark meet per document. Met het maandvolume van het scenario reken ik elke configuratie om (1 kWh = 3.600.000 J). |
| Meerdere categorieën voldoen | De kleinste wint: eerst *compact*, dan *medium* | De afwegingshulp laat de kleinste categorie zien die goed genoeg is. |
| Context niet gemeten | Foutmelding in plaats van *geen* | *Geen* zou betekenen dat het gemeten en afgekeurd is. Niet gemeten is iets anders. |
 
**Wat er gemaakt is.** Het script `src/candidate/data/labelling_rule.py` bevat de regel als losse functies (`maak_configuratietabel`, `energie_kwh_per_maand`, `voldoet_aan_eisen`, `voldoende_categorieen`, `bepaal_label`). De scenariogenerator en `train_candidate_selector.py` kunnen die later importeren.
 
```
python -m src.candidate.data.labelling_rule --benchmark outputs/candidate/dummy_benchmark.csv
```
 
**Controle.** Direct gedraaid print het script eerst de gemiddelde configuraties in de contexten van de voorbeelden. Daarna labelt het negen handgekozen scenario's op de dummy benchmark (seed 42, 72 configuraties) en vergelijkt die met het label dat ik vooraf uit die configuraties had afgeleid. Alle negen kloppen.
 
| Hardware | Inputlengte | Min. F1 | Max. latency (ms) | Volume (doc/maand) | Max. energie (kWh/maand) | Voldoet | Label |
|---|---|---|---|---|---|---|---|
| rtx3080 | 128 | 0,80 | 20 | 10.000 | geen budget | compact, medium | compact |
| rtx3080 | 128 | 0,85 | 20 | 10.000 | geen budget | medium | medium |
| rtx3080 | 128 | 0,95 | 50 | 10.000 | geen budget | niets | geen |
| rtx3080 | 512 | 0,85 | 5 | 10.000 | geen budget | niets | geen |
| rtx3080 | 128 | 0,85 | 20 | 100.000 | 0,01 | niets | geen |
| rtx3080 | 128 | 0,85 | 20 | 10.000 | 0,01 | medium | medium |
| ryzen5_3600x | 128 | 0,85 | 20 | 10.000 | geen budget | niets | geen |
| ryzen5_3600x | 128 | 0,80 | 20 | 10.000 | geen budget | compact | compact |
| ryzen5_3600x | 128 | 0,85 | 50 | 10.000 | 0,01 | medium | medium |
 
Het tweede scenario is het basisvoorbeeld: DistilBERT haalt een F1 van 0,85 niet, BERT-base wel, dus het label is *medium*. Het vijfde en zesde laten zien dat het volume meetelt. Hetzelfde budget van 0,01 kWh per maand is bij 100.000 documenten 0,36 J per document. Daar past geen medium-model in. Bij 10.000 documenten is het 3,6 J per document, en dan past BERT-base in fp16 er ruim in. Het zevende heeft dezelfde eisen als het basisvoorbeeld, maar op de CPU. Daar is het snelste medium-model te traag voor 20 ms, dus het label is *geen*. Zonder energiebudget doet het volume niet mee aan het label.
 
**Nog open.** (1) Gemiddelde over de herhalingen is een keuze. Een strengere variant gebruikt de slechtste herhaling. Dat geeft minder onveilige labels, maar ook vaker *geen*. (2) De labelregel labelt alleen gemeten contexten. Voor een inputlengte die niet gemeten is, geeft hij een foutmelding in plaats van een label. Ik rond zo'n lengte dus niet af naar een gemeten lengte. Voorspellen voor lengtes die niet gemeten zijn, is juist de taak van het model, niet van de regel. (3) Op de dummy data zitten scenario's die net op een grens liggen dicht bij de ruis. Met echte data kan het label daar per meetronde omslaan. (4) Latency is per document. Of een opstelling het maandvolume ook echt op tijd kan verwerken (doorvoer), toetst de regel niet.
 
#### Scenariogenerator
 
Status: geïmplementeerd en getest op de dummy benchmark. Dezelfde seed geeft dezelfde scenario's.
 
Een scenario is wat een gebruiker invult in de afwegingshulp: de eisen (minimale kwaliteit, maximale latency, eventueel een energiebudget per maand) en de context (hardware, inputlengte, aantal documenten per maand). De classifier heeft veel van zulke scenario's met een label nodig om van te leren. Die scenario's maak ik met een generator, omdat er nog geen echte gebruikers zijn die eisen invullen.
 
**Wat er gemaakt is.** Het script `src/candidate/data/generate_scenarios.py` trekt willekeurige scenario's en geeft elk scenario een label met de labelregel uit `labelling_rule.py`. De regel staat dus op één plek en wordt niet opnieuw geschreven. Alle grenswaarden staan als constanten bovenaan het script. Met dezelfde seed komt er altijd precies dezelfde CSV uit (`numpy.random.default_rng(seed)`, NumPy Developers, n.d.). Alle trekkingen gebeuren per scenario in een vaste volgorde, ook als een scenario geen budget krijgt.
 
```
python -m src.candidate.data.generate_scenarios --seed 42 --n 2000 --benchmark outputs/candidate/dummy_benchmark.csv --out outputs/candidate/scenarios.csv
```
 
**Grenswaarden.** Elke grens ligt net buiten het bereik van de dummy benchmark, zodat er scenario's zijn die elk model haalt en scenario's die geen model haalt. De grenzen gelden voor beide hardware-opstellingen: een gebruiker kiest zijn eisen niet op basis van de hardware.
 
| Invoer | Verdeling | Bereik | Waarom |
|---|---|---|---|
| Hardware | Gelijkmatig uit de gemeten waarden | `rtx3080`, `ryzen5_3600x` | Alleen gemeten contexten, anders kan de labelregel er niets over zeggen. |
| Inputlengte | Gelijkmatig uit de gemeten waarden | 64, 128, 192, 256, 384, 512 tokens | Idem. |
| Minimale macro-F1 | Gelijkmatig | 0,80 tot 0,895 | De modellen halen ongeveer 0,84 tot 0,89. De bovengrens ligt net boven het beste model. |
| Maximale latency | Log-uniform | 2 tot 1000 ms per document | Latency loopt over meerdere ordes van grootte (ongeveer 0,4 ms op de GPU tot 285 ms op de CPU). Log-uniform zorgt dat strenge en ruime eisen even vaak voorkomen. |
| Maandvolume | Log-uniform | 1.000 tot 1.000.000 documenten | Van een klein team tot een grote organisatie (verzonnen). |
| Energiebudget | In 50% van de scenario's. Dan eerst een verborgen doel in J per document (log-uniform), omgerekend naar kWh per maand | Doel 0,2 tot 30 J per document | Niet elke gebruiker heeft een budget. Door in J per document te trekken blijft het budget in het bereik van de benchmark (ongeveer 0,1 tot 18 J), bij elk volume. Alleen de kWh-waarde komt in de CSV, afgerond op 4 significante cijfers. |
 
Mijn eerste grenzen na het toevoegen van de CPU waren latency 1 tot 300 ms, energie 0,1 tot 20 J en F1 0,80 tot 0,90. Daarmee kreeg 55% van de scenario's het label *geen*, en op de CPU zelfs 73%. De CPU is zo traag dat een strenge latency-eis daar bijna altijd *geen* geeft. Met ruimere latencygrenzen en een iets hogere ondergrens voor energie is de verdeling minder scheef. Een nog hogere ondergrens voor latency (3 tot 5 ms) maakt de CPU iets evenwichtiger, maar dan zitten er bijna geen strenge latency-eisen meer in voor de GPU.
 
**Kolommen.** `scenario_id`, `family_id`, `hardware`, `input_length_tokens`, `min_macro_f1`, `max_latency_ms_per_doc`, `monthly_volume`, `has_energy_budget`, `max_energy_kwh_per_month` (leeg als er geen budget is), `label`, `is_dummy`. De dummy-markering wordt van de benchmark overgenomen: scenario's op nepdata zijn zelf ook nep. Er staan geen gemeten uitkomsten in het bestand, alleen eisen, context, de familie en het label.
 
**Controle.** Met seed 42 en 2000 scenario's (dummy data):
 
| Hardware | compact | medium | geen |
|---|---|---|---|
| rtx3080 | 392 (39%) | 406 (41%) | 198 (20%) |
| ryzen5_3600x | 239 (24%) | 193 (19%) | 572 (57%) |
| **Totaal** | **631 (31,6%)** | **599 (30,0%)** | **770 (38,5%)** |
 
| Inputlengte | compact | medium | geen |
|---|---|---|---|
| 64 | 146 | 133 | 71 |
| 128 | 107 | 106 | 109 |
| 192 | 101 | 121 | 114 |
| 256 | 110 | 85 | 133 |
| 384 | 93 | 81 | 158 |
| 512 | 74 | 73 | 185 |
 
- 981 van de 2000 scenario's hebben een energiebudget.
- Twee keer draaien met seed 42 geeft een byte-voor-byte identieke CSV.
- De labels uit de CSV opnieuw berekenen met de labelregel geeft precies dezelfde labels (2000 van 2000). Een lege kWh-cel wordt bij het inlezen NaN, en de labelregel leest dat als "geen budget".
- Geen klasse komt onder de 15%, en geen hardware-opstelling krijgt (bijna) alleen één label. De CPU krijgt wel vaker *geen*. Dat volgt direct uit de dummy benchmark: de CPU is daar 8 keer zo traag.
- Langere input geeft vaker *geen*: bij 512 tokens is elk model trager en kost het meer energie, dus halen minder modellen de eisen.
- De scenariogenerator trekt alleen gemeten inputlengtes, want alleen daarvoor kan de labelregel een label geven (zie de labelregel, punt 2).
 
**Nog open.** (1) De grenzen zijn afgestemd op het bereik van de dummy benchmark. Met echte data moet ik ze opnieuw kiezen, en dat vastleggen vóór ik de testset open. (2) De verdelingen zijn een aanname. Ze zijn niet gebaseerd op hoe medewerkers echt eisen invullen. (3) De klassen zijn niet gelijk verdeeld, en per hardware verschilt het flink. Bij de evaluatie moet ik op macro-F1 letten en niet op accuracy. (4) Het volume telt alleen mee via het energiebudget. Het model moet dus zelf leren dat de verhouding budget / volume (J per document) bepaalt of een budget haalbaar is. Mogelijk helpt het later om die verhouding, of de logaritme van volume en budget, als feature mee te geven. Dat is ook gebruikersinvoer, dus geen datalek.
 
#### Features
 
Status: geïmplementeerd en getest op de dummy benchmark.
 
**De regel.** Alleen wat de gebruiker invult is een feature, nooit een gemeten uitkomst. Het label wordt uit de gemeten kwaliteit, latency en energie berekend. Die als feature meegeven zou een datalek zijn. Het model krijgt dan het antwoord al mee, en scoort hoog zonder iets te leren.
 
**Wat er gemaakt is.** De lijst staat in een eigen bestand, `src/candidate/data/features.py`. Zo gebruiken de generator, de split en later `train_candidate_selector.py` dezelfde lijst, en hangt het trainingsscript niet af van de generator. Als de scenario's later op de echte benchmark worden gemaakt, blijft de lijst gewoon gelden. De functie `selecteer_features` geeft X (alleen de features) en y (het label). Voordat hij dat doet, roept hij `controleer_alleen_gebruikersinvoer` aan.
 
| Soort | Kolommen | Opmerking |
|---|---|---|
| Numeriek | `input_length_tokens`, `min_macro_f1`, `max_latency_ms_per_doc`, `monthly_volume`, `max_energy_kwh_per_month` | `input_length_tokens` blijft een getal (geen one-hot). De testfamilie heeft een inputlengte die het model nooit gezien heeft. Als getal kan het model daartussen interpoleren. `max_energy_kwh_per_month` is leeg als er geen budget is. |
| Categorisch | `hardware` | Komt in train en test allebei voor. |
| Budget-indicator | `has_energy_budget` | `True` als het scenario een budget heeft. Ook gebruikersinvoer. Zo blijft het verschil tussen "geen budget" en een budget zichtbaar, ook als de lege cel later wordt ingevuld. |
| Geen feature | `scenario_id`, `family_id`, `label`, `is_dummy` | Volgnummer, splitgroep, uitkomst en nepdata-markering. Met `family_id` als feature zou het model families uit het hoofd leren. |
| Nooit toegestaan | `model`, `n_params_m`, `category`, `precision`, `batch_size`, `repeat`, `macro_f1`, `latency_ms_per_doc`, `energy_j_per_doc` | Gemeten uitkomsten en eigenschappen van één benchmarkconfiguratie. Vooral `category` zou een direct lek zijn: dat is bijna het label zelf. `hardware` en `input_length_tokens` staan ook in de benchmark, maar die vult de gebruiker zelf in. |
 
**Controle.** `controleer_alleen_gebruikersinvoer` stopt met een foutmelding als (1) een feature in de lijst "geen feature" of "nooit toegestaan" staat, (2) de scenario-CSV een gemeten kolom bevat, of (3) een feature ontbreekt. Om te zien of de controle echt iets vangt, heb ik expres fouten ingebouwd. Alle drie werden gevangen:
 
| Expres ingebouwde fout | Uitkomst |
|---|---|
| Kolom `macro_f1` toegevoegd aan de scenario's | Foutmelding: de scenario's bevatten gemeten kolommen |
| `category` als feature toegevoegd | Foutmelding: geen gebruikersinvoer |
| `family_id` als feature toegevoegd | Foutmelding: geen gebruikersinvoer |
 
Op de scenario's van seed 42 geeft `selecteer_features` precies de zeven kolommen uit de tabel en het label.
 
#### Groepsgewijze split
 
Status: geïmplementeerd en getest op de dummy benchmark.
 
**Waarom groepsgewijs.** Scenario's uit dezelfde familie (hardware × inputlengte) worden met dezelfde benchmarkconfiguraties gelabeld. Ze lijken dus sterk op elkaar. Bij een willekeurige split zitten er bijna dezelfde scenario's in train en test, en is de score te optimistisch. Daarom blijven hele families bij elkaar (scikit-learn developers, n.d.).
 
**Wat er gemaakt is.** Het script `src/candidate/data/split_scenarios.py` verdeelt de families en schrijft een splitmanifest (`split_manifest.json`). Het trainingsscript leest het manifest en rekent de split niet opnieuw uit. Zo gebruikt elk experiment precies dezelfde split. Het script gebruikt alleen pandas en de standaardbibliotheek, en traint niets.
 
```
python -m src.candidate.data.split_scenarios --scenarios outputs/candidate/scenarios.csv --benchmark outputs/candidate/dummy_benchmark.csv --out outputs/candidate/split_manifest.json
```
 
| Onderdeel | Keuze | Waarom |
|---|---|---|
| Testset | Beide families met 256 tokens: `rtx3080__256` en `ryzen5_3600x__256` | De inputlengte 256 komt dan in de training nergens voor. 256 ligt tussen 192 en 384, dus het model moet interpoleren, niet extrapoleren. Vastgelegd vóór er een model getraind is. |
| Cross-validatie | Vijf folds, één per overgebleven inputlengte (64, 128, 192, 384, 512), met steeds beide hardware-families | Dit is GroupKFold met de inputlengte als groep. Elke validatiefold vraagt dan hetzelfde als de testset: voorspellen bij een lengte die in de rest niet voorkomt. Bij GroupKFold op losse families kan `rtx3080__64` in validatie zitten terwijl `ryzen5_3600x__64` in de training zit. Dan is de lengte niet echt nieuw. |
| Controle testfamilies | Het script stopt als een testfamilie niet bestaat of niet tussen de trainlengtes ligt | Zo kan ik niet per ongeluk een randlengte (64 of 512) als test kiezen. |
| Splitmanifest | Testfamilies, families per fold, features, bestandsnaam en SHA-256 van de scenario- en benchmark-CSV, aantallen en labelverdeling per deel | Legt vast op welke versie van de data de split hoort. Geen tijdstempel en geen volledig pad, zodat dezelfde input altijd hetzelfde bestand geeft, uit welke map je het script ook draait. |
| Regeleinde CSV | De generators schrijven altijd `\n` | Anders schrijft Windows een ander regeleinde dan Mac/Linux, en verschilt de SHA-256 per computer terwijl de data hetzelfde is. |
 
**Datalekcontroles.** Voordat het manifest wordt geschreven, voert het script negen controles uit. Mislukt er één, dan stopt het zonder manifest.
 
| # | Controle |
|---|---|
| 1 | Features zijn alleen gebruikersinvoer (zie hierboven) |
| 2 | Geen familie zit in train en test tegelijk |
| 3 | Elk trainscenario zit in een fold |
| 4 | Geen testscenario zit in een fold (anders komt de test via de cross-validatie toch in de training) |
| 5 | Elke trainfamilie zit in precies één fold |
| 6 | De testinputlengte komt niet voor in train |
| 7 | De validatielengte van een fold komt niet voor in de rest van die fold |
| 8 | Geen `scenario_id` zit in train en test tegelijk |
| 9 | Elke klasse komt voor in de test en in elke validatiefold (geen lek, maar anders kan ik de macro-F1 niet goed uitrekenen) |
 
Ook hier heb ik expres fouten ingebouwd. Een randlengte (512) als test en een niet-bestaande familie gaven allebei een foutmelding. Alleen `rtx3080__256` als test liet controle 6 mislukken, omdat 256 dan via de CPU nog in de training zit. Een testfamilie die ook in een fold stond, liet controle 4 mislukken. Een familie in twee folds liet controle 5 mislukken.
 
**Controle.** Met de scenario's van seed 42 (2000 scenario's, dummy data) slagen alle negen controles. Twee keer de hele pipeline draaien (benchmark, scenario's, split) geeft een byte-voor-byte identiek manifest.
 
| Deel | Families | compact | medium | geen | Totaal |
|---|---|---|---|---|---|
| Test | `rtx3080__256`, `ryzen5_3600x__256` | 110 | 85 | 133 | 328 |
| Fold 0 (64 tokens) | `rtx3080__64`, `ryzen5_3600x__64` | 146 | 133 | 71 | 350 |
| Fold 1 (128 tokens) | `rtx3080__128`, `ryzen5_3600x__128` | 107 | 106 | 109 | 322 |
| Fold 2 (192 tokens) | `rtx3080__192`, `ryzen5_3600x__192` | 101 | 121 | 114 | 336 |
| Fold 3 (384 tokens) | `rtx3080__384`, `ryzen5_3600x__384` | 93 | 81 | 158 | 332 |
| Fold 4 (512 tokens) | `rtx3080__512`, `ryzen5_3600x__512` | 74 | 73 | 185 | 332 |
 
De testset is 16% van de scenario's (328 van 2000). Elke klasse komt in elk deel voor. De verdeling verschilt per fold, omdat langere input vaker *geen* geeft (zie de scenariogenerator).
 
*Deze split is herzien (zie *Groepsgewijze split herzien voor de echte benchmark*): de testlengtes worden nu willekeurig gekozen met GroupShuffleSplit en zijn niet bekend.*

**Nog open.** (1) Fold 0 (64 tokens) en fold 4 (512 tokens) liggen aan de rand. Daar moet het model extrapoleren, en de testset vraagt alleen interpoleren. De scores van die folds zullen waarschijnlijk lager zijn. Bij het kiezen van een model kijk ik daarom ook naar de score per fold, niet alleen naar het gemiddelde. (2) De grenzen van de scenariogenerator heb ik gekozen met de labelverdeling van alle families, dus ook de testfamilies. Op dummy data maakt dat niet uit. Met echte data kies ik de grenzen alleen op de trainfamilies. (3) Met maar één testlengte is de eindtest klein (twee families). Als de echte benchmark meer inputlengtes of hardware heeft, kies ik opnieuw welke families in de test gaan, en leg ik dat vast vóór het trainen. (4) Hoe de lege `max_energy_kwh_per_month` wordt ingevuld, kies ik bij het trainen. Dat hangt af van het model: niet elk model kan met lege waarden werken.
 
#### Tests
 
Status: alle tests slagen.
 
**Wat er gemaakt is.** Het script `tests/test_candidate_pipeline.py` test de hele pipeline van deze story in één keer. Het draait de vier scripts zoals ik ze zelf ook draai, maar schrijft alles naar een tijdelijke map. De bestanden in `outputs/candidate` worden dus niet aangeraakt. Het script gebruikt alleen numpy, pandas en de standaardbibliotheek.
 
```
python tests/test_candidate_pipeline.py
```
 
| Onderdeel | Wat de tests controleren |
|---|---|
| Dummy benchmark | Aantal rijen, configuraties en families. Geen fp16 op de CPU. Geldige waarden. Categoriegrenzen (99M, 100M, 400M, 401M). De verhoudingen uit de formules (CPU 8×, fp16 0,6×, energie = tijd × vermogen). Zelfde seed geeft dezelfde data. |
| Labelregel | De omrekening naar kWh. De negen voorbeelden. Een niet-gemeten context geeft een fout. Een waarde precies op de grens voldoet. Een leeg budget (None of NaN) betekent geen eis. De kleinste categorie wint. De volgorde van de benchmarkrijen maakt niet uit. |
| Labelregel, los nagerekend | Alle 2000 labels opnieuw berekend met een tweede, los geschreven versie van de regel (gewone Python-lussen in plaats van pandas). Een fout in `labelling_rule.py` zit dan niet ook in de controle. |
| Eigenschappen | Een ruimere eis (hogere latency, lagere F1, groter budget, lager volume) geeft nooit minder voldoende categorieën. Zonder budget verandert het volume het label niet, met budget wel. |
| Scenariogenerator | Kolommen, grenzen, het verborgen doel in J per document, 4 significante cijfers, ongeveer 50% met budget, en de labelverdeling. Ook dat de random trekkingen vast zijn: met kans 0 en kans 1 op een budget zijn alle andere kolommen gelijk. |
| Features | X bevat precies de zeven features. Ingebouwde lekken (`macro_f1`, `category`, `family_id`, `label`, `energy_j_per_doc`, een ontbrekende kolom) geven een fout. X is een kopie. |
| Split en manifest | Families gescheiden, testscenario's zonder fold, de aantallen in het manifest kloppen, alle negen controles slagen, en ingebouwde fouten worden gevangen. Bij een mislukte controle wordt er geen manifest geschreven. |
| Hele pipeline | Twee keer draaien geeft byte-voor-byte dezelfde bestanden. Het manifest is hetzelfde uit welke map je het script ook draait. De CSV's hebben altijd hetzelfde regeleinde. |
 
**Testen de tests wel iets?** Een test die altijd slaagt, bewijst niets. Daarom heb ik in een kopie van de scripts 23 keer expres één fout ingebouwd, en gekeken of de tests dat merken (mutatietesten). Voorbeelden: `>` in plaats van `>=` bij de F1-grens, Wh in plaats van kWh, de grootste categorie laten winnen, de herhalingen niet middelen, het budget alleen trekken als het nodig is, CRLF in de CSV, `family_id` als feature, en een manifest schrijven terwijl een controle mislukt.
 
In de eerste ronde vingen de tests 20 van de 22 fouten. Twee bleven onopgemerkt. (1) De kans op een budget op 0,8 zetten: de test vergeleek met de constante zelf, dus een verkeerde constante slaagde ook. Nu vergelijkt de test met de eis uit dit verslag (50%). (2) Een manifest schrijven terwijl een controle mislukt: mijn test liet een controle een foutmelding geven, waardoor het script al eerder stopte. Nu is er ook een test waarin een controle alleen "mislukt" teruggeeft (in de test ontbreekt een klasse). In de tweede ronde, met één extra fout erbij, vingen de tests alle 23.
 
**Gevonden en opgelost.** Het testen leverde ook echte verbeteringen op. (1) Het manifest bevatte het volledige pad naar de CSV's. Draaide ik het script vanuit een andere map, dan kreeg ik een ander manifest voor dezelfde data. Nu staat alleen de bestandsnaam erin. De SHA-256 legt de versie vast. (2) De scenariogenerator maakte zonder `--n` maar 1000 scenario's, terwijl dit verslag er 2000 gebruikt. De standaard is nu 2000. (3) Bij het naast elkaar leggen van dit verslag en de data vond ik twee afrondfouten in de tabellen (96,10 moest 96,09 zijn, en 38,5% stond er als 39%). Die zijn verbeterd. Alle getallen in deze stap zijn nagerekend.
 
#### Conclusie
 
 De pipeline werkt van begin tot eind op de dummy benchmark: benchmark, labelregel, scenario's, features en een groepsgewijze split met splitmanifest en datalekcontroles. Alles is met een seed reproduceerbaar en wordt getest met `tests/test_candidate_pipeline.py`. De getallen zelf zeggen niets over echte modellen. Als de echte benchmark klaar is, vervang ik de dummy-CSV en draai ik dezelfde scripts opnieuw. De grenzen van de generator en de keuze van de testfamilies moet ik dan wel opnieuw bekijken, vóór ik een model train. De volgende stap is de baseline-regel en de trainbare modellen in `train_candidate_selector.py`.
 
**Bronnen:**
 
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv. https://arxiv.org/abs/1810.04805
- Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv. https://arxiv.org/abs/1907.11692
- NumPy Developers. (n.d.). Random Generator. https://numpy.org/doc/stable/reference/random/generator.html
- scikit-learn developers. (n.d.). Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv. https://arxiv.org/abs/1910.01108
 
### Baseline en evaluatiemethode
 
Status: klaar op de dummy benchmark.
 
**Waarom dit nodig is.** De trainbare modellen hebben een referentiepunt nodig. Een model is pas nuttig als het beter is dan een simpele regel die iedereen kan narekenen.
 
**De regel.** De baseline staat in het notebook `notebooks/01_baseline.ipynb`, zodat ik hem stap voor stap kan bekijken. De drempels zijn per hardware en categorie de mediaan van de macro-F1 en de latency in de trainfamilies. De regel kiest de kleinste categorie waarvan de drempels de eisen halen: de gevraagde F1 is niet hoger dan de drempel en de maximale latency is niet lager dan de drempel. Past er geen categorie, dan is de uitkomst *geen*. De regel kijkt bewust niet naar het energiebudget, de inputlengte of het maandvolume.
 
| Hardware | Categorie | F1-drempel | Latencydrempel (ms) |
|---|---|---|---|
| rtx3080 | compact | 0,838 | 1,92 |
| rtx3080 | medium | 0,879 | 5,39 |
| ryzen5_3600x | compact | 0,841 | 17,46 |
| ryzen5_3600x | medium | 0,879 | 70,53 |
 
**Evaluatie.** Ik evalueer op de vijf cross-validatiefolds uit het splitmanifest. Per fold komen de drempels alleen uit de andere trainfamilies. De tabel hierboven gebruikt alle trainfamilies samen. De testfamilies blijven dicht tot de eindvergelijking met de modellen.
 
| Maat | Betekenis |
|---|---|
| Macro-F1 | De F1 per klasse, gemiddeld. Elke klasse telt even zwaar. |
| Confusion matrix | Rijen zijn het echte label, kolommen de voorspelling. |
| Constraint violation rate | Violations gedeeld door het aantal keer dat er een categorie gekozen is. Een violation is een gekozen categorie die de eisen volgens de benchmark niet haalt. Kiest de regel *medium* bij label *compact*, dan is dat dus alleen een violation als *medium* de eisen niet haalt. |
| Gemiste kansen | Aantal keer *geen* terwijl er wel een categorie voldoet, gedeeld door het aantal scenario's waarin een categorie voldoet. Fout, maar niet onveilig. |
 
**Resultaat (dummy data).** Alle folds samen, 1672 scenario's. De getallen zeggen niets over echte modellen en zijn alleen het referentiepunt voor de modellen op dezelfde data.
 
| Inputlengte | Macro-F1 | Violations | Violation rate | Gemiste kansen | Gemiste-kansrate |
|---|---|---|---|---|---|
| 64 | 0,716 | 5 van 203 | 2,5% | 81 van 279 | 29,0% |
| 128 | 0,801 | 16 van 180 | 8,9% | 48 van 213 | 22,5% |
| 192 | 0,834 | 22 van 212 | 10,4% | 31 van 222 | 14,0% |
| 384 | 0,777 | 61 van 221 | 27,6% | 14 van 174 | 8,0% |
| 512 | 0,727 | 68 van 198 | 34,3% | 17 van 147 | 11,6% |
| **Totaal** | **0,774** | **172 van 1014** | **17,0%** | **191 van 1035** | **18,5%** |
 
| Echt \ voorspeld | compact | medium | geen |
|---|---|---|---|
| compact | 472 | 19 | 30 |
| medium | 1 | 352 | 161 |
| geen | 89 | 81 | 467 |
  
**Nog open.** (1) De score op de testfamilies volgt bij de eindvergelijking met de modellen. (2) Met de echte benchmark draai ik het notebook opnieuw. Dan veranderen de drempels en de scores.
 
### Eerste trainbare modellen
 
Status: klaar op de dummy benchmark.
 
**Waarom dit nodig is.** De baseline uit de vorige stap is het referentiepunt. Nu wil ik weten of simpele trainbare modellen daar beter op scoren, op dezelfde data en met dezelfde maten.
 
**Wat er gemaakt is.** Er zijn twee notebooks: `notebooks/02_logistic_regression.ipynb` en `notebooks/03_decision_tree.ipynb`. Ze gebruiken dezelfde cross-validatiefolds uit het splitmanifest als de baseline. Elke fold is één keer de validatieset. Het model traint op de andere folds. De testfamilies blijven dicht.
 
De features komen alleen uit `features.py`. Een lege budgetcel wordt 0, en `has_energy_budget` laat zien dat er geen budget was. `hardware` is one-hot. De logistische regressie gebruikt een `StandardScaler` in een pipeline, met `max_iter=1000`. De beslisboom gebruikt `max_depth=5` en `random_state=42`.
 
**Resultaat (dummy data).** Alle folds samen, 1672 scenario's. De getallen komen uit de opgeslagen outputs van de notebooks. Ze zeggen niets over echte modellen.
 
| Model | Macro-F1 | Violations | Violation rate | Gemiste kansen | Gemiste-kansrate |
|---|---|---|---|---|---|
| Baseline | 0,774 | 172 van 1014 | 17,0% | 191 van 1035 | 18,5% |
| Logistische regressie | 0,763 | 188 van 1011 | 18,6% | 201 van 1035 | 19,4% |
| Beslisboom | 0,795 | 193 van 1076 | 17,9% | 144 van 1035 | 13,9% |

Logistische regressie:

| Echt \ voorspeld | compact | medium | geen |
|---|---|---|---|
| compact | 434 | 14 | 73 |
| medium | 11 | 375 | 128 |
| geen | 85 | 92 | 460 |

Beslisboom:

| Echt \ voorspeld | compact | medium | geen |
|---|---|---|---|
| compact | 468 | 10 | 43 |
| medium | 8 | 405 | 101 |
| geen | 90 | 95 | 452 |
 
Per inputlengte. 64 en 512 zijn de randfolds.
 
| Inputlengte | Macro-F1 baseline | Macro-F1 log. regressie | Macro-F1 beslisboom | Violation rate baseline | Violation rate log. regressie | Violation rate beslisboom |
|---|---|---|---|---|---|---|
| 64 (randfold) | 0,716 | 0,740 | 0,756 | 2,5% | 10,6% | 3,6% |
| 128 | 0,801 | 0,748 | 0,870 | 8,9% | 21,7% | 15,4% |
| 192 | 0,834 | 0,732 | 0,833 | 10,4% | 23,0% | 19,1% |
| 384 | 0,777 | 0,781 | 0,761 | 27,6% | 18,3% | 18,7% |
| 512 (randfold) | 0,727 | 0,716 | 0,700 | 34,3% | 21,8% | 35,8% |
 
**Interpretatie.** De beslisboom scoort het hoogst op macro-F1: 0,795 tegenover 0,774 voor de baseline. Geen model heeft duidelijk minder violations dan de baseline: de beslisboom zit op 17,9% en de logistische regressie op 18,6%, tegenover 17,0%. De beslisboom mist wel minder kansen (13,9% tegenover 18,5%). De logistische regressie is op alle drie de maten slechter dan de baseline.
 
**Nog open.** (1) Een lege budgetcel is nu 0. Dat betekent ook "strengst mogelijke budget". Afgesproken is een groot getal plus `has_energy_budget`. (2) De logistische regressie gebruikt nog geen log10 van `monthly_volume` en `max_energy_kwh_per_month`. (3) Budget gedeeld door volume (J per document) is een mogelijke feature. (4) Ik kies het model vooral op de middelste folds (128, 192 en 384).
 
----
### Groepsgewijze split herzien voor de echte benchmark

Status: split herzien en vastgelegd in de code.

**Waarom dit nodig is.** In de eerste versie had ik de testlengte (256) zelf gekozen. Maar je mag helemaal niet weten wat er in de testset zit. Dat past bij CRISP-ML(Q): de testdata moet onafhankelijk blijven van alle keuzes tijdens het modelleren, anders zegt de eindscore niet hoe het model het op nieuwe data doet (Studer et al., 2021). Bij mij was dat niet zo. Ik kende de testlengte, en keuzes zoals features en drempels zijn gemaakt terwijl ik wist welke lengte de test was. Daarnaast past de oude opzet niet bij de echte benchmark. Die meet 8 inputlengtes (32, 64, 96, 128, 192, 256, 384 en 512) en 4 batchgroottes (1, 2, 4 en 8), en geen CPU meer.

**Vergelijking.** Ik heb vier opties vergeleken voor de 8 lengtes van de echte benchmark. Hiervoor zijn alleen de lengtes nodig, geen resultaten. De groep blijft de inputlengte: alle batchgroottes van een lengte zitten samen in train óf test. De kortste en langste lengte (32 en 512) zijn nooit test, zodat het model op de test interpoleert. Een lengte is 4 contexten (lengte × batch) en 40 benchmarkrijen (320 rijen in `complete` / 8 lengtes).

| Optie | Testset | CV-folds | Middelste folds | Train per fold | Eindscore zonder lek? |
|---|---|---|---|---|---|
| A. Oud: 256 met de hand gekozen | 1 lengte, 40 rijen | 7 | 5 | 6 lengtes (240 rijen) | Nee. De testlengte is bekend en al eens bekeken. |
| B. GroupShuffleSplit, 1 testlengte | 1 lengte, 40 rijen | 7 | 5 | 6 lengtes (240 rijen) | Ja, maar de eindscore hangt af van één lengte. |
| C. GroupShuffleSplit, 2 testlengtes | 2 lengtes, 80 rijen | 6 | 4 | 5 lengtes (200 rijen) | Ja. |
| D. `train_test_split` per rij | Willekeurige rijen | n.v.t. | n.v.t. | n.v.t. | Nee. Dezelfde lengte en batch zitten in train en test. |

Bij optie D zit bijvoorbeeld een scenario met 128 tokens en batch 4 in de training, en een bijna gelijk scenario met 128 tokens en batch 4 in de test. Beide worden gelabeld met dezelfde benchmarkrijen. Het model hoeft dan niets nieuws te voorspellen, en de score is te optimistisch. Opties A en B hebben evenveel folds. Het verschil is alleen dat bij B niemand de testlengte heeft gekozen.

**Keuze: optie C.** GroupShuffleSplit met de inputlengte als groep, seed 42, 2 testlengtes, en de kortste en langste lengte uitgesloten. Redenen:

1. Niemand kiest de testlengtes en niemand kent ze. De seed maakt de split wel reproduceerbaar.
2. Met 2 testlengtes hangt de eindscore niet af van één lengte. De test heeft 8 contexten in plaats van 4.
3. Er blijven 6 folds over, waarvan 4 in het midden. Dat is genoeg om een model op te kiezen. Met 1 testlengte zouden het er 5 zijn. Dat ene extra fold weegt minder zwaar dan een twee keer zo grote eindtest.

**Vastgelegd.** In `src/candidate/data/split_scenarios.py` staan `SPLIT_SEED = 42` en `AANTAL_TESTLENGTES = 2`, met de regel dat de randen niet meedoen. De middelste folds en de randfolds worden in het trainingsscript en de notebooks uit de folds afgeleid, niet meer met de hand ingevuld. Om de testset onbekend te houden:

- het script print de testfamilies niet, en het manifest bevat geen labelverdeling van de test
- het splitmanifest wordt lokaal gegenereerd en niet gecommit. Het is met de code en de seed altijd opnieuw te maken
- ik open het manifest en de testdata niet
- de test wordt één keer gebruikt, aan het eind, op de echte benchmark.

**Op de dummy data.** De dummy benchmark heeft nu dezelfde 8 lengtes als de echte benchmark. Er blijven dus 6 folds over, waarvan 4 in het midden. Alle tests slagen met de nieuwe instellingen (pipeline 39/39, selector 9/9, predict 7/7). In de run van 20 november (dummy data, test niet geopend) halen 7 van de 11 kandidaten de grens van 5% op de middelste folds. De selectieregel kiest HistGradientBoosting (macro-F1 0,938, violation rate 4,8%). Die zit net onder de grens. In de run van 18 november, met 6 lengtes en 1 testlengte, haalde alleen de conservatieve opzoekregel de grens. De keuze hangt dus sterk af van de data en de split. Dit zijn dummy getallen. Ze zeggen niets over echte modellen.

**Nog open.** (1) De grenzen van de scenariogenerator bepalen op alleen de trainlengtes. (2) Batchgrootte als gebruikersinvoer toevoegen, en de hardware eruit halen. (3) De data opnieuw genereren en het notebook `04_modelverbeteringen.ipynb` opnieuw draaien op de nieuwe folds.

**Bronnen:**

- Studer, S., et al. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. *Machine Learning and Knowledge Extraction, 3*(2), 392 tot 413. https://doi.org/10.3390/make3020020
- scikit-learn developers. (n.d.). GroupShuffleSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

----
### App-koppeling van de kandidaat-categorie opschonen

Status: klaar op de dummy benchmark.

**Waarom dit nodig is.** De voorspelling werd op twee plekken in elkaar gezet: in `predict_candidate.py` (command line) en in `koppeling.py` (app). Dat is dubbel werk, en de twee kunnen uit elkaar gaan lopen. Daarnaast lekten twee meldingen informatie over de testset. Bij een testlengte zei de app "Dit is een testfamilie", en bij een niet-gemeten lengte noemde de waarschuwing alle lengtes uit de training. Met 8 gemeten lengtes kun je daaruit afleiden welke lengtes in de test zitten. Dat mag niet (zie de herziene split hierboven).

**Wat er veranderd is.**

| Onderdeel | Voor | Na |
|---|---|---|
| Voorspellen | Losse functies, in de app opnieuw samengevoegd | Eén `voorspel_scenario()` in `predict_candidate.py`. De command line en de app gebruiken hem allebei. |
| Laden | Twee keer geschreven, met de scikit-learn-controle alleen in de app | Eén `laad()` in `predict_candidate.py`, met de versie- en SHA-256-controle. `koppeling.py` vertaalt alleen nog de invoer en het resultaat. |
| Label uit de benchmark | Altijd getoond, met een extra melding bij testfamilies | Niet getoond zolang de test dicht is (`test_opened: false`), voor **geen enkel** scenario. |
| Waarschuwing bij de inputlengte | Noemde de lengtes uit de training | Kijkt alleen of de lengte gemeten is. Bij een niet-gemeten lengte: "is niet gemeten, de voorspelling is een schatting tussen de gemeten lengtes 192 en 256 tokens". |
| Documentatie | Geen | `src/candidate/README.md`: hoe je alles draait, wat de opties en uitvoer zijn, de tests, en hoe de testset dicht blijft. |

**Waarom het label voor elk scenario verborgen is.** Het eerste idee was om het label alleen bij de testfamilies te verbergen. Maar als het label alleen bij de testlengtes ontbreekt, zie je juist daaraan welke lengtes in de test zitten. Daarom is het label verborgen voor elk scenario, zolang de test dicht is. Na `--open-test` staat `test_opened` op `true` en is het label weer zichtbaar. De app toont in het blok "Vergelijk de modelcategorieën" wel de metingen voor elke lengte. Dat gebeurt voor alle lengtes op dezelfde manier, dus daaruit is niet af te leiden welke lengtes in de test zitten.

**Tests.** Twee nieuwe tests: een testlengte geeft dezelfde waarschuwingen als een trainlengte en geen label, en het label verschijnt pas als de test open is. Twee app-tests faalden al door een eerdere aanpassing van de app (nieuwe koppen en een andere opmaak). Die zijn bijgewerkt. Alle tests slagen: pipeline 39/39, selector 9/9, predict 7/7, app 18/18 en data quality 3/3.

----
### Categorieën compact, medium en groot met de standaardafwijking

Status: geïmplementeerd en getest op de dummy benchmark.

**Waarom dit nodig is.** Tot nu toe had de pipeline twee categorieën met vaste grenzen (*compact* < 100M, *medium* 100M tot 400M). De echte benchmark heeft meer en grotere modellen. De opdrachtgever wilde dat de grenzen met de standaardafwijking bepaald worden, zodat ze uit de gemeten modellen volgen en niet uit een gekozen getal. Daarmee komt er een derde categorie bij: *groot*.

**De regel** (`src/candidate/data/categorieen.py`). Neem per model log10(aantal parameters), elk model één keer. Dan:

| Categorie | Regel |
|---|---|
| compact | kleiner dan gemiddelde − 0,5σ |
| medium | tussen gemiddelde − 0,5σ en gemiddelde + 0,5σ (grenzen tellen mee) |
| groot | groter dan gemiddelde + 0,5σ |

| Keuze | Waarom |
|---|---|
| log10 | Modelgroottes lopen over ordes van grootte. Zonder log trekt één groot model het gemiddelde en σ ver omhoog, en wordt bijna alles compact. |
| 0,5σ | Bij een ongeveer normale verdeling geeft dat drie ongeveer even grote groepen (31%, 38%, 31%). Met 1σ zou medium ongeveer 68% zijn. |
| Elk model één keer | Anders telt een model met meer configuraties zwaarder mee. |
| σ van de populatie (ddof = 0) | We beschrijven de gemeten modellen zelf, het is geen steekproef. |

De grenzen hangen alleen af van de modellen, niet van de scenario's of de split. Ze lekken dus niets uit de testset. Bij de vier dummy-modellen liggen de grenzen op 98,5M en 178,4M: DistilBERT (66M) is compact, BERT-base (110M) en RoBERTa-base (125M) zijn medium, BERT-large (340M) is groot.

**Wat er in de pipeline veranderd is.**

| Onderdeel | Verandering |
|---|---|
| Labelregel | De volgorde is compact, medium, groot: de kleinste categorie die voldoet wordt het label. |
| Ja/nee-model | Eén ja/nee-model per categorie (drie in plaats van twee). `kies_categorie` kiest de kleinste categorie met een kans van minstens t. |
| Split | De datalekcontrole eist nu dat ook *groot* in de test en in elke fold voorkomt. |
| Voorspelling en app | De kans wordt per categorie getoond. De app toont *groot* nu als gemeten categorie. |
| Dummy data | BERT-large heeft een basis-F1 van 0,92 in plaats van 0,89, en de bovengrens van de minimale kwaliteit in de scenariogenerator is 0,925. Met 0,89 kreeg maar 5% van de scenario's het label groot. Nu is dat 15%, de eis voor elke klasse. |

**Labelverdeling (seed 42, 2000 scenario's, dummy data):** geen 38,6%, compact 25,4%, medium 20,7%, groot 15,3%.

**Resultaat (dummy data, middelste folds).** Met vier klassen halen 4 van de 11 kandidaten de grens van 5%. De selectieregel kiest de ja/nee-HGB met t = 0,7 (macro-F1 0,934, violation rate 4,9%). De conservatieve opzoekregel blijft het veiligst (1,0% violations), maar mist veel kansen (11,8%). Dit zijn dummy getallen. Ze zeggen niets over echte modellen.

**Uitleg in de app.** Een gebruiker moet kunnen zien wat "compact", "medium" en "groot" betekenen, en waarom een model in een categorie valt. Daarom staat in de app, in het uitklapblok "Hoe werkt deze app? Achtergrond en aannames", een stuk "Modelgroottes: compact, medium en groot". Daarin staat:

- de definitie in gewone woorden: de categorie hangt af van het aantal parameters, en de grenzen volgen uit de gemeten modellen met de standaardafwijking
- de drie stappen (log10, gemiddelde en σ, indelen met een halve σ) en waarom log10 en een halve σ
- dat "groot" betekent: groot ten opzichte van de gemeten modellen, niet groot in het algemeen
- een tabel met de grenzen en de categorie per model. De app rekent die zelf uit met dezelfde functie (`categoriegrenzen`) uit de benchmark die hij gebruikt. Met de dummy benchmark staat er: grenzen op 98,5 en 178,4 miljoen, DistilBERT compact, BERT-base en RoBERTa-base medium, BERT-large groot. Met de echte benchmark verschijnen daar vanzelf de nieuwe grenzen.

Zo kan de app nooit een andere indeling uitleggen dan de pipeline gebruikt.

**Tests.** De grenzen zijn met de hand nagerekend. Er is getest dat een waarde precies op de grens medium is, dat de eenheid (miljoenen of losse parameters) niets uitmaakt, dat te weinig of even grote modellen een fout geven, en dat de kleinste categorie wint. Alle tests slagen.

**Nog open.** (1) Op de echte benchmark worden de grenzen berekend uit `parameter_count` van de gemeten modellen. (2) De notebooks (`01_baseline.ipynb`, `04_modelverbeteringen.ipynb` en andere) gaan nog uit van twee categorieën. Die heb ik niet aangepast. (3) Of de kleinste categorie moet winnen, is nog een open vraag.

## Conclusie

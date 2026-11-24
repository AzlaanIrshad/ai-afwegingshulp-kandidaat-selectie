"""
categorieen.py

De modelcategorieën: compact, medium en groot.

Een model krijgt een categorie op basis van zijn aantal parameters. De grenzen
liggen niet vast, maar worden berekend uit de modellen in de benchmark, met de
standaardafwijking:

    x = log10(aantal parameters), één waarde per model
    compact:  x <  gemiddelde - 0,5 sigma
    medium:   gemiddelde - 0,5 sigma <= x <= gemiddelde + 0,5 sigma
    groot:    x >  gemiddelde + 0,5 sigma

Waarom zo:
- log10, omdat modelgroottes over ordes van grootte lopen (66 miljoen tot
  honderden miljoenen). Zonder log trekt één groot model het gemiddelde en de
  standaardafwijking ver omhoog, en wordt bijna alles "compact".
- Elk model telt één keer, hoe vaak het ook gemeten is. Anders telt een model
  met meer configuraties zwaarder mee.
- 0,5 sigma, zodat de drie groepen bij een ongeveer normale verdeling even groot
  zijn (ongeveer 31%, 38% en 31%). Met 1 sigma zou medium ongeveer 68% zijn.
- De standaardafwijking van de populatie (ddof = 0): we beschrijven de gemeten
  modellen zelf, het is geen steekproef.

De grenzen hangen alleen af van de modellen, niet van de scenario's of de split.
Ze lekken dus geen informatie uit de testset.
"""

import numpy as np


# Volgorde van klein naar groot. labelling_rule.py gebruikt dezelfde volgorde:
# de kleinste categorie die voldoet wordt het label.
CATEGORIEEN = ["compact", "medium", "groot"]

# Hoeveel standaardafwijkingen de grenzen van het gemiddelde af liggen.
AANTAL_SIGMA = 0.5


def categoriegrenzen(aantallen_parameters):
    """Geef de ondergrens en bovengrens van medium, in dezelfde eenheid als de invoer.

    aantallen_parameters: het aantal parameters per model, elk model één keer.
    """
    aantallen = np.asarray(list(aantallen_parameters), dtype=float)
    if len(aantallen) < 3:
        raise ValueError(f"Minstens 3 modellen nodig voor drie categorieën, er zijn er {len(aantallen)}.")
    if (aantallen <= 0).any():
        raise ValueError("Een aantal parameters moet groter zijn dan 0.")

    x = np.log10(aantallen)
    gemiddelde, sigma = x.mean(), x.std(ddof=0)
    if sigma == 0:
        raise ValueError("Alle modellen zijn even groot: er zijn geen drie categorieën te maken.")

    return 10 ** (gemiddelde - AANTAL_SIGMA * sigma), 10 ** (gemiddelde + AANTAL_SIGMA * sigma)


def bepaal_categorie(aantal_parameters, grenzen):
    """Geef de categorie van één model bij de grenzen uit categoriegrenzen()."""
    ondergrens, bovengrens = grenzen
    if aantal_parameters < ondergrens:
        return "compact"
    if aantal_parameters <= bovengrens:
        return "medium"
    return "groot"


def categoriseer(parameters_per_model):
    """Geef per model de categorie. parameters_per_model: {model: aantal parameters}."""
    grenzen = categoriegrenzen(parameters_per_model.values())
    return {model: bepaal_categorie(aantal, grenzen) for model, aantal in parameters_per_model.items()}

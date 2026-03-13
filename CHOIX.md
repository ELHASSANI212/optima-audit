# CHOIX.md

## Méthode

Similarité cosinus avec `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers).
Chaque REQ est encodée, chaque champ de la fiche aussi — sous forme de phrases complètes
pour que le modèle capte le contexte (ex. "schémas électriques = circuits de commande pour
une machine tout-électrique").

Seuils : sim >= 0.55 → SATISFAIT, sim < 0.30 → NON SATISFAIT, entre les deux → AMBIGU.

## Pourquoi une couche de contraintes en plus

Les embeddings ratent les cas logiques que la sémantique ne suffit pas à trancher :
- "EN COURS" ne veut pas dire signé (REQ-01/02)
- marquage sur le tableau de commande ≠ sur la machine (REQ-03)
- évaluation "phase d'utilisation uniquement" ≠ cycle de vie complet (REQ-08)
- notice en français ≠ toutes les langues requises (REQ-06)

Pour ces 5 cas, une fonction de contrainte force le bon statut après le calcul de similarité.



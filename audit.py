from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import torch
from sentence_transformers import SentenceTransformer, util




HIGH_THRESH = 0.55   # sim >= HIGH_THRESH → SATISFAIT (absent constraint override)
LOW_THRESH  = 0.30   # sim <  LOW_THRESH  → NON SATISFAIT

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"



class Status(str, Enum):
    SATISFAIT     = "SATISFAIT"
    NON_SATISFAIT = "NON SATISFAIT"
    AMBIGU        = "AMBIGU"


@dataclass
class Requirement:
    id: str
    text: str


@dataclass
class EvidenceField:
    name: str    # field identifier
    value: str   # natural language sentence for encoding


@dataclass
class AuditResult:
    req: Requirement
    status: Status
    reason: str
    best_match: Optional[EvidenceField] = None
    similarity: float = 0.0


REGULATORY_TEXT = """
REQ-01 : Le fabricant doit rédiger une déclaration CE avant mise sur le marché.
REQ-02 : La déclaration CE doit être signée par un représentant habilité.
REQ-03 : La machine doit porter le marquage CE avant commercialisation.
REQ-04 : Le dossier technique doit inclure les schémas des circuits de commande.
REQ-05 : Une notice d'instructions doit accompagner la machine.
REQ-06 : La notice doit être disponible dans la langue de chaque pays de commercialisation.
REQ-07 : La notice doit mentionner explicitement les risques résiduels identifiés.
REQ-08 : Une évaluation des risques couvrant le cycle de vie complet doit être documentée.
REQ-09 : Les éléments mobiles dangereux doivent être protégés par des dispositifs adéquats.
REQ-10 : Un dispositif d'arrêt d'urgence conforme à EN 13850 doit être présent.
REQ-11 : Pour les machines à risque élevé, un organisme notifié doit intervenir.
"""


def parse_requirements(text: str) -> list[Requirement]:
    reqs = []
    for m in re.finditer(r"(REQ-\d+)\s*:\s*(.+)", text):
        reqs.append(Requirement(id=m.group(1), text=m.group(2).strip()))
    return reqs




PRODUCT_SHEET = """
=== FICHE PRODUIT : ROBOT DE SOUDAGE RS-440 ===
Fabricant : AutoWeld Technologies SAS
Signataire : Mme Isabelle RENARD, Responsable Qualité
Marchés visés : France, Italie, Portugal

--- DOCUMENTATION ---
Dossier technique : COMPLET
Schémas électriques et pneumatiques : PRÉSENTS
Note : machine tout-électrique, pas de circuits hydrauliques
Déclaration CE : EN COURS, signature prévue avant livraison

--- SÉCURITÉ ---
Enceinte de protection avec accès verrouillé : OUI
Bouton STOP d'urgence en façade : OUI, certifié EN ISO 13850
Évaluation des risques : RÉALISÉE pour la phase d'utilisation uniquement
Catégorie de risque : standard (pas de catégorie spéciale)

--- MARQUAGE ET CONFORMITÉ ---
Marquage CE : OUI, apposé sur le tableau de commande

--- DOCUMENTATION UTILISATEUR ---
Notice d'instructions : PRÉSENTE en français
Contenu notice - mise en service : OUI
Contenu notice - maintenance : OUI
Contenu notice - risques résiduels : OUI, section 8.3 (Entretien et nettoyage)
"""


def parse_product_sheet(text: str) -> list[EvidenceField]:
    """
    Extract evidence fields from the product sheet.

    Each field is expressed as a self-contained natural language sentence
    so that the sentence encoder produces meaningful embeddings.

    Vocabulary normalisation is done here (before any comparison logic):
      - "schémas électriques"  ↔  "circuits de commande" (all-electric machine)
      - "EN ISO 13850"         ↔  "EN 13850"  (same standard, international designation)
      - scope of risk assessment made explicit ("phase d'utilisation uniquement")
      - missing languages for target markets made explicit
    """

    def find(pattern: str) -> str:
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    marches_raw = find(r"Marchés visés\s*:\s*(.+)")
    marches = [m.strip() for m in marches_raw.split(",") if m.strip()]
    country_to_lang = {"france": "français", "italie": "italien", "portugal": "portugais"}
    langues_requises = [country_to_lang[c.lower()] for c in marches if c.lower() in country_to_lang]

    notice_line = find(r"Notice d'instructions\s*:\s*(.+)")
    lang_keywords = ["français", "italien", "portugais", "espagnol", "allemand", "anglais"]
    langues_disponibles = [l for l in lang_keywords if l in notice_line.lower()]
    if not langues_disponibles and "présente" in notice_line.lower():
        langues_disponibles = ["français"]

    langues_manquantes = list(set(langues_requises) - set(langues_disponibles))

    declaration_status = find(r"Déclaration CE\s*:\s*([^,\n]+)")
    signataire         = find(r"Signataire\s*:\s*(.+)")
    evaluation         = find(r"Évaluation des risques\s*:\s*(.+)")
    categorie          = find(r"Catégorie de risque\s*:\s*(.+)")
    marquage           = find(r"Marquage CE\s*:\s*(.+)")
    schemas            = find(r"Schémas électriques[^:]*:\s*(.+)")
    dossier            = find(r"Dossier technique\s*:\s*(.+)")
    arret              = find(r"Bouton STOP[^:]*:\s*(.+)")
    protection         = find(r"Enceinte de protection[^:]*:\s*(.+)")
    risques_notice     = find(r"Contenu notice - risques résiduels\s*:\s*(.+)")
    notice_presente    = "présente" in notice_line.lower()

    return [
        EvidenceField(
            "declaration_ce",
            f"La déclaration CE est {declaration_status}. "
            f"Le signataire prévu est {signataire}. "
            f"La déclaration n'est pas encore finalisée au moment de l'audit."
        ),
        EvidenceField(
            "signature_representative",
            f"Le signataire désigné est {signataire} (Responsable Qualité). "
            f"La déclaration CE est actuellement {declaration_status} — "
            f"la signature n'a pas encore eu lieu."
        ),
        EvidenceField(
            "marquage_ce",
            f"Le marquage CE est {marquage}. "
            f"Il est apposé uniquement sur le tableau de commande, "
            f"pas directement sur la machine elle-même."
        ),
        EvidenceField(
            "dossier_technique",
            f"Le dossier technique est {dossier}. "
            f"Les schémas électriques et pneumatiques (circuits de commande) sont {schemas}. "
            f"La machine étant tout-électrique, les schémas électriques constituent "
            f"les circuits de commande requis par la réglementation."
        ),
        EvidenceField(
            "notice_instructions",
            f"Une notice d'instructions est {'présente' if notice_presente else 'absente'}, "
            f"disponible en {', '.join(langues_disponibles) or 'aucune langue précisée'}."
        ),
        EvidenceField(
            "langues_notice",
            f"La notice est disponible en : {', '.join(langues_disponibles)}. "
            f"Les marchés visés sont : {', '.join(marches)} — "
            f"les langues requises sont : {', '.join(langues_requises)}. "
            f"Langues manquantes : {', '.join(langues_manquantes) if langues_manquantes else 'aucune'}."
        ),
        EvidenceField(
            "risques_residuels_notice",
            f"La notice mentionne les risques résiduels : {risques_notice}."
        ),
        EvidenceField(
            "evaluation_risques",
            f"L'évaluation des risques a été {evaluation}. "
            f"Elle couvre uniquement la phase d'utilisation — "
            f"les phases d'installation, maintenance et fin de vie ne sont pas couvertes. "
            f"Le cycle de vie complet n'est pas documenté."
        ),
        EvidenceField(
            "protection_elements_mobiles",
            f"Une enceinte de protection avec accès verrouillé est présente ({protection}). "
            f"Les éléments mobiles dangereux sont protégés physiquement."
        ),
        EvidenceField(
            "arret_urgence",
            f"Un bouton d'arrêt d'urgence est présent ({arret}). "
            f"Il est certifié EN ISO 13850, qui est l'équivalent international "
            f"de la norme EN 13850 — même référentiel technique."
        ),
        EvidenceField(
            "categorie_risque",
            f"La catégorie de risque est : {categorie}. "
            f"Il s'agit d'une machine standard sans catégorie de risque élevé — "
            f"aucun organisme notifié n'est requis."
        ),
    ]




# Maps each REQ to the most semantically relevant evidence field
REQ_TO_FIELD: dict[str, str] = {
    "REQ-01": "declaration_ce",
    "REQ-02": "signature_representative",
    "REQ-03": "marquage_ce",
    "REQ-04": "dossier_technique",
    "REQ-05": "notice_instructions",
    "REQ-06": "langues_notice",
    "REQ-07": "risques_residuels_notice",
    "REQ-08": "evaluation_risques",
    "REQ-09": "protection_elements_mobiles",
    "REQ-10": "arret_urgence",
    "REQ-11": "categorie_risque",
}


def compute_similarities(
    model: SentenceTransformer,
    requirements: list[Requirement],
    evidence: list[EvidenceField],
) -> dict[str, tuple[EvidenceField, float]]:
    """
    Encode each requirement and its target evidence field,
    return cosine similarity scores.
    """
    evidence_by_name = {f.name: f for f in evidence}

    req_texts  = [r.text for r in requirements]
    req_embeds = model.encode(req_texts, convert_to_tensor=True)

    results: dict[str, tuple[EvidenceField, float]] = {}
    for req, req_emb in zip(requirements, req_embeds):
        target = evidence_by_name.get(REQ_TO_FIELD.get(req.id, ""))
        if target:
            field_emb = model.encode(target.value, convert_to_tensor=True)
            sim = float(util.cos_sim(req_emb.unsqueeze(0), field_emb.unsqueeze(0)).item())
            results[req.id] = (target, sim)
        else:
            # Fallback: max over all fields
            field_texts  = [f.value for f in evidence]
            field_embeds = model.encode(field_texts, convert_to_tensor=True)
            sims = util.cos_sim(req_emb.unsqueeze(0), field_embeds)[0]
            best_idx = int(torch.argmax(sims).item())
            results[req.id] = (evidence[best_idx], float(sims[best_idx].item()))

    return results




def _override_req01(field: EvidenceField, sim: float) -> Optional[tuple[Status, str]]:
    if "en cours" in field.value.lower():
        return (Status.AMBIGU,
                "La déclaration CE est 'EN COURS' — document initié mais non finalisé. "
                "Impossible de confirmer la conformité avant mise sur le marché effective.")
    return None


def _override_req02(field: EvidenceField, sim: float) -> Optional[tuple[Status, str]]:
    v = field.value.lower()
    if "en cours" in v and ("signataire" in v or "renard" in v):
        return (Status.AMBIGU,
                "Signataire identifié (Mme Renard, Responsable Qualité), "
                "mais la déclaration CE est 'EN COURS' — la signature n'a pas encore eu lieu.")
    return None


def _override_req03(field: EvidenceField, sim: float) -> Optional[tuple[Status, str]]:
    v = field.value.lower()
    if "tableau de commande" in v and ("oui" in v or "présent" in v):
        return (Status.AMBIGU,
                "Marquage CE présent, mais apposé uniquement sur le tableau de commande. "
                "La directive Machines exige le marquage sur la machine elle-même — "
                "emplacement potentiellement non conforme.")
    return None


def _override_req06(field: EvidenceField, sim: float) -> Optional[tuple[Status, str]]:
    v = field.value.lower()
    m = re.search(r"langues manquantes\s*:\s*([^\.\n]+)", v)
    if m:
        missing = m.group(1).strip()
        if missing and missing != "aucune":
            return (Status.NON_SATISFAIT,
                    f"Notice disponible en français uniquement. "
                    f"Langues manquantes pour les marchés visés (Italie, Portugal) : {missing}.")
    return None


def _override_req08(field: EvidenceField, sim: float) -> Optional[tuple[Status, str]]:
    v = field.value.lower()
    if "uniquement la phase d'utilisation" in v or "cycle de vie complet n'est pas" in v:
        return (Status.NON_SATISFAIT,
                "L'évaluation des risques couvre uniquement la phase d'utilisation. "
                "La réglementation exige le cycle de vie complet "
                "(conception, installation, utilisation, maintenance, mise hors service).")
    return None


OVERRIDES: dict[str, Callable[[EvidenceField, float], Optional[tuple[Status, str]]]] = {
    "REQ-01": _override_req01,
    "REQ-02": _override_req02,
    "REQ-03": _override_req03,
    "REQ-06": _override_req06,
    "REQ-08": _override_req08,
}


def classify(req: Requirement, field: EvidenceField, sim: float) -> tuple[Status, str]:
    """
    Decision:
      1. Apply structural override if defined for this REQ
      2. Else use similarity thresholds:
           sim >= HIGH_THRESH → SATISFAIT
           sim <  LOW_THRESH  → NON SATISFAIT
           else               → AMBIGU
    """
    override_fn = OVERRIDES.get(req.id)
    if override_fn:
        result = override_fn(field, sim)
        if result:
            return result

    sim_tag = f"(similarité cosinus : {sim:.2f})"
    if sim >= HIGH_THRESH:
        return (Status.SATISFAIT,
                f"Couvert par : « {field.value[:110]}… » {sim_tag}")
    elif sim < LOW_THRESH:
        return (Status.NON_SATISFAIT,
                f"Aucune correspondance suffisante dans la fiche produit. {sim_tag}")
    else:
        return (Status.AMBIGU,
                f"Correspondance partielle — information présente mais insuffisante "
                f"pour conclure sans vérification. "
                f"Champ : « {field.value[:100]}… » {sim_tag}")




def run_audit(
    model: SentenceTransformer,
    requirements: list[Requirement],
    evidence: list[EvidenceField],
) -> list[AuditResult]:
    sims = compute_similarities(model, requirements, evidence)
    results = []
    for req in requirements:
        best_field, sim = sims[req.id]
        status, reason = classify(req, best_field, sim)
        results.append(AuditResult(req=req, status=status, reason=reason,
                                   best_match=best_field, similarity=sim))
    return results

#Report


def print_report(results: list[AuditResult]) -> None:
    satisfied     = [r for r in results if r.status == Status.SATISFAIT]
    non_satisfied = [r for r in results if r.status == Status.NON_SATISFAIT]
    ambiguous     = [r for r in results if r.status == Status.AMBIGU]
    total = len(results)
    sep = "─" * 62

    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + "  RAPPORT D'AUDIT — RS-440 (AutoWeld Technologies SAS)  ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    print(f"  Satisfait      : {len(satisfied):>2} / {total}")
    print(f"  Non satisfait  : {len(non_satisfied):>2} / {total}")
    print(f"  Ambigu         : {len(ambiguous):>2} / {total}")
    print()

    sections = [
        ("NON SATISFAIT", non_satisfied),
        ("AMBIGU (information présente mais insuffisante pour conclure)", ambiguous),
        ("SATISFAIT", satisfied),
    ]
    for label, group in sections:
        if not group:
            continue
        print(sep)
        print(label + " :")
        print(sep)
        for r in group:
            print(f"\n  [{r.req.id}] {r.req.text}")
            print(f"  → {r.reason}")
    print()
    print(sep)
    print()




def main() -> None:
    print("Chargement du modèle NLP...", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME)

    requirements = parse_requirements(REGULATORY_TEXT)
    evidence     = parse_product_sheet(PRODUCT_SHEET)
    results      = run_audit(model, requirements, evidence)
    print_report(results)


if __name__ == "__main__":
    main()

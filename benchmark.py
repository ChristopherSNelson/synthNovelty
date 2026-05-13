"""
Curated face-validity benchmark for the novelty scorer.

Tests whether the model assigns higher novelty scores to reaction types that
are genuinely unusual vs. those that dominate the USPTO training corpus.

This is a FACE VALIDITY check, not ground-truth validation. The reactions are
chosen to represent reaction-type diversity, not any specific literature claim.
Verify SMILES independently before citing in publication.

Run:
    python benchmark.py
"""

import numpy as np
import sys
import os

# ---- Curated reactions -------------------------------------------------------
# Each entry: (name, SMILES, category)
#   category: "routine" | "uncommon" | "novel"
#
# Routine   = high-frequency Schneider classes; should score LOW
# Uncommon  = present but lower frequency; intermediate scores expected
# Novel     = reaction types rare or absent in USPTO training; should score HIGH
#
# CRITICAL CAVEAT: If the model was trained on USPTO-50K which includes many
# of these reaction types, the discrimination may be weak. High novelty score
# for "routine" reactions would indicate the model is miscalibrated or the
# conditioning is too coarse (see CLAUDE.md - conditioning limitation at inference).

BENCHMARK = [
    # --- ROUTINE ---
    # Amide bond formation is the single most common reaction in medicinal chemistry.
    ("Amide bond formation (acyl chloride)", "CC(=O)Cl.NCC>>CC(=O)NCC", "routine"),

    # Suzuki coupling: the dominant C-C bond-forming reaction in modern drug synthesis.
    ("Suzuki-Miyaura coupling", "OB(O)c1ccccc1.Brc1ccccc1>>c1ccc(-c2ccccc2)cc1", "routine"),

    # SN2 N-alkylation: textbook nucleophilic substitution.
    ("N-Alkylation (SN2)", "CCBr.Nc1ccccc1>>CCNc1ccccc1", "routine"),

    # Esterification via acyl chloride.
    ("O-Acylation (ester formation)", "CC(=O)Cl.OCCO>>CC(=O)OCCO", "routine"),

    # SNAr: electron-poor arene + amine nucleophile.
    ("Nucleophilic aromatic substitution (SNAr)",
     "Fc1ccc([N+](=O)[O-])cc1.Nc1ccccc1>>c1ccc(Nc2ccc([N+](=O)[O-])cc2)cc1", "routine"),

    # --- UNCOMMON ---
    # Buchwald-Hartwig: Pd-catalyzed C-N coupling; well-known but lower frequency.
    ("Buchwald-Hartwig C-N coupling",
     "Brc1ccc(OC)cc1.Nc1ccccc1>>COc1ccc(Nc2ccccc2)cc1", "uncommon"),

    # Heck: Pd-catalyzed C-C via beta-hydride elimination.
    ("Heck olefination",
     "Brc1ccccc1.C=CC(=O)OCC>>CCOC(=O)/C=C/c1ccccc1", "uncommon"),

    # --- NOVEL ---
    # CuAAC "click" chemistry: [3+2] azide-alkyne cycloaddition -> 1,2,3-triazole.
    # Bergingly unusual bond disconnection pattern vs. standard C-C/C-N couplings.
    ("CuAAC click cycloaddition (triazole formation)",
     "CCCCN=[N+]=[N-].C#Cc1ccccc1>>CCCCn1cc(-c2ccccc2)nn1", "novel"),

    # Diels-Alder [4+2]: pericyclic reaction; rare in pharma synthesis routes.
    # Product: cyclohex-3-ene-1-carbaldehyde from butadiene + acrolein.
    ("Diels-Alder [4+2] cycloaddition",
     "C=CC=C.C=CC=O>>O=CC1CC=CCC1", "novel"),

    # Metal-carbene cyclopropanation: ethyl diazoacetate + styrene -> cyclopropane.
    # Unusual three-membered ring formation; strain-release disconnection.
    ("Carbene cyclopropanation",
     "CCOC(=O)C=[N+]=[N-].C=Cc1ccccc1>>CCOC(=O)C1CC1c1ccccc1", "novel"),
]

CATEGORY_ORDER = ["routine", "uncommon", "novel"]
CATEGORY_LABEL = {"routine": "ROUTINE (expected LOW)", "uncommon": "UNCOMMON (expected MED)",
                  "novel": "NOVEL (expected HIGH)"}


def run_benchmark():
    # Lazy import so import errors are clear
    try:
        from score_reaction import load_model, score_reactions
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)

    if not os.path.exists("model.pt") or not os.path.exists("metrics.pt"):
        print("model.pt or metrics.pt not found. Run the full pipeline first:")
        print("  python dataset_setup.py && python precompute_routes.py && python train.py")
        sys.exit(1)

    print("Loading model (this may take ~30s for RXNFP)...")
    model, mean, std, rxnfp_gen, mean_freq, device = load_model()
    print(f"Device: {device}\n")

    reactions = [r[1] for r in BENCHMARK]
    names = [r[0] for r in BENCHMARK]
    categories = [r[2] for r in BENCHMARK]

    print("Scoring benchmark reactions...")
    scores = score_reactions(reactions, model, mean, std, rxnfp_gen, mean_freq, device)

    # ---- Results by category ----
    print("\n" + "="*72)
    print("BENCHMARK RESULTS")
    print("="*72)

    cat_scores = {c: [] for c in CATEGORY_ORDER}
    for name, smiles, cat, score in zip(names, reactions, categories, scores):
        cat_scores[cat].append(score)

    for cat in CATEGORY_ORDER:
        print(f"\n{CATEGORY_LABEL[cat]}")
        print("-"*72)
        for name, smiles, c, score in zip(names, reactions, categories, scores):
            if c != cat:
                continue
            bar = "#" * int(score * 5)
            print(f"  {score:6.3f}  {bar:<30}  {name}")

    # ---- Summary statistics ----
    print("\n" + "="*72)
    print("CATEGORY SUMMARY")
    print("="*72)
    print(f"\n  {'Category':<12}  {'Mean':>7}  {'Std':>7}  {'n':>4}")
    for cat in CATEGORY_ORDER:
        s = np.array(cat_scores[cat])
        print(f"  {cat:<12}  {s.mean():>7.3f}  {s.std():>7.3f}  {len(s):>4}")

    # ---- Interpretation ----
    routine_mean = np.mean(cat_scores["routine"])
    novel_mean = np.mean(cat_scores["novel"])
    delta = novel_mean - routine_mean

    print(f"\n  Novel - Routine delta: {delta:+.3f}")
    if delta > 0.5:
        print("  Face validity: GOOD - model assigns higher scores to unusual reactions.")
    elif delta > 0:
        print("  Face validity: WEAK - correct direction but small separation.")
        print("  The conditioning limitation at inference (mean_freq for all reactions)")
        print("  may be suppressing sensitivity. See CLAUDE.md for details.")
    else:
        print("  Face validity: FAILED - routine reactions score as high or higher than novel.")
        print("  This is a significant concern. Check model training and conditioning.")

    # ---- Ranked list ----
    print("\n" + "="*72)
    print("ALL REACTIONS RANKED BY NOVELTY SCORE")
    print("="*72)
    ranked = sorted(zip(scores, names, categories), reverse=True)
    print(f"\n  {'Score':>7}  {'Category':<12}  Name")
    for score, name, cat in ranked:
        print(f"  {score:>7.3f}  {cat:<12}  {name}")


if __name__ == "__main__":
    run_benchmark()

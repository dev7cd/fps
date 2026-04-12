"""
@file run_campaign.py
@brief Automatisation de la campagne de génération RVE multi-seeds / multi-résolutions.

@details
Génère une matrice de configurations (N_seeds × N_résolutions) en appelant
main.py via subprocess. Chaque run est isolé dans son propre répertoire de
sortie, structuré comme suit :

    campaign/
    ├── cfg_42/
    │   ├── res_064/
    │   │   └── RVE_cfg42_r064.*
    │   ├── res_128/
    │   ├── res_256/
    │   ├── res_512/
    │   └── res_700/
    ├── cfg_43/
    │   └── ...
    └── ...

Paramètres globaux fixés (identiques pour toutes les configurations) :
  - Boîte cubique          : Lx = Ly = Lz = 0.612
  - Fraction volumique     : Vf = 0.18
  - Rayon fibre            : r  = 0.02
  - Section                : circulaire
  - Clearance inter-fibres : 0.01
  - RSDA                   : activé
  - Orientation            : uniaxiale suivant (1, 0, 0), force = 0.22
  - Export GMSH / Nastran  : désactivé

Variables de campagne :
  - Seeds  : 42 → 51 inclus (10 configurations géométriques distinctes)
  - Résolutions FFT : 64, 128, 256, 512, 700

Usage :
    python run_campaign.py               # lance tout
    python run_campaign.py --dry_run     # affiche les commandes sans les exécuter
    python run_campaign.py --seeds 42 43 --resolutions 64 128  # sous-campagne

@author  Devine Ngouloubi — LMNO, Université de Caen
"""

import argparse
import subprocess
import sys
import os
import time
import logging
from itertools import product
from pathlib import Path

# ── Configuration de la campagne ──────────────────────────────────────────────

# Paramètres géométriques et physiques — INVARIANTS
GLOBAL_PARAMS = {
    "--dims":       "0.612 0.612 0.612",
    "--vf":         "0.18",
    "--radius":     "0.02",
    "--section":    "circular",
    "--clearance":  "0.01",
    "--bias_type":  "uniaxial",
    "--bias_vec1":  "1.0 0.0 0.0",
    "--strength":   "0.22",
}

# Flags booléens (sans valeur associée)
GLOBAL_FLAGS = [
    "--rsda",       # RSDA activé
    "--no_mesh",    # Pas de maillage GMSH
]
# --nastran est omis (désactivé par défaut dans cli.py)

# Variables de campagne
DEFAULT_SEEDS       = list(range(42, 52))          # 42..51 inclus
DEFAULT_RESOLUTIONS = [64, 128, 256, 512, 700]

# Répertoire racine de la campagne
CAMPAIGN_ROOT = Path("campaign")

# Chemin vers main.py (relatif à l'emplacement de ce script)
MAIN_SCRIPT = Path(__file__).parent / "main.py"

# ── Utilitaires ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Campaign")


def build_command(seed: int, resolution: int, output_prefix: str) -> list[str]:
    """
    Construit la liste d'arguments pour un appel à main.py.

    @param seed          Graine aléatoire de la configuration.
    @param resolution    Résolution de la grille FFT (cube de côté `resolution`).
    @param output_prefix Préfixe complet (chemin inclus) des fichiers de sortie.
    @return Liste de chaînes constituant la commande complète.
    """
    cmd = [sys.executable, str(MAIN_SCRIPT)]

    # Paramètres invariants clé/valeur
    for flag, value in GLOBAL_PARAMS.items():
        cmd += [flag] + value.split()

    # Flags booléens
    cmd += GLOBAL_FLAGS

    # Paramètres variables
    cmd += ["--seed",    str(seed)]
    cmd += ["--res_fft", str(resolution)]
    cmd += ["--output",  output_prefix]

    return cmd


def run_single(seed: int, resolution: int, dry_run: bool = False) -> bool:
    """
    Exécute (ou simule) la génération d'une configuration unique.

    @param seed       Graine de la configuration.
    @param resolution Résolution FFT.
    @param dry_run    Si True, affiche la commande sans l'exécuter.
    @return True si succès (ou dry_run), False en cas d'erreur subprocess.
    """
    # Construction de l'arborescence de sortie
    run_dir = CAMPAIGN_ROOT / f"cfg_{seed:02d}" / f"res_{resolution:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(run_dir / f"RVE_cfg{seed:02d}_r{resolution:04d}")
    cmd    = build_command(seed, resolution, prefix)

    label = f"[seed={seed:02d} | res={resolution:4d}^3]"

    if dry_run:
        logger.info(f"DRY-RUN {label}  →  {' '.join(cmd)}")
        return True

    logger.info(f"START   {label}  →  {run_dir}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,   # laisser stdout/stderr s'afficher en direct
            text=True,
        )
        elapsed = time.time() - t0
        logger.info(f"OK      {label}  ({elapsed:.1f}s)")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        logger.error(f"FAIL    {label}  (code={e.returncode}, {elapsed:.1f}s)")
        return False

    except FileNotFoundError:
        logger.critical(f"main.py introuvable : {MAIN_SCRIPT}")
        sys.exit(1)


# ── Point d'entrée ────────────────────────────────────────────────────────────

def parse_campaign_args():
    p = argparse.ArgumentParser(
        description="Campagne de génération RVE multi-seeds / multi-résolutions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Liste des graines à utiliser (ex: 42 43 44)",
    )
    p.add_argument(
        "--resolutions", type=int, nargs="+", default=DEFAULT_RESOLUTIONS,
        help="Liste des résolutions FFT à générer (ex: 64 128 256)",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Affiche les commandes sans les exécuter",
    )
    p.add_argument(
        "--sequential", action="store_true", default=True,
        help="Exécution séquentielle (défaut). Réservé pour une future extension parallèle.",
    )
    return p.parse_args()


def main():
    args = parse_campaign_args()

    seeds       = sorted(set(args.seeds))
    resolutions = sorted(set(args.resolutions))
    n_total     = len(seeds) * len(resolutions)

    logger.info("=" * 60)
    logger.info("  CAMPAGNE RVE — LMNO / Université de Caen")
    logger.info(f"  Seeds       : {seeds}")
    logger.info(f"  Résolutions : {resolutions}")
    logger.info(f"  Total runs  : {n_total}")
    logger.info(f"  Racine      : {CAMPAIGN_ROOT.resolve()}")
    if args.dry_run:
        logger.info("  MODE        : DRY-RUN (aucun fichier produit)")
    logger.info("=" * 60)

    t_campaign_start = time.time()
    results = {"ok": 0, "fail": 0, "fail_list": []}

    # Boucle principale : on itère d'abord sur les seeds, puis sur les résolutions.
    # Cela garantit que les 5 résolutions d'une même configuration géométrique
    # sont générées consécutivement — utile si l'on veut inspecter une config
    # avant de passer à la suivante.
    for seed, res in product(seeds, resolutions):
        success = run_single(seed, res, dry_run=args.dry_run)
        if success:
            results["ok"] += 1
        else:
            results["fail"] += 1
            results["fail_list"].append((seed, res))

    elapsed_total = time.time() - t_campaign_start

    logger.info("=" * 60)
    logger.info(f"  BILAN : {results['ok']} succès / {results['fail']} échec(s)")
    logger.info(f"  Durée totale : {elapsed_total:.1f}s  "
                f"({elapsed_total / 60:.1f} min)")
    if results["fail_list"]:
        logger.warning("  Configurations échouées :")
        for s, r in results["fail_list"]:
            logger.warning(f"    seed={s:02d}, res={r}")
    logger.info("=" * 60)

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()

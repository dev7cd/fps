<p align="center">
  <img src="fps_logo.png" alt="Logo FPS" width="400">
</p>

# Fiber Packing System (FPS)

*Langue : **Français** · [English](README.md)*

[![Licence : AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![tests](https://github.com/dev7cd/fps/actions/workflows/tests.yml/badge.svg)](https://github.com/dev7cd/fps/actions/workflows/tests.yml)

**FPS** est un paquet Python libre qui génère des **microstructures
triphasées périodiques** (matrice, fibres, pores sphériques) pour
l'homogénéisation numérique, avec la particularité de traiter des **fibres
longues et courbes**. Chaque VER est audité (aucun export si fibres et pores ne
sont pas disjoints, images périodiques comprises), reproductible à partir d'une
graine et décrit par un enregistrement paramétrique compact à partir duquel les
exports CAO, éléments finis et voxel peuvent être régénérés.

Ce README est une présentation en français. La référence complète (tous les
paramètres, l'export Abaqus, le contrôle des exports, les algorithmes) est le
[README anglais](README.md).

## Points clés

- **Fibres courbes lisses** : lignes moyennes en splines de Catmull–Rom portant
  un **repère de Bishop** (sans torsion) ; sections circulaires (un profil
  super-elliptique existe dans `geometry/sections.py`, mais il n'est encore ni
  audité ni voxelisé : `main.py` le refuse).
- **Empilement dense admissible** : une **marche auto-évitante constructive
  (CSAW)** suivie d'une densification par **réarrangement collectif** dépasse la
  limite de blocage du placement séquentiel, sans modifier le rayon des fibres
  (temps et fractions volumiques mesurés : [README anglais, *Performance*](README.md#performance)).
- **Sorties certifiées** : un **audit géométrique bloquant**, avec son propre
  noyau de distance, refuse tout export si deux fibres (ou une fibre et ses
  images périodiques) s'intersectent, ou si un pore chevauche une fibre ou un
  autre pore.
- **Microstructures triphasées** : pores sphériques optionnels qui ne
  chevauchent ni les fibres (images périodiques comprises) ni les autres pores.
- **Reproductible par construction** : une graine (tirée et enregistrée si elle
  n'est pas donnée) reproduit un VER bit à bit, et `regenerate.py` refait tous
  les exports à partir de l'enregistrement paramétrique, sans relancer le placement.
- **Exports** : JSON paramétrique, CAO (STEP), maillage éléments finis conforme
  (Gmsh, périodique en option), champ voxel pour FFT, Nastran `.bdf`, maillage
  voxel Abaqus `.inp` avec équations de périodicité.

## Chaîne de traitement

```
Étape 1 : placement constructif (CSAW)            -> generation/generator.py
Étape 2 : densification (réarrangement collectif) -> generation/optimizer_dynamic.py
Étape 3 : audit géométrique bloquant des fibres   -> validation/geometric_audit.py
Étape 4 : porosité (option) + audit des pores     -> generation/porosity_gen.py, validation/geometric_audit.py
Étape 5 : statistiques et exports                 -> main.py
```

## Installation

Python ≥ 3.9 (suite de tests exécutée sous Python 3.9, 3.11, 3.12 et 3.13 sous
Linux, et 3.12 sous macOS et Windows, avec NumPy 1.x ou 2.x).

```bash
git clone https://github.com/dev7cd/fps.git
cd fps
pip install -r requirements.txt        # ou : pip install -e ".[gui,dev]"
```

L'installation en paquet fournit trois commandes : `fps` (= `python main.py`),
`fps-regenerate` (= `python regenerate.py`) et `fps-gui` (= `python gui.py`).
Numba compile ses noyaux au premier lancement et les met en cache sur disque.
Tests : `pytest`.

## Utilisation

```bash
# VER minimal
python main.py --dims 1 1 1 --vf 0.05 --radius 0.02 --seed 42

# VER orienté, poreux, avec tous les exports
python main.py --dims 1 1 1 --vf 0.30 --radius 0.02 --seed 7 \
    --bias_type uniaxial --bias_vec1 1 0 0 --strength 0.3 --curvature 60 \
    --porosity --void_vf 0.02 --void_mean 0.03 --void_std 0.006 \
    --res_fft 256 --abaqus --res_abaqus 64 --periodic_mesh --nastran --output RVE_UD_30
```

Avec la trajectoire par défaut (29 à 30 points de contrôle, pas de 0,053), les
fibres mesurent environ 1,53, soit *L/D* ≈ 38 pour `--radius 0.02` ; le rapport
d'aspect se règle avec `--min_pts`, `--max_pts`, `--step` et `--radius`.
`python main.py --help` liste tous les paramètres et leurs valeurs par défaut.

## Fichiers de sortie

| Fichier | Contenu |
|---------|---------|
| `{prefix}_validity.json` | statut des audits géométriques (`passed`, `failed`, `undetermined`), écrit à chaque tentative |
| `{prefix}_parametric.json` | arguments CLI complets, versions logicielles, durées des phases, statistiques, points de contrôle et rayons des fibres, pores, bloc `validity` |
| `{prefix}_roots.csv`, `{prefix}_full_periodic.csv` | points de contrôle des fibres racines / avec images périodiques |
| `{prefix}_voxelmap.npy`, `{prefix}_slice****.pgm` | grille voxel uint8 (matrice 0, fibre 128, pore 255) et coupes PGM |
| `{prefix}.step`, `{prefix}.msh` | CAO et maillage Gmsh (sauf `--no_mesh`) |
| `{prefix}.bdf`, `{prefix}.inp` | Nastran (`--nastran`), Abaqus voxel (`--abaqus`) |
| `{prefix}_export_check.json` | contrôle des exports CAO et maillage |

## Reproductibilité

- Tous les tirages aléatoires (placement, densification, pores) viennent de
  générateurs initialisés par `--seed` ; sans `--seed`, une graine est tirée,
  affichée et enregistrée. Même commande, même graine et même version de FPS :
  même VER, bit à bit, sur une même plateforme (`tests/test_reproducibility.py`).
  Les arrondis flottants varient légèrement d'un processeur à l'autre ; pour
  partager un VER entre machines, on partage donc son enregistrement.
- `regenerate.py` reconstruit la configuration, les fibres et les pores depuis
  le JSON paramétrique, vérifie que l'enregistrement a le SHA-256 de la
  géométrie auditée et que les fibres reconstruites ont exactement ses points de
  contrôle et ses rayons, relance les deux audits bloquants et réécrit tous les
  exports. Les volumes des fibres, recalculés à partir des splines, peuvent
  différer dans leurs derniers bits sur un autre processeur : ils sont comparés
  avec une tolérance relative de 1e-9, si bien qu'un enregistrement écrit sous
  Linux se régénère sous macOS ou Windows (`tests/test_examples.py`). Seules les
  options d'export peuvent changer :

  ```bash
  python regenerate.py RVE_UD_30_parametric.json --res_fft 512 --no_mesh --output RVE_UD_30_512
  ```

## Performance

Mesuré avec FPS v7.0.0 sur un portable (Intel Core i5-1335U), un fil d'exécution par
VER, cellule unité, `--radius 0.02`, trajectoire par défaut (*L/D* ≈ 38) : 26 fibres
à V_f = 5 % en ≈ 5 s, 104 fibres à 20 % en ≈ 3 min, 155 fibres à 30 % en ≈ 19 min.
Le placement séquentiel seul cale vers V_f ≈ 23 %. Scripts et résultats bruts :
`benchmarks/` (détail dans le [README anglais](README.md#performance)).

## Interface graphique

`python gui.py` (ou `fps-gui`) : panneau de paramètres, vue 3D, console en temps
réel, barre de progression ; la génération tourne dans un sous-processus.

## Documentation

- Référence de l'API (Doxygen) : `docs/html/index.html` (régénérer avec `doxygen Doxyfile`)
- Benchmarks et scripts des figures de l'article : `benchmarks/`

## Auteur

**Devine Ngouloubi**, développeur (concept, conception et implémentation).
Laboratoire de Mathématiques Nicolas Oresme (LMNO, UMR 6139), Université de
Normandie, UNICAEN, CNRS, Caen, France.

## Citation

Merci de citer le logiciel avec sa version (voir [CITATION.cff](CITATION.cff)).
L'algorithme de placement constructif (CSAW) de l'étape 1 a été décrit pour la
première fois dans :

```bibtex
@article{Ngouloubi2026CSAW,
  title   = {A geometric algorithm for dense 3D RVEs of long curved fiber composites},
  author  = {Ngouloubi, Devine and Cho{\"\i}, Daniel and Karamian-Surville, Philippe},
  journal = {Composite Structures},
  volume  = {391},
  pages   = {120524},
  year    = {2026},
  doi     = {10.1016/j.compstruct.2026.120524}
}
```

## Licence

FPS est distribué sous **GNU Affero General Public License v3.0 ou ultérieure
(AGPL-3.0-or-later)**, voir [LICENSE](LICENSE) (texte identique dans
[Licence.txt](Licence.txt)). Vous êtes libre d'utiliser, d'étudier, de modifier
et de partager le logiciel ; si vous distribuez une version modifiée **ou
l'exploitez comme service réseau**, vous devez en publier le code source sous la
même licence. Les dépendances gardent leurs propres licences, voir [NOTICE](NOTICE).

import gmsh
import numpy as np
import logging
from generation.periodicity import PeriodicManager

## @var logger
#  @brief Logger for the gmsh_exporter module.
logger = logging.getLogger(__name__)

class GmshExporter:
    """!
    @class GmshExporter
    @brief Classe responsable de la génération de fichiers .step et .msh via GMSH.
    """

    def __init__(self, config):
        """!
        @brief Initialise l'exportateur avec la configuration du RVE.
        @param config FiberPackingConfig Objet de configuration contenant box_dims et les paramètres de maillage.
        """
        self.config = config ##< Configuration de la simulation

    def generate_mesh(self, fibers, voids, output_path) -> None:
        """!
        @brief Génère la géométrie CAO (.step) et le maillage (.msh) de manière robuste.
        
        Réalise les étapes suivantes :
        1. Création du cube matrice.
        2. Création des fibres (balayage/pipe) et pores (sphères).
        3. Fragmentation booléenne pour assurer la conformité des interfaces.
        4. Tri des fragments (clipping) et définition des groupes physiques.
        5. Maillage volumique avec champs de taille adaptatifs.

        @param fibers List[Fiber] Liste des fibres à mailler.
        @param voids List[Void] Liste des pores à mailler.
        @param output_path str Chemin (sans extension) pour les fichiers de sortie.
        @return None
        """
        # Initialisation propre
        try:
            gmsh.finalize()
        except:
            pass
            
        gmsh.initialize()
        gmsh.model.add("RVE_Composite_Final")
        occ = gmsh.model.occ # Raccourci vers le noyau
        dims = self.config.box_dims

        logger.info("Démarrage de la construction géométrique...")

        # --- 1. CRÉATION DU DOMAINE MATRICE ---
        try:
            matrix_tag = occ.addBox(0, 0, 0, dims[0], dims[1], dims[2])
        except Exception as e:
            logger.error(f"Erreur création matrice: {e}")
            return

        inclusion_volumes = [] 

        # --- 2. CRÉATION DES INCLUSIONS (Fibres Mères + Ghosts) ---
        count_f = 0
        for f in fibers:
            # Récupérer les vecteurs de décalage périodiques
            shifts = PeriodicManager.generate_ghosts(f, dims)
            shifts.append(np.array([0, 0, 0], dtype=float))

            for s in shifts:
                try:
                    # Construction du chemin
                    # On crée les points et la spline directement
                    path_pts = [occ.addPoint(*(p + s)) for p in f.centerline]
                    if len(path_pts) < 2: continue
                    
                    path_spline = occ.addSpline(path_pts)
                    path_wire = occ.addWire([path_spline])
                    
                    # Section
                    section_face = self._create_section_face(f, occ, shift=s)
                    
                    # Extrusion
                    # out est [(dim, tag)]
                    pipe_out = occ.addPipe([(2, section_face)], path_wire)
                    if pipe_out:
                        inclusion_volumes.append(pipe_out[0][1])
                        
                    # Nettoyage immédiat des outils temporaires pour soulager OCC
                    occ.remove([(2, section_face), (1, path_wire)], recursive=True)
                    
                except Exception as e:
                    # On continue même si un fragment échoue
                    continue
            count_f += 1

        logger.info(f"{len(inclusion_volumes)} volumes de fibres générés (avant découpe).")

        # --- 3. CRÉATION DES POROSITÉS ---
        for v in voids:
            try:
                sphere = occ.addSphere(v.center[0], v.center[1], v.center[2], v.radius)
                inclusion_volumes.append(sphere)
            except:
                pass

        # --- 4. FRAGMENTATION (Booléenne) ---
        occ.synchronize()
        logger.info("Fragmentation booléenne en cours...")
        
        try:
            # Fragment retourne (nouveaux_objets, mapping_parents->enfants)
            out, out_map = occ.fragment([(3, matrix_tag)], [(3, i) for i in inclusion_volumes])
        except Exception as e:
            logger.error(f"Echec critique fragmentation GMSH: {e}")
            gmsh.finalize()
            return

        occ.synchronize()

        # --- 5. TRI ROBUSTE DES FRAGMENTS ---
        # C'est ici que l'erreur "Unknown entity" se produisait.
        # Stratégie : On interroge GMSH pour savoir ce qui existe VRAIMENT maintenant.
        
        # Récupération de toutes les entités 3D vivantes après fragment
        all_live_entities_3d = set(tag for dim, tag in gmsh.model.getEntities(3))
        
        matrix_tags = []
        inclusion_tags = []
        to_remove = []
        
        eps = 1e-5
        
        # out_map[0] -> Enfants de la Matrice (Index 0 du 1er argument de fragment)
        # out_map[1:] -> Enfants des Inclusions
        
        for input_index, child_list in enumerate(out_map):
            for dim, tag in child_list:
                # ÉTAPE CLÉ DE SÉCURITÉ : Vérifier si le tag est vivant
                if tag not in all_live_entities_3d:
                    continue 
                
                try:
                    # Calcul du centre de gravité
                    com = occ.getCenterOfMass(dim, tag)
                    
                    # Clipping : Est-ce dans la boîte [0, dims] ?
                    is_inside = (
                        -eps <= com[0] <= dims[0] + eps and
                        -eps <= com[1] <= dims[1] + eps and
                        -eps <= com[2] <= dims[2] + eps
                    )
                    
                    if is_inside:
                        if input_index == 0:
                            matrix_tags.append(tag)
                        else:
                            inclusion_tags.append(tag)
                    else:
                        to_remove.append((dim, tag))
                except Exception as e:
                    # Si une erreur survient ici, le volume est probablement corrompu/supprimé
                    continue

        # Suppression propre de ce qui dépasse
        if to_remove:
            occ.remove(to_remove, recursive=True)
            
        occ.synchronize()

        # Groupes Physiques
        if matrix_tags:
            gmsh.model.addPhysicalGroup(3, matrix_tags, tag=2, name="MATRIX")
        if inclusion_tags:
            gmsh.model.addPhysicalGroup(3, inclusion_tags, tag=1, name="INCLUSIONS")

        # Stocker les tags d'inclusions pour les champs de taille
        self._inclusion_tags = inclusion_tags

        # --- 6. EXPORTS ---
        logger.info(f"Finalisation : {len(matrix_tags)} fragments matrice, {len(inclusion_tags)} fragments inclusions.")

        # CAO
        gmsh.write(f"{output_path}.step")

        # Maillage
        logger.info("Maillage adaptatif...")

        # Tailles de base
        min_size = self.config.fiber_radius / 6.0
        max_size = max(self.config.box_dims) / 8.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

        # Champs de taille adaptatifs (Distance + Threshold autour des interfaces)
        if getattr(self.config, 'enable_adaptive_mesh', True):
            self._setup_size_fields(dims, min_size, max_size)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 15)

        # Optimiseur maillage 3D
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # 1=Delaunay, 10=HXT (Rapide/Parallèle)

        try:
            gmsh.model.mesh.generate(3)

            # Maillage périodique (contraintes sur faces opposées)
            if getattr(self.config, 'enable_periodic_mesh', False):
                self._apply_periodic_constraints(dims)
                # Regénérer après contraintes périodiques
                gmsh.model.mesh.generate(3)

            gmsh.write(f"{output_path}.msh")
        except Exception as e:
            logger.error(f"Echec du maillage : {e}")

        gmsh.finalize()

    def _create_section_face(self, fiber, occ, shift):
        """!
        @brief Crée une face 2D représentant la section de la fibre.
        @param fiber Fiber Instance de la fibre.
        @param occ Any Raccourci vers gmsh.model.occ.
        @param shift np.ndarray Vecteur de translation (pour les ghosts).
        @return int Tag de la face créée.
        """
        p0 = fiber.centerline[0] + shift
        if fiber.section_type == 'circular':
            # Primitive disque : très stable
            # zAxis = direction tangente T, xAxis = normale N
            return occ.addDisk(p0[0], p0[1], p0[2], fiber.radius, fiber.radius,
                               zAxis=fiber.T[0], xAxis=fiber.N[0])
        else:
            # Spline complexe
            pts = fiber.section_profile.generate_points(32)
            pt_tags = []
            for loc in pts:
                # P = P0 + u*N + v*B
                w = p0 + loc[0]*fiber.N[0] + loc[1]*fiber.B[0]
                pt_tags.append(occ.addPoint(*w))
            
            spl = occ.addSpline(pt_tags)
            lin = occ.addLine(pt_tags[-1], pt_tags[0])
            wire = occ.addWire([spl, lin])
            return occ.addPlaneSurface([wire])

    def _setup_size_fields(self, dims, min_size, max_size):
        """!
        @brief Configure des champs de taille adaptatifs Distance + Threshold
        autour des interfaces fibre/matrice.
        @param dims np.ndarray Dimensions de la boîte.
        @param min_size float Taille de maille minimale (aux interfaces).
        @param max_size float Taille de maille maximale (loin des fibres).
        """
        # Récupérer les surfaces des inclusions
        inclusion_surfaces = []
        for tag in self._inclusion_tags:
            try:
                bounds = gmsh.model.getBoundary([(3, tag)], oriented=False)
                inclusion_surfaces.extend([b[1] for b in bounds])
            except:
                continue

        if not inclusion_surfaces:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 15)
            return

        # Champ 1 : Distance aux surfaces d'inclusion
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", inclusion_surfaces)

        # Champ 2 : Seuil basé sur la distance
        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", min_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", max_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", self.config.fiber_radius * 0.5)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", self.config.fiber_radius * 3.0)

        # Utiliser comme maillage de fond
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

        # Désactiver les sources de taille par défaut pour éviter les conflits
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def _apply_periodic_constraints(self, dims):
        """!
        @brief Applique les contraintes de maillage périodique sur les 3 paires de faces.
        Face pairs : (X=0, X=Lx), (Y=0, Y=Ly), (Z=0, Z=Lz)
        @param dims np.ndarray Dimensions (Lx, Ly, Lz) du domaine.
        """
        Lx, Ly, Lz = dims
        surfaces = gmsh.model.getEntities(2)

        eps = 1e-4

        # Pour chaque axe : identifier faces master (position ~0) et slave (position ~L)
        face_pairs = [
            (0, Lx, [1, 0, 0, Lx,  0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1]),  # X
            (1, Ly, [1, 0, 0, 0,   0, 1, 0, Ly,   0, 0, 1, 0,   0, 0, 0, 1]),  # Y
            (2, Lz, [1, 0, 0, 0,   0, 1, 0, 0,    0, 0, 1, Lz,  0, 0, 0, 1]),  # Z
        ]

        for axis_idx, length, transform in face_pairs:
            master_tags = []
            slave_tags = []

            for dim, tag in surfaces:
                try:
                    com = gmsh.model.occ.getCenterOfMass(dim, tag)
                except:
                    continue
                if abs(com[axis_idx]) < eps:
                    master_tags.append(tag)
                elif abs(com[axis_idx] - length) < eps:
                    slave_tags.append(tag)

            # Associer chaque face slave à son master correspondant
            for st in slave_tags:
                try:
                    st_com = gmsh.model.occ.getCenterOfMass(2, st)
                except:
                    continue
                best_master = None
                best_dist = float('inf')

                for mt in master_tags:
                    try:
                        mt_com = gmsh.model.occ.getCenterOfMass(2, mt)
                    except:
                        continue
                    # Le master doit correspondre géométriquement (même position sauf sur l'axe)
                    diff = 0.0
                    for k in range(3):
                        if k == axis_idx:
                            continue
                        diff += (st_com[k] - mt_com[k]) ** 2
                    diff = diff ** 0.5
                    if diff < best_dist:
                        best_dist = diff
                        best_master = mt

                if best_master is not None and best_dist < eps * 10:
                    try:
                        gmsh.model.mesh.setPeriodic(2, [st], [best_master], transform)
                    except Exception as e:
                        logger.debug(f"setPeriodic échoué pour face {st}->{best_master}: {e}")
"""
gui.py
Interface graphique PyQt6 pour Fiber Packing System v6.1.
Lance la generation de VER via QProcess (sous-processus non-bloquant).
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QPlainTextEdit, QLineEdit,
    QScrollArea, QSplitter, QProgressBar, QSlider, QFormLayout,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QProcess, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QIcon

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


# ---------------------------------------------------------------------------
#  Section collapsible
# ---------------------------------------------------------------------------

class CollapsibleSection(QGroupBox):
    """QGroupBox avec titre cliquable pour plier/deplier le contenu."""

    def __init__(self, title, description="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)
        if description:
            self.setToolTip(description)

        self._content = QWidget()
        self._layout = QFormLayout(self._content)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(6)

        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._content)
        self.setLayout(box)

    def form(self) -> QFormLayout:
        return self._layout

    def _on_toggle(self, checked):
        self._content.setVisible(checked)


# ---------------------------------------------------------------------------
#  Vue 3D Matplotlib
# ---------------------------------------------------------------------------

class Preview3DCanvas(FigureCanvasQTAgg):
    """Canvas matplotlib pour afficher la boite RVE et les fibres."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        self._style_axes()
        self.draw_box(1, 1, 1)

    def _style_axes(self):
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.label.set_color('white')
            axis.set_tick_params(colors='white')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def draw_box(self, lx, ly, lz):
        """Dessine un wireframe de la boite RVE."""
        self.ax.cla()
        self._style_axes()

        # 12 aretes du cube
        corners = np.array([
            [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
            [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
        ])
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i, j in edges:
            self.ax.plot3D(*zip(corners[i], corners[j]), color='cyan', alpha=0.4, linewidth=0.8)

        self.ax.set_xlim(0, lx)
        self.ax.set_ylim(0, ly)
        self.ax.set_zlim(0, lz)
        self.draw()

    def draw_fibers(self, json_path, box_dims):
        """Charge le fichier parametric JSON et dessine les fibres."""
        import json
        self.ax.cla()
        self._style_axes()

        lx, ly, lz = box_dims
        # Redessiner la boite
        corners = np.array([
            [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
            [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz]
        ])
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for i, j in edges:
            self.ax.plot3D(*zip(corners[i], corners[j]), color='cyan', alpha=0.3, linewidth=0.8)

        try:
            with open(json_path) as f:
                data = json.load(f)

            fibers = data.get("microstructure_data", [])
            cmap = matplotlib.colormaps['tab20']
            for idx, fib in enumerate(fibers):
                pts = np.array(fib["control_points"])
                color = cmap(idx % 20)
                self.ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                               color=color, linewidth=1.2, alpha=0.8)

            self.ax.set_title(f"{len(fibers)} fibres", color='white', fontsize=10)
        except Exception as e:
            self.ax.set_title(f"Erreur: {e}", color='red', fontsize=9)

        self.ax.set_xlim(0, lx)
        self.ax.set_ylim(0, ly)
        self.ax.set_zlim(0, lz)
        self.draw()


# ---------------------------------------------------------------------------
#  Fenetre principale
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Packing System v6.1")
        self.setMinimumSize(1100, 750)
        self.process = None

        self._build_ui()
        self._apply_style()

    # -----------------------------------------------------------------------
    #  Construction de l'UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- PANNEAU GAUCHE : Parametres (scrollable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(370)
        scroll.setMaximumWidth(480)

        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setSpacing(6)

        self._build_section_domain(params_layout)
        self._build_section_fibers(params_layout)
        self._build_section_orientation(params_layout)
        self._build_section_optimizer(params_layout)
        self._build_section_porosity(params_layout)
        self._build_section_export(params_layout)

        params_layout.addStretch()
        scroll.setWidget(params_widget)
        splitter.addWidget(scroll)

        # --- PANNEAU DROIT ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Vue 3D
        self.canvas = Preview3DCanvas()
        right_layout.addWidget(self.canvas, stretch=3)

        # Console
        console_label = QLabel("Console de sortie")
        console_label.setStyleSheet("font-weight: bold; color: #aaa; margin-top: 4px;")
        right_layout.addWidget(console_label)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Monospace", 9))
        self.console.setMaximumBlockCount(2000)
        right_layout.addWidget(self.console, stretch=2)

        # Barre de progression
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminee
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Generation en cours...")
        right_layout.addWidget(self.progress)

        # Boutons d'action
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Generer le VER")
        self.btn_generate.setFixedHeight(40)
        self.btn_generate.clicked.connect(self._on_generate)

        self.btn_stop = QPushButton("Arreter")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_refresh = QPushButton("Rafraichir la vue 3D")
        self.btn_refresh.setFixedHeight(40)
        self.btn_refresh.clicked.connect(self._on_refresh_3d)

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_refresh)
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right)
        splitter.setSizes([380, 700])

    # -----------------------------------------------------------------------
    #  Section 1 : Domaine et Cibles
    # -----------------------------------------------------------------------

    def _build_section_domain(self, parent_layout):
        sec = CollapsibleSection(
            "1. Domaine et Cibles",
            "Definit la taille de l'echantillon numerique et le taux de remplissage en fibres."
        )

        self.spin_lx = self._double_spin(0.01, 100.0, 1.0, 3, "Longueur selon l'axe X")
        self.spin_ly = self._double_spin(0.01, 100.0, 1.0, 3, "Longueur selon l'axe Y")
        self.spin_lz = self._double_spin(0.01, 100.0, 1.0, 3, "Longueur selon l'axe Z")

        dims_layout = QHBoxLayout()
        for label, spin in [("Lx", self.spin_lx), ("Ly", self.spin_ly), ("Lz", self.spin_lz)]:
            w = QWidget()
            vl = QVBoxLayout(w)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.addWidget(QLabel(label))
            vl.addWidget(spin)
            dims_layout.addWidget(w)

        dims_widget = QWidget()
        dims_widget.setLayout(dims_layout)
        sec.form().addRow("Dimensions du volume :", dims_widget)

        # Vf avec slider synchronise
        self.spin_vf = self._double_spin(0.01, 0.80, 0.30, 2,
            "Proportion du volume occupee par les fibres.\nExemple : 0.55 = 55% de fibres.")
        self.slider_vf = QSlider(Qt.Orientation.Horizontal)
        self.slider_vf.setRange(1, 80)
        self.slider_vf.setValue(30)
        self.slider_vf.setToolTip("Glisser pour ajuster le taux de fibres")
        self.slider_vf.valueChanged.connect(lambda v: self.spin_vf.setValue(v / 100.0))
        self.spin_vf.valueChanged.connect(lambda v: self.slider_vf.setValue(int(v * 100)))

        vf_widget = QWidget()
        vf_layout = QHBoxLayout(vf_widget)
        vf_layout.setContentsMargins(0, 0, 0, 0)
        vf_layout.addWidget(self.slider_vf, stretch=3)
        vf_layout.addWidget(self.spin_vf, stretch=1)
        sec.form().addRow("Taux de fibres :", vf_widget)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setSpecialValueText("Aleatoire")
        self.spin_seed.setToolTip("Fixe la graine aleatoire pour obtenir des resultats reproductibles.\n0 = generation differente a chaque fois.")
        sec.form().addRow("Graine aleatoire :", self.spin_seed)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Section 2 : Geometrie des Fibres
    # -----------------------------------------------------------------------

    def _build_section_fibers(self, parent_layout):
        sec = CollapsibleSection(
            "2. Geometrie des Fibres",
            "Definit la taille et la forme de la section transversale de chaque fibre."
        )

        self.spin_radius = self._double_spin(0.001, 1.0, 0.02, 4,
            "Rayon de l'enveloppe circulaire utilisee pour la detection de collision.")
        sec.form().addRow("Rayon de la fibre :", self.spin_radius)

        self.combo_section = QComboBox()
        self.combo_section.addItem("Circulaire", "circular")
        self.combo_section.addItem("Superelliptique", "superelliptical")
        self.combo_section.setToolTip("Forme de la section reelle de la fibre.\nCirculaire = rond parfait.\nSuperelliptique = forme ovale a coins ajustables.")
        sec.form().addRow("Type de section :", self.combo_section)

        self.spin_clearance = self._double_spin(0.0, 1.0, 0.0, 4,
            "Distance minimale imposee entre deux fibres voisines.")
        sec.form().addRow("Espacement minimal :", self.spin_clearance)

        self.spin_sec_n = self._double_spin(1.0, 10.0, 2.5, 1,
            "Exposant de la superellipse.\n2.0 = ellipse, >2 = coins plus carres.")
        sec.form().addRow("Exposant forme :", self.spin_sec_n)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Section 3 : Orientation
    # -----------------------------------------------------------------------

    def _build_section_orientation(self, parent_layout):
        sec = CollapsibleSection(
            "3. Orientation des Fibres",
            "Controle la direction preferentielle des fibres dans le volume."
        )

        self.combo_bias = QComboBox()
        self.combo_bias.addItem("Aleatoire 3D (aucune direction preferee)", "free")
        self.combo_bias.addItem("Unidirectionnel (fibres paralleles)", "uniaxial")
        self.combo_bias.addItem("Dans un plan (fibres orientees dans un plan)", "planar")
        self.combo_bias.setCurrentIndex(2)
        self.combo_bias.setToolTip("Mode d'orientation des fibres.")
        self.combo_bias.currentIndexChanged.connect(self._on_bias_changed)
        sec.form().addRow("Mode d'orientation :", self.combo_bias)

        self.lbl_bias_desc = QLabel("Les fibres restent dans le plan XY.")
        self.lbl_bias_desc.setWordWrap(True)
        self.lbl_bias_desc.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        sec.form().addRow(self.lbl_bias_desc)

        # Strength slider
        self.spin_strength = self._double_spin(0.0, 1.0, 0.0, 2,
            "0 = orientation completement libre\n1 = alignement strict sur la direction choisie")
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(0)
        self.slider_strength.valueChanged.connect(lambda v: self.spin_strength.setValue(v / 100.0))
        self.spin_strength.valueChanged.connect(lambda v: self.slider_strength.setValue(int(v * 100)))

        str_widget = QWidget()
        str_layout = QHBoxLayout(str_widget)
        str_layout.setContentsMargins(0, 0, 0, 0)
        str_layout.addWidget(self.slider_strength, stretch=3)
        str_layout.addWidget(self.spin_strength, stretch=1)
        sec.form().addRow("Force d'alignement :", str_widget)

        self.spin_curvature = self._double_spin(1.0, 180.0, 60.0, 1,
            "Angle maximal autorise entre deux segments consecutifs d'une fibre.\nPlus petit = fibres plus droites. Plus grand = fibres plus sinueuses.")
        sec.form().addRow("Courbure max (deg) :", self.spin_curvature)

        self.spin_min_pts = QSpinBox()
        self.spin_min_pts.setRange(3, 100)
        self.spin_min_pts.setValue(15)
        self.spin_min_pts.setToolTip("Nombre minimal de points de controle par fibre.\nPlus de points = fibres plus longues.")
        sec.form().addRow("Min points/fibre :", self.spin_min_pts)

        self.spin_max_pts = QSpinBox()
        self.spin_max_pts.setRange(3, 200)
        self.spin_max_pts.setValue(20)
        self.spin_max_pts.setToolTip("Nombre maximal de points de controle par fibre.")
        sec.form().addRow("Max points/fibre :", self.spin_max_pts)

        self.spin_step = self._double_spin(0.001, 1.0, 0.07, 3,
            "Longueur de chaque segment de la fibre.")
        sec.form().addRow("Longueur segment :", self.spin_step)

        self.chk_rsda = QCheckBox("Activer (meilleur remplissage)")
        self.chk_rsda.setToolTip("RSDA : permet de deplacer legerement les fibres existantes\npour faire de la place aux nouvelles.\nAide a depasser la limite de remplissage standard (~55%).")
        sec.form().addRow("Rearrangement :", self.chk_rsda)

        parent_layout.addWidget(sec)

    def _on_bias_changed(self, index):
        descriptions = {
            0: "Les fibres partent dans toutes les directions sans preference.",
            1: "Les fibres s'alignent selon un axe principal (par defaut X).",
            2: "Les fibres restent dans le plan XY."
        }
        self.lbl_bias_desc.setText(descriptions.get(index, ""))

    # -----------------------------------------------------------------------
    #  Section 4 : Optimisation
    # -----------------------------------------------------------------------

    def _build_section_optimizer(self, parent_layout):
        sec = CollapsibleSection(
            "4. Densification (Phase 2)",
            "Apres le placement initial, cette etape tasse les fibres\npour augmenter le taux de remplissage."
        )

        self.chk_optimize = QCheckBox("Activer la densification")
        self.chk_optimize.setChecked(True)
        self.chk_optimize.setToolTip("Active une phase d'optimisation qui secoue et comprime\nles fibres pour remplir davantage le volume.")
        self.chk_optimize.toggled.connect(self._on_optimize_toggled)
        sec.form().addRow(self.chk_optimize)

        self.spin_opt_iters = QSpinBox()
        self.spin_opt_iters.setRange(1, 500)
        self.spin_opt_iters.setValue(10)
        self.spin_opt_iters.setToolTip("Nombre de cycles d'optimisation.\nPlus de cycles = meilleur remplissage mais plus long.")
        sec.form().addRow("Cycles :", self.spin_opt_iters)

        self.spin_jitter = self._double_spin(0.0, 1.0, 0.05, 3,
            "Amplitude des vibrations aleatoires appliquees aux fibres\npour les reorganiser (fraction du rayon).")
        sec.form().addRow("Vibrations :", self.spin_jitter)

        self.spin_compression = self._double_spin(0.0, 0.5, 0.01, 3,
            "Intensite de la compression vers le centre du volume\npour rapprocher les fibres.")
        sec.form().addRow("Compression :", self.spin_compression)

        self.spin_injection = QSpinBox()
        self.spin_injection.setRange(0, 500)
        self.spin_injection.setValue(50)
        self.spin_injection.setToolTip("Nombre de tentatives d'ajout de nouvelles fibres\na chaque cycle d'optimisation.")
        sec.form().addRow("Injections/cycle :", self.spin_injection)

        # Stocker les widgets pour activer/desactiver
        self._opt_widgets = [self.spin_opt_iters, self.spin_jitter,
                             self.spin_compression, self.spin_injection]
        parent_layout.addWidget(sec)

    def _on_optimize_toggled(self, checked):
        for w in self._opt_widgets:
            w.setEnabled(checked)

    # -----------------------------------------------------------------------
    #  Section 5 : Porosite
    # -----------------------------------------------------------------------

    def _build_section_porosity(self, parent_layout):
        sec = CollapsibleSection(
            "5. Porosite (defauts)",
            "Ajoute des bulles d'air spheriques dans la matrice\npour simuler les defauts de fabrication."
        )

        self.chk_porosity = QCheckBox("Ajouter des pores")
        self.chk_porosity.setToolTip("Active la generation de bulles d'air\ndans la matrice du composite.")
        self.chk_porosity.toggled.connect(self._on_porosity_toggled)
        sec.form().addRow(self.chk_porosity)

        self.spin_void_vf = self._double_spin(0.0, 0.10, 0.01, 3,
            "Proportion du volume occupee par les bulles d'air.\nExemple : 0.01 = 1% de porosite.")
        sec.form().addRow("Taux de porosite :", self.spin_void_vf)

        self.spin_void_mean = self._double_spin(0.001, 0.5, 0.01, 3,
            "Rayon moyen des bulles d'air.")
        sec.form().addRow("Rayon moyen pores :", self.spin_void_mean)

        self.spin_void_std = self._double_spin(0.0, 0.1, 0.002, 4,
            "Variabilite du rayon des pores (ecart-type).")
        sec.form().addRow("Variabilite rayon :", self.spin_void_std)

        self._poro_widgets = [self.spin_void_vf, self.spin_void_mean, self.spin_void_std]
        for w in self._poro_widgets:
            w.setEnabled(False)
        parent_layout.addWidget(sec)

    def _on_porosity_toggled(self, checked):
        for w in self._poro_widgets:
            w.setEnabled(checked)

    # -----------------------------------------------------------------------
    #  Section 6 : Exports
    # -----------------------------------------------------------------------

    def _build_section_export(self, parent_layout):
        sec = CollapsibleSection(
            "6. Exports et Resolution",
            "Configure les fichiers de sortie et la finesse de la discretisation."
        )

        self.edit_output = QLineEdit("RVE_Result")
        self.edit_output.setToolTip("Prefixe utilise pour nommer tous les fichiers de sortie.")
        sec.form().addRow("Nom du projet :", self.edit_output)

        self.spin_res_fft = QComboBox()
        for r in [32, 64, 128, 256, 512]:
            self.spin_res_fft.addItem(f"{r}x{r}x{r}", r)
        self.spin_res_fft.setCurrentIndex(2)  # 128
        self.spin_res_fft.setToolTip("Resolution de la grille de voxels pour la simulation FFT.\nPlus eleve = plus precis mais plus lent et plus de memoire.")
        sec.form().addRow("Resolution voxels :", self.spin_res_fft)

        self.chk_mesh = QCheckBox("Maillage elements finis (GMSH)")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.setToolTip("Genere un maillage tetraedrique conforme\npour la simulation par elements finis.")
        sec.form().addRow(self.chk_mesh)

        self.chk_periodic_mesh = QCheckBox("Maillage periodique")
        self.chk_periodic_mesh.setToolTip("Force les noeuds sur les faces opposees du volume\na etre identiques (pour conditions aux limites periodiques).")
        sec.form().addRow(self.chk_periodic_mesh)

        self.chk_nastran = QCheckBox("Export Nastran (.bdf)")
        self.chk_nastran.setToolTip("Convertit le maillage au format Nastran Bulk Data\npour les solveurs compatibles (Nastran, Optistruct...).")
        sec.form().addRow(self.chk_nastran)

        self.chk_stats = QCheckBox("Statistiques spatiales")
        self.chk_stats.setChecked(True)
        self.chk_stats.setToolTip("Calcule des descripteurs de validation :\n- Distance au plus proche voisin\n- Fonction K de Ripley\n- Statistiques de Voronoi")
        sec.form().addRow(self.chk_stats)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Utilitaires de creation de widgets
    # -----------------------------------------------------------------------

    def _double_spin(self, vmin, vmax, default, decimals, tooltip=""):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (-decimals))
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    # -----------------------------------------------------------------------
    #  Construction de la commande CLI
    # -----------------------------------------------------------------------

    def _build_command(self):
        """Construit la liste d'arguments pour main.py a partir des widgets."""
        args = [sys.executable, "main.py"]

        # Domaine
        args += ["--dims", str(self.spin_lx.value()),
                 str(self.spin_ly.value()), str(self.spin_lz.value())]
        args += ["--vf", str(self.spin_vf.value())]
        seed = self.spin_seed.value()
        if seed > 0:
            args += ["--seed", str(seed)]

        # Fibres
        args += ["--radius", str(self.spin_radius.value())]
        args += ["--section", self.combo_section.currentData()]
        if self.spin_clearance.value() > 0:
            args += ["--clearance", str(self.spin_clearance.value())]
        args += ["--sec_n", str(self.spin_sec_n.value())]

        # Orientation
        bias = self.combo_bias.currentData()
        args += ["--bias_type", bias]
        args += ["--strength", str(self.spin_strength.value())]
        args += ["--curvature", str(self.spin_curvature.value())]
        args += ["--min_pts", str(self.spin_min_pts.value())]
        args += ["--max_pts", str(self.spin_max_pts.value())]
        args += ["--step", str(self.spin_step.value())]
        if self.chk_rsda.isChecked():
            args += ["--rsda"]

        # Optimiseur
        if not self.chk_optimize.isChecked():
            args += ["--no_optimize"]
        else:
            args += ["--opt_iters", str(self.spin_opt_iters.value())]
            args += ["--jitter", str(self.spin_jitter.value())]
            args += ["--compression", str(self.spin_compression.value())]
            args += ["--injection", str(self.spin_injection.value())]

        # Porosite
        if self.chk_porosity.isChecked():
            args += ["--porosity"]
            args += ["--void_vf", str(self.spin_void_vf.value())]
            args += ["--void_mean", str(self.spin_void_mean.value())]
            args += ["--void_std", str(self.spin_void_std.value())]

        # Export
        args += ["--output", self.edit_output.text()]
        args += ["--res_fft", str(self.spin_res_fft.currentData())]
        if not self.chk_mesh.isChecked():
            args += ["--no_mesh"]
        if self.chk_periodic_mesh.isChecked():
            args += ["--periodic_mesh"]
        if self.chk_nastran.isChecked():
            args += ["--nastran"]
        if not self.chk_stats.isChecked():
            args += ["--no_spatial_stats"]
        if self.chk_mesh.isChecked() and not self.chk_periodic_mesh.isChecked():
            pass  # adaptive mesh par defaut

        return args

    # -----------------------------------------------------------------------
    #  Lancement / arret du processus
    # -----------------------------------------------------------------------

    def _on_generate(self):
        self.console.clear()
        args = self._build_command()

        self.console.appendPlainText(f"$ {' '.join(args)}\n")

        self.process = QProcess()
        self.process.setWorkingDirectory(os.path.dirname(os.path.abspath(__file__)))
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.readyReadStandardError.connect(self._on_stderr)
        self.process.finished.connect(self._on_finished)

        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setVisible(True)

        self.process.start(args[0], args[1:])

    def _on_stop(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.console.appendPlainText("\n--- Generation interrompue par l'utilisateur ---")

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        for line in data.splitlines():
            self.console.appendPlainText(line)
            self._parse_progress(line)

    def _on_stderr(self):
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        for line in data.splitlines():
            self.console.appendPlainText(line)

    def _on_finished(self, exit_code, exit_status):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)

        if exit_code == 0:
            self.console.appendPlainText("\n=== Generation terminee avec succes ===")
            # Charger automatiquement la vue 3D
            QTimer.singleShot(500, self._on_refresh_3d)
        else:
            self.console.appendPlainText(f"\n=== Erreur (code {exit_code}) ===")

    def _parse_progress(self, line):
        """Met a jour le texte de la barre de progression selon la phase detectee."""
        line_lower = line.lower()
        if "phase 1" in line_lower:
            self.progress.setFormat("Phase 1 : Placement des fibres...")
        elif "phase 2" in line_lower:
            self.progress.setFormat("Phase 2 : Densification...")
        elif "phase 3" in line_lower:
            self.progress.setFormat("Phase 3 : Validation topologique...")
        elif "phase 4" in line_lower:
            self.progress.setFormat("Phase 4 : Generation des pores...")
        elif "phase 5" in line_lower or "export" in line_lower:
            self.progress.setFormat("Phase 5 : Exports...")
        elif "termin" in line_lower:
            self.progress.setFormat("Termine !")

    def _on_refresh_3d(self):
        """Charge le fichier JSON parametric et affiche les fibres."""
        prefix = self.edit_output.text()
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{prefix}_parametric.json"
        )
        box = (self.spin_lx.value(), self.spin_ly.value(), self.spin_lz.value())

        if os.path.exists(json_path):
            self.canvas.draw_fibers(json_path, box)
        else:
            self.canvas.draw_box(*box)

    # -----------------------------------------------------------------------
    #  Style
    # -----------------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ddd;
                font-size: 13px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #5dade2;
            }
            QGroupBox::indicator {
                width: 13px;
                height: 13px;
            }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px 6px;
                color: #eee;
                min-height: 22px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {
                border-color: #5dade2;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                margin: -5px 0;
                background: #5dade2;
                border-radius: 8px;
            }
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #5dade2;
            }
            QPushButton:pressed {
                background-color: #5dade2;
                color: #1e1e1e;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QPlainTextEdit {
                background-color: #1a1a1a;
                border: 1px solid #444;
                border-radius: 3px;
                color: #0f0;
                font-family: "Courier New", monospace;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #333;
                color: #eee;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background-color: #5dade2;
                border-radius: 3px;
            }
            QScrollArea {
                border: none;
            }
            QLabel {
                background: transparent;
            }
            QToolTip {
                background-color: #3c3c3c;
                color: #eee;
                border: 1px solid #5dade2;
                padding: 4px;
                font-size: 12px;
            }
        """)


# ---------------------------------------------------------------------------
#  Point d'entree
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Fiber Packing System v6")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

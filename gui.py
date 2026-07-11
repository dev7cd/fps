## @file gui.py
#  @brief Graphical User Interface for the Fiber Packing System.
#  @details Built with PyQt6, this interface allows users to configure RVE parameters,
#  visualize the generation process in real-time via a console, and preview the
#  resulting fiber microstructure in 3D.


import sys
import os
from PyQt6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QPlainTextEdit, QLineEdit,
    QScrollArea, QSplitter, QProgressBar, QSlider, QFormLayout,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QProcess, QTimer # type: ignore
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QIcon # type: ignore

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


# ---------------------------------------------------------------------------
#  Collapsible section
# ---------------------------------------------------------------------------

class CollapsibleSection(QGroupBox):
    """!
    @class CollapsibleSection
    @brief A custom QGroupBox that can be toggled to show or hide its content.
    """

    def __init__(self, title, description="", parent=None):
        """!
        @brief Initializes the collapsible section.
        @param title str The title displayed on the group box.
        @param description str Tooltip description for the section.
        @param parent QWidget Parent widget.
        """
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)
        if description:
            self.setToolTip(description)

        ## @var _content
        #  @brief Widget holding the section content.
        self._content = QWidget()
        ## @var _layout
        #  @brief Internal form layout.
        self._layout = QFormLayout(self._content)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(6)

        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._content)
        self.setLayout(box)

    def form(self) -> QFormLayout:
        """!
        @brief Returns the internal form layout for adding rows.
        @return QFormLayout instance.
        """
        return self._layout

    def _on_toggle(self, checked):
        """!
        @brief Handles the toggling of the section visibility.
        """
        self._content.setVisible(checked)


# ---------------------------------------------------------------------------
#  Matplotlib 3D view
# ---------------------------------------------------------------------------
class Preview3DCanvas(FigureCanvasQTAgg):
    """!
    @class Preview3DCanvas
    @brief Matplotlib canvas integrated into PyQt for 3D RVE visualization.
    """

    def __init__(self, parent=None):
        """!
        @brief Initializes the 3D canvas with a dark theme.
        @param parent QWidget The parent widget.
        """
        ## @var fig
        #  @brief The matplotlib Figure instance.
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        super().__init__(self.fig)
        ## @var ax
        #  @brief The 3D axes for plotting.
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#2b2b2b')
        self._style_axes()
        self.draw_box(1, 1, 1)

    def _style_axes(self):
        """!
        @brief Applies scientific styling to the 3D axes.
        """
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis.label.set_color('white')
            axis.set_tick_params(colors='white')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def draw_box(self, lx, ly, lz):
        """!
        @brief Draws the wireframe of the RVE bounding box.
        @param lx float Length in X.
        @param ly float Length in Y.
        @param lz float Length in Z.
        @return None
        """
        self.ax.cla()
        self._style_axes()

        # 12 edges of the cube
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
        """!
        @brief Loads fiber data from a JSON file and renders them in 3D.
        @param json_path str Path to the parametric JSON record.
        @param box_dims tuple Tuple of (Lx, Ly, Lz).
        @return None
        """
        import json
        self.ax.cla()
        self._style_axes()

        lx, ly, lz = box_dims
        # Redraw the box
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

            self.ax.set_title(f"{len(fibers)} fibers", color='white', fontsize=10)
        except Exception as e:
            self.ax.set_title(f"Error: {e}", color='red', fontsize=9)

        self.ax.set_xlim(0, lx)
        self.ax.set_ylim(0, ly)
        self.ax.set_zlim(0, lz)
        self.draw()


# ---------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """!
    @class MainWindow
    @brief Main application window for the Fiber Packing System GUI.
    """

    def __init__(self):
        """!
        @brief Initializes the main window, UI components, and styles.
        """
        super().__init__()
        self.setWindowTitle("Fiber Packing System v6.1")
        self.setMinimumSize(1100, 750)
        ## @var process
        #  @brief External process for running main.py.
        self.process = None

        self._build_ui()
        self._apply_style()

    # -----------------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        """!
        @brief Orchestrates the construction of the sidebar and the main view.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT PANEL: parameters (scrollable) ---
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

        # --- RIGHT PANEL ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 3D view
        ## @var canvas
        #  @brief 3D visualization canvas.
        self.canvas = Preview3DCanvas()
        right_layout.addWidget(self.canvas, stretch=3)

        # Console
        console_label = QLabel("Output console")
        console_label.setStyleSheet("font-weight: bold; color: #aaa; margin-top: 4px;")
        right_layout.addWidget(console_label)

        ## @var console
        #  @brief Text area for the console.
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Monospace", 9))
        self.console.setMaximumBlockCount(2000)
        right_layout.addWidget(self.console, stretch=2)

        # Progress bar
        ## @var progress
        #  @brief Progress bar.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Generation in progress...")
        right_layout.addWidget(self.progress)

        # Action buttons
        btn_layout = QHBoxLayout()
        ## @var btn_generate
        #  @brief Button to start the generation.
        self.btn_generate = QPushButton("Generate the RVE")
        self.btn_generate.setFixedHeight(40)
        self.btn_generate.clicked.connect(self._on_generate)

        ## @var btn_stop
        #  @brief Button to stop the generation.
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)

        ## @var btn_refresh
        #  @brief Button to refresh the 3D view.
        self.btn_refresh = QPushButton("Refresh 3D view")
        self.btn_refresh.setFixedHeight(40)
        self.btn_refresh.clicked.connect(self._on_refresh_3d)

        btn_layout.addWidget(self.btn_generate)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_refresh)
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right)
        splitter.setSizes([380, 700])

    # -----------------------------------------------------------------------
    #  Section 1: Domain and targets
    # -----------------------------------------------------------------------

    def _build_section_domain(self, parent_layout):
        """!
        @brief Builds the UI section for domain dimensions and target volume fraction.
        """
        sec = CollapsibleSection(
            "1. Domain and targets",
            "Defines the size of the numerical sample and the fiber filling ratio."
        )

        ## @var spin_lx
        #  @brief Length in X.
        self.spin_lx = self._double_spin(0.01, 200.0, 1.0, 4, "Length along the X axis")
        ## @var spin_ly
        #  @brief Length in Y.
        self.spin_ly = self._double_spin(0.01, 200.0, 1.0, 4, "Length along the Y axis")
        ## @var spin_lz
        #  @brief Length in Z.
        self.spin_lz = self._double_spin(0.01, 200.0, 1.0, 4, "Length along the Z axis")

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
        sec.form().addRow("Volume dimensions:", dims_widget)

        # Vf with synchronised slider
        ## @var spin_vf
        #  @brief Target volume fraction.
        self.spin_vf = self._double_spin(0.001, 0.801, 0.30, 4,
            "Fraction of the volume occupied by the fibers.\nExample: 0.55 = 55% fibers.")
        ## @var slider_vf
        #  @brief Slider for volume fraction.
        self.slider_vf = QSlider(Qt.Orientation.Horizontal)
        self.slider_vf.setRange(1, 80)
        self.slider_vf.setValue(30)
        self.slider_vf.setToolTip("Drag to adjust the fiber ratio")
        self.slider_vf.valueChanged.connect(lambda v: self.spin_vf.setValue(v / 100.0))
        self.spin_vf.valueChanged.connect(lambda v: self.slider_vf.setValue(int(v * 100)))

        vf_widget = QWidget()
        vf_layout = QHBoxLayout(vf_widget)
        vf_layout.setContentsMargins(0, 0, 0, 0)
        vf_layout.addWidget(self.slider_vf, stretch=3)
        vf_layout.addWidget(self.spin_vf, stretch=1)
        sec.form().addRow("Fiber ratio:", vf_widget)

        ## @var spin_seed
        #  @brief Random seed.
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        self.spin_seed.setSpecialValueText("Random")
        self.spin_seed.setToolTip("Fixes the random seed to obtain reproducible results.\n0 = different generation every time.")
        sec.form().addRow("Random seed:", self.spin_seed)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Section 2: Fiber geometry
    # -----------------------------------------------------------------------

    def _build_section_fibers(self, parent_layout):
        """!
        @brief Builds the UI section for fiber geometry and cross-section parameters.
        @param parent_layout QVBoxLayout The layout to add this section to.
        """
        sec = CollapsibleSection(
            "2. Fiber geometry",
            "Defines the size and shape of the cross-section of each fiber."
        )

        ## @var spin_radius
        #  @brief Fiber radius.
        self.spin_radius = self._double_spin(0.001, 1.0, 0.02, 4,
            "Radius of the circular envelope used for collision detection.")
        sec.form().addRow("Fiber radius:", self.spin_radius)

        ## @var combo_section
        #  @brief Cross-section type.
        self.combo_section = QComboBox()
        self.combo_section.addItem("Circular", "circular")
        self.combo_section.addItem("Super-elliptical", "superelliptical")
        self.combo_section.setToolTip("Shape of the real fiber section.\nCircular = perfect round.\nSuper-elliptical = oval shape with adjustable corners.")
        sec.form().addRow("Section type:", self.combo_section)

        ## @var spin_clearance
        #  @brief Minimum clearance.
        self.spin_clearance = self._double_spin(0.0, 1.0, 0.0, 4,
            "Minimum distance imposed between two neighbouring fibers.")
        sec.form().addRow("Minimum spacing:", self.spin_clearance)

        ## @var spin_sec_n
        #  @brief Superellipse exponent.
        self.spin_sec_n = self._double_spin(1.0, 10.0, 2.5, 1,
            "Super-ellipse exponent.\n2.0 = ellipse, >2 = more square corners.")
        sec.form().addRow("Shape exponent:", self.spin_sec_n)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Section 3: Orientation
    # -----------------------------------------------------------------------

    def _build_section_orientation(self, parent_layout):
        """!
        @brief Builds the UI section for orientation bias and trajectory constraints.
        """
        sec = CollapsibleSection(
            "3. Fiber orientation",
            "Controls the preferred direction of the fibers in the volume."
        )

        self.combo_bias = QComboBox()
        ## @var combo_bias
        #  @brief Orientation bias mode selector.
        self.combo_bias.addItem("Random 3D (no preferred direction)", "free")
        self.combo_bias.addItem("Unidirectional (parallel fibers)", "uniaxial")
        self.combo_bias.addItem("In a plane (fibers oriented in a plane)", "planar")
        self.combo_bias.setCurrentIndex(2)
        self.combo_bias.setToolTip("Fiber orientation mode.")
        self.combo_bias.currentIndexChanged.connect(self._on_bias_changed)
        sec.form().addRow("Orientation mode:", self.combo_bias)

        self.lbl_bias_desc = QLabel("The fibers stay in the XY plane.")
        ## @var lbl_bias_desc
        #  @brief Description label for orientation bias.
        self.lbl_bias_desc.setWordWrap(True)
        self.lbl_bias_desc.setStyleSheet("color: #888; font-style: italic; padding: 2px;")
        sec.form().addRow(self.lbl_bias_desc)

        # Strength slider
        self.spin_strength = self._double_spin(0.0, 1.0, 0.0, 2,
            "0 = completely free orientation\n1 = strict alignment with the chosen direction")
        ## @var spin_strength
        #  @brief Alignment strength selector.
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        ## @var slider_strength
        #  @brief Slider for alignment strength.
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(0)
        self.slider_strength.valueChanged.connect(lambda v: self.spin_strength.setValue(v / 100.0))
        self.spin_strength.valueChanged.connect(lambda v: self.slider_strength.setValue(int(v * 100)))

        str_widget = QWidget()
        str_layout = QHBoxLayout(str_widget)
        str_layout.setContentsMargins(0, 0, 0, 0)
        str_layout.addWidget(self.slider_strength, stretch=3)
        str_layout.addWidget(self.spin_strength, stretch=1)
        sec.form().addRow("Alignment strength:", str_widget)

        self.spin_curvature = self._double_spin(1.0, 180.0, 60.0, 1,
            "Maximum angle allowed between two consecutive segments of a fiber.\nSmaller = straighter fibers. Larger = more wavy fibers.")
        ## @var spin_curvature
        #  @brief Maximum curvature selector.
        sec.form().addRow("Max curvature (deg):", self.spin_curvature)

        self.spin_min_pts = QSpinBox()
        ## @var spin_min_pts
        #  @brief Minimum control points per fiber.
        self.spin_min_pts.setRange(3, 100)
        self.spin_min_pts.setValue(15)
        self.spin_min_pts.setToolTip("Minimum number of control points per fiber.\nMore points = longer fibers.")
        sec.form().addRow("Min points/fiber:", self.spin_min_pts)

        self.spin_max_pts = QSpinBox()
        ## @var spin_max_pts
        #  @brief Maximum control points per fiber.
        self.spin_max_pts.setRange(3, 200)
        self.spin_max_pts.setValue(20)
        self.spin_max_pts.setToolTip("Maximum number of control points per fiber.")
        sec.form().addRow("Max points/fiber:", self.spin_max_pts)

        self.spin_step = self._double_spin(0.001, 10.0, 0.07, 3,
            "Length of each fiber segment.")
        ## @var spin_step
        #  @brief Fiber segment length selector.
        sec.form().addRow("Segment length:", self.spin_step)

        self.chk_rsda = QCheckBox("Enable (better filling)")
        ## @var chk_rsda
        #  @brief Checkbox for RSDA (rearrangement) activation.
        self.chk_rsda.setToolTip("RSDA: allows the existing fibers to be moved slightly\nto make room for the new ones.\nHelps to exceed the standard filling limit (~55%).")
        sec.form().addRow("Rearrangement:", self.chk_rsda)

        parent_layout.addWidget(sec)

    def _on_bias_changed(self, index):
        """!
        @brief Updates the description label when the orientation bias mode changes.
        @param index int The index of the selected bias mode.
        """
        descriptions = {
            0: "The fibers go in all directions with no preference.",
            1: "The fibers align along a main axis (X by default).",
            2: "The fibers stay in the XY plane."
        }
        self.lbl_bias_desc.setText(descriptions.get(index, ""))

    # -----------------------------------------------------------------------
    #  Section 4: Optimisation
    # -----------------------------------------------------------------------

    def _build_section_optimizer(self, parent_layout):
        """!
        @brief Builds the UI section for Phase 2 (Dynamic Densification).
        @param parent_layout QVBoxLayout The layout to add this section to.
        """
        sec = CollapsibleSection(
            "4. Densification (Phase 2)",
            "After the initial placement, this step compacts the fibers\nto increase the filling ratio."
        )

        self.chk_optimize = QCheckBox("Enable densification")
        ## @var chk_optimize
        #  @brief Checkbox for densification phase activation.
        self.chk_optimize.setChecked(True)
        self.chk_optimize.setToolTip("Enables an optimisation phase that shakes and compresses\nthe fibers to fill the volume further.")
        self.chk_optimize.toggled.connect(self._on_optimize_toggled)
        sec.form().addRow(self.chk_optimize)

        self.spin_opt_iters = QSpinBox()
        ## @var spin_opt_iters
        #  @brief Optimization cycles selector.
        self.spin_opt_iters.setRange(1, 500)
        self.spin_opt_iters.setValue(10)
        self.spin_opt_iters.setToolTip("Number of optimisation cycles.\nMore cycles = better filling but slower.")
        sec.form().addRow("Cycles:", self.spin_opt_iters)

        self.spin_jitter = self._double_spin(0.0, 1.0, 0.05, 3,
            "Amplitude of the random vibrations applied to the fibers\nto reorganise them (fraction of the radius).")
        ## @var spin_jitter
        #  @brief Vibration amplitude selector.
        sec.form().addRow("Vibrations:", self.spin_jitter)

        self.spin_compression = self._double_spin(0.0, 0.5, 0.01, 3,
            "Intensity of the compression towards the volume centre\nto bring the fibers closer.")
        ## @var spin_compression
        #  @brief Compression intensity selector.
        sec.form().addRow("Compression:", self.spin_compression)

        self.spin_injection = QSpinBox()
        ## @var spin_injection
        #  @brief Fiber injection count selector.
        self.spin_injection.setRange(0, 500)
        self.spin_injection.setValue(50)
        self.spin_injection.setToolTip("Number of attempts to add new fibers\nat each optimisation cycle.")
        sec.form().addRow("Injections/cycle:", self.spin_injection)

        # Store the widgets to enable/disable them
        self._opt_widgets = [self.spin_opt_iters, self.spin_jitter,
                             self.spin_compression, self.spin_injection]
        parent_layout.addWidget(sec)

    def _on_optimize_toggled(self, checked):
        """!
        @brief Enables or disables optimizer inputs based on the checkbox state.
        @param checked bool True if checked, False otherwise.
        """
        for w in self._opt_widgets:
            w.setEnabled(checked)

    # -----------------------------------------------------------------------
    #  Section 5: Porosity
    # -----------------------------------------------------------------------

    def _build_section_porosity(self, parent_layout):
        """!
        @brief Builds the UI section for void/porosity generation.
        @param parent_layout QVBoxLayout The layout to add this section to.
        """
        sec = CollapsibleSection(
            "5. Porosity (defects)",
            "Adds spherical air bubbles in the matrix\nto simulate manufacturing defects."
        )

        self.chk_porosity = QCheckBox("Add pores")
        ## @var chk_porosity
        #  @brief Checkbox for porosity activation.
        self.chk_porosity.setToolTip("Enables the generation of air bubbles\nin the composite matrix.")
        self.chk_porosity.toggled.connect(self._on_porosity_toggled)
        sec.form().addRow(self.chk_porosity)

        self.spin_void_vf = self._double_spin(0.0, 0.10, 0.01, 3,
            "Fraction of the volume occupied by the air bubbles.\nExample: 0.01 = 1% porosity.")
        ## @var spin_void_vf
        #  @brief Target porosity volume fraction selector.
        sec.form().addRow("Porosity ratio:", self.spin_void_vf)

        self.spin_void_mean = self._double_spin(0.001, 0.5, 0.01, 3,
            "Mean radius of the air bubbles.")
        ## @var spin_void_mean
        #  @brief Mean pore radius selector.
        sec.form().addRow("Mean pore radius:", self.spin_void_mean)

        self.spin_void_std = self._double_spin(0.0, 0.1, 0.002, 4,
            "Variability of the pore radius (standard deviation).")
        ## @var spin_void_std
        #  @brief Pore radius variability (std dev) selector.
        sec.form().addRow("Radius variability:", self.spin_void_std)

        self._poro_widgets = [self.spin_void_vf, self.spin_void_mean, self.spin_void_std]
        for w in self._poro_widgets:
            w.setEnabled(False)
        parent_layout.addWidget(sec)

    def _on_porosity_toggled(self, checked):
        """!
        @brief Enables or disables porosity inputs based on the checkbox state.
        @param checked bool True if checked, False otherwise.
        """
        for w in self._poro_widgets:
            w.setEnabled(checked)

    # -----------------------------------------------------------------------
    #  Section 6: Exports
    # -----------------------------------------------------------------------

    def _build_section_export(self, parent_layout):
        """!
        @brief Builds the UI section for export formats and mesh resolution.
        @param parent_layout QVBoxLayout The layout to add this section to.
        """
        sec = CollapsibleSection(
            "6. Exports and resolution",
            "Configures the output files and the discretisation fineness."
        )

        self.edit_output = QLineEdit("RVE_Result")
        ## @var edit_output
        #  @brief Output project name edit field.
        self.edit_output.setToolTip("Prefix used to name all the output files.")
        sec.form().addRow("Project name:", self.edit_output)

        self.spin_res_fft = QComboBox()
        ## @var spin_res_fft
        #  @brief Voxel resolution selector for FFT.
        for r in [32, 64, 128, 256, 512]:
            self.spin_res_fft.addItem(f"{r}x{r}x{r}", r)
        self.spin_res_fft.setCurrentIndex(2)  # 128
        self.spin_res_fft.setToolTip("Resolution of the voxel grid for the FFT simulation.\nHigher = more accurate but slower and more memory.")
        sec.form().addRow("Voxel resolution:", self.spin_res_fft)

        self.chk_mesh = QCheckBox("Finite-element mesh (GMSH)")
        ## @var chk_mesh
        #  @brief Checkbox for FEA mesh generation.
        self.chk_mesh.setChecked(True)
        self.chk_mesh.setToolTip("Generates a conformal tetrahedral mesh\nfor finite-element simulation.")
        sec.form().addRow(self.chk_mesh)

        self.chk_periodic_mesh = QCheckBox("Periodic mesh")
        ## @var chk_periodic_mesh
        #  @brief Checkbox for periodic mesh constraints.
        self.chk_periodic_mesh.setToolTip("Forces the nodes on opposite faces of the volume\nto be identical (for periodic boundary conditions).")
        sec.form().addRow(self.chk_periodic_mesh)

        self.chk_nastran = QCheckBox("Nastran export (.bdf)")
        ## @var chk_nastran
        #  @brief Checkbox for Nastran export.
        self.chk_nastran.setToolTip("Converts the mesh to the Nastran Bulk Data format\nfor compatible solvers (Nastran, Optistruct...).")
        sec.form().addRow(self.chk_nastran)

        self.chk_stats = QCheckBox("Spatial statistics")
        ## @var chk_stats
        #  @brief Checkbox for spatial statistics validation.
        self.chk_stats.setChecked(True)
        self.chk_stats.setToolTip("Computes validation descriptors:\n- Nearest-neighbour distance\n- Ripley's K function\n- Voronoi statistics")
        sec.form().addRow(self.chk_stats)

        parent_layout.addWidget(sec)

    # -----------------------------------------------------------------------
    #  Widget creation helpers
    # -----------------------------------------------------------------------

    def _double_spin(self, vmin, vmax, default, decimals, tooltip=""):
        """!
        @brief Helper to create a configured QDoubleSpinBox.
        @param vmin float Minimum value.
        @param vmax float Maximum value.
        @param default float Default value.
        @param decimals int Number of decimals.
        @param tooltip str Optional tooltip text.
        @return QDoubleSpinBox instance.
        """
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(default)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (-decimals))
        if tooltip:
            spin.setToolTip(tooltip)
        return spin

    # -----------------------------------------------------------------------
    #  CLI command construction
    # -----------------------------------------------------------------------

    def _build_command(self):
        """!
        @brief Maps GUI widget values to CLI arguments for the backend orchestrator.
        @return List of strings representing the command to execute.
        """
        args = [sys.executable, "main.py"]

        # Domain
        args += ["--dims", str(self.spin_lx.value()),
                 str(self.spin_ly.value()), str(self.spin_lz.value())]
        args += ["--vf", str(self.spin_vf.value())]
        seed = self.spin_seed.value()
        if seed > 0:
            args += ["--seed", str(seed)]

        # Fibers
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

        # Optimiser
        if not self.chk_optimize.isChecked():
            args += ["--no_optimize"]
        else:
            args += ["--opt_iters", str(self.spin_opt_iters.value())]
            args += ["--jitter", str(self.spin_jitter.value())]
            args += ["--compression", str(self.spin_compression.value())]
            args += ["--injection", str(self.spin_injection.value())]

        # Porosity
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
            pass  # adaptive mesh by default

        return args

    # -----------------------------------------------------------------------
    #  Process launch / stop
    # -----------------------------------------------------------------------

    def _on_generate(self):
        """!
        @brief Starts the RVE generation process using QProcess.
        """
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
        """!
        @brief Terminates the running generation process.
        """
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.console.appendPlainText("\n--- Generation interrupted by the user ---")

    def _on_stdout(self):
        """!
        @brief Reads standard output from the process and updates the console.
        """
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        for line in data.splitlines():
            self.console.appendPlainText(line)
            self._parse_progress(line)

    def _on_stderr(self):
        """!
        @brief Reads standard error from the process and updates the console.
        """
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        for line in data.splitlines():
            self.console.appendPlainText(line)

    def _on_finished(self, exit_code, exit_status):
        """!
        @brief Handles process completion, updating UI state and refreshing the 3D view.
        """
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)

        if exit_code == 0:
            self.console.appendPlainText("\n=== Generation finished successfully ===")
            # Automatically load the 3D view
            QTimer.singleShot(500, self._on_refresh_3d)
        else:
            self.console.appendPlainText(f"\n=== Error (code {exit_code}) ===")

    def _parse_progress(self, line):
        """!
        @brief Parses the console output to update the progress bar text.
        """
        line_lower = line.lower()
        if "phase 1" in line_lower:
            self.progress.setFormat("Phase 1: Fiber placement...")
        elif "phase 2" in line_lower:
            self.progress.setFormat("Phase 2: Densification...")
        elif "phase 3" in line_lower:
            self.progress.setFormat("Phase 3: Topological validation...")
        elif "phase 4" in line_lower:
            self.progress.setFormat("Phase 4: Pore generation...")
        elif "phase 5" in line_lower or "export" in line_lower:
            self.progress.setFormat("Phase 5: Exports...")
        elif "finished" in line_lower:
            self.progress.setFormat("Done!")

    def _on_refresh_3d(self):
        """!
        @brief Refreshes the 3D preview by loading the latest generated JSON record.
        """
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
        """!
        @brief Applies a global dark-themed stylesheet to the application.
        """
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
#  Entry point
# ---------------------------------------------------------------------------

def main():
    """!
    @brief Entry point for the GUI application.
    @return None
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Fiber Packing System v6")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

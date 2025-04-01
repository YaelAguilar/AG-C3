import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, 
                            QPushButton, QTabWidget, QScrollArea, QGroupBox, QSlider, 
                            QCheckBox, QRadioButton, QSplitter, QFrame, QGridLayout, 
                            QButtonGroup, QFileDialog, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QColor, QPalette
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Estilo y colores para la aplicación
STYLE = """
    QMainWindow {
        background-color: #f8f9fa;
    }
    QTabWidget::pane {
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: white;
    }
    QTabBar::tab {
        background-color: #e9ecef;
        padding: 8px 16px;
        margin-right: 2px;
        border: 1px solid #ddd;
        border-bottom: 0px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #007bff;
        color: white;
    }
    QPushButton {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #0069d9;
    }
    QPushButton:pressed {
        background-color: #0062cc;
    }
    QGroupBox {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
        background-color: white;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 6px;
        background-color: white;
    }
    QComboBox::drop-down {
        border: 0px;
    }
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
        border: 1px solid #80bdff;
    }
    QSlider::groove:horizontal {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #007bff;
        border: 1px solid #007bff;
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }
    QSlider::sub-page:horizontal {
        background: #007bff;
        border-radius: 4px;
    }
    QScrollArea {
        border: none;
    }
    QLabel {
        color: #212529;
    }
"""

class ResultCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(ResultCanvas, self).__init__(fig)
        self.setParent(parent)
        
        # Configuración inicial del gráfico
        self.axes.set_title('Evolución de Aptitud')
        self.axes.set_xlabel('Generaciones')
        self.axes.set_ylabel('Aptitud')
        self.axes.grid(True)

class OptilensApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptiLens - Sistema Inteligente para Lentes Terapéuticos")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(STYLE)
        
        # Cargar datos CSV
        self.cargar_datos()
        
        # Inicializar interfaz
        self.setup_ui()
        
    def cargar_datos(self):
        # Esta función cargaría los datos CSV. Temporalmente usamos datos de prueba
        try:
            self.df_padecimientos = pd.read_csv('padecimientos.csv')
            self.df_monturas = pd.read_csv('monturas.csv')
            self.df_lentes = pd.read_csv('lentes.csv')
            self.df_capas = pd.read_csv('capas.csv')
            self.df_filtros = pd.read_csv('filtros.csv')
        except Exception as e:
            # Si los archivos no existen, creamos listas temporales para la interfaz
            self.df_padecimientos = pd.DataFrame({
                'nombre_padecimiento': ['Miopía', 'Hipermetropía', 'Astigmatismo', 'Presbicia', 
                                       'Fotofobia', 'Sequedad Ocular', 'Retinopatía Diabética',
                                       'Glaucoma', 'Degeneración Macular', 'Fatiga Visual Digital']
            })
            
    def setup_ui(self):
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Header con logo y título
        header_layout = QHBoxLayout()
        
        # Logo placeholder (cambiar por logo real)
        logo_label = QLabel()
        logo_label.setFixedSize(80, 80)
        logo_label.setStyleSheet("background-color: #007bff; border-radius: 40px; color: white; font-size: 16px; font-weight: bold;")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setText("OptiLens")
        header_layout.addWidget(logo_label)
        
        # Título y descripción
        title_layout = QVBoxLayout()
        title_label = QLabel("OptiLens")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        subtitle_label = QLabel("Sistema Inteligente para la Configuración Óptima de Lentes Terapéuticos")
        subtitle_label.setFont(QFont("Arial", 12))
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Panel de pestañas
        tab_widget = QTabWidget()
        
        # Pestaña de configuración
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        
        # Panel de entrada dividido en dos columnas
        input_layout = QHBoxLayout()
        
        # Columna izquierda - Información del paciente
        patient_group = QGroupBox("Información del Paciente")
        patient_layout = QVBoxLayout(patient_group)
        
        # Padecimiento
        pad_layout = QHBoxLayout()
        pad_label = QLabel("Padecimiento:")
        pad_label.setFixedWidth(130)
        self.pad_combo = QComboBox()
        for pad in self.df_padecimientos['nombre_padecimiento']:
            self.pad_combo.addItem(pad)
        pad_layout.addWidget(pad_label)
        pad_layout.addWidget(self.pad_combo)
        patient_layout.addLayout(pad_layout)
        
        # Rango de precio
        price_group = QGroupBox("Rango de Precio")
        price_layout = QVBoxLayout(price_group)
        
        slider_layout = QHBoxLayout()
        min_price_label = QLabel("Mínimo:")
        self.min_price_spin = QSpinBox()
        self.min_price_spin.setRange(100, 1000)
        self.min_price_spin.setValue(200)
        self.min_price_spin.setSingleStep(50)
        self.min_price_spin.setSuffix(" MXN")
        
        max_price_label = QLabel("Máximo:")
        self.max_price_spin = QSpinBox()
        self.max_price_spin.setRange(100, 2000)
        self.max_price_spin.setValue(800)
        self.max_price_spin.setSingleStep(50)
        self.max_price_spin.setSuffix(" MXN")
        
        slider_layout.addWidget(min_price_label)
        slider_layout.addWidget(self.min_price_spin)
        slider_layout.addWidget(max_price_label)
        slider_layout.addWidget(self.max_price_spin)
        
        price_layout.addLayout(slider_layout)
        
        # Conectar los spinBoxes y el slider
        self.min_price_spin.valueChanged.connect(self.update_price_range)
        self.max_price_spin.valueChanged.connect(self.update_price_range)
        
        patient_layout.addWidget(price_group)
        
        # Restricciones médicas adicionales
        medical_group = QGroupBox("Restricciones Médicas Adicionales")
        medical_layout = QVBoxLayout(medical_group)
        
        self.light_sensitivity = QCheckBox("Sensibilidad a la luz")
        self.screen_time = QCheckBox("Uso prolongado de pantallas")
        self.outdoor_activities = QCheckBox("Actividades al aire libre")
        self.night_driving = QCheckBox("Conducción nocturna")
        
        medical_layout.addWidget(self.light_sensitivity)
        medical_layout.addWidget(self.screen_time)
        medical_layout.addWidget(self.outdoor_activities)
        medical_layout.addWidget(self.night_driving)
        
        patient_layout.addWidget(medical_group)
        
        # Columna derecha - Disponibilidad de componentes
        availability_group = QGroupBox("Disponibilidad de Componentes")
        availability_layout = QVBoxLayout(availability_group)
        
        # Monturas
        frame_group = QGroupBox("Tipos de Montura")
        frame_layout = QVBoxLayout(frame_group)
        self.full_frame = QCheckBox("Full-Frame")
        self.semi_rimless = QCheckBox("Semi-Rimless")
        self.rimless = QCheckBox("Rimless")
        
        self.full_frame.setChecked(True)
        self.semi_rimless.setChecked(True)
        self.rimless.setChecked(True)
        
        frame_layout.addWidget(self.full_frame)
        frame_layout.addWidget(self.semi_rimless)
        frame_layout.addWidget(self.rimless)
        availability_layout.addWidget(frame_group)
        
        # Material Disponible
        material_group = QGroupBox("Materiales Disponibles")
        material_layout = QVBoxLayout(material_group)
        self.acetato = QCheckBox("Acetato")
        self.metal = QCheckBox("Metal")
        self.titanio = QCheckBox("Titanio")
        self.tr90 = QCheckBox("TR-90 (Nylon)")
        
        self.acetato.setChecked(True)
        self.metal.setChecked(True)
        self.titanio.setChecked(True)
        self.tr90.setChecked(True)
        
        material_layout.addWidget(self.acetato)
        material_layout.addWidget(self.metal)
        material_layout.addWidget(self.titanio)
        material_layout.addWidget(self.tr90)
        availability_layout.addWidget(material_group)
        
        # Capas y Filtros
        coating_group = QGroupBox("Disponibilidad de Capas y Filtros")
        coating_layout = QGridLayout(coating_group)
        
        # Primera columna - Capas
        coating_layout.addWidget(QLabel("<b>Capas:</b>"), 0, 0)
        self.antireflejo = QCheckBox("Antirreflejante")
        self.hidrofobica = QCheckBox("Hidrofóbica")
        self.fotocrom = QCheckBox("Fotocromática")
        self.endurecida = QCheckBox("Endurecida")
        
        self.antireflejo.setChecked(True)
        self.hidrofobica.setChecked(True)
        self.fotocrom.setChecked(True)
        self.endurecida.setChecked(True)
        
        coating_layout.addWidget(self.antireflejo, 1, 0)
        coating_layout.addWidget(self.hidrofobica, 2, 0)
        coating_layout.addWidget(self.fotocrom, 3, 0)
        coating_layout.addWidget(self.endurecida, 4, 0)
        
        # Segunda columna - Filtros
        coating_layout.addWidget(QLabel("<b>Filtros:</b>"), 0, 1)
        self.uv400 = QCheckBox("UV400")
        self.blue_light = QCheckBox("Anti Luz Azul")
        self.polarizado = QCheckBox("Polarizado")
        self.hd = QCheckBox("Alta Definición")
        
        self.uv400.setChecked(True)
        self.blue_light.setChecked(True)
        self.polarizado.setChecked(True)
        self.hd.setChecked(True)
        
        coating_layout.addWidget(self.uv400, 1, 1)
        coating_layout.addWidget(self.blue_light, 2, 1)
        coating_layout.addWidget(self.polarizado, 3, 1)
        coating_layout.addWidget(self.hd, 4, 1)
        
        availability_layout.addWidget(coating_group)
        
        # Añadir ambas columnas al layout de entrada
        input_layout.addWidget(patient_group, 1)
        input_layout.addWidget(availability_group, 1)
        
        config_layout.addLayout(input_layout)
        
        # Parámetros de algoritmo genético
        algo_group = QGroupBox("Parámetros del Algoritmo Genético")
        algo_layout = QHBoxLayout(algo_group)
        
        # Tamaño de población
        pop_label = QLabel("Tamaño de Población:")
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(10, 500)
        self.pop_spin.setValue(100)
        self.pop_spin.setSingleStep(10)
        
        # Número de generaciones
        gen_label = QLabel("Generaciones:")
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(10, 1000)
        self.gen_spin.setValue(50)
        self.gen_spin.setSingleStep(10)
        
        # Tasa de mutación
        mut_label = QLabel("Tasa de Mutación:")
        self.mut_spin = QDoubleSpinBox()
        self.mut_spin.setRange(0.001, 0.5)
        self.mut_spin.setValue(0.05)
        self.mut_spin.setSingleStep(0.01)
        
        # Tasa de elitismo 
        elite_label = QLabel("Elitismo (%):")
        self.elite_spin = QSpinBox()
        self.elite_spin.setRange(0, 50)
        self.elite_spin.setValue(10)
        self.elite_spin.setSingleStep(5)
        self.elite_spin.setSuffix("%")
        
        algo_layout.addWidget(pop_label)
        algo_layout.addWidget(self.pop_spin)
        algo_layout.addWidget(gen_label)
        algo_layout.addWidget(self.gen_spin)
        algo_layout.addWidget(mut_label)
        algo_layout.addWidget(self.mut_spin)
        algo_layout.addWidget(elite_label)
        algo_layout.addWidget(self.elite_spin)
        
        config_layout.addWidget(algo_group)
        
        # Botones de acción
        button_layout = QHBoxLayout()
        
        self.optimize_btn = QPushButton("Optimizar Configuración")
        self.optimize_btn.setMinimumHeight(40)
        self.optimize_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.optimize_btn.clicked.connect(self.optimize_configuration)
        
        self.reset_btn = QPushButton("Reiniciar")
        self.reset_btn.setStyleSheet("background-color: #6c757d;")
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.optimize_btn)
        
        config_layout.addLayout(button_layout)
        
        # Pestaña de resultados
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Dividir pantalla horizontalmente en resultados
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Gráfica de evolución
        graph_frame = QFrame()
        graph_layout = QVBoxLayout(graph_frame)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        
        graph_title = QLabel("Evolución del Algoritmo Genético")
        graph_title.setFont(QFont("Arial", 14, QFont.Bold))
        graph_title.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(graph_title)
        
        self.canvas = ResultCanvas()
        graph_layout.addWidget(self.canvas)
        
        results_splitter.addWidget(graph_frame)
        
        # Panel derecho - Mejores soluciones
        solutions_frame = QFrame()
        solutions_layout = QVBoxLayout(solutions_frame)
        
        solutions_title = QLabel("Mejores Configuraciones")
        solutions_title.setFont(QFont("Arial", 14, QFont.Bold))
        solutions_title.setAlignment(Qt.AlignCenter)
        solutions_layout.addWidget(solutions_title)
        
        # ScrollArea para las soluciones
        solutions_scroll = QScrollArea()
        solutions_scroll.setWidgetResizable(True)
        solutions_content = QWidget()
        solutions_scroll_layout = QVBoxLayout(solutions_content)
        
        # Añadir placeholders para las 3 mejores soluciones
        for i in range(1, 4):
            solution_group = QGroupBox(f"Configuración #{i}")
            solution_layout = QVBoxLayout(solution_group)
            
            # Detalles de la configuración
            details = QTextEdit()
            details.setReadOnly(True)
            details.setFixedHeight(150)
            
            if i == 1:
                details.setText("Aquí se mostrará la información detallada de la mejor configuración encontrada.")
            else:
                details.setText(f"Aquí se mostrará la información detallada de la configuración alternativa #{i}.")
            
            solution_layout.addWidget(details)
            
            # Métricas de la solución
            metrics_layout = QHBoxLayout()
            
            aptitude_label = QLabel(f"<b>Aptitud:</b> --")
            price_label = QLabel(f"<b>Precio:</b> -- MXN")
            
            metrics_layout.addWidget(aptitude_label)
            metrics_layout.addWidget(price_label)
            
            solution_layout.addLayout(metrics_layout)
            
            solutions_scroll_layout.addWidget(solution_group)
        
        solutions_scroll.setWidget(solutions_content)
        solutions_layout.addWidget(solutions_scroll)
        
        # Botones para resultados
        result_buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Exportar Resultados")
        self.export_btn.setStyleSheet("background-color: #28a745;")
        
        self.compare_btn = QPushButton("Comparar Soluciones")
        self.compare_btn.setStyleSheet("background-color: #17a2b8;")
        
        result_buttons_layout.addWidget(self.compare_btn)
        result_buttons_layout.addWidget(self.export_btn)
        
        solutions_layout.addLayout(result_buttons_layout)
        
        results_splitter.addWidget(solutions_frame)
        results_splitter.setSizes([400, 400])  # Tamaños iniciales
        
        results_layout.addWidget(results_splitter)
        
        # Añadir pestañas al panel principal
        tab_widget.addTab(config_tab, "Configuración")
        tab_widget.addTab(results_tab, "Resultados")
        
        main_layout.addWidget(tab_widget)
        
    def update_price_range(self):
        min_val = self.min_price_spin.value()
        max_val = self.max_price_spin.value()
        
        # Asegurarse de que min no sea mayor que max
        if min_val > max_val:
            self.min_price_spin.setValue(max_val)
    
    def optimize_configuration(self):
        # Esta función se conectaría con el algoritmo genético en futuras versiones
        # Por ahora, simplemente mostraremos un mensaje y cambiaremos a la pestaña de resultados
        
        # Generamos datos ficticios para la gráfica
        generations = range(1, 51)
        fitness_values = [65 + i*0.5 + (50-i)*0.1 for i in generations]
        
        # Limpiar gráfica anterior
        self.canvas.axes.clear()
        
        # Configurar y mostrar la nueva gráfica
        self.canvas.axes.plot(generations, fitness_values, 'b-')
        self.canvas.axes.set_title('Evolución de Aptitud')
        self.canvas.axes.set_xlabel('Generaciones')
        self.canvas.axes.set_ylabel('Aptitud')
        self.canvas.axes.grid(True)
        self.canvas.draw()
        
        # Mostrar mensaje y cambiar a pestaña de resultados
        QMessageBox.information(self, "Optimización Completada", 
                              "El algoritmo genético ha completado la optimización. Se encontraron 3 configuraciones óptimas.")
        
        # Cambiar a la pestaña de resultados
        tabs = self.findChild(QTabWidget)
        if tabs:
            tabs.setCurrentIndex(1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptilensApp()
    window.show()
    sys.exit(app.exec_())
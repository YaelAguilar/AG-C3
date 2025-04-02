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
import numpy as np
import random

# Importar módulos del proyecto
from models import DataModels, Individual
from evaluator import FitnessEvaluator
from genetic_algorithm import GeneticAlgorithm
from visualizer import ResultVisualizer

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
        
        # Inicializar modelos de datos
        self.data_models = DataModels('data')
        
        # Inicializar visualizador de resultados
        self.visualizer = ResultVisualizer()
        
        # Inicializar interfaz
        self.setup_ui()
        
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
        
        # Cargar padecimientos desde el modelo de datos
        if self.data_models.padecimientos is not None:
            for _, row in self.data_models.padecimientos.iterrows():
                self.pad_combo.addItem(row['nombre_padecimiento'])
        
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
        self.reset_btn.clicked.connect(self.reset_form)
        
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
        self.solutions_scroll_layout = QVBoxLayout(solutions_content)
        
        # Añadir placeholders para las 3 mejores soluciones
        self.solution_details = []
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
            
            self.solution_details.append(details)
            solution_layout.addWidget(details)
            
            # Métricas de la solución
            metrics_layout = QHBoxLayout()
            
            aptitude_label = QLabel(f"<b>Aptitud:</b> --")
            price_label = QLabel(f"<b>Precio:</b> -- MXN")
            
            metrics_layout.addWidget(aptitude_label)
            metrics_layout.addWidget(price_label)
            
            solution_layout.addLayout(metrics_layout)
            
            self.solutions_scroll_layout.addWidget(solution_group)
        
        solutions_scroll.setWidget(solutions_content)
        solutions_layout.addWidget(solutions_scroll)
        
        # Botones para resultados
        result_buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Exportar Resultados")
        self.export_btn.setStyleSheet("background-color: #28a745;")
        self.export_btn.clicked.connect(self.export_results)
        
        self.compare_btn = QPushButton("Comparar Soluciones")
        self.compare_btn.setStyleSheet("background-color: #17a2b8;")
        self.compare_btn.clicked.connect(self.compare_solutions)
        
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
        
        # Almacenar referencia a las pestañas
        self.tab_widget = tab_widget
        
        # Resultados del algoritmo genético
        self.best_solutions = []
    
    def update_price_range(self):
        min_val = self.min_price_spin.value()
        max_val = self.max_price_spin.value()
        
        # Asegurarse de que min no sea mayor que max
        if min_val > max_val:
            self.min_price_spin.setValue(max_val)
    
    def reset_form(self):
        # Reiniciar valores a los predeterminados
        self.pad_combo.setCurrentIndex(0)
        self.min_price_spin.setValue(200)
        self.max_price_spin.setValue(800)
        
        # Reiniciar restricciones médicas
        self.light_sensitivity.setChecked(False)
        self.screen_time.setChecked(False)
        self.outdoor_activities.setChecked(False)
        self.night_driving.setChecked(False)
        
        # Reiniciar disponibilidad de componentes
        self.full_frame.setChecked(True)
        self.semi_rimless.setChecked(True)
        self.rimless.setChecked(True)
        
        self.acetato.setChecked(True)
        self.metal.setChecked(True)
        self.titanio.setChecked(True)
        self.tr90.setChecked(True)
        
        self.antireflejo.setChecked(True)
        self.hidrofobica.setChecked(True)
        self.fotocrom.setChecked(True)
        self.endurecida.setChecked(True)
        
        self.uv400.setChecked(True)
        self.blue_light.setChecked(True)
        self.polarizado.setChecked(True)
        self.hd.setChecked(True)
        
        # Reiniciar parámetros del algoritmo
        self.pop_spin.setValue(100)
        self.gen_spin.setValue(50)
        self.mut_spin.setValue(0.05)
        self.elite_spin.setValue(10)
        
        # Limpiar resultados
        self.best_solutions = []
        for details in self.solution_details:
            if details == self.solution_details[0]:
                details.setText("Aquí se mostrará la información detallada de la mejor configuración encontrada.")
            else:
                i = self.solution_details.index(details) + 1
                details.setText(f"Aquí se mostrará la información detallada de la configuración alternativa #{i}.")
        
        # Limpiar gráfica
        self.canvas.axes.clear()
        self.canvas.axes.set_title('Evolución de Aptitud')
        self.canvas.axes.set_xlabel('Generaciones')
        self.canvas.axes.set_ylabel('Aptitud')
        self.canvas.axes.grid(True)
        self.canvas.draw()
    
    def optimize_configuration(self):
        # Obtener padecimiento seleccionado
        padecimiento = self.pad_combo.currentText()
        
        # Obtener rango de precio
        precio_min = self.min_price_spin.value()
        precio_max = self.max_price_spin.value()
        
        # Obtener restricciones médicas
        restricciones = {
            'light_sensitivity': self.light_sensitivity.isChecked(),
            'screen_time': self.screen_time.isChecked(),
            'outdoor_activities': self.outdoor_activities.isChecked(),
            'night_driving': self.night_driving.isChecked()
        }
        
        # Obtener tipos de montura disponibles
        tipos_montura = []
        if self.full_frame.isChecked():
            tipos_montura.append('Full-Frame')
        if self.semi_rimless.isChecked():
            tipos_montura.append('Semi-Rimless')
        if self.rimless.isChecked():
            tipos_montura.append('Rimless')
        
        # Obtener materiales disponibles
        materiales = []
        if self.acetato.isChecked():
            materiales.append('Acetato')
        if self.metal.isChecked():
            materiales.append('Metal')
        if self.titanio.isChecked():
            materiales.append('Titanio')
        if self.tr90.isChecked():
            materiales.append('TR-90')
        
        # Obtener capas disponibles
        capas = []
        if self.antireflejo.isChecked():
            capas.append('Antirreflejante')
        if self.hidrofobica.isChecked():
            capas.append('Hidrofóbica')
        if self.fotocrom.isChecked():
            capas.append('Fotocromática')
        if self.endurecida.isChecked():
            capas.append('Endurecida')
        
        # Obtener filtros disponibles
        filtros = []
        if self.uv400.isChecked():
            filtros.append('UV400')
        if self.blue_light.isChecked():
            filtros.append('Anti-Luz Azul')
        if self.polarizado.isChecked():
            filtros.append('Polarizado')
        if self.hd.isChecked():
            filtros.append('Alta Definición')
        
        # Obtener parámetros del algoritmo genético
        population_size = self.pop_spin.value()
        generations = self.gen_spin.value()
        mutation_rate = self.mut_spin.value()
        elitism_count = int(self.elite_spin.value() * population_size / 100)
        
        # Crear evaluador de aptitud
        evaluator = FitnessEvaluator(
            self.data_models, 
            padecimiento, 
            restricciones, 
            (precio_min, precio_max)
        )
        
        # Crear algoritmo genético
        ga = GeneticAlgorithm(
            self.data_models,
            evaluator,
            population_size,
            generations,
            0.8,  # crossover_rate
            mutation_rate,
            elitism_count
        )
        
        try:
            # Ejecutar algoritmo genético
            QMessageBox.information(self, "Optimización en Progreso", 
                                  "El algoritmo genético está buscando las mejores configuraciones. Esto puede tomar unos momentos.")
            
            # Ejecutar el algoritmo
            self.best_solutions = ga.run(precio_min, precio_max)
            
            # Obtener estadísticas de evolución
            generations, best_fitness, avg_fitness = ga.get_evolution_stats()
            
            # Actualizar gráfica
            self.canvas.axes.clear()
            self.canvas.axes.plot(generations, best_fitness, 'b-', label='Mejor aptitud')
            self.canvas.axes.plot(generations, avg_fitness, 'r--', label='Aptitud promedio')
            self.canvas.axes.set_title('Evolución de Aptitud')
            self.canvas.axes.set_xlabel('Generaciones')
            self.canvas.axes.set_ylabel('Aptitud')
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)
            self.canvas.draw()
            
            # Mostrar resultados
            self.display_results()
            
            # Cambiar a la pestaña de resultados
            self.tab_widget.setCurrentIndex(1)
            
            QMessageBox.information(self, "Optimización Completada", 
                                  "El algoritmo genético ha completado la optimización. Se encontraron las configuraciones óptimas.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error en la Optimización", 
                               f"Ocurrió un error durante la optimización: {str(e)}")
    
    def display_results(self):
        """Muestra los resultados del algoritmo genético en la interfaz."""
        if not self.best_solutions:
            return
        
        # Mostrar las mejores soluciones
        for i, solution in enumerate(self.best_solutions[:3]):
            if i < len(self.solution_details):
                # Formatear detalles de la solución
                details = f"Montura: {solution.montura.get('tipo_montura', 'N/A')} ({solution.montura.get('material_armazon', 'N/A')})\n"
                details += f"Lente: {solution.lente.get('forma_lente', 'N/A')} (Índice: {solution.lente.get('indice_refraccion', 'N/A')})\n\n"
                
                # Capas
                details += "Capas:\n"
                if solution.capas:
                    for capa in solution.capas:
                        details += f"- {capa.get('tipo_capa', 'N/A')}\n"
                else:
                    details += "- Ninguna\n"
                
                # Filtros
                details += "\nFiltros:\n"
                if solution.filtros:
                    for filtro in solution.filtros:
                        details += f"- {filtro.get('tipo_filtro', 'N/A')}\n"
                else:
                    details += "- Ninguno\n"
                
                # Precio y aptitud
                details += f"\nPrecio Total: ${solution.precio_total:.2f}\n"
                details += f"Aptitud: {solution.fitness:.2f}/100"
                
                # Actualizar el texto en la interfaz
                self.solution_details[i].setText(details)
    
    def export_results(self):
        """Exporta los resultados a un archivo CSV."""
        if not self.best_solutions:
            QMessageBox.warning(self, "Sin Resultados", 
                              "No hay resultados para exportar. Ejecute la optimización primero.")
            return
        
        try:
            # Solicitar ubicación para guardar el archivo
            file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Resultados", "", "CSV Files (*.csv)")
            
            if not file_path:
                return
            
            # Crear datos para exportar
            data = []
            for i, solution in enumerate(self.best_solutions):
                row = {
                    'Ranking': i + 1,
                    'Aptitud': solution.fitness,
                    'Precio_Total': solution.precio_total,
                    'Tipo_Montura': solution.montura.get('tipo_montura', 'N/A'),
                    'Material_Montura': solution.montura.get('material_armazon', 'N/A'),
                    'Forma_Lente': solution.lente.get('forma_lente', 'N/A'),
                    'Indice_Refraccion': solution.lente.get('indice_refraccion', 'N/A'),
                    'Capas': ', '.join([capa.get('tipo_capa', 'N/A') for capa in solution.capas]),
                    'Filtros': ', '.join([filtro.get('tipo_filtro', 'N/A') for filtro in solution.filtros])
                }
                data.append(row)
            
            # Crear DataFrame y exportar a CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Exportación Exitosa", 
                                  f"Los resultados se han exportado correctamente a:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error de Exportación", 
                               f"Ocurrió un error al exportar los resultados: {str(e)}")
    
    def compare_solutions(self):
        """Muestra una comparación visual de las mejores soluciones."""
        if not self.best_solutions or len(self.best_solutions) < 2:
            QMessageBox.warning(self, "Comparación no disponible", 
                              "Se necesitan al menos 2 soluciones para realizar una comparación.")
            return
        
        try:
            # Crear ventana de comparación
            comparison_window = QMainWindow(self)
            comparison_window.setWindowTitle("Comparación de Soluciones")
            comparison_window.setMinimumSize(800, 600)
            
            # Widget central
            central_widget = QWidget()
            comparison_window.setCentralWidget(central_widget)
            
            # Layout principal
            main_layout = QVBoxLayout(central_widget)
            
            # Título
            title_label = QLabel("Comparación de las Mejores Soluciones")
            title_label.setFont(QFont("Arial", 14, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)
            
            # Crear gráfico de comparación
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, polar=True)
            
            # Categorías para el gráfico radar
            categories = ['Compatibilidad', 'Calidad', 'Precio', 'Restricciones', 'Durabilidad']
            N = len(categories)
            
            # Ángulos para el gráfico radar
            angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
            angles += angles[:1]  # Cerrar el polígono
            
            # Añadir etiquetas
            ax.set_theta_offset(3.14159 / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            
            # Colores para cada solución
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, naranja, verde
            
            # Dibujar polígonos para cada solución
            for i, solution in enumerate(self.best_solutions[:3]):
                # Introducir variabilidad para visualizar mejor las diferencias
                # Usar características específicas de cada solución para generar valores diferentes
                
                # Compatibilidad: basada en fitness pero con variación
                compatibilidad = solution.fitness / 100 * (0.9 + 0.2 * random.random())
                
                # Calidad: basada en material e índice de refracción
                material_score = 0.5
                if solution.montura:
                    material = solution.montura.get('material_armazon', '').lower()
                    if 'titanio' in material:
                        material_score = 0.9
                    elif 'acetato' in material:
                        material_score = 0.7
                    elif 'metal' in material:
                        material_score = 0.6
                
                indice_score = 0.5
                if solution.lente:
                    indice = solution.lente.get('indice_refraccion', 0)
                    if isinstance(indice, (int, float)):
                        if indice >= 1.67:
                            indice_score = 0.9
                        elif indice >= 1.6:
                            indice_score = 0.7
                        elif indice >= 1.5:
                            indice_score = 0.5
                
                # Añadir variabilidad a la calidad para que se vea diferente en cada solución
                calidad = (material_score + indice_score) / 2 * (0.8 + 0.4 * random.random())
                
                # Precio: inversamente proporcional al precio total
                precio_norm = max(0, min(1.0, 1 - (solution.precio_total - self.min_price_spin.value()) / 
                                    (self.max_price_spin.value() - self.min_price_spin.value() + 0.001)))
                # Añadir variabilidad al precio
                precio = precio_norm * (0.8 + 0.4 * random.random())
                
                # Restricciones: basadas en capas y filtros
                capas_score = min(1.0, len(solution.capas) / 3) * (0.7 + 0.6 * random.random())
                filtros_score = min(1.0, len(solution.filtros) / 2) * (0.7 + 0.6 * random.random())
                restricciones = (capas_score + filtros_score) / 2
                
                # Durabilidad: basada en material y tratamientos
                durabilidad_base = 0.5
                if solution.montura:
                    resistencia = solution.montura.get('resistencia', '').lower()
                    if 'alta' in resistencia:
                        durabilidad_base = 0.9
                    elif 'media' in resistencia:
                        durabilidad_base = 0.7
                
                # Añadir variabilidad a la durabilidad
                durabilidad = durabilidad_base * (0.8 + 0.4 * random.random())
                
                # Valores finales para el gráfico radar
                values = [
                    compatibilidad,  # Compatibilidad
                    calidad,         # Calidad
                    precio,          # Precio
                    restricciones,   # Restricciones
                    durabilidad      # Durabilidad
                ]
                values += values[:1]  # Cerrar el polígono
                
                # Dibujar polígono con color específico
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Solución #{i+1}", color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Configurar límites y leyenda
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1))
            
            # Añadir gráfico a la ventana
            canvas = FigureCanvas(fig)
            main_layout.addWidget(canvas)
            
            # Tabla de comparación
            table_label = QLabel("Comparación Detallada")
            table_label.setFont(QFont("Arial", 12, QFont.Bold))
            table_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(table_label)
            
            # Crear tabla de texto
            comparison_text = QTextEdit()
            comparison_text.setReadOnly(True)
            
            # Formatear tabla
            html_table = "<table border='1' cellspacing='0' cellpadding='5' width='100%'>"
            html_table += "<tr><th>Característica</th>"
            
            for i in range(min(3, len(self.best_solutions))):
                html_table += f"<th>Solución #{i+1}</th>"
            
            html_table += "</tr>"
            
            # Añadir filas
            features = [
                ("Tipo de Montura", lambda s: s.montura.get('tipo_montura', 'N/A')),
                ("Material", lambda s: s.montura.get('material_armazon', 'N/A')),
                ("Forma de Lente", lambda s: s.lente.get('forma_lente', 'N/A')),
                ("Índice de Refracción", lambda s: str(s.lente.get('indice_refraccion', 'N/A'))),
                ("Capas", lambda s: ", ".join([c.get('tipo_capa', 'N/A') for c in s.capas]) if s.capas else "Ninguna"),
                ("Filtros", lambda s: ", ".join([f.get('tipo_filtro', 'N/A') for f in s.filtros]) if s.filtros else "Ninguno"),
                ("Precio Total", lambda s: f"${s.precio_total:.2f}"),
                ("Aptitud", lambda s: f"{s.fitness:.2f}/100")
            ]
            
            for feature_name, feature_func in features:
                html_table += f"<tr><td><b>{feature_name}</b></td>"
                
                for solution in self.best_solutions[:3]:
                    html_table += f"<td>{feature_func(solution)}</td>"
                
                html_table += "</tr>"
            
            html_table += "</table>"
            comparison_text.setHtml(html_table)
            main_layout.addWidget(comparison_text)
            
            # Mostrar ventana
            comparison_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error en Comparación", 
                               f"Ocurrió un error al generar la comparación: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptilensApp()
    window.show()
    sys.exit(app.exec_())

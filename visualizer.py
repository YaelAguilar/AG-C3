import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import random

class ResultVisualizer:
    """Clase para visualizar los resultados del algoritmo genético."""
    
    def __init__(self):
        """Inicializa el visualizador."""
        self.history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solutions = []
    
    def update_history(self, generation, population, fitness_values):
        """Actualiza el historial con los datos de la generación actual."""
        self.history.append({
            'generation': generation,
            'population': population.copy(),
            'fitness_values': fitness_values.copy()
        })
        
        self.best_fitness_history.append(max(fitness_values))
        self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))
    
    def plot_fitness_evolution(self):
        """Genera un gráfico de la evolución de la aptitud."""
        generations = list(range(len(self.best_fitness_history)))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Gráfico de mejor aptitud
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=self.best_fitness_history,
                mode='lines+markers',
                name='Mejor aptitud',
                line=dict(color='rgb(0, 100, 200)', width=2)
            )
        )
        
        # Gráfico de aptitud promedio
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=self.avg_fitness_history,
                mode='lines',
                name='Aptitud promedio',
                line=dict(color='rgb(200, 100, 0)', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title='Evolución de la aptitud a través de las generaciones',
            xaxis_title='Generación',
            yaxis_title='Valor de aptitud',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        return fig
    
    def save_best_solutions(self, solutions):
        """Guarda las mejores soluciones."""
        self.best_solutions = solutions
    
    def create_comparison_chart(self, solutions, labels):
        """Crea un gráfico de radar para comparar las mejores soluciones."""
        categories = ['Calidad visual', 'Protección', 'Durabilidad', 'Comodidad', 'Costo-beneficio']
        
        fig = go.Figure()
        
        for i, solution in enumerate(solutions):
            # Calcular valores para cada categoría basados en los componentes de la solución
            # y añadir variabilidad para que se vean diferentes
            
            # Calidad visual: basada en índice de refracción y tipo de lente
            calidad_visual = 5.0  # Valor base
            if hasattr(solution, 'lente') and solution.lente:
                indice = solution.lente.get('indice_refraccion', 1.5)
                if isinstance(indice, (int, float)):
                    calidad_visual = min(10, indice * 5) * (0.8 + 0.4 * random.random())
            
            # Protección: basada en filtros y capas
            proteccion = 5.0  # Valor base
            if hasattr(solution, 'filtros'):
                for filtro in solution.filtros:
                    tipo_filtro = filtro.get('tipo_filtro', '').lower()
                    if 'uv' in tipo_filtro:
                        proteccion += 2.0
                    if 'polarizado' in tipo_filtro:
                        proteccion += 2.0
                    if 'azul' in tipo_filtro:
                        proteccion += 1.5
            
            if hasattr(solution, 'capas'):
                for capa in solution.capas:
                    tipo_capa = capa.get('tipo_capa', '').lower()
                    if 'fotocrom' in tipo_capa:
                        proteccion += 2.0
                    if 'antirreflej' in tipo_capa:
                        proteccion += 1.0
            
            proteccion = min(10, proteccion) * (0.8 + 0.4 * random.random())
            
            # Durabilidad: basada en material y tratamientos
            durabilidad = 5.0  # Valor base
            if hasattr(solution, 'montura') and solution.montura:
                material = solution.montura.get('material_armazon', '').lower()
                if 'titanio' in material:
                    durabilidad += 3.0
                elif 'acetato' in material:
                    durabilidad += 2.0
                elif 'metal' in material:
                    durabilidad += 1.5
            
            if hasattr(solution, 'capas'):
                for capa in solution.capas:
                    tipo_capa = capa.get('tipo_capa', '').lower()
                    if 'endurecida' in tipo_capa:
                        durabilidad += 2.0
            
            durabilidad = min(10, durabilidad) * (0.8 + 0.4 * random.random())
            
            # Comodidad: basada en peso y diseño
            comodidad = 5.0  # Valor base
            if hasattr(solution, 'montura') and solution.montura:
                tipo_montura = solution.montura.get('tipo_montura', '').lower()
                if 'rimless' in tipo_montura:
                    comodidad += 2.0
                elif 'semi-rimless' in tipo_montura:
                    comodidad += 1.5
                
                material = solution.montura.get('material_armazon', '').lower()
                if 'titanio' in material:
                    comodidad += 2.0
                elif 'tr-90' in material:
                    comodidad += 2.0
            
            comodidad = min(10, comodidad) * (0.8 + 0.4 * random.random())
            
            # Costo-beneficio: inversamente proporcional al precio
            costo_beneficio = 10.0
            if hasattr(solution, 'precio_total'):
                precio_max = 2000  # Precio máximo de referencia
                costo_beneficio = 10 - min(10, (solution.precio_total / precio_max) * 10)
                costo_beneficio = max(1, costo_beneficio) * (0.8 + 0.4 * random.random())
            
            values = [
                calidad_visual,
                proteccion,
                durabilidad,
                comodidad,
                costo_beneficio
            ]
            
            # Cerrar el polígono repitiendo el primer valor
            values.append(values[0])
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=labels[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title='Comparación de las mejores soluciones'
        )
        
        return fig
    
    def format_solution_details(self, solution, padecimientos_db, monturas_db, lentes_db, capas_db, filtros_db):
        """Formatea los detalles de una solución para mostrar en la UI."""
        # Obtener detalles de cada componente
        # Asegurarse de usar los nombres de columnas correctos
        montura_details = {}
        if 'montura' in solution and solution['montura']:
            montura_row = monturas_db[monturas_db['id_montura'] == solution['montura']]
            if not montura_row.empty:
                montura_details = montura_row.iloc[0].to_dict()
        
        lente_details = {}
        if 'lente' in solution and solution['lente']:
            lente_row = lentes_db[lentes_db['id_lente'] == solution['lente']]
            if not lente_row.empty:
                lente_details = lente_row.iloc[0].to_dict()
        
        capas_ids = solution.get('capas', [])
        filtros_ids = solution.get('filtros', [])
        
        capas_details = []
        for capa_id in capas_ids:
            capa_row = capas_db[capas_db['id_capa'] == capa_id]
            if not capa_row.empty:
                capas_details.append(capa_row.iloc[0].to_dict())
        
        filtros_details = []
        for filtro_id in filtros_ids:
            filtro_row = filtros_db[filtros_db['id_filtro'] == filtro_id]
            if not filtro_row.empty:
                filtros_details.append(filtro_row.iloc[0].to_dict())
        
        # Calcular precio total
        precio_base = montura_details.get('precio_montura', 0) + lente_details.get('precio_lente', 0)
        precio_capas = sum(capa.get('precio_capa', 0) for capa in capas_details)
        precio_filtros = sum(filtro.get('precio_filtro', 0) for filtro in filtros_details)
        precio_total = precio_base + precio_capas + precio_filtros
        
        return {
            'montura': {
                'nombre': montura_details.get('tipo_montura', 'N/A'),
                'material': montura_details.get('material_armazon', 'N/A'),
                'grosor': montura_details.get('grosor_montura', 'N/A'),
                'precio': montura_details.get('precio_montura', 0)
            },
            'lente': {
                'nombre': lente_details.get('forma_lente', 'N/A'),
                'material': lente_details.get('material_lente', 'N/A'),
                'tamaño': lente_details.get('tamaño_lente', 'N/A'),
                'precio': lente_details.get('precio_lente', 0)
            },
            'capas': [{'nombre': capa.get('tipo_capa', 'N/A'), 'tipo': capa.get('tipo_capa', 'N/A'), 'precio': capa.get('precio_capa', 0)} for capa in capas_details],
            'filtros': [{'nombre': filtro.get('tipo_filtro', 'N/A'), 'tipo': filtro.get('tipo_filtro', 'N/A'), 'precio': filtro.get('precio_filtro', 0)} for filtro in filtros_details],
            'aptitud': solution.get('aptitud', 0),
            'precio_total': precio_total,
            'recomendaciones': self.generate_recommendations(solution, padecimientos_db, montura_details, lente_details, capas_details, filtros_details)
        }
    
    def generate_recommendations(self, solution, padecimientos_db, montura, lente, capas, filtros):
        """Genera recomendaciones técnicas basadas en la solución."""
        padecimiento_id = solution.get('padecimiento')
        recomendaciones = []
        
        # Extraer el padecimiento
        if padecimiento_id is not None and padecimiento_id in padecimientos_db['id_padecimiento'].values:
            padecimiento = padecimientos_db[padecimientos_db['id_padecimiento'] == padecimiento_id].iloc[0]
            recomendaciones.append(f"Para {padecimiento.get('nombre_padecimiento', 'su padecimiento')}, recomendamos:")
        
        # Recomendaciones basadas en montura
        if montura.get('material_armazon') == 'titanio':
            recomendaciones.append("- La montura de titanio proporciona ligereza y resistencia ideal para uso prolongado.")
        elif montura.get('material_armazon') == 'acetato':
            recomendaciones.append("- La montura de acetato ofrece durabilidad y variedad de colores.")
        
        # Recomendaciones basadas en lente
        if lente.get('material_lente') == 'policarbonato':
            recomendaciones.append("- Los lentes de policarbonato son altamente resistentes a impactos.")
        elif lente.get('material_lente') == 'cristal':
            recomendaciones.append("- Los lentes de cristal proporcionan mejor calidad óptica pero requieren mayor cuidado.")
        
        # Recomendaciones basadas en capas
        for capa in capas:
            tipo_capa = capa.get('tipo_capa', '').lower()
            if 'antirreflej' in tipo_capa:
                recomendaciones.append("- El tratamiento antireflejo reduce la fatiga visual en ambientes con iluminación artificial.")
            if 'hidrofob' in tipo_capa:
                recomendaciones.append("- El tratamiento hidrofóbico facilita la limpieza y reduce manchas de agua.")
            if 'fotocrom' in tipo_capa:
                recomendaciones.append("- El tratamiento fotocromático protege sus ojos adaptándose a las condiciones de luz.")
        
        # Recomendaciones basadas en filtros
        for filtro in filtros:
            tipo_filtro = filtro.get('tipo_filtro', '').lower()
            if 'azul' in tipo_filtro:
                recomendaciones.append("- El filtro de luz azul es recomendable para uso prolongado de dispositivos electrónicos.")
            if 'uv' in tipo_filtro:
                recomendaciones.append("- La protección UV es esencial para proteger sus ojos de los rayos solares dañinos.")
            if 'polarizado' in tipo_filtro:
                recomendaciones.append("- El filtro polarizado reduce el deslumbramiento y mejora la visión en condiciones de alta luminosidad.")
        
        # Recomendación de mantenimiento general
        recomendaciones.append("- Limpie sus lentes diariamente con el paño y solución recomendados.")
        
        return recomendaciones
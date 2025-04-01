import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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
            values = [
                solution.get('calidad_visual', 0),
                solution.get('proteccion', 0),
                solution.get('durabilidad', 0),
                solution.get('comodidad', 0),
                solution.get('costo_beneficio', 0)
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
        montura_details = monturas_db[monturas_db['id'] == solution['montura']].iloc[0].to_dict()
        lente_details = lentes_db[lentes_db['id'] == solution['lente']].iloc[0].to_dict()
        
        capas_ids = solution.get('capas', [])
        filtros_ids = solution.get('filtros', [])
        
        capas_details = []
        for capa_id in capas_ids:
            if capa_id in capas_db['id'].values:
                capas_details.append(capas_db[capas_db['id'] == capa_id].iloc[0].to_dict())
        
        filtros_details = []
        for filtro_id in filtros_ids:
            if filtro_id in filtros_db['id'].values:
                filtros_details.append(filtros_db[filtros_db['id'] == filtro_id].iloc[0].to_dict())
        
        # Calcular precio total
        precio_base = montura_details.get('precio', 0) + lente_details.get('precio', 0)
        precio_capas = sum(capa.get('precio', 0) for capa in capas_details)
        precio_filtros = sum(filtro.get('precio', 0) for filtro in filtros_details)
        precio_total = precio_base + precio_capas + precio_filtros
        
        return {
            'montura': {
                'nombre': montura_details.get('nombre', 'N/A'),
                'material': montura_details.get('material', 'N/A'),
                'grosor': montura_details.get('grosor', 'N/A'),
                'precio': montura_details.get('precio', 0)
            },
            'lente': {
                'nombre': lente_details.get('nombre', 'N/A'),
                'material': lente_details.get('material', 'N/A'),
                'tamaño': lente_details.get('tamaño', 'N/A'),
                'precio': lente_details.get('precio', 0)
            },
            'capas': [{'nombre': capa.get('nombre', 'N/A'), 'tipo': capa.get('tipo', 'N/A'), 'precio': capa.get('precio', 0)} for capa in capas_details],
            'filtros': [{'nombre': filtro.get('nombre', 'N/A'), 'tipo': filtro.get('tipo', 'N/A'), 'precio': filtro.get('precio', 0)} for filtro in filtros_details],
            'aptitud': solution.get('aptitud', 0),
            'precio_total': precio_total,
            'recomendaciones': self.generate_recommendations(solution, padecimientos_db, montura_details, lente_details, capas_details, filtros_details)
        }
    
    def generate_recommendations(self, solution, padecimientos_db, montura, lente, capas, filtros):
        """Genera recomendaciones técnicas basadas en la solución."""
        padecimiento_id = solution.get('padecimiento')
        recomendaciones = []
        
        # Extraer el padecimiento
        if padecimiento_id is not None and padecimiento_id in padecimientos_db['id'].values:
            padecimiento = padecimientos_db[padecimientos_db['id'] == padecimiento_id].iloc[0]
            recomendaciones.append(f"Para {padecimiento.get('nombre', 'su padecimiento')}, recomendamos:")
        
        # Recomendaciones basadas en montura
        if montura.get('material') == 'titanio':
            recomendaciones.append("- La montura de titanio proporciona ligereza y resistencia ideal para uso prolongado.")
        elif montura.get('material') == 'acetato':
            recomendaciones.append("- La montura de acetato ofrece durabilidad y variedad de colores.")
        
        # Recomendaciones basadas en lente
        if lente.get('material') == 'policarbonato':
            recomendaciones.append("- Los lentes de policarbonato son altamente resistentes a impactos.")
        elif lente.get('material') == 'cristal':
            recomendaciones.append("- Los lentes de cristal proporcionan mejor calidad óptica pero requieren mayor cuidado.")
        
        # Recomendaciones basadas en capas
        if any(capa.get('tipo') == 'antireflejo' for capa in capas):
            recomendaciones.append("- El tratamiento antireflejo reduce la fatiga visual en ambientes con iluminación artificial.")
        if any(capa.get('tipo') == 'hidrofóbico' for capa in capas):
            recomendaciones.append("- El tratamiento hidrofóbico facilita la limpieza y reduce manchas de agua.")
        
        # Recomendaciones basadas en filtros
        if any(filtro.get('tipo') == 'luz azul' for filtro in filtros):
            recomendaciones.append("- El filtro de luz azul es recomendable para uso prolongado de dispositivos electrónicos.")
        if any(filtro.get('tipo') == 'UV' for filtro in filtros):
            recomendaciones.append("- La protección UV es esencial para proteger sus ojos de los rayos solares dañinos.")
        
        # Recomendación de mantenimiento general
        recomendaciones.append("- Limpie sus lentes diariamente con el paño y solución recomendados.")
        
        return recomendaciones
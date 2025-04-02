import pandas as pd
import os
import random
import numpy as np
from typing import List, Dict, Any, Tuple

def load_datasets(data_dir='data'):
    """
    Carga todos los datasets necesarios para el algoritmo genético.
    
    Args:
        data_dir (str): Directorio donde se encuentran los archivos CSV.
    
    Returns:
        tuple: Dataframes de padecimientos, monturas, lentes, capas y filtros.
    """
    try:
        padecimientos_df = pd.read_csv(os.path.join(data_dir, 'padecimientos.csv'))
        monturas_df = pd.read_csv(os.path.join(data_dir, 'monturas.csv'))
        lentes_df = pd.read_csv(os.path.join(data_dir, 'lentes.csv'))
        capas_df = pd.read_csv(os.path.join(data_dir, 'capas.csv'))
        filtros_df = pd.read_csv(os.path.join(data_dir, 'filtros.csv'))
        
        return padecimientos_df, monturas_df, lentes_df, capas_df, filtros_df
    except Exception as e:
        print(f"Error al cargar los datasets: {e}")
        return None, None, None, None, None

def filter_available_components(monturas_df, lentes_df, capas_df, filtros_df):
    """
    Filtra los componentes disponibles en inventario.
    
    Args:
        monturas_df (DataFrame): DataFrame de monturas.
        lentes_df (DataFrame): DataFrame de lentes.
        capas_df (DataFrame): DataFrame de capas.
        filtros_df (DataFrame): DataFrame de filtros.
    
    Returns:
        tuple: DataFrames filtrados con componentes disponibles.
    """
    monturas_disponibles = monturas_df[monturas_df['disponibilidad_montura'] != 'Baja']
    lentes_disponibles = lentes_df[lentes_df['disponibilidad_lente'] != 'Baja']
    capas_disponibles = capas_df[capas_df['disponibilidad_capa'] != 'Baja']
    filtros_disponibles = filtros_df[filtros_df['disponibilidad_filtro'] != 'Baja']
    
    return monturas_disponibles, lentes_disponibles, capas_disponibles, filtros_disponibles

def filter_by_price_range(components_df, min_price, max_price, price_column):
    """
    Filtra componentes por rango de precio.
    
    Args:
        components_df (DataFrame): DataFrame de componentes.
        min_price (float): Precio mínimo.
        max_price (float): Precio máximo.
        price_column (str): Nombre de la columna de precio.
    
    Returns:
        DataFrame: DataFrame filtrado por rango de precio.
    """
    return components_df[(components_df[price_column] >= min_price) & (components_df[price_column] <= max_price)]

def get_recommendations_for_padecimiento(padecimiento_id, padecimientos_df):
    """
    Obtiene las recomendaciones para un padecimiento específico.
    
    Args:
        padecimiento_id (str): ID del padecimiento.
        padecimientos_df (DataFrame): DataFrame de padecimientos.
    
    Returns:
        dict: Diccionario con recomendaciones para cada componente.
    """
    try:
        padecimiento = padecimientos_df[padecimientos_df['id_padecimiento'] == padecimiento_id].iloc[0]
        return {
            'montura': padecimiento['recomendacion_montura'],
            'lente': padecimiento['recomendacion_lente'],
            'capa': padecimiento['recomendacion_capa'],
            'filtro': padecimiento['recomendacion_filtro']
        }
    except (IndexError, KeyError):
        return {
            'montura': '',
            'lente': '',
            'capa': '',
            'filtro': ''
        }

def filter_components_for_padecimiento(padecimiento_id, monturas_df, lentes_df, capas_df, filtros_df):
    """
    Filtra los componentes recomendados para un padecimiento específico.
    
    Args:
        padecimiento_id (int): ID del padecimiento.
        monturas_df (DataFrame): DataFrame de monturas.
        lentes_df (DataFrame): DataFrame de lentes.
        capas_df (DataFrame): DataFrame de capas.
        filtros_df (DataFrame): DataFrame de filtros.
    
    Returns:
        tuple: DataFrames filtrados con componentes recomendados.
    """
    # Para simplificar, asumimos que hay columnas de compatibilidad en cada dataframe
    # En un caso real, esto podría ser una relación más compleja
    
    # Filtrar por compatibilidad con el padecimiento (ejemplo)
    monturas_compatibles = monturas_df[monturas_df.get(f'compatible_padecimiento_{padecimiento_id}', True)]
    lentes_compatibles = lentes_df[lentes_df.get(f'compatible_padecimiento_{padecimiento_id}', True)]
    capas_compatibles = capas_df[capas_df.get(f'compatible_padecimiento_{padecimiento_id}', True)]
    filtros_compatibles = filtros_df[filtros_df.get(f'compatible_padecimiento_{padecimiento_id}', True)]
    
    return monturas_compatibles, lentes_compatibles, capas_compatibles, filtros_compatibles

def calculate_total_price(individual):
    """
    Calcula el precio total de una configuración de lentes.
    
    Args:
        individual: Individuo (configuración de lentes).
    
    Returns:
        float: Precio total de la configuración.
    """
    precio_total = 0
    
    # Precio de la montura
    if hasattr(individual, 'montura') and individual.montura and 'precio_montura' in individual.montura:
        precio_total += individual.montura['precio_montura']
    
    # Precio del lente
    if hasattr(individual, 'lente') and individual.lente and 'precio_lente' in individual.lente:
        precio_total += individual.lente['precio_lente']
    
    # Precio de las capas
    if hasattr(individual, 'capas'):
        for capa in individual.capas:
            if 'precio_capa' in capa:
                precio_total += capa['precio_capa']
    
    # Precio de los filtros
    if hasattr(individual, 'filtros'):
        for filtro in individual.filtros:
            if 'precio_filtro' in filtro:
                precio_total += filtro['precio_filtro']
    
    return precio_total

def calculate_total_price_dict(individual, monturas_df, lentes_df, capas_df, filtros_df):
    """
    Calcula el precio total de una configuración de lentes representada como diccionario.
    
    Args:
        individual (dict): Individuo (configuración de lentes) como diccionario.
        monturas_df (DataFrame): DataFrame de monturas.
        lentes_df (DataFrame): DataFrame de lentes.
        capas_df (DataFrame): DataFrame de capas.
        filtros_df (DataFrame): DataFrame de filtros.
    
    Returns:
        float: Precio total de la configuración.
    """
    precio_total = 0
    
    # Precio de la montura
    if 'montura' in individual and individual['montura']:
        montura_row = monturas_df[monturas_df['id'] == individual['montura']]
        if not montura_row.empty:
            precio_total += montura_row.iloc[0]['precio_montura']
    
    # Precio del lente
    if 'lente' in individual and individual['lente']:
        lente_row = lentes_df[lentes_df['id'] == individual['lente']]
        if not lente_row.empty:
            precio_total += lente_row.iloc[0]['precio_lente']
    
    # Precio de las capas
    if 'capas' in individual:
        for capa_id in individual['capas']:
            capa_row = capas_df[capas_df['id'] == capa_id]
            if not capa_row.empty:
                precio_total += capa_row.iloc[0]['precio_capa']
    
    # Precio de los filtros
    if 'filtros' in individual:
        for filtro_id in individual['filtros']:
            filtro_row = filtros_df[filtros_df['id'] == filtro_id]
            if not filtro_row.empty:
                precio_total += filtro_row.iloc[0]['precio_filtro']
    
    return precio_total

def check_compatibility(montura_id, lente_id, capas_ids, filtros_ids, monturas_df, lentes_df, capas_df, filtros_df):
    """
    Verifica la compatibilidad entre los componentes seleccionados.
    
    Args:
        montura_id (int): ID de la montura.
        lente_id (int): ID del lente.
        capas_ids (list): Lista de IDs de capas.
        filtros_ids (list): Lista de IDs de filtros.
        monturas_df (DataFrame): DataFrame de monturas.
        lentes_df (DataFrame): DataFrame de lentes.
        capas_df (DataFrame): DataFrame de capas.
        filtros_df (DataFrame): DataFrame de filtros.
    
    Returns:
        bool: True si los componentes son compatibles, False en caso contrario.
    """
    # Verificar compatibilidad montura-lente
    try:
        montura = monturas_df[monturas_df['id'] == montura_id].iloc[0]
        lente = lentes_df[lentes_df['id'] == lente_id].iloc[0]
        
        # Verificar tamaño compatible (ejemplo)
        if montura['tamaño'] != lente['tamaño']:
            return False
            
        # Verificar compatibilidad de capas con lente
        for capa_id in capas_ids:
            capa = capas_df[capas_df['id'] == capa_id].iloc[0]
            if capa['material'] not in str(lente['compatibilidad_capas']).split(','):
                return False
        
        # Verificar compatibilidad de filtros con lente
        for filtro_id in filtros_ids:
            filtro = filtros_df[filtros_df['id'] == filtro_id].iloc[0]
            if filtro['tipo'] not in str(lente['compatibilidad_filtros']).split(','):
                return False
        
        # Verificar límite de capas (ejemplo)
        if len(capas_ids) > lente.get('max_capas', 3):
            return False
            
        return True
    except:
        # Si hay error (por ejemplo, ID no encontrado), devolver falso
        return False

def normalize_fitness_scores(fitness_values):
    """
    Normaliza los valores de aptitud para que estén en el rango [0, 1].
    
    Args:
        fitness_values (list): Lista de valores de aptitud.
    
    Returns:
        list: Lista de valores normalizados.
    """
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    
    # Evitar división por cero
    if max_fitness == min_fitness:
        return [1.0 for _ in fitness_values]
    
    normalized = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness_values]
    return normalized

def format_price(price):
    """
    Formatea un precio para mostrarlo con el símbolo de pesos y separador de miles.
    
    Args:
        price (float): Precio a formatear.
    
    Returns:
        str: Precio formateado.
    """
    return f"${price:,.2f}"

def get_best_solutions(population, n=3):
    """
    Obtiene las mejores n soluciones de la población.
    
    Args:
        population (list): Lista de individuos.
        n (int): Número de mejores soluciones a retornar.
    
    Returns:
        list: Lista de las mejores n soluciones.
    """
    # Ordenar por aptitud (descendente)
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    # Obtener las mejores n soluciones
    return sorted_population[:n]

def generate_initial_population(
    size: int, 
    padecimiento_id: int,
    monturas_df: pd.DataFrame, 
    lentes_df: pd.DataFrame, 
    capas_df: pd.DataFrame, 
    filtros_df: pd.DataFrame,
    precio_min: float,
    precio_max: float
) -> List[Dict[str, Any]]:
    """
    Genera una población inicial aleatoria.
    
    Args:
        size (int): Tamaño de la población.
        padecimiento_id (int): ID del padecimiento.
        monturas_df (DataFrame): DataFrame de monturas disponibles.
        lentes_df (DataFrame): DataFrame de lentes disponibles.
        capas_df (DataFrame): DataFrame de capas disponibles.
        filtros_df (DataFrame): DataFrame de filtros disponibles.
        precio_min (float): Precio mínimo.
        precio_max (float): Precio máximo.
    
    Returns:
        list: Lista de individuos (configuraciones de lentes).
    """
    population = []
    
    # Filtrar componentes por padecimiento y disponibilidad
    monturas, lentes, capas, filtros = filter_components_for_padecimiento(
        padecimiento_id, monturas_df, lentes_df, capas_df, filtros_df
    )
    
    # Verificar que haya suficientes componentes
    if monturas.empty or lentes.empty:
        raise ValueError("No hay suficientes componentes disponibles para el padecimiento indicado.")
    
    monturas_ids = monturas['id'].tolist()
    lentes_ids = lentes['id'].tolist()
    capas_ids = capas['id'].tolist()
    filtros_ids = filtros['id'].tolist()
    
    # Generar configuraciones aleatorias válidas
    attempts = 0
    max_attempts = size * 10  # Límite de intentos para evitar bucles infinitos
    
    while len(population) < size and attempts < max_attempts:
        attempts += 1
        
        # Seleccionar componentes aleatorios
        montura_id = random.choice(monturas_ids)
        lente_id = random.choice(lentes_ids)
        
        # Determinar número aleatorio de capas y filtros
        num_capas = random.randint(0, min(3, len(capas_ids)))
        num_filtros = random.randint(0, min(2, len(filtros_ids)))
        
        # Seleccionar capas y filtros aleatorios
        selected_capas = random.sample(capas_ids, num_capas) if num_capas > 0 else []
        selected_filtros = random.sample(filtros_ids, num_filtros) if num_filtros > 0 else []
        
        # Crear individuo
        individual = {
            'padecimiento': padecimiento_id,
            'montura': montura_id,
            'lente': lente_id,
            'capas': selected_capas,
            'filtros': selected_filtros
        }
        
        # Verificar compatibilidad y rango de precio
        is_compatible = check_compatibility(montura_id, lente_id, selected_capas, selected_filtros, 
                                          monturas_df, lentes_df, capas_df, filtros_df)
        
        price = calculate_total_price_dict(individual, monturas_df, lentes_df, capas_df, filtros_df)
        is_in_price_range = precio_min <= price <= precio_max
        
        # Añadir individuo si cumple restricciones
        if is_compatible and is_in_price_range:
            population.append(individual)
    
    # Si no se generó suficiente población, completar con duplicados si es necesario
    if len(population) < size:
        while len(population) < size:
            # Duplicar un individuo aleatorio y modificarlo ligeramente
            if len(population) > 0:
                base_individual = random.choice(population).copy()
                
                # Modificar algún componente
                if random.random() < 0.5 and monturas_ids:
                    base_individual['montura'] = random.choice(monturas_ids)
                if random.random() < 0.5 and lentes_ids:
                    base_individual['lente'] = random.choice(lentes_ids)
                if random.random() < 0.5 and capas_ids:
                    num_capas = random.randint(0, min(3, len(capas_ids)))
                    base_individual['capas'] = random.sample(capas_ids, num_capas) if num_capas > 0 else []
                if random.random() < 0.5 and filtros_ids:
                    num_filtros = random.randint(0, min(2, len(filtros_ids)))
                    base_individual['filtros'] = random.sample(filtros_ids, num_filtros) if num_filtros > 0 else []
                
                population.append(base_individual)
            else:
                # Si no hay individuos, crear uno básico
                population.append({
                    'padecimiento': padecimiento_id,
                    'montura': monturas_ids[0] if monturas_ids else None,
                    'lente': lentes_ids[0] if lentes_ids else None,
                    'capas': [],
                    'filtros': []
                })
    
    return population[:size]  # Asegurarse de devolver exactamente el tamaño solicitado
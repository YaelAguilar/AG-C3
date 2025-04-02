import pandas as pd
import os

class DataModels:
    """
    Clase para manejar los modelos de datos del sistema OptiLens.
    Carga y proporciona acceso a los diferentes conjuntos de datos necesarios.
    """
    def __init__(self, data_dir='data'):
        """
        Inicializa el modelo de datos cargando los CSV desde el directorio especificado.
        
        Args:
            data_dir (str): Directorio donde se encuentran los archivos CSV
        """
        self.data_dir = data_dir
        self.padecimientos = None
        self.monturas = None
        self.lentes = None
        self.capas = None
        self.filtros = None
        self.load_data()
    
    def load_data(self):
        """Carga todos los archivos CSV necesarios."""
        try:
            self.padecimientos = pd.read_csv(os.path.join(self.data_dir, 'padecimientos.csv'))
            self.monturas = pd.read_csv(os.path.join(self.data_dir, 'monturas.csv'))
            self.lentes = pd.read_csv(os.path.join(self.data_dir, 'lentes.csv'))
            self.capas = pd.read_csv(os.path.join(self.data_dir, 'capas.csv'))
            self.filtros = pd.read_csv(os.path.join(self.data_dir, 'filtros.csv'))
            return True
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return False
    
    def get_padecimiento_data(self, nombre_padecimiento):
        """
        Obtiene datos específicos de un padecimiento.
        
        Args:
            nombre_padecimiento (str): Nombre del padecimiento a buscar
            
        Returns:
            dict: Datos del padecimiento o None si no se encuentra
        """
        if self.padecimientos is not None:
            try:
                padecimiento = self.padecimientos[self.padecimientos['nombre_padecimiento'] == nombre_padecimiento]
                if not padecimiento.empty:
                    return padecimiento.iloc[0].to_dict()
            except Exception as e:
                print(f"Error al buscar padecimiento: {e}")
        return None
    
    def get_available_monturas(self, tipos=None, materiales=None, min_precio=None, max_precio=None):
        """
        Filtra monturas disponibles según criterios.
        
        Args:
            tipos (list): Lista de tipos de montura permitidos
            materiales (list): Lista de materiales permitidos
            min_precio (float): Precio mínimo
            max_precio (float): Precio máximo
            
        Returns:
            DataFrame: Monturas filtradas
        """
        if self.monturas is None:
            return pd.DataFrame()
        
        filtered_monturas = self.monturas.copy()
        
        if tipos:
            filtered_monturas = filtered_monturas[filtered_monturas['tipo_montura'].isin(tipos)]
        
        if materiales:
            filtered_monturas = filtered_monturas[filtered_monturas['material_armazon'].isin(materiales)]
        
        if min_precio is not None:
            filtered_monturas = filtered_monturas[filtered_monturas['precio_montura'] >= min_precio]
        
        if max_precio is not None:
            filtered_monturas = filtered_monturas[filtered_monturas['precio_montura'] <= max_precio]
        
        # Filtrar por disponibilidad
        filtered_monturas = filtered_monturas[filtered_monturas['disponibilidad_montura'] != 'Baja']
        
        return filtered_monturas
    
    def get_available_lentes(self, min_precio=None, max_precio=None):
        """
        Filtra lentes disponibles según criterios.
        
        Args:
            min_precio (float): Precio mínimo
            max_precio (float): Precio máximo
            
        Returns:
            DataFrame: Lentes filtrados
        """
        if self.lentes is None:
            return pd.DataFrame()
        
        filtered_lentes = self.lentes.copy()
        
        if min_precio is not None:
            filtered_lentes = filtered_lentes[filtered_lentes['precio_lente'] >= min_precio]
        
        if max_precio is not None:
            filtered_lentes = filtered_lentes[filtered_lentes['precio_lente'] <= max_precio]
        
        # Filtrar por disponibilidad
        filtered_lentes = filtered_lentes[filtered_lentes['disponibilidad_lente'] != 'Baja']
        
        return filtered_lentes
    
    def get_available_capas(self, tipos=None, min_precio=None, max_precio=None):
        """
        Filtra capas disponibles según criterios.
        
        Args:
            tipos (list): Lista de tipos de capas permitidas
            min_precio (float): Precio mínimo
            max_precio (float): Precio máximo
            
        Returns:
            DataFrame: Capas filtradas
        """
        if self.capas is None:
            return pd.DataFrame()
        
        filtered_capas = self.capas.copy()
        
        if tipos:
            filtered_capas = filtered_capas[filtered_capas['tipo_capa'].isin(tipos)]
        
        if min_precio is not None:
            filtered_capas = filtered_capas[filtered_capas['precio_capa'] >= min_precio]
        
        if max_precio is not None:
            filtered_capas = filtered_capas[filtered_capas['precio_capa'] <= max_precio]
        
        # Filtrar por disponibilidad
        filtered_capas = filtered_capas[filtered_capas['disponibilidad_capa'] != 'Baja']
        
        return filtered_capas
    
    def get_available_filtros(self, tipos=None, min_precio=None, max_precio=None):
        """
        Filtra filtros disponibles según criterios.
        
        Args:
            tipos (list): Lista de tipos de filtros permitidos
            min_precio (float): Precio mínimo
            max_precio (float): Precio máximo
            
        Returns:
            DataFrame: Filtros disponibles
        """
        if self.filtros is None:
            return pd.DataFrame()
        
        filtered_filtros = self.filtros.copy()
        
        if tipos:
            filtered_filtros = filtered_filtros[filtered_filtros['tipo_filtro'].isin(tipos)]
        
        if min_precio is not None:
            filtered_filtros = filtered_filtros[filtered_filtros['precio_filtro'] >= min_precio]
        
        if max_precio is not None:
            filtered_filtros = filtered_filtros[filtered_filtros['precio_filtro'] <= max_precio]
        
        # Filtrar por disponibilidad
        filtered_filtros = filtered_filtros[filtered_filtros['disponibilidad_filtro'] != 'Baja']
        
        return filtered_filtros

class Individual:
    """
    Representa una configuración de lentes (individuo del algoritmo genético).
    """
    def __init__(self, montura=None, lente=None, capas=None, filtros=None):
        """
        Inicializa un individuo.
        
        Args:
            montura (dict): Datos de la montura seleccionada
            lente (dict): Datos del lente seleccionado
            capas (list): Lista de capas seleccionadas
            filtros (list): Lista de filtros seleccionados
        """
        self.montura = montura or {}
        self.lente = lente or {}
        self.capas = capas or []
        self.filtros = filtros or []
        self.fitness = 0
        self.precio_total = 0
        self.calculate_precio_total()
    
    def calculate_precio_total(self):
        """Calcula el precio total de la configuración."""
        precio = 0
        
        # Suma precio de montura
        if self.montura and 'precio_montura' in self.montura:
            precio += self.montura['precio_montura']
        
        # Suma precio de lente
        if self.lente and 'precio_lente' in self.lente:
            precio += self.lente['precio_lente']
        
        # Suma precio de capas
        for capa in self.capas:
            if 'precio_capa' in capa:
                precio += capa['precio_capa']
        
        # Suma precio de filtros
        for filtro in self.filtros:
            if 'precio_filtro' in filtro:
                precio += filtro['precio_filtro']
        
        self.precio_total = precio
        return precio
    
    def to_dict(self):
        """
        Convierte el individuo a un diccionario.
        
        Returns:
            dict: Representación del individuo como diccionario
        """
        return {
            'montura': self.montura,
            'lente': self.lente,
            'capas': self.capas,
            'filtros': self.filtros,
            'fitness': self.fitness,
            'precio_total': self.precio_total
        }
    
    def __str__(self):
        """Representación en cadena del individuo."""
        montura_info = f"Montura: {self.montura.get('id_montura', 'N/A')} - {self.montura.get('tipo_montura', 'N/A')}"
        lente_info = f"Lente: {self.lente.get('id_lente', 'N/A')} - {self.lente.get('forma_lente', 'N/A')}"
        
        capas_str = ", ".join([f"{capa.get('id_capa', 'N/A')} ({capa.get('tipo_capa', 'N/A')})" for capa in self.capas])
        filtros_str = ", ".join([f"{filtro.get('id_filtro', 'N/A')} ({filtro.get('tipo_filtro', 'N/A')})" for filtro in self.filtros])
        
        return (f"{montura_info}\n"
                f"{lente_info}\n"
                f"Capas: {capas_str if capas_str else 'Ninguna'}\n"
                f"Filtros: {filtros_str if filtros_str else 'Ninguno'}\n"
                f"Precio Total: ${self.precio_total:.2f}\n"
                f"Aptitud: {self.fitness:.2f}")


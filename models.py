import pandas as pd
import os

class DataModels:
    """
    Clase para manejar los modelos de datos del sistema OptiLens.
    Carga y proporciona acceso a los diferentes conjuntos de datos necesarios.
    """
    def __init__(self, data_dir='ds'):
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
            filtered_monturas = filtered_monturas[filtered_monturas['tipo'].isin(tipos)]
        
        if materiales:
            filtered_monturas = filtered_monturas[filtered_monturas['material'].isin(materiales)]
        
        if min_precio is not None:
            filtered_monturas = filtered_monturas[filtered_monturas['precio'] >= min_precio]
        
        if max_precio is not None:
            filtered_monturas = filtered_monturas[filtered_monturas['precio'] <= max_precio]
        
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
            filtered_lentes = filtered_lentes[filtered_lentes['precio'] >= min_precio]
        
        if max_precio is not None:
            filtered_lentes = filtered_lentes[filtered_lentes['precio'] <= max_precio]
        
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
            filtered_capas = filtered_capas[filtered_capas['tipo'].isin(tipos)]
        
        if min_precio is not None:
            filtered_capas = filtered_capas[filtered_capas['precio'] >= min_precio]
        
        if max_precio is not None:
            filtered_capas = filtered_capas[filtered_capas['precio'] <= max_precio]
        
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
            filtered_filtros = filtered_filtros[filtered_filtros['tipo'].isin(tipos)]
        
        if min_precio is not None:
            filtered_filtros = filtered_filtros[filtered_filtros['precio'] >= min_precio]
        
        if max_precio is not None:
            filtered_filtros = filtered_filtros[filtered_filtros['precio'] <= max_precio]
        
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
        if self.montura and 'precio' in self.montura:
            precio += self.montura['precio']
        
        # Suma precio de lente
        if self.lente and 'precio' in self.lente:
            precio += self.lente['precio']
        
        # Suma precio de capas
        for capa in self.capas:
            if 'precio' in capa:
                precio += capa['precio']
        
        # Suma precio de filtros
        for filtro in self.filtros:
            if 'precio' in filtro:
                precio += filtro['precio']
        
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
        capas_str = ", ".join([capa.get('nombre', 'Sin nombre') for capa in self.capas])
        filtros_str = ", ".join([filtro.get('nombre', 'Sin nombre') for filtro in self.filtros])
        
        return (f"Montura: {self.montura.get('nombre', 'Sin montura')} ({self.montura.get('material', 'N/A')})\n"
                f"Lente: {self.lente.get('nombre', 'Sin lente')}\n"
                f"Capas: {capas_str if capas_str else 'Ninguna'}\n"
                f"Filtros: {filtros_str if filtros_str else 'Ninguno'}\n"
                f"Precio Total: {self.precio_total:.2f} MXN\n"
                f"Aptitud: {self.fitness:.2f}")
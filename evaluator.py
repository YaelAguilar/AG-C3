class FitnessEvaluator:
    """
    Evaluador de aptitud para configuraciones de lentes terapéuticos.
    Calcula la aptitud de un individuo basado en múltiples factores.
    """
    def __init__(self, data_models, padecimiento, restricciones=None, precio_objetivo=None):
        """
        Inicializa el evaluador de aptitud.
        
        Args:
            data_models (DataModels): Instancia del modelo de datos
            padecimiento (str): Nombre del padecimiento a tratar
            restricciones (dict): Restricciones médicas adicionales
            precio_objetivo (tuple): Rango de precio objetivo (min, max)
        """
        self.data_models = data_models
        self.padecimiento_data = data_models.get_padecimiento_data(padecimiento)
        self.restricciones = restricciones or {}
        self.precio_min, self.precio_max = precio_objetivo or (0, float('inf'))
        
        # Pesos para la función de aptitud
        self.weights = {
            'compatibilidad_padecimiento': 0.35,
            'calidad_componentes': 0.20,
            'precio': 0.25,
            'restricciones_adicionales': 0.20
        }
    
    def evaluate(self, individual):
        """
        Evalúa la aptitud de un individuo.
        
        Args:
            individual (Individual): Individuo a evaluar
        
        Returns:
            float: Valor de aptitud (0-100)
        """
        if not individual or not self.padecimiento_data:
            return 0
        
        # Evaluar cada componente de la aptitud
        comp_padecimiento = self._evaluar_compatibilidad_padecimiento(individual)
        calidad = self._evaluar_calidad_componentes(individual)
        precio = self._evaluar_precio(individual)
        restricciones = self._evaluar_restricciones_adicionales(individual)
        
        # Calcular aptitud ponderada
        fitness = (
            self.weights['compatibilidad_padecimiento'] * comp_padecimiento + 
            self.weights['calidad_componentes'] * calidad +
            self.weights['precio'] * precio +
            self.weights['restricciones_adicionales'] * restricciones
        )
        
        # Normalizar a rango 0-100
        fitness = max(0, min(100, fitness * 100))
        
        # Actualizar la aptitud del individuo
        individual.fitness = fitness
        
        return fitness
    
    def _evaluar_compatibilidad_padecimiento(self, individual):
        """
        Evalúa la compatibilidad de la configuración con el padecimiento.
        
        Args:
            individual (Individual): Individuo a evaluar
        
        Returns:
            float: Puntuación de compatibilidad (0-1)
        """
        puntuacion = 0.0
        
        # Verificar si hay padecimiento
        if not self.padecimiento_data:
            return 0.5  # Valor neutro si no hay padecimiento específico
        
        # Obtener recomendaciones para el padecimiento
        recomendacion_montura = self.padecimiento_data.get('recomendacion_montura', '')
        recomendacion_lente = self.padecimiento_data.get('recomendacion_lente', '')
        recomendacion_capa = self.padecimiento_data.get('recomendacion_capa', '')
        recomendacion_filtro = self.padecimiento_data.get('recomendacion_filtro', '')
        
        # 1. Evaluar la montura
        if individual.montura:
            tipo_montura = individual.montura.get('tipo_montura', '')
            if recomendacion_montura.lower() in tipo_montura.lower():
                puntuacion += 0.25
        
        # 2. Evaluar el lente
        if individual.lente:
            forma_lente = individual.lente.get('forma_lente', '')
            if recomendacion_lente.lower() in forma_lente.lower():
                puntuacion += 0.25
        
        # 3. Evaluar capas
        if individual.capas and recomendacion_capa:
            for capa in individual.capas:
                tipo_capa = capa.get('tipo_capa', '')
                if recomendacion_capa.lower() in tipo_capa.lower():
                    puntuacion += 0.25
                    break
        
        # 4. Evaluar filtros
        if individual.filtros and recomendacion_filtro:
            for filtro in individual.filtros:
                tipo_filtro = filtro.get('tipo_filtro', '')
                if recomendacion_filtro.lower() in tipo_filtro.lower():
                    puntuacion += 0.25
                    break
        
        return min(1.0, puntuacion)
    
    def _evaluar_calidad_componentes(self, individual):
        """
        Evalúa la calidad general de los componentes.
        
        Args:
            individual (Individual): Individuo a evaluar
        
        Returns:
            float: Puntuación de calidad (0-1)
        """
        puntuacion = 0.0
        componentes_evaluados = 0
        
        # Evaluar calidad de la montura
        if individual.montura:
            componentes_evaluados += 1
            # Evaluar por material y resistencia
            material = individual.montura.get('material_armazon', '').lower()
            resistencia = individual.montura.get('resistencia', '').lower()
            
            # Puntuación por material
            if 'titanio' in material:
                puntuacion += 1.0
            elif 'acetato' in material:
                puntuacion += 0.8
            elif 'metal' in material:
                puntuacion += 0.7
            else:
                puntuacion += 0.5
            
            # Puntuación por resistencia
            if 'alta' in resistencia:
                puntuacion += 1.0
            elif 'media' in resistencia:
                puntuacion += 0.7
            else:
                puntuacion += 0.4
        
        # Evaluar calidad del lente
        if individual.lente:
            componentes_evaluados += 1
            # Evaluar por índice de refracción
            indice = individual.lente.get('indice_refraccion', 0)
            
            if indice >= 1.67:
                puntuacion += 1.0
            elif indice >= 1.6:
                puntuacion += 0.8
            elif indice >= 1.5:
                puntuacion += 0.6
            else:
                puntuacion += 0.4
        
        # Evaluar calidad de capas
        if individual.capas:
            for capa in individual.capas:
                componentes_evaluados += 1
                durabilidad = capa.get('durabilidad', '').lower()
                
                if 'alta' in durabilidad:
                    puntuacion += 1.0
                elif 'media' in durabilidad:
                    puntuacion += 0.7
                else:
                    puntuacion += 0.4
        
        # Evaluar calidad de filtros
        if individual.filtros:
            for filtro in individual.filtros:
                componentes_evaluados += 1
                selectividad = filtro.get('selectividad', '').lower()
                
                if 'alta' in selectividad:
                    puntuacion += 1.0
                elif 'media' in selectividad:
                    puntuacion += 0.7
                else:
                    puntuacion += 0.4
        
        # Calcular promedio
        total_evaluaciones = componentes_evaluados * 1.0  # Cada componente tiene 1 evaluación
        if componentes_evaluados > 0:
            return puntuacion / total_evaluaciones
        return 0.5  # Valor neutro si no hay componentes
    
    def _evaluar_precio(self, individual):
        """
        Evalúa qué tan bien se ajusta el precio al rango objetivo.
        
        Args:
            individual (Individual): Individuo a evaluar
        
        Returns:
            float: Puntuación de precio (0-1)
        """
        precio = individual.precio_total
        
        # Si el precio está dentro del rango, puntuación máxima
        if self.precio_min <= precio <= self.precio_max:
            # Mejor puntuación para precios más cercanos al mínimo dentro del rango
            return 1.0 - 0.3 * ((precio - self.precio_min) / (self.precio_max - self.precio_min + 0.001))
        
        # Si está por debajo del mínimo, penalizar ligeramente (podría indicar baja calidad)
        elif precio < self.precio_min:
            return 0.7 * (precio / (self.precio_min + 0.001))
        
        # Si está por encima del máximo, penalizar significativamente
        else:
            exceso = precio - self.precio_max
            # Cuánto más excede, peor puntuación
            return max(0, 0.5 - (exceso / (self.precio_max + 0.001)) * 0.5)
    
    def _evaluar_restricciones_adicionales(self, individual):
        """
        Evalúa el cumplimiento de restricciones médicas adicionales.
        
        Args:
            individual (Individual): Individuo a evaluar
        
        Returns:
            float: Puntuación de restricciones (0-1)
        """
        if not self.restricciones:
            return 1.0  # Si no hay restricciones, puntuación máxima
        
        puntuacion = 0.0
        num_restricciones = 0
        
        # Evaluación para sensibilidad a la luz
        if self.restricciones.get('light_sensitivity', False):
            num_restricciones += 1
            # Buscar capas fotocromáticas o filtros polarizados/anti-UV
            tiene_solucion_luz = False
            
            for capa in individual.capas:
                if 'fotocrom' in capa.get('tipo_capa', '').lower():
                    tiene_solucion_luz = True
                    break
            
            if not tiene_solucion_luz:
                for filtro in individual.filtros:
                    if any(t in filtro.get('tipo_filtro', '').lower() for t in ['polarizado', 'uv']):
                        tiene_solucion_luz = True
                        break
            
            puntuacion += 1.0 if tiene_solucion_luz else 0.0
        
        # Evaluación para uso prolongado de pantallas
        if self.restricciones.get('screen_time', False):
            num_restricciones += 1
            # Buscar filtros de luz azul
            tiene_filtro_azul = any('azul' in filtro.get('tipo_filtro', '').lower() 
                                   for filtro in individual.filtros)
            puntuacion += 1.0 if tiene_filtro_azul else 0.0
        
        # Evaluación para actividades al aire libre
        if self.restricciones.get('outdoor_activities', False):
            num_restricciones += 1
            # Buscar protección UV y/o polarizado
            tiene_proteccion_exterior = False
            
            for filtro in individual.filtros:
                if any(t in filtro.get('tipo_filtro', '').lower() for t in ['uv', 'polarizado']):
                    tiene_proteccion_exterior = True
                    break
            
            # También considerar capas fotocromáticas
            if not tiene_proteccion_exterior:
                for capa in individual.capas:
                    if 'fotocrom' in capa.get('tipo_capa', '').lower():
                        tiene_proteccion_exterior = True
                        break
            
            puntuacion += 1.0 if tiene_proteccion_exterior else 0.0
        
        # Evaluación para conducción nocturna
        if self.restricciones.get('night_driving', False):
            num_restricciones += 1
            # Buscar antirreflejante y alta definición
            tiene_antirreflejo = any('antirreflej' in capa.get('tipo_capa', '').lower() 
                                    for capa in individual.capas)
            tiene_alta_def = any('alta definición' in filtro.get('tipo_filtro', '').lower() 
                                for filtro in individual.filtros)
            
            if tiene_antirreflejo:
                puntuacion += 0.7  # Antirreflejo es importante para conducción nocturna
            if tiene_alta_def:
                puntuacion += 0.3  # Alta definición complementa, pero es menos crucial
        
        # Calcular puntuación promedio
        if num_restricciones > 0:
            return puntuacion / num_restricciones
        return 1.0  # Si no se evaluaron restricciones, puntuación máxima


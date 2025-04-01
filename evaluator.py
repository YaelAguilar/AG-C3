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
        
        # 1. Evaluar la montura
        if 'montura' in individual.montura:
            # Ejemplos de criterios:
            # - Miopía/Hipermetropía: Preferible monturas ligeras
            # - Astigmatismo: Preferible monturas estables
            if self.padecimiento_data['nombre_padecimiento'] in ['Miopía', 'Hipermetropía']:
                if individual.montura.get('material') in ['Titanio', 'TR-90']:
                    puntuacion += 0.15
            elif self.padecimiento_data['nombre_padecimiento'] == 'Astigmatismo':
                if individual.montura.get('tipo') in ['Full-Frame']:
                    puntuacion += 0.15
        
        # 2. Evaluar el lente
        if individual.lente:
            # Evaluar material del lente para el padecimiento
            materiales_recomendados = {
                'Miopía': ['Policarbonato', 'Alto índice'],
                'Hipermetropía': ['Alto índice', 'Cristal'],
                'Astigmatismo': ['Alto índice', 'Policarbonato'],
                'Presbicia': ['Progresivo', 'Bifocal'],
                'Glaucoma': ['Policarbonato', 'Fotocromático'],
                'Degeneración Macular': ['Alto índice', 'Fotocromático'],
                'Fotofobia': ['Fotocromático', 'Polarizado'],
                'Sequedad Ocular': ['Hidrofóbico', 'Antirreflejo'],
                'Retinopatía Diabética': ['Alto índice', 'Protección UV'],
                'Fatiga Visual Digital': ['Antirreflejo', 'Filtro luz azul']
            }
            
            if self.padecimiento_data['nombre_padecimiento'] in materiales_recomendados:
                if individual.lente.get('material') in materiales_recomendados[self.padecimiento_data['nombre_padecimiento']]:
                    puntuacion += 0.20
        
        # 3. Evaluar capas
        if individual.capas:
            capas_recomendadas = {
                'Miopía': ['Antirreflejo', 'Endurecida'],
                'Hipermetropía': ['Antirreflejo', 'Endurecida'],
                'Astigmatismo': ['Antirreflejo', 'Endurecida'],
                'Presbicia': ['Antirreflejo', 'Endurecida'],
                'Glaucoma': ['Fotocromática', 'Antirreflejo'],
                'Degeneración Macular': ['Fotocromática', 'Protección UV'],
                'Fotofobia': ['Fotocromática', 'Polarizada'],
                'Sequedad Ocular': ['Hidrofóbica'],
                'Retinopatía Diabética': ['Antirreflejo', 'Protección UV'],
                'Fatiga Visual Digital': ['Antirreflejo', 'Filtro luz azul']
            }
            
            if self.padecimiento_data['nombre_padecimiento'] in capas_recomendadas:
                recomendadas = capas_recomendadas[self.padecimiento_data['nombre_padecimiento']]
                capas_nombres = [capa.get('tipo', '') for capa in individual.capas]
                
                # Calcular cuántas de las capas recomendadas están presentes
                coincidencias = sum(1 for rec in recomendadas if any(rec.lower() in c.lower() for c in capas_nombres))
                puntuacion += (0.30 * coincidencias / max(1, len(recomendadas)))
        
        # 4. Evaluar filtros
        if individual.filtros:
            filtros_recomendados = {
                'Miopía': ['UV400'],
                'Hipermetropía': ['UV400'],
                'Astigmatismo': ['UV400'],
                'Presbicia': ['UV400', 'Anti Luz Azul'],
                'Glaucoma': ['UV400', 'Polarizado'],
                'Degeneración Macular': ['UV400', 'Alta Definición'],
                'Fotofobia': ['Polarizado', 'UV400'],
                'Sequedad Ocular': ['UV400'],
                'Retinopatía Diabética': ['UV400', 'Polarizado'],
                'Fatiga Visual Digital': ['Anti Luz Azul', 'UV400']
            }
            
            if self.padecimiento_data['nombre_padecimiento'] in filtros_recomendados:
                recomendados = filtros_recomendados[self.padecimiento_data['nombre_padecimiento']]
                filtros_nombres = [filtro.get('tipo', '') for filtro in individual.filtros]
                
                # Calcular cuántos de los filtros recomendados están presentes
                coincidencias = sum(1 for rec in recomendados if any(rec.lower() in f.lower() for f in filtros_nombres))
                puntuacion += (0.35 * coincidencias / max(1, len(recomendados)))
        
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
        total_componentes = 0
        
        # Evaluar calidad de la montura (basado en material)
        if individual.montura:
            total_componentes += 1
            calidad_materiales = {
                'Acetato': 0.7,
                'Metal': 0.8,
                'Titanio': 0.95,
                'TR-90': 0.9
            }
            puntuacion += calidad_materiales.get(individual.montura.get('material', ''), 0.5)
        
        # Evaluar calidad del lente
        if individual.lente:
            total_componentes += 1
            calidad_lentes = {
                'Cristal': 0.75,
                'Policarbonato': 0.85,
                'Alto índice': 0.95,
                'Trivex': 0.9,
                'CR-39': 0.7
            }
            puntuacion += calidad_lentes.get(individual.lente.get('material', ''), 0.5)
        
        # Evaluar calidad promedio de capas
        if individual.capas:
            for capa in individual.capas:
                total_componentes += 1
                # Asumimos que capas más caras son de mayor calidad
                precio_normalizado = min(1.0, capa.get('precio', 0) / 500)
                puntuacion += 0.5 + (precio_normalizado * 0.5)  # 0.5 a 1.0 basado en precio
        
        # Evaluar calidad promedio de filtros
        if individual.filtros:
            for filtro in individual.filtros:
                total_componentes += 1
                # Asumimos que filtros más caros son de mayor calidad
                precio_normalizado = min(1.0, filtro.get('precio', 0) / 400)
                puntuacion += 0.5 + (precio_normalizado * 0.5)  # 0.5 a 1.0 basado en precio
        
        # Calcular promedio
        if total_componentes > 0:
            return puntuacion / total_componentes
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
            return 1.0 - 0.3 * ((precio - self.precio_min) / (self.precio_max - self.precio_min))
        
        # Si está por debajo del mínimo, penalizar ligeramente (podría indicar baja calidad)
        elif precio < self.precio_min:
            return 0.7 * (precio / self.precio_min)
        
        # Si está por encima del máximo, penalizar significativamente
        else:
            exceso = precio - self.precio_max
            # Cuánto más excede, peor puntuación
            return max(0, 0.5 - (exceso / self.precio_max) * 0.5)
    
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
                if 'fotocrom' in capa.get('tipo', '').lower():
                    tiene_solucion_luz = True
                    break
            
            if not tiene_solucion_luz:
                for filtro in individual.filtros:
                    if any(t in filtro.get('tipo', '').lower() for t in ['polarizado', 'uv']):
                        tiene_solucion_luz = True
                        break
            
            puntuacion += 1.0 if tiene_solucion_luz else 0.0
        
        # Evaluación para uso prolongado de pantallas
        if self.restricciones.get('screen_time', False):
            num_restricciones += 1
            # Buscar filtros de luz azul
            tiene_filtro_azul = any('azul' in filtro.get('tipo', '').lower() 
                                   for filtro in individual.filtros)
            puntuacion += 1.0 if tiene_filtro_azul else 0.0
        
        # Evaluación para actividades al aire libre
        if self.restricciones.get('outdoor_activities', False):
            num_restricciones += 1
            # Buscar protección UV y/o polarizado
            tiene_proteccion_exterior = False
            
            for filtro in individual.filtros:
                if any(t in filtro.get('tipo', '').lower() for t in ['uv', 'polarizado']):
                    tiene_proteccion_exterior = True
                    break
            
            # También considerar capas fotocromáticas
            if not tiene_proteccion_exterior:
                for capa in individual.capas:
                    if 'fotocrom' in capa.get('tipo', '').lower():
                        tiene_proteccion_exterior = True
                        break
            
            puntuacion += 1.0 if tiene_proteccion_exterior else 0.0
        
        # Evaluación para conducción nocturna
        if self.restricciones.get('night_driving', False):
            num_restricciones += 1
            # Buscar antirreflejante y alta definición
            tiene_antirreflejo = any('antirreflej' in capa.get('tipo', '').lower() 
                                    for capa in individual.capas)
            tiene_alta_def = any('alta definición' in filtro.get('tipo', '').lower() 
                                for filtro in individual.filtros)
            
            if tiene_antirreflejo:
                puntuacion += 0.7  # Antirreflejo es importante para conducción nocturna
            if tiene_alta_def:
                puntuacion += 0.3  # Alta definición complementa, pero es menos crucial
        
        # Calcular puntuación promedio
        if num_restricciones > 0:
            return puntuacion / num_restricciones
        return 1.0  # Si no se evaluaron restricciones, puntuación máxima
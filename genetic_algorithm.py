import random
import numpy as np
from models import Individual

class GeneticAlgorithm:
    """
    Implementación del algoritmo genético para encontrar configuraciones óptimas de lentes terapéuticos.
    """
    def __init__(self, data_models, evaluator, population_size=50, generations=30, 
                crossover_rate=0.8, mutation_rate=0.2, elitism_count=2):
        """
        Inicializa el algoritmo genético.
        
        Args:
            data_models (DataModels): Instancia con acceso a los datos
            evaluator (FitnessEvaluator): Evaluador de aptitud
            population_size (int): Tamaño de la población
            generations (int): Número de generaciones
            crossover_rate (float): Tasa de cruce (0-1)
            mutation_rate (float): Tasa de mutación (0-1)
            elitism_count (int): Número de mejores individuos que pasan directamente a la siguiente generación
        """
        self.data_models = data_models
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.current_generation = 0
    
    def initialize_population(self, precio_min=None, precio_max=None):
        """
        Inicializa una población aleatoria de individuos.
        
        Args:
            precio_min (float): Precio mínimo para los componentes
            precio_max (float): Precio máximo para los componentes
            
        Returns:
            list: Población inicial
        """
        self.population = []
        
        # Obtener monturas, lentes, capas y filtros disponibles
        monturas = self.data_models.get_available_monturas(min_precio=precio_min, max_precio=precio_max)
        lentes = self.data_models.get_available_lentes(min_precio=precio_min, max_precio=precio_max)
        capas = self.data_models.get_available_capas(min_precio=precio_min, max_precio=precio_max)
        filtros = self.data_models.get_available_filtros(min_precio=precio_min, max_precio=precio_max)
        
        # Crear individuos aleatorios
        for _ in range(self.population_size):
            # Seleccionar una montura aleatoria
            montura = None
            if not monturas.empty:
                montura_row = monturas.sample(1).iloc[0]
                montura = montura_row.to_dict()
            
            # Seleccionar un lente aleatorio
            lente = None
            if not lentes.empty:
                lente_row = lentes.sample(1).iloc[0]
                lente = lente_row.to_dict()
            
            # Seleccionar capas aleatorias (0-3 capas)
            selected_capas = []
            if not capas.empty:
                num_capas = random.randint(0, min(3, len(capas)))
                if num_capas > 0:
                    selected_capas_rows = capas.sample(num_capas)
                    selected_capas = [row.to_dict() for _, row in selected_capas_rows.iterrows()]
            
            # Seleccionar filtros aleatorios (0-2 filtros)
            selected_filtros = []
            if not filtros.empty:
                num_filtros = random.randint(0, min(2, len(filtros)))
                if num_filtros > 0:
                    selected_filtros_rows = filtros.sample(num_filtros)
                    selected_filtros = [row.to_dict() for _, row in selected_filtros_rows.iterrows()]
            
            # Crear el individuo
            individuo = Individual(montura, lente, selected_capas, selected_filtros)
            self.population.append(individuo)
        
        # Evaluar la aptitud inicial de la población
        self.evaluate_population()
        
        return self.population
    
    def evaluate_population(self):
        """
        Evalúa la aptitud de todos los individuos en la población.
        
        Returns:
            list: Lista de valores de aptitud
        """
        fitness_values = []
        for individual in self.population:
            fitness = self.evaluator.evaluate(individual)
            fitness_values.append(fitness)
        
        # Registrar estadísticas
        if fitness_values:
            avg_fitness = sum(fitness_values) / len(fitness_values)
            best_fitness = max(fitness_values)
            self.avg_fitness_history.append(avg_fitness)
            self.best_fitness_history.append(best_fitness)
            self.fitness_history.append(fitness_values)
        
        return fitness_values
    
    def select_parents(self, num_parents):
        """
        Selecciona padres para reproducción usando selección por torneo.
        
        Args:
            num_parents (int): Número de padres a seleccionar
            
        Returns:
            list: Individuos seleccionados como padres
        """
        parents = []
        for _ in range(num_parents):
            # Selección por torneo (tamaño 3)
            tournament_size = min(3, len(self.population))
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents
    
    def select_parents_with_diversity(self, num_parents):
        """
        Selecciona padres para reproducción usando selección por torneo con penalización por similitud.
        
        Args:
            num_parents (int): Número de padres a seleccionar
            
        Returns:
            list: Individuos seleccionados como padres
        """
        parents = []
        selected_genotypes = set()
        
        for _ in range(num_parents):
            # Selección por torneo con penalización por similitud
            tournament_size = min(3, len(self.population))
            tournament = random.sample(self.population, tournament_size)
            
            # Crear copias para no modificar los originales
            tournament_with_penalties = []
            for individual in tournament:
                # Crear una copia del individuo para aplicar penalización
                ind_copy = Individual(
                    individual.montura.copy() if individual.montura else None,
                    individual.lente.copy() if individual.lente else None,
                    [capa.copy() for capa in individual.capas],
                    [filtro.copy() for filtro in individual.filtros]
                )
                ind_copy.fitness = individual.fitness
                
                # Generar una representación del genotipo
                genotype = (
                    str(individual.montura.get('id_montura', 0)) if individual.montura else "None",
                    str(individual.lente.get('id_lente', 0)) if individual.lente else "None",
                    ",".join(sorted([str(capa.get('id_capa', 0)) for capa in individual.capas])),
                    ",".join(sorted([str(filtro.get('id_filtro', 0)) for filtro in individual.filtros]))
                )
                
                # Penalizar si es similar a uno ya seleccionado
                if genotype in selected_genotypes:
                    ind_copy.fitness *= 0.7  # Penalización fuerte por similitud
                
                tournament_with_penalties.append((ind_copy, genotype))
            
            # Seleccionar el ganador después de aplicar penalizaciones
            winner, winner_genotype = max(tournament_with_penalties, key=lambda x: x[0].fitness)
            
            # Encontrar el individuo original correspondiente
            for individual in tournament:
                genotype = (
                    str(individual.montura.get('id_montura', 0)) if individual.montura else "None",
                    str(individual.lente.get('id_lente', 0)) if individual.lente else "None",
                    ",".join(sorted([str(capa.get('id_capa', 0)) for capa in individual.capas])),
                    ",".join(sorted([str(filtro.get('id_filtro', 0)) for filtro in individual.filtros]))
                )
                if genotype == winner_genotype:
                    parents.append(individual)
                    selected_genotypes.add(genotype)
                    break
        
        return parents
    
    def crossover(self, parent1, parent2):
        """
        Realiza operación de cruce entre dos padres para crear descendencia.
        
        Args:
            parent1 (Individual): Primer padre
            parent2 (Individual): Segundo padre
            
        Returns:
            tuple: Dos nuevos individuos (descendencia)
        """
        if random.random() > self.crossover_rate:
            # Si no se realiza cruce, devolver copias de los padres
            child1 = Individual(
                parent1.montura.copy() if parent1.montura else None,
                parent1.lente.copy() if parent1.lente else None,
                [capa.copy() for capa in parent1.capas],
                [filtro.copy() for filtro in parent1.filtros]
            )
            
            child2 = Individual(
                parent2.montura.copy() if parent2.montura else None,
                parent2.lente.copy() if parent2.lente else None,
                [capa.copy() for capa in parent2.capas],
                [filtro.copy() for filtro in parent2.filtros]
            )
            return child1, child2
        
        # Cruce de componentes
        # Montura: intercambio directo
        if random.random() < 0.5:
            child1_montura = parent1.montura.copy() if parent1.montura else None
            child2_montura = parent2.montura.copy() if parent2.montura else None
        else:
            child1_montura = parent2.montura.copy() if parent2.montura else None
            child2_montura = parent1.montura.copy() if parent1.montura else None
        
        # Lente: intercambio directo
        if random.random() < 0.5:
            child1_lente = parent1.lente.copy() if parent1.lente else None
            child2_lente = parent2.lente.copy() if parent2.lente else None
        else:
            child1_lente = parent2.lente.copy() if parent2.lente else None
            child2_lente = parent1.lente.copy() if parent1.lente else None
        
        # Capas: intercambio parcial
        combined_capas = parent1.capas + parent2.capas
        if combined_capas:
            # Asegurar que no se repitan capas del mismo tipo
            unique_capas = {}
            for capa in combined_capas:
                tipo = capa.get('tipo_capa', '')
                if tipo not in unique_capas or random.random() < 0.5:
                    unique_capas[tipo] = capa.copy()
            
            # Distribuir aleatoriamente las capas entre los hijos
            unique_capas_list = list(unique_capas.values())
            random.shuffle(unique_capas_list)
            split_point = random.randint(0, len(unique_capas_list))
            
            child1_capas = unique_capas_list[:split_point]
            child2_capas = unique_capas_list[split_point:]
        else:
            child1_capas = []
            child2_capas = []
        
        # Filtros: intercambio parcial
        combined_filtros = parent1.filtros + parent2.filtros
        if combined_filtros:
            # Asegurar que no se repitan filtros del mismo tipo
            unique_filtros = {}
            for filtro in combined_filtros:
                tipo = filtro.get('tipo_filtro', '')
                if tipo not in unique_filtros or random.random() < 0.5:
                    unique_filtros[tipo] = filtro.copy()
            
            # Distribuir aleatoriamente los filtros entre los hijos
            unique_filtros_list = list(unique_filtros.values())
            random.shuffle(unique_filtros_list)
            split_point = random.randint(0, len(unique_filtros_list))
            
            child1_filtros = unique_filtros_list[:split_point]
            child2_filtros = unique_filtros_list[split_point:]
        else:
            child1_filtros = []
            child2_filtros = []
        
        # Crear nuevos individuos
        child1 = Individual(child1_montura, child1_lente, child1_capas, child1_filtros)
        child2 = Individual(child2_montura, child2_lente, child2_capas, child2_filtros)
        
        return child1, child2
    
    def mutate(self, individual):
        """
        Aplica mutación a un individuo con una probabilidad determinada.
        
        Args:
            individual (Individual): Individuo a mutar
            
        Returns:
            Individual: Individuo mutado
        """
        if random.random() > self.mutation_rate:
            return individual
        
        # Seleccionar aleatoriamente qué componente mutar
        mutation_component = random.choice(['montura', 'lente', 'capas', 'filtros'])
        
        if mutation_component == 'montura':
            # Mutar montura
            monturas = self.data_models.get_available_monturas()
            if not monturas.empty:
                montura_row = monturas.sample(1).iloc[0]
                individual.montura = montura_row.to_dict()
        
        elif mutation_component == 'lente':
            # Mutar lente
            lentes = self.data_models.get_available_lentes()
            if not lentes.empty:
                lente_row = lentes.sample(1).iloc[0]
                individual.lente = lente_row.to_dict()
        
        elif mutation_component == 'capas':
            # Mutar capas
            capas = self.data_models.get_available_capas()
            if not capas.empty:
                # Operaciones posibles: agregar, eliminar o reemplazar
                operacion = random.choice(['agregar', 'eliminar', 'reemplazar'])
                
                if operacion == 'agregar' and len(individual.capas) < 3:
                    # Agregar una nueva capa
                    capa_row = capas.sample(1).iloc[0]
                    nueva_capa = capa_row.to_dict()
                    # Evitar duplicados
                    if not any(c.get('id_capa') == nueva_capa.get('id_capa') for c in individual.capas):
                        individual.capas.append(nueva_capa)
                
                elif operacion == 'eliminar' and individual.capas:
                    # Eliminar una capa aleatoria
                    idx = random.randint(0, len(individual.capas) - 1)
                    individual.capas.pop(idx)
                
                elif operacion == 'reemplazar' and individual.capas:
                    # Reemplazar una capa aleatoria
                    idx = random.randint(0, len(individual.capas) - 1)
                    capa_row = capas.sample(1).iloc[0]
                    individual.capas[idx] = capa_row.to_dict()
        
        elif mutation_component == 'filtros':
            # Mutar filtros
            filtros = self.data_models.get_available_filtros()
            if not filtros.empty:
                # Operaciones posibles: agregar, eliminar o reemplazar
                operacion = random.choice(['agregar', 'eliminar', 'reemplazar'])
                
                if operacion == 'agregar' and len(individual.filtros) < 2:
                    # Agregar un nuevo filtro
                    filtro_row = filtros.sample(1).iloc[0]
                    nuevo_filtro = filtro_row.to_dict()
                    # Evitar duplicados
                    if not any(f.get('id_filtro') == nuevo_filtro.get('id_filtro') for f in individual.filtros):
                        individual.filtros.append(nuevo_filtro)
                
                elif operacion == 'eliminar' and individual.filtros:
                    # Eliminar un filtro aleatorio
                    idx = random.randint(0, len(individual.filtros) - 1)
                    individual.filtros.pop(idx)
                
                elif operacion == 'reemplazar' and individual.filtros:
                    # Reemplazar un filtro aleatorio
                    idx = random.randint(0, len(individual.filtros) - 1)
                    filtro_row = filtros.sample(1).iloc[0]
                    individual.filtros[idx] = filtro_row.to_dict()
        
        # Recalcular precio total
        individual.calculate_precio_total()
        
        return individual
    
    def evolve(self):
        """
        Ejecuta una generación del algoritmo genético.
        
        Returns:
            list: Nueva población después de la evolución
        """
        # Ordenar población por aptitud (mayor a menor)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Preservar los mejores individuos (elitismo)
        elite = self.population[:self.elitism_count]
        elite_copies = []
        for e in elite:
            elite_copy = Individual(
                e.montura.copy() if e.montura else None,
                e.lente.copy() if e.lente else None,
                [capa.copy() for capa in e.capas],
                [filtro.copy() for filtro in e.filtros]
            )
            elite_copy.fitness = e.fitness
            elite_copies.append(elite_copy)
        
        # Crear nueva población
        new_population = elite_copies.copy()
        
        # Generar el resto de la población mediante cruce y mutación
        num_offspring = self.population_size - len(elite_copies)
        num_parents_needed = (num_offspring + 1) // 2 * 2  # Asegurar número par
        
        # Usar selección con diversidad para mejorar la variedad de soluciones
        parents = self.select_parents_with_diversity(num_parents_needed)
        
        # Cruce para generar descendencia
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i+1])
                
                # Mutación
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
        
        # Actualizar población
        self.population = new_population
        
        # Evaluar nueva población
        self.evaluate_population()
        
        # Incrementar contador de generación
        self.current_generation += 1
        
        return self.population
    
    def run(self, precio_min=None, precio_max=None):
        """
        Ejecuta el algoritmo genético completo.
        
        Args:
            precio_min (float): Precio mínimo para los componentes
            precio_max (float): Precio máximo para los componentes
            
        Returns:
            list: Mejores individuos encontrados
        """
        # Inicializar población
        self.initialize_population(precio_min, precio_max)
        
        # Reiniciar historial
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.current_generation = 0
        
        # Evaluar población inicial
        self.evaluate_population()
        
        # Evolucionar por el número especificado de generaciones
        for _ in range(self.generations):
            self.evolve()
        
        # Ordenar población final por aptitud
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Devolver los mejores individuos
        return self.population[:5]
    
    def get_best_individual(self):
        """
        Devuelve el mejor individuo de la población actual.
        
        Returns:
            Individual: Mejor individuo
        """
        if not self.population:
            return None
        
        return max(self.population, key=lambda x: x.fitness)
    
    def get_top_n(self, n=3):
        """
        Devuelve los N mejores individuos de la población actual.
        
        Args:
            n (int): Número de individuos a devolver
            
        Returns:
            list: Lista de los N mejores individuos
        """
        if not self.population:
            return []
        
        # Ordenar por aptitud (mayor a menor)
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Devolver los primeros N
        return sorted_population[:n]
    
    def get_evolution_stats(self):
        """
        Obtiene estadísticas de la evolución.
        
        Returns:
            tuple: (generaciones, mejor aptitud, aptitud promedio)
        """
        generations = list(range(len(self.best_fitness_history)))
        return generations, self.best_fitness_history, self.avg_fitness_history
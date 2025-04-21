import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import time

class SimAnnealMochila:
    def __init__(self, items, capacidad_maxima, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.items = items
        self.capacidad_maxima = capacidad_maxima
        self.num_items = len(items)
        
        # Inicializa la peor solución
        self.worst_solution = None
        self.worst_value = float("Inf")

        # Parámetros del enfriamiento simulado
        self.T = 1000 if T == -1 else T
        self.T_save = self.T
        self.alpha = 0.95 if alpha == -1 else alpha
        self.stopping_temperature = 0.1 if stopping_T == -1 else stopping_T
        self.stopping_iter = 10000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        
        # Mejores soluciones encontradas
        self.best_solution = None
        self.best_value = -float("Inf")
        self.value_list = []
        
        # Solución actual
        self.cur_solution = None
        self.cur_value = -float("Inf")

    def initial_solution(self):
        # Genera una solución inicial aleatoria válida
        solution = []
        peso_total = 0
        
        for item in self.items:
            max_posible = min(int(item['cantidad']),
                  int((self.capacidad_maxima - peso_total) / item['peso']) if item['peso'] > 0 else 0)
            cantidad = random.randint(0, max_posible)
            solution.append(cantidad)
            peso_total += cantidad * item['peso']
        
        
        current_value = self.fitness(solution)
        
        if current_value > self.best_value:
            self.best_value = current_value
            self.best_solution = solution.copy()
        
        self.value_list.append(current_value)
        return solution, current_value

    def fitness(self, solution):
        # Calcula el valor total de la solución (a maximizar)
        valor_total = 0
        peso_total = 0
        
        for i in range(self.num_items):
            valor_total += solution[i] * self.items[i]['valor']
            peso_total += solution[i] * self.items[i]['peso']
        
        if peso_total > self.capacidad_maxima:
            return -float("Inf")
        
        return valor_total

    def p_accept(self, candidate_value):
        # Probabilidad de aceptar una solución peor
        return math.exp((candidate_value - self.cur_value) / self.T)

    def accept(self, candidate):
        # Decide si aceptar la nueva solución
        candidate_value = self.fitness(candidate)
        
        if candidate_value > self.cur_value:
            self.cur_value, self.cur_solution = candidate_value, candidate
            if candidate_value > self.best_value:
                self.best_value, self.best_solution = candidate_value, candidate
        else:
            if random.random() < self.p_accept(candidate_value):
                self.cur_value, self.cur_solution = candidate_value, candidate
        if self.cur_value < self.worst_value and self.cur_value != -float("Inf"):
            self.worst_value = self.cur_value
            self.worst_solution = self.cur_solution.copy()


    def generate_neighbor(self):
        # Genera una solución vecina modificando aleatoriamente un elemento
        neighbor = self.cur_solution.copy()
        idx = random.randint(0, self.num_items-1)
        
        cambio = random.choice([-1, 1])
        nueva_cantidad = neighbor[idx] + cambio
        nueva_cantidad = max(0, min(nueva_cantidad, self.items[idx]['cantidad']))
        
        neighbor[idx] = nueva_cantidad
        return neighbor

    print("\n" + "="*70)
    print("  Iniciando enfriamiento simulado para el problema de la mochila...")
    print("="*70)
    
    def anneal(self):
        # Ejecuta el algoritmo de enfriamiento simulado
        self.cur_solution, self.cur_value = self.initial_solution()

        start_time = time.perf_counter()
        
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = self.generate_neighbor()
            self.accept(candidate)
            
            self.T *= self.alpha
            self.iteration += 1
            self.value_list.append(self.cur_value)
        
        execution_time = time.perf_counter() - start_time
        print(f"\nMejor valor encontrado: {self.best_value:,.0f}")
        print(f"Peor valor encontrado: {self.worst_value:,.0f}")
        print(f"Tiempo de ejecución: {execution_time:.6f} segundos")
        print(f"Iteraciones totales: {self.iteration}")
        
        return {
            'solution': self.best_solution,
            'value': self.best_value,
            'time': execution_time,
            'iterations': self.iteration,
            'value_list': self.value_list
        }

    def plot_learning(self):
        # Grafica la evolución del valor encontrado
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(len(self.value_list))], self.value_list)
        plt.title('Evolución del Valor de la Mochila')
        plt.ylabel('Valor Total')
        plt.xlabel('Iteración')
        plt.grid(True)
        plt.show()

    def print_solution(self):
        # Muestra la solución encontrada de forma detallada
        print("\nMejor solución encontrada:")
        peso_total = 0
        for i in range(self.num_items):
            if self.best_solution[i] > 0:
                print(f"Item {self.items[i]['id']}: {self.best_solution[i]} unidades - "
                      f"Peso: {self.items[i]['peso']*self.best_solution[i]:.2f} kg - "
                      f"Valor: ${self.items[i]['valor']*self.best_solution[i]:,}")
                peso_total += self.items[i]['peso'] * self.best_solution[i]
        
        print(f"\nPeso total: {peso_total:.2f} kg / {self.capacidad_maxima} kg")
        print(f"Valor total: ${self.best_value:,}")
        
        # Peor solución
        if self.worst_solution:
            print("\nPeor solución encontrada:")
            peso_total_peor = 0
            for i in range(self.num_items):
                if self.worst_solution[i] > 0:
                    print(f"Item {self.items[i]['id']}: {self.worst_solution[i]} unidades - "
                        f"Peso: {self.items[i]['peso']*self.worst_solution[i]:.2f} kg - "
                        f"Valor: ${self.items[i]['valor']*self.worst_solution[i]:,}")
                    peso_total_peor += self.items[i]['peso'] * self.worst_solution[i]
            
            print(f"\nPeso total: {peso_total_peor:.2f} kg / {self.capacidad_maxima} kg")
            print(f"Valor total: ${self.worst_value:,}")
            
    print("\n" + "="*50)
    print("  Los resultados se podrán ver una vez se cierren todas las gráficas, dirigirse a resultados.txt")
    print("="*50)
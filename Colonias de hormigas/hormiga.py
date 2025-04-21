import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import statistics
from openpyxl import load_workbook
from pathlib import Path  # <-- Agregado para rutas relativas

class AntColonyKnapsack:
    def __init__(self, objetos, peso_max, n_hormigas=10, n_iter=100, alpha=1, beta=2, rho=0.5, q=100):
        self.objetos = objetos
        self.n = len(objetos)
        self.peso_max = peso_max
        self.n_hormigas = n_hormigas
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = np.ones(self.n) * 0.1
        self.heuristic = np.array([obj['valor'] / obj['peso'] for obj in objetos])
        self.global_best = {'solution': None, 'value': 0, 'weight': 0}

    def run(self):
        convergence = []
        gen_best = 0
        start_time = time.time()

        for it in range(self.n_iter):
            solutions = []
            values = []
            weights = []

            for _ in range(self.n_hormigas):
                solution, weight = self._construct_solution()
                value = self._calculate_value(solution)

                solutions.append(solution)
                values.append(value)
                weights.append(weight)

                if weight <= self.peso_max and value > self.global_best['value']:
                    self.global_best = {
                        'solution': solution.copy(),
                        'value': value,
                        'weight': weight
                    }
                    gen_best = it

            self._update_pheromone(solutions, values)
            convergence.append(self.global_best['value'])

        exec_time = time.time() - start_time
        return self.global_best['solution'], self.global_best['value'], convergence, gen_best, exec_time

    def _construct_solution(self):
        solution = np.zeros(self.n, dtype=int)
        remaining_weight = self.peso_max
        available_items = list(range(self.n))
        np.random.shuffle(available_items)

        for i in available_items:
            if remaining_weight <= 0:
                break

            obj = self.objetos[i]
            max_possible = min(obj['cantidad'], int(remaining_weight / obj['peso']))

            if max_possible <= 0:
                continue

            probabilities = []
            for q in range(1, max_possible + 1):
                prob = (self.pheromone[i] ** self.alpha) * ((self.heuristic[i] * q) ** self.beta)
                probabilities.append(prob)

            probabilities = np.array(probabilities)
            if probabilities.sum() > 0:
                probabilities /= probabilities.sum()
            else:
                probabilities = np.ones_like(probabilities) / len(probabilities)

            chosen_qty = np.random.choice(range(1, max_possible + 1), p=probabilities)
            solution[i] = chosen_qty
            remaining_weight -= obj['peso'] * chosen_qty

        return solution, self.peso_max - remaining_weight

    def _calculate_value(self, solution):
        return sum(solution[i] * self.objetos[i]['valor'] for i in range(self.n))

    def _update_pheromone(self, solutions, values):
        self.pheromone *= (1 - self.rho)
        if len(solutions) > 0:
            best_idx = np.argmax(values)
            best_sol = solutions[best_idx]
            best_val = values[best_idx]
            for i in range(self.n):
                if best_sol[i] > 0:
                    self.pheromone[i] += (self.q * best_val) / (1 + sum(
                        self.objetos[j]['peso'] * best_sol[j] for j in range(self.n)
                    ))

def cargar_datos(archivo):
    df = pd.read_excel(archivo)
    objetos = []
    for _, row in df.iterrows():
        objetos.append({
            'id': int(row['Id']),
            'peso': float(row['Peso_kg']),
            'valor': float(row['Valor']),
            'cantidad': int(row['Cantidad'])
        })
    return objetos

def ejecutar_experimentos(objetos, peso_max, configs, n_ejecuciones=30):
    resultados_completos = []
    todas_convergencias = []

    for config in configs:
        resultados_config = []
        convergencias_config = []
        tiempos_config = []

        for ejec in range(n_ejecuciones):
            aco = AntColonyKnapsack(objetos, peso_max, **config)
            solucion, valor, convergencia, iter_mejor, tiempo = aco.run()

            resultados_config.append({
                'Configuración': str(config),
                'Ejecución': ejec + 1,
                'Valor total': valor,
                'Peso total': sum(solucion[i] * objetos[i]['peso'] for i in range(len(objetos))),
                'Iteración mejor': iter_mejor,
                'Tiempo (s)': round(tiempo, 4),
                'Solución': solucion
            })
            convergencias_config.append(convergencia)
            tiempos_config.append(tiempo)

        valores = [r['Valor total'] for r in resultados_config]
        stats = {
            'Configuración': str(config),
            'Valor Promedio': round(statistics.mean(valores), 2),
            'Valor Máximo': max(valores),
            'Valor Mínimo': min(valores),
            'Desviación Estándar': round(statistics.stdev(valores), 2) if len(valores) > 1 else 0,
            'Tiempo Promedio': round(statistics.mean(tiempos_config), 4),
            'Iteración Promedio': round(statistics.mean([r['Iteración mejor'] for r in resultados_config]), 2)
        }

        resultados_completos.extend(resultados_config)
        todas_convergencias.append((config, convergencias_config))

        df_resultados = pd.DataFrame(resultados_config)
        df_stats = pd.DataFrame([stats])

        filename = f"resultados_aco_config_{config['alpha']}{config['beta']}{config['rho']}.xlsx"
        with pd.ExcelWriter(filename) as writer:
            df_resultados.to_excel(writer, sheet_name="Ejecuciones", index=False)
            df_stats.to_excel(writer, sheet_name="Estadísticas", index=False)

    return resultados_completos, todas_convergencias

def generar_graficas(convergencias, peso_max):
    plt.figure(figsize=(12, 6))
    for config, convs in convergencias:
        mean_conv = np.mean(convs, axis=0)
        label = f"α={config['alpha']}, β={config['beta']}, ρ={config['rho']}"
        plt.plot(mean_conv, label=label)

    plt.title(f"Convergencia Promedio de ACO (Peso máximo: {peso_max} kg)")
    plt.xlabel("Iteración")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergencia_comparativa.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    data = []
    labels = []
    for config, convs in convergencias:
        final_values = [conv[-1] for conv in convs]
        data.append(final_values)
        labels.append(f"α={config['alpha']}\nβ={config['beta']}\nρ={config['rho']}")

    plt.boxplot(data, labels=labels)
    plt.title("Distribución de Valores Finales por Configuración")
    plt.ylabel("Valor")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig("boxplot_valores_finales.png", dpi=300, bbox_inches='tight')
    plt.show()

# Obtener ruta del archivo actual y construir ruta relativa al Excel
archivo_excel = Path(__file__).parent / "datos_mochila.xlsx"
peso_maximo = 10

# Cargar datos
objetos = cargar_datos(archivo_excel)

# Configuraciones a probar
configuraciones = [
    {'n_hormigas': 10, 'n_iter': 100, 'alpha': 1, 'beta': 2, 'rho': 0.5},
    {'n_hormigas': 15, 'n_iter': 100, 'alpha': 2, 'beta': 1, 'rho': 0.3},
    {'n_hormigas': 20, 'n_iter': 100, 'alpha': 1, 'beta': 3, 'rho': 0.7}
]

# Ejecutar experimentos
resultados, convergencias = ejecutar_experimentos(objetos, peso_maximo, configuraciones)

# Generar gráficas
generar_graficas(convergencias, peso_maximo)

# Guardar resultados finales
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel("resultados_finales_aco.xlsx", index=False)
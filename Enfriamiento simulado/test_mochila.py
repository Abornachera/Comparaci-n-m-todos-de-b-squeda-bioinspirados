from anneal_mochila import SimAnnealMochila
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sys
import os
from contextlib import redirect_stdout

def guardar_grafica(nombre_archivo):
    carpeta = "graficas_resultados"
    os.makedirs(carpeta, exist_ok=True)
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    plt.savefig(ruta_completa)
    print(f"\n[INFO] Gráfica guardada como: {ruta_completa}")

def leer_datos_excel(archivo):
    df = pd.read_excel(archivo)
    items = []
    for _, row in df.iterrows():
        items.append({
            'id': row['Id'],
            'peso': row['Peso_kg'],
            'valor': row['Valor'],
            'cantidad': row['Cantidad']
        })
    capacidad_maxima = 10
    return items, capacidad_maxima

def ejecutar_experimentos(items, capacidad, num_ejecuciones=30, stdout_file=None):
    valores = []
    tiempos = []
    iteraciones = []
    mejores_soluciones = []
    convergencias = []
    mejores_iteraciones = []  # Lista para almacenar las iteraciones de las mejores soluciones
    peores_iteraciones = []   # Lista para almacenar las iteraciones de las peores soluciones

    print("-= MÉTODOS DE BUSQUEDA BIOINSPIRADOS: ENFRIAMIENTO SIMULADO - PROBLEMA DE LA MOCHILA =-")
    print(f"\n-= Ejecutando {num_ejecuciones} experimentos, capacidad máxima de la mochila: {capacidad} kg, cantidad de items: {len(items)} =-")
    
    for i in range(num_ejecuciones):
        print(f"\n-= Ejecución {i+1}/{num_ejecuciones} =-")

        # Lógica adaptativa: aumentar las iteraciones si la variación es alta
        if i >= 10:
            desviacion = statistics.stdev(valores)
            promedio = statistics.mean(valores)
            variacion_relativa = desviacion / promedio if promedio > 0 else 0

            if variacion_relativa > 0.20:
                stopping_iter = 5370
                print(f"[INFO] Muy alta variación detectada ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
            elif variacion_relativa > 0.15:
                stopping_iter = 4500
                print(f"[INFO] Alta variación detectada ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
            else:
                stopping_iter = 4000
                print(f"[INFO] Variación aceptable ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
        else:
            stopping_iter = 2000

        sa = SimAnnealMochila(
            items,
            capacidad,
            T=5000,
            alpha=0.995,
            stopping_T=1e-8,
            stopping_iter=stopping_iter
        )

        resultado = sa.anneal()
        valores.append(resultado['value'])
        tiempos.append(resultado['time'])
        iteraciones.append(resultado['iterations'])
        mejores_soluciones.append(resultado['solution'])
        convergencias.append(resultado['value_list'])

        # Encontrar la iteración de la mejor y peor solución
        mejor_idx = valores.index(max(valores))
        peor_idx = valores.index(min(valores))
        mejores_iteraciones.append(mejor_idx + 1)  # Iteración de la mejor solución (empezamos desde 1)
        peores_iteraciones.append(peor_idx + 1)    # Iteración de la peor solución (empezamos desde 1)

    print("\n" + "="*50)
    print("Resumen Estadístico de las Ejecuciones")
    print("="*50)
    print(f"Valor promedio: {statistics.mean(valores):,.0f}")
    print(f"Mejor valor: {max(valores):,.0f}")
    print(f"Peor valor: {min(valores):,.0f}")
    print(f"Desviación estándar de valores: {statistics.stdev(valores) if len(valores) > 1 else 0:,.2f}")
    print(f"\nTiempo promedio: {statistics.mean(tiempos):.4f} segundos")
    print(f"Iteraciones promedio: {statistics.mean(iteraciones):,.0f}")
    
    # Mostrar las iteraciones donde se encuentran las mejores y peores soluciones
    print(f"\nIteración de la mejor solución: {mejores_iteraciones[valores.index(max(valores))]}")
    print(f"Iteración de la peor solución: {peores_iteraciones[valores.index(min(valores))]}")

    # Mostrar gráfico (fuera del redireccionamiento para que se vea)
    if stdout_file:
        sys.stdout = sys.__stdout__
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_ejecuciones + 1), valores, 'o-')
    plt.title('Valores Obtenidos por Ejecución')
    plt.xlabel('Ejecución')
    plt.ylabel('Valor')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(valores, bins=10, edgecolor='black')
    plt.title('Distribución de Valores')
    plt.xlabel('Valor')
    plt.ylabel('Frecuencia')
    plt.grid(True)

    plt.tight_layout()
    guardar_grafica("valores_y_distribucion_config_base.png")
    plt.show()


    # Generación del gráfico de convergencia promedio (dentro de la función)
    max_len = max(len(c) for c in convergencias)
    suma_iteraciones = [0] * max_len
    conteo = [0] * max_len

    for conv in convergencias:
        for i in range(len(conv)):
            suma_iteraciones[i] += conv[i]
            conteo[i] += 1

    promedio_convergencia = [suma_iteraciones[i] / conteo[i] for i in range(max_len)]

    plt.figure(figsize=(10, 5))
    plt.plot(promedio_convergencia, label="Valor Promedio por Iteración", color='purple')
    plt.title("Convergencia Promedio - Enfriamiento Simulado (Configuración Base)")
    guardar_grafica("convergencia_promedio_config_base.png")
    plt.xlabel("Iteración")
    plt.ylabel("Valor Total")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {
        'valores': valores,
        'tiempos': tiempos,
        'iteraciones': iteraciones,
        'mejores_soluciones': mejores_soluciones
    }

# Comparar configuraciones de parámetros - Por defecto el número de ejecuciones es 30, abajo se puede cambiar.
def comparar_configuraciones(items, capacidad, num_ejecuciones=30, stdout_file=None):
    configuraciones = [
        {"nombre": "Config A", "T": 5000, "alpha": 0.995, "stopping_T": 1e-8},
        {"nombre": "Config B", "T": 4000, "alpha": 0.999, "stopping_T": 1e-6}
    ]
    
    resultados_config = []

    for config in configuraciones:
        print(f"\n" + "="*60)
        print(f"  INICIANDO EJECUCIONES PARA: {config['nombre']}")
        print("="*60)

        valores = []
        tiempos = []
        iteraciones = []
        soluciones = []
        mejores_iteraciones = []  # Lista para almacenar las iteraciones de las mejores soluciones
        peores_iteraciones = []   # Lista para almacenar las iteraciones de las peores soluciones

        for i in range(num_ejecuciones):
            print(f"\n[{config['nombre']}] Ejecución {i+1}/{num_ejecuciones}")

            # Lógica adaptativa de iteraciones
            if i >= 10:
                desviacion = statistics.stdev(valores)
                promedio = statistics.mean(valores)
                variacion_relativa = desviacion / promedio if promedio > 0 else 0

                if variacion_relativa > 0.20:
                    stopping_iter = 5370
                    print(f"[INFO] Muy alta variación detectada ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
                elif variacion_relativa > 0.15:
                    stopping_iter = 4500
                    print(f"[INFO] Alta variación detectada ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
                else:
                    stopping_iter = 4000
                    print(f"[INFO] Variación aceptable ({variacion_relativa:.2%}). Usando {stopping_iter} iteraciones.")
            else:
                stopping_iter = 2000

            sa = SimAnnealMochila(
                items, capacidad,
                T=config["T"],
                alpha=config["alpha"],
                stopping_T=config["stopping_T"],
                stopping_iter=stopping_iter
            )
            resultado = sa.anneal()
            valores.append(resultado['value'])
            tiempos.append(resultado['time'])
            iteraciones.append(resultado['iterations'])
            soluciones.append(resultado['solution'])

            # Encontrar la iteración de la mejor y peor solución
            mejor_idx = valores.index(max(valores))
            peor_idx = valores.index(min(valores))
            mejores_iteraciones.append(mejor_idx + 1)  # Iteración de la mejor solución (empezamos desde 1)
            peores_iteraciones.append(peor_idx + 1)    # Iteración de la peor solución (empezamos desde 1)

        resultados_config.append((config['nombre'], valores))

        # Estadísticas
        print("\n" + "="*50)
        print("Resumen Estadístico de las Ejecuciones")
        print("="*50)
        print(f"Valor promedio: {statistics.mean(valores):,.0f}")
        print(f"Mejor valor: {max(valores):,.0f}") 
        print(f"Peor valor: {min(valores):,.0f}")
        print(f"Desviación estándar de valores: {statistics.stdev(valores):,.2f}" if len(valores) > 1 else "Desviación estándar de valores: 0.00")
        print(f"\nTiempo promedio: {statistics.mean(tiempos):.4f} segundos")
        print(f"Iteraciones promedio: {statistics.mean(iteraciones):,.0f}")   
        
        # Mostrar iteraciones de las mejores y peores soluciones
        print(f"\nIteración de la mejor solución: {mejores_iteraciones[valores.index(max(valores))]}")
        print(f"Iteración de la peor solución: {peores_iteraciones[valores.index(min(valores))]}")

        # Mejor solución
        mejor_idx = valores.index(max(valores))
        mejor_sol = soluciones[mejor_idx]
        print("\n" + "="*50)
        print("Mejor Solución Encontrada en Todas las Ejecuciones")
        print("="*50)
        peso_total = 0
        valor_total = 0
        for i in range(len(items)):
            if mejor_sol[i] > 0:
                print(f"Item {items[i]['id']}: {mejor_sol[i]} unidades - Peso: {items[i]['peso']*mejor_sol[i]:.3f} kg - Valor: ${items[i]['valor']*mejor_sol[i]:,.0f}")
                peso_total += items[i]['peso'] * mejor_sol[i]
                valor_total += items[i]['valor'] * mejor_sol[i]
        print(f"\nPeso total: {peso_total:.3f} kg / {capacidad} kg")
        print(f"Valor total: ${valor_total:,.0f}")

        # Peor solución
        peor_idx = valores.index(min(valores))
        peor_sol = soluciones[peor_idx]
        print("\n" + "="*50)
        print("Peor Solución Encontrada en Todas las Ejecuciones")
        print("="*50)
        peso_total = 0
        valor_total = 0
        for i in range(len(items)):
            if peor_sol[i] > 0:
                print(f"Item {items[i]['id']}: {peor_sol[i]} unidades - Peso: {items[i]['peso']*peor_sol[i]:.3f} kg - Valor: ${items[i]['valor']*peor_sol[i]:,.0f}")
                peso_total += items[i]['peso'] * peor_sol[i]
                valor_total += items[i]['valor'] * peor_sol[i]
        print(f"\nPeso total: {peso_total:.3f} kg / {capacidad} kg")
        print(f"Valor total: ${valor_total:,.0f}")



    # Mostrar gráfico comparativo
    if stdout_file:
        sys.stdout = sys.__stdout__

    plt.figure(figsize=(10, 5))
    for nombre, valores in resultados_config:
        plt.plot(valores, label=nombre)
    plt.title("Comparación de Configuraciones - Valor Obtenido por Ejecución")
    plt.xlabel("Ejecución")
    plt.ylabel("Valor Total de la Mochila")
    plt.grid(True)
    plt.legend()
    guardar_grafica("comparacion_configuraciones.png")
    plt.show()



if __name__ == "__main__":
    archivo_excel = "datos_mochila.xlsx"
    items, capacidad = leer_datos_excel(archivo_excel)

    with open("resultados.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # Análisis único de una configuración
            resultados = ejecutar_experimentos(items, capacidad, num_ejecuciones=35)

            mejor_idx = resultados['valores'].index(max(resultados['valores']))
            mejor_solucion = resultados['mejores_soluciones'][mejor_idx]

            print("\n" + "="*50)
            print("Mejor Solución Encontrada en Todas las Ejecuciones")
            print("="*50)

            peso_total = 0
            valor_total = 0
            for i in range(len(items)):
                if mejor_solucion[i] > 0:
                    print(f"Item {items[i]['id']}: {mejor_solucion[i]} unidades - "
                          f"Peso: {items[i]['peso']*mejor_solucion[i]:.3f} kg - "
                          f"Valor: ${items[i]['valor']*mejor_solucion[i]:,.0f}")
                    peso_total += items[i]['peso'] * mejor_solucion[i]
                    valor_total += items[i]['valor'] * mejor_solucion[i]

            print(f"\nPeso total: {peso_total:.3f} kg / {capacidad} kg")
            print(f"Valor total: ${valor_total:,.0f}")

            peor_idx = resultados['valores'].index(min(resultados['valores']))
            peor_solucion = resultados['mejores_soluciones'][peor_idx]

            print("\n" + "="*50)
            print("Peor Solución Encontrada en Todas las Ejecuciones")
            print("="*50)

            peso_total = 0
            valor_total = 0
            for i in range(len(items)):
                if peor_solucion[i] > 0:
                    print(f"Item {items[i]['id']}: {peor_solucion[i]} unidades - "
                          f"Peso: {items[i]['peso']*peor_solucion[i]:.3f} kg - "
                          f"Valor: ${items[i]['valor']*peor_solucion[i]:,.0f}")
                    peso_total += items[i]['peso'] * peor_solucion[i]
                    valor_total += items[i]['valor'] * peor_solucion[i]

            print(f"\nPeso total: {peso_total:.3f} kg / {capacidad} kg")
            print(f"Valor total: ${valor_total:,.0f}")
            
            # Compara dos configuraciones distintas
            print("\n" + "="*80)
            print("  RESULTADOS EXPERIMENTALES - MÉTODO DE ENFRIAMIENTO SIMULADO (MOCHILA) ")
            print("="*80 + "\n")
            print("Comparando configuraciones de parámetros:")
            comparar_configuraciones(items, capacidad, num_ejecuciones=40) #Aquí cambiamos el número de ejecuciones experimentales a 40.
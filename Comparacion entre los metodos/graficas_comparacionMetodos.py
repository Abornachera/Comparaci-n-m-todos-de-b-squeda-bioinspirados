import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Cargar los datos de ACO
df_aco = pd.read_excel("resultados_finales_aco.xlsx")

# Datos de SA (los extraemos directamente del txt)
valores_sa = [
    726971, 603778, 399067, 845136, 615980, 821057, 728552, 663785, 317637, 583020,
    567336, 732390, 748038, 958887, 665432, 788826, 588113, 363922, 579820, 546666,
    366583, 633382, 540061, 974301, 919169, 594930, 711082, 905693, 583498, 902555,
    797919, 632343, 896149, 995772, 731889
]

# Convergencia de ACO (usaremos promedios aproximados, simplificado aquí)
# Aquí simularemos para tener la idea general
convergencia_aco = np.linspace(100000, 703942, 100)
convergencia_sa = np.linspace(150000, 686564, 100)

#Gráfica de Comparación de Convergencia

plt.figure(figsize=(10,6))
plt.plot(convergencia_aco, label="ACO", color='blue')
plt.plot(convergencia_sa, label="SA", color='green')
plt.title("Comparación de Convergencia: ACO vs SA")
plt.xlabel("Iteraciones")
plt.ylabel("Valor Acumulado")
plt.legend()
plt.grid(True)
plt.savefig("comparacion_convergencia_aco_sa.png", dpi=300)
plt.show()

#Gráfica de Boxplot de Valores Finales

# Obtener valores finales ACO
valores_finales_aco = df_aco['Valor total'].tolist()

data = [valores_finales_aco, valores_sa]
labels = ['ACO', 'SA']

plt.figure(figsize=(8,6))
plt.boxplot(data, labels=labels)
plt.title("Distribución de Valores Finales: ACO vs SA")
plt.ylabel("Valor")
plt.grid(True)
plt.savefig("boxplot_valores_aco_sa.png", dpi=300)
plt.show()

#Librerías necesarias para el procedimiento
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Herramientas para preprocesamiento y división de archivo para entrenamiento y prueba
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Librería para entrenamiento de algoritmo K-MEANS
from sklearn.cluster import KMeans

print("\n|__________________EXPLORACIÓN DE DATOS MEDIANTE ALGORITMO K-MEANS___________________|\n")

#Carga y lectura de archivo csv a implementar
df = pd.read_csv("Mall_Customers.csv", index_col = 0)
print(df.head(5))


#Información de variables del archivo
print("\nVariables: \n")
df.info()


#Renombrar columnas necesarias para el modelo
df.rename({'Annual Income (k$)':'Ingreso_Anual','Spending Score (1-100)':'Puntaje_Gasto'}, axis=1, inplace=True)


#Verificar la información
print("\nVariables renombradas: \n")
df.info()


#Visualización de estadísticas
print("\nEstadisticas: \n", df.describe())


#Eliminación de columnas no necesarias para el entrenamiento del modelo
x = df.iloc[:,[2,3]].values


#Asiganación de valores a arreglo x
print("\nArreglo de columnas 2 y 3: \n", x)

#Escalamiento por columna con StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)


'''___________ SECCIÓN  DE ENTRENAMIENTO DE ALGORITMO K-MEANS____________'''

#División aleatoria en entrenamiento y prueba
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42, shuffle=True)


#Ajustes del algoritmo k-means
y=[]
for i in range(1, 13):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train)
    y.append(kmeans.inertia_)

plt.plot(range(1,13), y)
plt.show()


#Ajuste de modelo definitivo con partición de prueba
kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X_train)
labels = kmeans.labels_
centers = kmeans.cluster_centers_


#Ver coordenadas de los centroides escalados
print("\nCoordenadas de centroides escaladas: \n", centers)


#Graficación de puntos a analizar
df.plot.scatter(x ='Ingreso_Anual', y = "Puntaje_Gasto")
plt.show()


#Gráfica de datos de entrenamiento con sus respectivos centroides
centers_orig = scaler.inverse_transform(centers)
points_orig = scaler.inverse_transform(X_train)
color = ['red','orange','yellow','green','blue','purple']
for clust in range(6):
    plt.scatter(points_orig[labels == clust, 0], points_orig[labels == clust, 1],
                c=color[clust], label=f"Cluster {clust}", s=20)
plt.scatter(centers_orig[:, 0], centers_orig[:, 1], s=200, c='cyan', edgecolor='k', marker='o')
plt.xlabel('Ingreso Anual')
plt.ylabel('Puntaje de Gasto')
plt.legend()
plt.show()


#Ver coordenadas de los centroides en escala original
print("\nCoordenadas de centroides a escala original: \n", centers_orig)


# predecir etiquetas para la muestra de prueba
prueba = kmeans.predict(X_test)


#Valores predichos por el algoritmo K-Means
#Muestra los valores de los clusters a los que pertenecen
print("\nValores predichos de los clusteres a los que los puntos pertenecen: \n", prueba)
print("\n|__________________FIN DEL ALGORITMO___________________|\n")
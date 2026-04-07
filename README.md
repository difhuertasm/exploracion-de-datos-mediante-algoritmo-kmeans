Exploración de Datos Mediante Algoritmo K-Means

Este proyecto implementa un modelo de aprendizaje no supervisado utilizando el algoritmo K-Means para segmentar clientes de un centro comercial. El objetivo es agrupar a los clientes basándose en su Ingreso Anual y su Puntaje de Gasto, permitiendo identificar patrones de comportamiento útiles para estrategias de marketing.


🚀 Estructura del Proyecto
El proyecto se divide en dos scripts principales:

Datos.py: Utiliza la librería kagglehub para descargar automáticamente la última versión del dataset Customer Segmentation Tutorial desde Kaggle.

Ejemplo_KMeans.py: Realiza el preprocesamiento, escalamiento, entrenamiento del modelo y la visualización de los clusters.


📊 Dataset
El dataset utilizado es el Mall Customer Segmentation Data, que contiene información básica sobre los clientes:

  * CustomerID: ID único del cliente.

  * Gender: Género.

  * Age: Edad.

  * Annual Income (k$): Ingreso anual en miles de dólares.

  * Spending Score (1-100) : Puntaje asignado por el centro comercial basado en el comportamiento del cliente.


🛠️ Tecnologías Utilizadas

1. Python 3.x

2. Pandas & Numpy: Manipulación de datos.

3. Matplotlib: Visualización de gráficos.

4. Scikit-Learn:

  * KMeans para el agrupamiento.

  * StandardScaler para el escalado de características.

  * train_test_split para la validación del modelo.


⚙️ Instalación y Uso

Clonar el repositorio mediante :

  git clone https://github.com/difhuertasm/exploracion-de-datos-mediante-algoritmo-kmeans.git

Instalar dependencias:

  pip install numpy pandas matplotlib scikit-learn kagglehub

Descargar los datos:

Ejecuta el primer script para obtener el archivo Mall_Customers.csv mediante:
  
  python Datos.py

Ejecutar el análisis:

  python Ejemplo_KMeans


📈 Proceso del Modelo

1. Método del Codo (Elbow Method)
Se realiza una iteración de 1 a 12 clusters para identificar el punto donde la inercia disminuye de forma menos significativa. Según el código, se ha seleccionado un valor de K = 6.

2. Preprocesamiento
Se renombran las columnas para facilitar la lectura.

Se aplica StandardScaler para normalizar los datos, asegurando que el algoritmo de K-Means (basado en distancias euclidianas) no se vea sesgado por las escalas de las variables.

3. Visualización de Resultados
El script genera una gráfica final donde se muestran los clientes agrupados por colores y la ubicación de los centroides en su escala original.

Nota: El modelo divide los datos en entrenamiento (70%) y prueba (30%) para validar la consistencia de los clusters asignados.


📋 Resultados
El algoritmo logra identificar 6 grupos distintos de clientes, lo cual permite diferenciar, por ejemplo:

Clientes con ingresos altos y gastos altos.

Clientes con ingresos bajos pero gastos elevados.

Clientes con perfiles de ahorro moderado.


Desarrollado por: Diego Fernando Huertas M.

# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar los datos en un DataFrame (suponiendo que ya tienes tus datos en un archivo CSV)
data = pd.read_csv('Book1.csv')

# Seleccionar las variables a utilizar para la agrupación
selected_features = ['edad', 'ingresos', 'gasto_promedio', 'numero_de_compras']
X = data[selected_features]

# Normalizar los datos para asegurarnos de que todas las variables estén en la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elegir el número de clusters (k)
k = 3

# Crear y ajustar el modelo de K-means
kmeans_model = KMeans(n_clusters=k, random_state=42)
kmeans_model.fit(X_scaled)

# Obtener las etiquetas de los clusters y agregarlas al DataFrame original
data['cluster'] = kmeans_model.labels_

# Ver los centroides de cada cluster (representan el centro de cada agrupación)
centroids = scaler.inverse_transform(kmeans_model.cluster_centers_)
print('Centroides de los clusters:')
print(pd.DataFrame(centroids, columns=selected_features))

# Reportar el conteo de elementos en cada cluster
print('Conteo de elementos en cada cluster:')
print(data['cluster'].value_counts())

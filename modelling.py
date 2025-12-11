import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def codo(scaled_df):
    """Determinar cuántos clusters obtendremos utilizando el "Método del codo" """
    inertia = []

    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled_df)
        inertia.append(km.inertia_)

    plt.plot(range(2, 10), inertia, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inercia')
    plt.title('Método del codo')
    plt.show()

# ============================================

def training_kmeans(df_identification, scaled_df, df_cluster_data):
    k = 4

    # 1. Ejecutar el modelo K-Means y obtener las etiquetas
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_df)

    # 2. Concatenar para crear el DataFrame de resultados
    df_results = pd.concat([
        df_identification.reset_index(drop=True),
        df_cluster_data.reset_index(drop=True),
        pd.Series(labels, name='cluster')  # Ahora 'labels' existe
    ], axis=1)

    # 3.Análisis de los clusters
    print("\nPROMEDIOS POR CLUSTER:\n")
    df_results.groupby("cluster").mean(numeric_only=True).round(2)

    return df_results


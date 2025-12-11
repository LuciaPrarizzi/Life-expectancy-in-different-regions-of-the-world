# Importar librerías principales para procesar datos y generar visualizaciones
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importar librerías para modelado
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuración visual de gráficos realizados con Seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (8,5)

#Importar funciones de otros módulos
from data_processing import *
from modelling import *
from reporting_and_visual import *

#Fuente de datos
data_csv = ("Life Expectancy Data.csv")

#Creación de una lista conformada por variables que serán consideradas para el análisis.
variables = [
    "schooling",
    "income_composition",
    "HIV/AIDS",
    "dipht",
    "health_expend",
    "total_spend",
    "polio",
    "under_five_deaths",
    "BMI",
    "measles",
    "alcohol",
    "infant_deaths",
    "adult_mort",
    "life_exp",
    "status"
]

#Carga de datos
df_raw = data_load(data_csv)

#Vista general de los datos
df_raw.info()
df_raw.head()

#Exploración de datos faltantes
percent_missings = analize_missings(df_raw)

#Limpieza de la variable 'GDP' (PBI)
# Se identificaron los países que no contienen en el data frame la información sobre GDP.
# Nota: La variable fue excluida del análisis por severas inconsistencias.
df_cleaned_gdp = clean_gdp_data(df_raw)

#Transformación categórica de variables
df_transformed = transform_status(df_cleaned_gdp)

#Preparación para Clustering (selección de columnas y funciones de eliminación de NA cuando no es
# recomendable la imputación de variables)
variables_for_cluster = variables
df_cluster_data, df_identification = prepare_clustering_data(df_transformed, variables_for_cluster)

#Estandarización
scaled_df = standardize_data(df_cluster_data)

#Identificación del "codo"
codo(scaled_df)

#Entrenamiento
df_results = training_kmeans(df_cluster_data, scaled_df, df_identification)

#Resultados correspondientes al último año
df_latest_results = filter_latest_year(df_results)

#Visualización de los resultados
visualization(scaled_df,df_results)

#Generación de reporte
final_clusters_report(df_results, variables_for_cluster)

#Creación de mapa que muestra los grupos de países
create_map(df_latest_results)

#Creación de mapa interactivo
interactive_map(df_latest_results)





import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_load(data):
    """Carga de los datos"""
    df = pd.read_csv(data)

    # Renombra a algunas columnas para simplificar comandos
    df = df.rename(columns={
        'Country': 'country',
        'Year': 'year',
        'Status': 'status',
        'Life expectancy ': 'life_exp',
        'Adult Mortality': 'adult_mort',
        'infant deaths': 'infant_deaths',
        'Alcohol': 'alcohol',
        'percentage expenditure': 'health_expend',
        'Hepatitis B': 'hepb',
        'Measles ': 'measles',
        'under-five deaths': 'under_five_deaths',
        'Polio': 'polio',
        'Total expenditure': 'total_spend',
        'Diphtheria ': 'dipht',
        'Population': 'population',
        'Income composition of resources': 'income_composition',
        'Schooling': 'schooling'
    })
    return df

def analize_missings(data_missings):
    """Identificación de valores faltantes, determinando el porcentaje por columna, para evaluar la exclusión de
    variables"""
    # Este paso es necesario para la imputación de columnas de acuerdo al tipo de dato,
    # cantidad de faltantes y lógica epidemiológica/estadística.
    missing_percent = data_missings.isna().mean().sort_values(ascending=False) * 100
    return missing_percent

def clean_gdp_data(data: pd.DataFrame) -> pd.DataFrame:
    """Identificación de países que no contienen la información sobre la variable GDP"""
    # Importante: La variable finalmente fue excluida del análisis por hallarse numerosas inconsistencias en la carga
    # de datos.
    countries_all_null = data.groupby('country')['GDP'].apply(lambda x: x.isna().all())
    countries_to_remove = countries_all_null[countries_all_null == True].index
    df = data[~data['country'].isin(countries_to_remove)].copy()
    df['GDP'] = df.groupby('country')['GDP'].transform(
        lambda x: x.where((x > 200) & (x < 150000), x.median(numeric_only=True))
    )
    return df


def prepare_clustering_data(data: pd.DataFrame, variables: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara los datos para el clustering:
    1. Selecciona solo las variables de clustering.
    2. Elimina filas con NA en esas variables.
    3. Separa y devuelve los datos de clustering y los datos de identificación.
    """
    # Selección de variables y eliminación de NA
    df_cluster_data = data[variables].dropna()

    # Separación e identificación usando el índice de las filas limpias
    indices_finales = df_cluster_data.index
    df_identification = data.loc[indices_finales, ['country', 'year']].reset_index(drop=True)

    # Reseteo del índice de los datos para que coincida con la identificación
    df_cluster_data = df_cluster_data.reset_index(drop=True)

    return df_cluster_data, df_identification

def transform_status(data: pd.DataFrame):
    """Transforma la variable categórica 'status' a codificación binaria."""
    df = data.copy()
    df['status'] = df['status'].map({'Developing': 0, 'Developed': 1})
    return df

def standardize_data(data: pd.DataFrame):
    """Aplica la estandarización (StandardScaler) a las variables del DataFrame."""
    #Estandariza aplicando una transformación matemática para que todas las variables se midan en rangos
    # comparables.
    # Este paso evita que el algoritmo dé mayor importancia a una variable sólo por tener valores más grandes.

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Creamos un nuevo DataFrame con los mismos nombres de columna
    scaled_df = pd.DataFrame(scaled, columns=data.columns)

    return scaled_df

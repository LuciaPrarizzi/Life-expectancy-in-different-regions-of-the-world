import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import geopandas as gpd
import pycountry
import numpy as np

def filter_latest_year(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra el DataFrame para obtener solo la observación del último año para cada país."""
    df_latest = df.sort_values("year").drop_duplicates("country", keep="last")
    return df_latest

def visualization(scaled_df, df_cluster):
    """Visualización PCA 2D"""
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(scaled_df)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=pca_2d[:,0],
        y=pca_2d[:,1],
        hue=df_cluster["cluster"],
        palette="Set2"
    )
    plt.title("Clusters de países (PCA 2D)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend(title="Cluster")
    plt.show()

# ============================
# Generador automático de informe de clusters
# ============================
from scipy import stats
import textwrap
import os
def final_clusters_report(df_results, vars_of_interest):
    """Genera un reporte con toda la información obtenida"""
    df = df_results.copy()
    output_dir = "cluster_report"
    os.makedirs(output_dir, exist_ok=True)

    # ---------- CHECKS ----------
    #Control preventivo para evitar errores silenciosos o mensajes confusos,
    #asegurando que el flujo del pipeline esté completo.
    if 'cluster' not in df.columns:
        raise ValueError("El DataFrame no tiene la columna 'cluster'. Ejecutá el clustering antes.")

    # Asegurar tipos numéricos
    for v in vars_of_interest:
        if v in df.columns:
            df[v] = pd.to_numeric(df[v], errors='coerce')

    # Resumen básico por cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_summary = df.groupby('cluster')[vars_of_interest].agg(['mean','std','median','count']).round(2)

    # Guardar resumen numérico
    cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))

    # Listar los 10 países más representativos por cluster
    # (según la variable expectativa de vida media en el cluster)
    topn = 10
    representatives = {}
    for c in sorted(df['cluster'].unique()):
        sub = df[df['cluster'] == c]
        # países con más observaciones y mayor representatividad
        country_counts = sub['country'].value_counts().head(50)
        # usa país-medio ordenado por life_expectancy dentro del cluster
        country_means = sub.groupby('country')['life_exp'].mean().sort_values(ascending=False)
        reps = country_means.head(topn).index.tolist()
        representatives[c] = reps

    # Tests estadísticos: comparar life_expectancy entre clusters
    groups = [df[df['cluster'] == c]['life_exp'].dropna() for c in sorted(df['cluster'].unique())]
    # Test normalidad de cada grupo
    normal_flags = [stats.shapiro(g[:500]).pvalue > 0.05 if len(g) >= 3 else False for g in groups]  # limit shapiro size
    use_anova = all(normal_flags) and all(len(g) >= 3 for g in groups)

    if use_anova:
        fstat, p_anova = stats.f_oneway(*groups)
        stat_text = f"ANOVA F={fstat:.3f}, p={p_anova:.3g}"
    else:
        hstat, p_kw = stats.kruskal(*groups)
        stat_text = f"Kruskal-Wallis H={hstat:.3f}, p={p_kw:.3g}"

    # Comparaciones por par (opcional): t-test or Mann-Whitney
    pairwise = []
    from itertools import combinations
    for a,b in combinations(sorted(df['cluster'].unique()), 2):
        ga = df[df['cluster']==a]['life_exp'].dropna()
        gb = df[df['cluster']==b]['life_exp'].dropna()
        if len(ga)>2 and len(gb)>2:
            stat, p = stats.mannwhitneyu(ga, gb, alternative='two-sided')
            pairwise.append(((a,b), stat, p))

    # Generar texto del informe
    lines = []
    lines.append("# Informe de clusters — Análisis automático\n")
    lines.append("## Resumen general\n")
    lines.append(f"- Observaciones totales usadas: {len(df):,}")
    lines.append(f"- Número de clusters: {df['cluster'].nunique()}")
    lines.append("\n## Tamaño por cluster\n")
    for c, cnt in cluster_counts.sort_index().items():
        lines.append(f"- Cluster {c}: {cnt} observaciones")

    lines.append("\n## Estadísticas por cluster (media ± std) — variables seleccionadas")
    lines.append("\n```csv\n")
    # incluir un snapshot de cluster_summary
    lines.append(cluster_summary.to_csv())
    lines.append("\n```\n")

    lines.append("## Países representativos por cluster (top {})\n".format(topn))
    for c, reps in representatives.items():
        lines.append(f"- Cluster {c}: {', '.join(reps)}")

    lines.append("\n## Comparación de expectativa de vida entre clusters")
    lines.append(f"- Test seleccionado: {stat_text}")
    lines.append("\n## Comparaciones por pares (Mann-Whitney U)\n")
    for (a,b), stat, p in pairwise:
        lines.append(f"- Cluster {a} vs {b}: U={stat:.2f}, p={p:.3g}")

    # Interpretación automatizada base
    lines.append("\n## Interpretación (resumen automático)\n")
    for c in sorted(df['cluster'].unique()):
        mean_life = df[df['cluster']==c]['life_exp'].mean()
        mean_spend = df[df['cluster']==c]['total_spend'].mean() if 'total_spend' in df.columns else np.nan
        schooling = df[df['cluster']==c]['schooling'].mean() if 'schooling' in df.columns else np.nan
        lines.append(f"- **Cluster {c}** — vida media ≈ {mean_life:.2f} años; porcentaje promedio del PBI per cápita destinado a salud  ≈ {mean_spend:.2f}% (gasto en salud); escolaridad promedio ≈ {schooling:.2f} años.")
        # heurística simple para tag
        if mean_life >= 75 and schooling >= 9:
            tag = "Países con alto desarrollo, representan un posible mercado maduro para la silver economy"
        elif mean_life < 65:
            tag = "Países con desafíos sanitarios. El foco debe situarse en la mejora socio-sanitaria, la reducción de mortalidad infantil y el fortalecimiento de cobertura vacunatoria."
        else:
            tag = "Países en transición. Las estrategias deben apuntar a la mejora en inversión sanitaria, prevención de enfermedades y acceso a oportunidades de crecimiento socio-económico."
        lines.append(f"  - Interpretación: {tag}\n")

    # Implicancias para Silver Economy y políticas
    lines.append("## Implicancias para la Silver Economy y políticas de salud\n")
    lines.append(textwrap.fill(
        "Los clusters permiten identificar grupos de países con perfiles sanitarios y socioeconómicos distintos. "
        "Para los clusters con mayor esperanza de vida y mejor escolaridad, las políticas pueden enfocarse en servicios y "
        "productos orientados a mejorar la calidad de vida de adultos mayores (atención domiciliaria, tecnologías de salud, financiamiento de pensiones). "
        "Para clusters con baja esperanza de vida, las prioridades son fortalecer la atención primaria de la salud, la cobertura de vacunación y reducir la mortalidad infantil,"
        " condiciones previas a la consolidación de una silver economy.", 80))

    # Guardar informe en Markdown
    md_path = os.path.join(output_dir, "cluster_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Informe generado en: {md_path}")
    print("Resumen corto por cluster:\n")
    print(cluster_counts)
    print("\nTest comparación expectativa de vida entre clusters:", stat_text)

# ============================
# Creación de mapa para reflejar la distribución de los clusteres
# ============================
def create_map(df_results):
    # 1. Cargar mapa mundial
    world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

    # 2. Alinear nombres de países (Si algunos nombres no coinciden entre el dataset y el mapa)
    mapping = {
        "United States": "United States of America",
        "Democratic Republic of the Congo": "Congo (Kinshasa)",
        "Republic of the Congo": "Congo (Brazzaville)",
        "Russian Federation": "Russia",
        "Viet Nam": "Vietnam",
        "Iran (Islamic Republic of)": "Iran",
        "Bolivia (Plurinational State of)": "Bolivia",
        "Tanzania, United Republic of": "Tanzania",
        "Venezuela (Bolivarian Republic of)": "Venezuela"
    }
    world = world.rename(columns={"ADMIN": "country"})

    # 3. Unir mapa + clusters
    map_df = world.merge(df_results, left_on="country", right_on="country", how="left")

    # 4. Dibujar mapa coloreado por cluster
    map_df.plot(column="cluster",
                cmap="tab20b",
                linewidth=0.6,
                edgecolor="black",
                legend=True)

    plt.title("Clusters de Expectativa de Vida y Variables Socio-Sanitarias")
    plt.axis("off")
    plt.show()
    plt.close()

# ============================
# Creación de un mapa interactivo
# ============================
def interactive_map(df_results):
    #Identificamos a los países según la denominación ISO 3
    def to_iso3(country):
        try:
            return pycountry.countries.lookup(country).alpha_3
        except:
            return None
    df_results["iso3"] = df_results["country"].apply(to_iso3)
    import plotly.express as px
    fig = px.choropleth(
        df_results,
        locations="iso3",           # ISO-3 en lugar de nombres
        color="cluster",
        hover_name="country",
        color_continuous_scale="Viridis"
    )
    fig.show()

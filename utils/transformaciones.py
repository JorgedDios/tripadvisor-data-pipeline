"""
Módulo: transformaciones.py
Descripción: Conjunto de utilidades de ingeniería de datos avanzado.
Este módulo contiene la lógica de limpieza (DLQ), cálculo de métricas 
competitivas (Window Functions) y feature engineering para Polars.
Autor: Jorge de Dios Orellana
"""

import polars as pl

def separar_dlq(df: pl.DataFrame, critical_cols: list) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Patrón DLQ (Dead Letter Queue): Audita registros corruptos en lugar de eliminarlos.
    
    Args:
        df: DataFrame original a evaluar.
        critical_cols: Lista de columnas donde no se permiten nulos.
        
    Returns:
        tuple: (df_bueno, df_corrupto)
    """
    # Identificamos filas con nulos en columnas críticas mediante lógica horizontal
    condicion_nulos = pl.any_horizontal([pl.col(c).is_null() for c in critical_cols])
    
    # Segregación de datos: el filtro ~ (NOT) aísla los registros válidos
    df_corrupto = df.filter(condicion_nulos)
    df_bueno = df.filter(~condicion_nulos)
    
    return df_bueno, df_corrupto

def aplicar_window_functions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Funciones de Ventana: Calcula métricas competitivas por ciudad.
    Realiza operaciones analíticas complejas sobre particiones lógicas.
    """
    # 1. Calculamos la media móvil o agregada por ciudad mediante el over()
    df = df.with_columns(
        pl.col("avg_rating").mean().over("city").alias("city_avg_rating")
    )
    
    # 2. Ingeniería de variable de contexto competitivo
    # Permite comparar el local respecto a su entorno urbano específico
    df = df.with_columns(
        (pl.col("avg_rating") - pl.col("city_avg_rating")).alias("rating_diff_city")
    )
    return df

def feature_engineering_avanzado(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """
    Aplica transformaciones de datos (Encoding) para preparar el ML.
    Convierte variables categóricas de texto a numéricas para Spark MLlib.
    """
    # Imputación de nulos en columnas numéricas clave
    df = df.with_columns(pl.col(config['numeric_columns']).fill_null(0))
    
    # Codificación ordinal para el nivel de precio
    df = df.with_columns(
        pl.when(pl.col("price_level").str.strip_chars() == "$").then(1)
        .when(pl.col("price_level").str.strip_chars() == "$$ - $$$").then(2)
        .when(pl.col("price_level").str.strip_chars() == "$$$$").then(3)
        .otherwise(0).alias("price_level_num")
    )
    
    # Codificación binaria (One-Hot Encoding simplificado) para banderas de calidad
    df = df.with_columns([
        pl.when(pl.col("claimed") == "Claimed").then(1).otherwise(0).alias("is_claimed"),
        pl.when(pl.col("vegetarian_friendly") == "Y").then(1).otherwise(0).alias("is_veg_friendly"),
        pl.when(pl.col("gluten_free") == "Y").then(1).otherwise(0).alias("is_gluten_free")
    ])
    
    return df
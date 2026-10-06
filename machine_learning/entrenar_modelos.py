"""
Script: entrenar_modelos.py
Fase 3: Modelado Predictivo con Apache Spark MLlib.
Autor: Jorge de Dios Orellana

Descripción:
Entrena y evalúa modelos de regresión sobre el Data Lake particionado para
predecir la nota media de un restaurante a partir de sus atributos estructurales.
Implementa ensamblaje de vectores (VectorAssembler), división de dataset
(train/test) y evaluación métrica (RMSE/R2) para dos modelos:
  - Regresión Lineal (baseline), interpretada mediante sus coeficientes.
  - Random Forest Regressor, que captura relaciones no lineales y se interpreta
    mediante 'Feature Importances'.

Uso:
    python machine_learning/entrenar_modelos.py --modelo lr      # Regresión Lineal
    python machine_learning/entrenar_modelos.py --modelo rf      # Random Forest
    python machine_learning/entrenar_modelos.py --modelo ambos   # Ambos y comparativa (por defecto)
"""

import argparse
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Raíz del repositorio en el sys.path para importar utils al ejecutar el script directamente
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.configuracion import cargar_config, comprobar_java_home

# Variables estructurales disponibles antes de abrir el restaurante (sin Data Leakage)
COLUMNAS_PREDICTORAS = [
    "total_reviews_count",
    "price_level_num",
    "is_claimed",
    "is_veg_friendly",
    "is_gluten_free",
    "city_avg_rating"
]


def cargar_y_preparar_datos(spark: SparkSession):
    """
    Lee el Data Lake, filtra el ruido, ensambla el vector de características
    y divide el dataset en train/test.

    Returns:
        tuple: (train_data, test_data)
    """
    print("1. Leyendo datos desde el Data Lake particionado...")
    # Lectura de particiones de manera transparente gracias a la estructura de directorios
    ruta_data_lake = str(Path(cargar_config()["paths"]["partitioned_dir"]) / "*" / "*.parquet")
    df = spark.read.parquet(ruta_data_lake)

    # LIMPIEZA Y PREPARACIÓN DEL DATASET
    # Filtrado de ruido: registros sin nota (nulos imputados) no aportan valor al aprendizaje
    # garbage in - garbage out
    df_clean = df.filter(col("avg_rating") > 0)
    print(f"Total de restaurantes útiles para entrenar: {df_clean.count()}")

    # ENSAMBLAJE DE VARIABLES (VectorAssembler)
    # Spark ML requiere un único vector de características (input) para el algoritmo
    assembler = VectorAssembler(
        inputCols=COLUMNAS_PREDICTORAS,
        outputCol="features"
    )
    df_features = assembler.transform(df_clean)

    # Estructura final: features (vectores) vs label (valor objetivo)
    df_ml = df_features.select(col("features"), col("avg_rating").alias("label"))

    # DIVISIÓN TRAIN / TEST
    # Split 80/20 con semilla fija para garantizar la reproducibilidad de resultados.
    # Se cachean para que ambos modelos se entrenen y evalúen exactamente sobre las mismas filas.
    train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    test_data.cache()
    print(f"Datos de Entrenamiento: {train_data.count()} | Datos de Test: {test_data.count()}")

    return train_data, test_data


def evaluar(modelo, test_data, titulo: str) -> dict:
    """
    Evalúa un modelo entrenado sobre los datos de test y muestra sus métricas.
    RMSE para la magnitud del error y R2 para la bondad del ajuste.
    """
    print("3. Evaluando el modelo...")
    # usamos el transformer para probar el modelo con los datos test (datos no vistos que han pasado el mismo pipeline que los de entrenamiento)
    predicciones = modelo.transform(test_data)

    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predicciones)
    r2 = evaluator_r2.evaluate(predicciones)

    print("-" * 50)
    print(f" {titulo}")
    print("-" * 50)
    print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Coeficiente de Determinación (R2): {r2:.4f}")
    print("-" * 50)

    return {"rmse": rmse, "r2": r2}


def entrenar_regresion_lineal(train_data, test_data) -> dict:
    """Entrena la Regresión Lineal (baseline) y la interpreta mediante sus coeficientes."""
    print("\n2. Entrenando el modelo de Regresión Lineal...")
    lr = LinearRegression(featuresCol="features", labelCol="label")
    # Entrenamos el estimator (modelo vacío)
    modelo = lr.fit(train_data)

    metricas = evaluar(modelo, test_data, "RESULTADOS DE LA REGRESIÓN LINEAL")

    # INTERPRETABILIDAD
    # Extracción de coeficientes para explicar el impacto de cada variable en el modelo
    print("Peso matemático de cada variable en la nota final:")
    for col_name, peso in zip(COLUMNAS_PREDICTORAS, modelo.coefficients):
        print(f" - {col_name}: {peso:.4f}")

    return metricas


def entrenar_random_forest(train_data, test_data) -> dict:
    """Entrena el Random Forest y lo interpreta mediante 'Feature Importances'."""
    print("\n2. Entrenando el modelo de Random Forest...")
    # Configuración: 50 árboles para estabilidad, profundidad 5 para evitar overfitting.
    # Semilla fija: sin ella PySpark la deriva del hash del nombre de la clase, que cambia
    # entre procesos de Python, y el bosque (y su RMSE) variaría de una ejecución a otra.
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=5, seed=42)
    modelo_rf = rf.fit(train_data)

    metricas = evaluar(modelo_rf, test_data, "RESULTADOS DEL RANDOM FOREST")

    # INTERPRETABILIDAD DE NEGOCIO (Feature Importance)
    print("Importancia de cada variable en las decisiones del modelo:")
    importancias = modelo_rf.featureImportances.toArray()
    for col_name, importancia in zip(COLUMNAS_PREDICTORAS, importancias):
        print(f" - {col_name}: {importancia * 100:.2f}%")

    return metricas


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de predicción de nota (Spark MLlib).")
    parser.add_argument(
        "--modelo",
        choices=["lr", "rf", "ambos"],
        default="ambos",
        help="lr = Regresión Lineal, rf = Random Forest, ambos = entrena los dos y los compara (por defecto)."
    )
    args = parser.parse_args()

    # Configuración del entorno de ejecución
    comprobar_java_home()

    # INICIALIZAR SPARK SESSION
    # Sesión configurada para el procesamiento distribuido del Data Lake
    spark = SparkSession.builder \
        .appName("TripAdvisor_ML_Models") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    train_data, test_data = cargar_y_preparar_datos(spark)

    resultados = {}
    if args.modelo in ("lr", "ambos"):
        resultados["Regresión Lineal"] = entrenar_regresion_lineal(train_data, test_data)
    if args.modelo in ("rf", "ambos"):
        resultados["Random Forest"] = entrenar_random_forest(train_data, test_data)

    # COMPARATIVA FINAL (solo si se han entrenado ambos modelos)
    if len(resultados) > 1:
        print("\n" + "=" * 50)
        print(" COMPARATIVA DE MODELOS")
        print("=" * 50)
        print(f"{'Modelo':<20}{'RMSE':>12}{'R2':>12}")
        for nombre, metricas in resultados.items():
            print(f"{nombre:<20}{metricas['rmse']:>12.4f}{metricas['r2']:>12.4f}")
        mejor = min(resultados, key=lambda nombre: resultados[nombre]["rmse"])
        print(f"\nMejor modelo por RMSE: {mejor}")

    spark.stop()


if __name__ == "__main__":
    main()

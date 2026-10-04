"""
Script: modelo_ml.py
Fase 3: Modelado Predictivo con Apache Spark MLlib.
Autor: Jorge de Dios Orellana

Descripción:
Realiza el entrenamiento y evaluación de un modelo de Regresión Lineal 
sobre datos particionados. Implementa ensamblaje de vectores (VectorAssembler), 
división de dataset (train/test) y evaluación métrica (RMSE/R2).
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Configuración del entorno de ejecución
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# 1. INICIALIZAR SPARK SESSION
# Sesión configurada para el procesamiento distribuido del Data Lake
spark = SparkSession.builder \
    .appName("TripAdvisor_ML_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("1. Leyendo datos desde el Data Lake particionado...")
# Lectura de particiones de manera transparente gracias a la estructura de directorios
ruta_data_lake = "/home/jorge/proyecto_sdpd2/data/processed_lake/*/*.parquet"
df = spark.read.parquet(ruta_data_lake)

# 2. LIMPIEZA Y PREPARACIÓN DEL DATASET
# Filtrado de ruido: registros sin nota (nulos imputados) no aportan valor al aprendizaje
# garbage in - garbage out
df_clean = df.filter(col("avg_rating") > 0)
print(f"Total de restaurantes útiles para entrenar: {df_clean.count()}")

# 3. ENSAMBLAJE DE VARIABLES (VectorAssembler)
# Spark ML requiere un único vector de características (input) para el algoritmo
columnas_predictoras = [
    "total_reviews_count", 
    "price_level_num", 
    "is_claimed", 
    "is_veg_friendly", 
    "is_gluten_free",
    "city_avg_rating"
]

assembler = VectorAssembler(
    inputCols=columnas_predictoras,
    outputCol="features"
)

df_features = assembler.transform(df_clean)

# Estructura final: features (vectores) vs label (valor objetivo)
df_ml = df_features.select(col("features"), col("avg_rating").alias("label"))

# 4. DIVISIÓN TRAIN / TEST
# Split 80/20 con semilla fija para garantizar la reproducibilidad de resultados
train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)
print(f"Datos de Entrenamiento: {train_data.count()} | Datos de Test: {test_data.count()}")

# 5. ENTRENAMIENTO DEL MODELO
print("\n2. Entrenando el modelo de Regresión Lineal...")
lr = LinearRegression(featuresCol="features", labelCol="label")
# Entrenamos el estimator (modelo vacío)
modelo = lr.fit(train_data)

# 6. EVALUACIÓN DE MÉTRICAS
print("3. Evaluando el modelo...")
# usamos el transformer para probar el modelo con los datos test (datos no vistos que han pasado el mismo pipeline que los de entrenamiento)
predicciones = modelo.transform(test_data)

# Métricas de error: RMSE para magnitud del error y R2 para bondad del ajuste
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predicciones)
r2 = evaluator_r2.evaluate(predicciones)

print("-" * 50)
print(" RESULTADOS DEL MODELO")
print("-" * 50)
print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coheficiente de Determinación (R2): {r2:.4f}")
print("-" * 50)

# 7. INTERPRETABILIDAD (Feature Importance)
# Extracción de coeficientes para explicar el impacto de cada variable en el modelo
print("Peso matemático de cada variable en la nota final:")
for col_name, peso in zip(columnas_predictoras, modelo.coefficients):
    print(f" - {col_name}: {peso:.4f}")

spark.stop()
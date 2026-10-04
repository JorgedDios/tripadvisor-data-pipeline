"""
Script: modelos_ml2.py
Fase 3: Modelado Predictivo Avanzado con Random Forest (Spark MLlib).
Autor: Jorge de Dios Orellana

Descripción:
Implementa un modelo de ensamble (Random Forest Regressor) para capturar 
relaciones no lineales entre las variables estructurales del restaurante y su nota. 
Incluye cálculo de 'Feature Importances' para interpretabilidad del negocio.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Configuración del entorno
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# 1. INICIALIZAR SPARK SESSION
spark = SparkSession.builder \
    .appName("TripAdvisor_RF_Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("1. Leyendo datos desde el Data Lake particionado...")
ruta_data_lake = "/home/jorge/proyecto_sdpd2/data/processed_lake/*/*.parquet"
df = spark.read.parquet(ruta_data_lake)

# 2. LIMPIEZA
df_clean = df.filter(col("avg_rating") > 0)
print(f"Total de restaurantes útiles para entrenar: {df_clean.count()}")

# 3. ENSAMBLAJE DE VARIABLES
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
df_ml = df_features.select(col("features"), col("avg_rating").alias("label"))

# 4. DIVISIÓN TRAIN / TEST
train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)

# 5. ENTRENAMIENTO (RANDOM FOREST)
print("\n2. Entrenando el modelo de Random Forest...")
# Configuración: 50 árboles para estabilidad, profundidad 5 para evitar overfitting
rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=5)
modelo_rf = rf.fit(train_data)

# 6. EVALUACIÓN
print("3. Evaluando el modelo...")
predicciones = modelo_rf.transform(test_data)

evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predicciones)
r2 = evaluator_r2.evaluate(predicciones)

print("-" * 50)
print(" RESULTADOS DEL RANDOM FOREST")
print("-" * 50)
print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R2): {r2:.4f}")
print("-" * 50)

# 7. INTERPRETABILIDAD DE NEGOCIO (Feature Importance)
# Esta es la parte más importante para tu defensa oral
print("Importancia de cada variable en las decisiones del modelo:")
importancias = modelo_rf.featureImportances.toArray()
for col_name, importancia in zip(columnas_predictoras, importancias):
    print(f" - {col_name}: {importancia * 100:.2f}%")

spark.stop()
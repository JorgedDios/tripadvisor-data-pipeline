"""
Script: consumidor_tripadvisor.py
Fase 2: Consumo en tiempo real con Spark Structured Streaming.
Autor: Jorge de Dios Orellana

Descripción:
Establece un consumidor de Apache Kafka que deserializa mensajes JSON, 
aplica transformaciones de filtrado (Modo Append) y agregaciones geográficas 
en tiempo real (Modo Complete), demostrando resiliencia y escalabilidad.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import from_json, col
import os

# Configuración del entorno: Definición del motor de ejecución de Spark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"


# 1. INICIALIZAR SPARK SESSION

# Se añade el conector de Kafka para integrar el clúster con Spark
spark = SparkSession.builder \
    .appName("TripAdvisorStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# 2. DEFINIR EL ESQUEMA (StructType)

# Esquema estricto para garantizar la calidad del dato en tiempo real
tripadvisor_schema = StructType([
    StructField("restaurant_name", StringType(), True),
    StructField("country", StringType(), True),
    StructField("city", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("avg_rating", DoubleType(), True),
    StructField("city_avg_rating", DoubleType(), True),
    StructField("rating_diff_city", DoubleType(), True),
    StructField("total_reviews_count", IntegerType(), True), 
    StructField("price_level_num", IntegerType(), True),
    StructField("is_claimed", IntegerType(), True),
    StructField("is_veg_friendly", IntegerType(), True),
    StructField("is_gluten_free", IntegerType(), True)
])

print("Iniciando la conexión con Kafka en localhost:9092...")


# 3. LEER EL FLUJO DESDE KAFKA (FASE DE INGESTA)

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "tripadvisor_restaurants") \
    .option("startingOffsets", "earliest") \
    .load()

# Parseo de JSON binario a DataFrame relacional
# vuelve a pasar de bytes a json y coge el json y lo convierte en una tabla de Spark
# 'Dataframe' con el StructType que he definido arriba
parsed_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), tripadvisor_schema).alias("data")) \
    .select("data.*")



# BLOQUE DE EJECUCIÓN DE CONSULTAS (STREAMING)
#   3  grifos distintos:

# ---> CONSULTA 0: Auditoría visual del flujo raw: escupe los datos por consola en modo append tal cual llegan
query_raw = parsed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .queryName("raw_data") \
    .start()

# ---> CONSULTA 1: Filtrado de nicho dietético (Modo Append)
# Escribe en CSV garantizando la inmutabilidad de los resultados
consulta1_df = parsed_df.filter((col("is_gluten_free") == 1) & (col("avg_rating") >= 4.5))

query1 = consulta1_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("path", "./salida1_txt") \
    .option("checkpointLocation", "./checkpoints/consulta1") \
    .start()

# ---> CONSULTA 2: Agregación analítica por país (Modo Complete) por consola
# Fuerza el re-cálculo global de la tabla con cada micro-lote
consulta2_df = parsed_df.groupBy("country").count()

query2 = consulta2_df.writeStream \
    .outputMode("complete") \
    .format("console") \
    .queryName("conteo_paises") \
    .start()


# 5. MANTENER EL SCRIPT VIVO

# Mantiene la sesión activa para procesar eventos indefinidamente
spark.streams.awaitAnyTermination()
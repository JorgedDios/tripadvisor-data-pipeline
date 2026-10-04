"""
DAG: tripadvisor_etl_pipeline_senior
Fase 1: Orquestación ETL con Apache Airflow
Autor: Jorge de Dios Orellana

Pipeline modular que gestiona la ingesta, validación (DLQ), 
transformación particionada y carga en Kafka del dataset de TripAdvisor.
Implementa idempotencia mediante Offset Watermarking.
"""

import os
import sys
import polars as pl
import pyarrow.dataset as ds
from datetime import datetime, timedelta
from pathlib import Path
from airflow.decorators import dag, task
from confluent_kafka import Producer
import json

# Airflow solo añade la carpeta dags/ al sys.path: añadimos la raíz del repositorio
# para poder importar el paquete utils sin instalarlo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.configuracion import cargar_config
from utils.transformaciones import separar_dlq, aplicar_window_functions, feature_engineering_avanzado

# Rutas resueltas contra SDPD2_HOME (ver utils/configuracion.py)
config = cargar_config()

default_args = {
    'owner': 'grupo_sdpd2',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

@dag(
    dag_id='tripadvisor_etl_pipeline_senior',
    default_args=default_args,
    description='Pipeline ETL de 3 Fases: Extracción, Transformación y Carga (Kafka)',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['SDPD2', 'ETL', 'Senior', 'Idempotente'],
)
def tripadvisor_pipeline():

    @task
    def extract_and_validate() -> str:
        """TAREA 1 (EXTRACT): Lee datos, aplica Watermark y audita nulos (DLQ)."""
        print("Fase 1: Extracción y Validación (DLQ) con Watermarking...")
        ruta_csv = config['paths']['raw_csv']
        ruta_dlq = config['paths']['dlq_parquet']
        watermark_path = config['paths']['watermark_file']
        ruta_clean = config['paths']['clean_parquet']
        
        os.makedirs(os.path.dirname(ruta_clean), exist_ok=True)
        os.makedirs(os.path.dirname(ruta_dlq), exist_ok=True)

        # Control de idempotencia (Watermarking)
        last_row = 0
        if os.path.exists(watermark_path):
            with open(watermark_path, 'r') as f:
                last_row = json.load(f).get('last_row', 0)

        df_raw = pl.read_csv(ruta_csv, ignore_errors=True)
        filas_totales = df_raw.height
        
        # Filtro de nuevos registros
        nuevas_filas = filas_totales - last_row
        if nuevas_filas <= 0:
            print(f"No hay datos nuevos. El Watermark está en la fila {last_row}.")
            return "NO_DATA"
            
        print(f"Procesando {nuevas_filas} filas nuevas.")
        df_incremental = df_raw.slice(last_row, nuevas_filas)
        
        # Auditoría de calidad y segregación DLQ
        df_good, df_bad = separar_dlq(df_incremental, config['processing']['critical_columns'])
        
        if df_bad.height > 0:
            df_bad.write_parquet(ruta_dlq)
            
        df_good.write_parquet(ruta_clean)
        
        # Actualización del punto de control
        with open(watermark_path, 'w') as f:
            json.dump({'last_row': filas_totales}, f)
        
        return ruta_clean

    @task
    def transform_and_partition(input_path: str) -> str:
        """TAREA 2 (TRANSFORM): Aplica Feature Engineering y particiona."""
        if input_path == "NO_DATA":
            return "NO_DATA"
            
        print(f"Fase 2: Feature Engineering leyendo desde {input_path}...")
        ruta_partitioned = config['paths']['partitioned_dir']
        os.makedirs(ruta_partitioned, exist_ok=True)

        df_good = pl.read_parquet(input_path)
            
        # Aplicación de lógica de negocio (Transformaciones vectorizadas)
        df_transformed = feature_engineering_avanzado(df_good, config['processing'])
        df_transformed = aplicar_window_functions(df_transformed)
        
        columnas_finales = [
            "restaurant_name", "country", "city", "latitude", "longitude", 
            "avg_rating", "city_avg_rating", "rating_diff_city", 
            "total_reviews_count", "price_level_num", 
            "is_claimed", "is_veg_friendly", "is_gluten_free"
        ]
        df_final = df_transformed.select(columnas_finales) # me quedo con las columans que me interesan enviar a Spark

        # Escritura en Data Lake con particionado físico (Partition Pruning)
        table_final = df_final.to_arrow() # lo pasamos al formato de memoria pyarrow
        ds.write_dataset(
            table_final, base_dir=ruta_partitioned, format="parquet", 
            partitioning=["country"], existing_data_behavior="overwrite_or_ignore" # crea las columnas de cada pais y borra la original
            # y no se crean carpetas duplicadas, se sobreescriben o se ignoran.
        )
        return ruta_partitioned

    @task
    def load_to_kafka(input_path: str):
        """TAREA 3 (LOAD): Carga las particiones en el tópico de Kafka."""
        if input_path == "NO_DATA":
            print("Carga en Kafka omitida. No hubo datos nuevos.")
            return

        print(f"Fase 3: Leyendo dataset particionado desde {input_path}...")
        df = pl.read_parquet(f"{input_path}/**/*.parquet") # coge todos los archivos parquet sueltos, los junta y monta un solo dataframe en memoria

        # Límite de demostración para el entorno local: un broker de Kafka de un solo nodo
        # no necesita el millón de registros para demostrar el flujo extremo a extremo.
        # Basta con subir 'max_records_per_run' en config.toml, o ponerlo a 0, para
        # publicar el dataset completo.
        max_records = config['kafka'].get('max_records_per_run', 0)
        df_safe = df.head(max_records) if max_records else df

        kafka_config = {
            'bootstrap.servers': config['kafka']['bootstrap_servers'],
            'acks': config['kafka']['acks'],
            'message.timeout.ms': 10000,
            'delivery.timeout.ms': 15000
        }
        producer = Producer(kafka_config)
        
        print(f"Enviando mensajes al topic '{config['kafka']['topic_name']}'...")
        records = df_safe.to_dicts() # Transforma el DataFrame de filas y columnas a una lista de diccionarios.
        count = 0
        
        # Producción de mensajes en formato JSON
        for record in records:
            producer.produce( # enviamos el paquete a kafka
                topic=config['kafka']['topic_name'], 
                value=json.dumps(record).encode('utf-8'), # json.dumps = convertir el diccionario a texto json para luego aplastarlo a bytes en UTF-8
                callback=lambda err, msg: None 
            )
            count += 1
            producer.poll(0)
            
        producer.flush() # Prohibido terminar el programa hasta que el último byte del último mensaje haya llegado sano y salvo a Kafka
        print(f"¡ÉXITO! {count} registros enviados a Kafka.")

    # Flujo de dependencias (Orquestación secuencial)
    ruta_limpia = extract_and_validate()
    ruta_particionada = transform_and_partition(ruta_limpia)
    load_to_kafka(ruta_particionada)

dag_instance = tripadvisor_pipeline() # Llamamos a la función principal para que Airflow registre el DAG
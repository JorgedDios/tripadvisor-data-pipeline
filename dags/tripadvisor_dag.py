"""
DAG: tripadvisor_etl_pipeline
Fase 1: Orquestación ETL con Apache Airflow
Autor: Jorge de Dios Orellana

Pipeline modular que gestiona la ingesta, validación (DLQ), 
transformación particionada y carga en Kafka del dataset de TripAdvisor.
Implementa idempotencia mediante Offset Watermarking: watermark.json guarda
cuántas filas del CSV de origen se han consumido ya (publicadas en Kafka o
desviadas a la DLQ). Solo se actualiza cuando Kafka ha confirmado la entrega
de todo el lote, de modo que cada fila del origen pasa por el pipeline una vez.
"""

import os
import sys
import polars as pl
import pyarrow as pa
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
    'owner': 'jorge-de-dios',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

@dag(
    dag_id='tripadvisor_etl_pipeline',
    default_args=default_args,
    description='Pipeline ETL de 3 Fases: Extracción, Transformación y Carga (Kafka)',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # dos ejecuciones simultáneas leerían el mismo watermark y duplicarían el lote
    tags=['SDPD2', 'ETL', 'Kafka', 'Idempotente'],
)
def tripadvisor_pipeline():

    @task
    def extract_and_validate() -> dict:
        """TAREA 1 (EXTRACT): Lee datos, aplica Watermark y audita nulos (DLQ)."""
        print("Fase 1: Extracción y Validación (DLQ) con Watermarking...")
        ruta_csv = config['paths']['raw_csv']
        ruta_dlq = config['paths']['dlq_dir']
        watermark_path = config['paths']['watermark_file']
        ruta_clean = config['paths']['clean_parquet']

        os.makedirs(os.path.dirname(ruta_clean), exist_ok=True)
        os.makedirs(ruta_dlq, exist_ok=True)

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
            return {"estado": "NO_DATA"}

        # Límite opcional de demostración para el entorno local (por defecto 0 = sin tope):
        # un broker de Kafka de un solo nodo no necesita el millón de registros para demostrar
        # el flujo extremo a extremo. Con tope, las window functions se calculan por lote.
        # El tope se aplica aquí, sobre las filas del origen, y no al publicar: así el lote
        # que recorre el pipeline es exactamente el tramo [inicio, fin) del CSV y el watermark
        # puede avanzar hasta 'fin' sin dejar filas leídas y nunca publicadas.
        # Con 'max_records_per_run' = 0 en config.toml se procesa el dataset completo en una sola ejecución.
        max_records = config['kafka'].get('max_records_per_run', 0)
        tamano_lote = min(nuevas_filas, max_records) if max_records else nuevas_filas
        inicio, fin = last_row, last_row + tamano_lote

        print(f"Procesando las filas {inicio + 1}-{fin} del origen ({nuevas_filas} pendientes).")
        df_incremental = df_raw.slice(inicio, tamano_lote)
        
        # Auditoría de calidad y segregación DLQ
        df_good, df_bad = separar_dlq(df_incremental, config['processing']['critical_columns'])
        
        # Un fichero por lote (lote-<fila de inicio>.parquet), con el mismo criterio que el Data Lake:
        # la DLQ acumula los registros corruptos de todas las ejecuciones, y un lote reintentado
        # (siempre empieza en la misma fila, la del watermark) sobrescribe su propio fichero en vez
        # de duplicar sus filas. Si el reintento ya no tiene filas corruptas (p. ej. porque cambió
        # el tope y el lote es más corto), se borra el fichero del intento anterior: esas filas
        # se volverán a evaluar, y a escribir, en el lote que las contenga.
        ruta_dlq_lote = os.path.join(ruta_dlq, f"lote-{inicio:010d}.parquet")
        if df_bad.height > 0:
            df_bad.write_parquet(ruta_dlq_lote)
            print(f"{df_bad.height} registros corruptos desviados a {ruta_dlq_lote}.")
        elif os.path.exists(ruta_dlq_lote):
            os.remove(ruta_dlq_lote)

        df_good.write_parquet(ruta_clean)

        # El watermark NO se actualiza aquí: las filas aún no han llegado a Kafka.
        # Se pasa la posición 'fin' a la carga, que la confirma tras publicar el lote.
        # Las filas desviadas a la DLQ también cuentan: ya se han evaluado y no se reintentan.
        return {"estado": "OK", "ruta_clean": ruta_clean, "inicio": inicio, "fin": fin}

    @task
    def transform_and_partition(lote: dict) -> dict:
        """TAREA 2 (TRANSFORM): Aplica Feature Engineering y particiona."""
        if lote["estado"] == "NO_DATA":
            return lote

        input_path = lote["ruta_clean"]
        print(f"Fase 2: Feature Engineering leyendo desde {input_path}...")
        ruta_partitioned = config['paths']['partitioned_dir']
        os.makedirs(ruta_partitioned, exist_ok=True)

        df_good = pl.read_parquet(input_path)
        if df_good.height == 0:
            # Todo el lote fue a la DLQ: no hay nada que escribir, pero la carga
            # debe confirmar igualmente el avance del watermark
            return {**lote, "base_dir": ruta_partitioned, "ficheros": []}
            
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
        # Cada lote escribe ficheros con nombre propio (lote-<fila de inicio>-N.parquet), de modo que
        # el Data Lake acumula los lotes en lugar de pisar el part-0.parquet de cada país, y repetir
        # el mismo lote tras un fallo sobrescribe sus propios ficheros en vez de duplicarlos.
        # file_visitor registra los ficheros escritos para que la carga publique solo este lote.
        ficheros = []
        ds.write_dataset(
            table_final, base_dir=ruta_partitioned, format="parquet",
            partitioning=["country"], existing_data_behavior="overwrite_or_ignore", # crea las columnas de cada pais y borra la original
            # y no se crean carpetas duplicadas, se sobreescriben o se ignoran.
            basename_template=f"lote-{lote['inicio']:010d}-{{i}}.parquet",
            file_visitor=lambda fichero: ficheros.append(fichero.path),
        )
        return {**lote, "base_dir": ruta_partitioned, "ficheros": ficheros}

    @task
    def load_to_kafka(lote: dict):
        """TAREA 3 (LOAD): Carga el lote en el tópico de Kafka y confirma el watermark."""
        if lote["estado"] == "NO_DATA":
            print("Carga en Kafka omitida. No hubo datos nuevos.")
            return

        input_path = lote["base_dir"]
        print(f"Fase 3: Leyendo el lote de las filas {lote['inicio'] + 1}-{lote['fin']} desde {input_path}...")
        # El Data Lake se escribió con partitioning=["country"] (particionado por directorio:
        # processed_lake/Spain/lote-0000000000-0.parquet). La columna 'country' ya NO está dentro de los
        # ficheros, solo en el nombre de la carpeta. Una lectura plana con glob la pierde, así que
        # leemos con pyarrow.dataset declarando el mismo esquema de partición para reconstruirla.
        # Solo se leen los ficheros de este lote (no todo el lago, que acumula lotes anteriores);
        # partition_base_dir permite reconstruir 'country' a partir de la ruta de cada fichero.
        particionado = ds.partitioning(pa.schema([("country", pa.string())]))
        if lote["ficheros"]:
            lake = ds.dataset(lote["ficheros"], format="parquet", partitioning=particionado,
                              partition_base_dir=input_path)
            df = pl.from_arrow(lake.to_table()) # junta todas las particiones en un solo dataframe en memoria
            if "country" not in df.columns:
                raise ValueError("No se ha reconstruido la columna de partición 'country' del Data Lake.")
        else:
            df = pl.DataFrame()

        kafka_config = {
            'bootstrap.servers': config['kafka']['bootstrap_servers'],
            'acks': config['kafka']['acks'],
            'enable.idempotence': True,  # los reintentos internos del productor no duplican mensajes
            'message.timeout.ms': 10000,
            'delivery.timeout.ms': 15000
        }
        producer = Producer(kafka_config)
        
        print(f"Enviando mensajes al topic '{config['kafka']['topic_name']}'...")
        records = df.to_dicts() # Transforma el DataFrame de filas y columnas a una lista de diccionarios.
        count = 0
        errores = []

        # Producción de mensajes en formato JSON
        for record in records:
            while True:
                try:
                    producer.produce( # enviamos el paquete a kafka
                        topic=config['kafka']['topic_name'],
                        value=json.dumps(record).encode('utf-8'), # json.dumps = convertir el diccionario a texto json para luego aplastarlo a bytes en UTF-8
                        callback=lambda err, msg: errores.append(err) if err is not None else None # anotamos los envíos fallidos
                    )
                    break
                except BufferError:
                    # Cola local del productor llena (100.000 mensajes por defecto; sin tope se envía el
                    # dataset completo): esperamos a que Kafka confirme entregas pendientes y reintentamos
                    producer.poll(1)
            count += 1
            producer.poll(0)

        pendientes = producer.flush(30) # Prohibido terminar el programa hasta que el último byte del último mensaje haya llegado sano y salvo a Kafka
        if pendientes or errores:
            # El watermark no avanza: la siguiente ejecución repetirá este mismo lote
            raise RuntimeError(
                f"Kafka no confirmó el lote completo ({len(errores)} errores, {pendientes} mensajes sin entregar). "
                f"El watermark se mantiene en la fila {lote['inicio']}."
            )
        print(f"¡ÉXITO! {count} registros enviados a Kafka.")

        # Confirmación del punto de control: solo ahora el lote [inicio, fin) cuenta como procesado.
        # Escritura atómica (fichero temporal + os.replace) para no dejar un watermark a medias.
        watermark_path = config['paths']['watermark_file']
        tmp_path = watermark_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump({'last_row': lote['fin']}, f)
        os.replace(tmp_path, watermark_path)
        print(f"Watermark actualizado: {lote['fin']} filas del origen procesadas.")

    # Flujo de dependencias (Orquestación secuencial)
    ruta_limpia = extract_and_validate()
    ruta_particionada = transform_and_partition(ruta_limpia)
    load_to_kafka(ruta_particionada)

dag_instance = tripadvisor_pipeline() # Llamamos a la función principal para que Airflow registre el DAG
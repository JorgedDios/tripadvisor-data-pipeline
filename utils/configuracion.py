"""
Módulo: configuracion.py
Descripción: Punto único de resolución de rutas y entorno del proyecto.
Carga config.toml y convierte sus rutas relativas en absolutas a partir de
la raíz del proyecto, de modo que el repositorio funcione en cualquier máquina
sin editar el código.
Autor: Jorge de Dios Orellana
"""

import os
import sys
from pathlib import Path

import tomli

# Raíz del proyecto: variable de entorno SDPD2_HOME o, si no existe,
# el directorio raíz del repositorio (este fichero vive en <raíz>/utils/)
PROJECT_ROOT = Path(
    os.environ.get("SDPD2_HOME") or Path(__file__).resolve().parent.parent
).expanduser().resolve()

CONFIG_PATH = PROJECT_ROOT / "config.toml"


def cargar_config() -> dict:
    """
    Lee config.toml y resuelve las rutas de las secciones [paths] y [streaming].

    'data_dir' se resuelve contra PROJECT_ROOT y el resto de rutas de [paths]
    contra 'data_dir'. Las rutas de [streaming] (claves terminadas en '_dir') se
    resuelven contra PROJECT_ROOT. Si alguna ruta ya es absoluta se respeta tal cual.
    """
    with open(CONFIG_PATH, "rb") as f:
        config = tomli.load(f)

    paths = config["paths"]
    data_dir = PROJECT_ROOT / paths.pop("data_dir", "data")
    paths["data_dir"] = str(data_dir)
    for clave, valor in paths.items():
        if clave != "data_dir":
            paths[clave] = str(data_dir / valor)

    streaming = config.get("streaming", {})
    for clave, valor in streaming.items():
        if clave.endswith("_dir"):
            streaming[clave] = str(PROJECT_ROOT / valor)

    return config


def comprobar_java_home() -> None:
    """
    Spark necesita una JVM. No se fuerza ninguna ruta concreta: si JAVA_HOME
    ya está definida se respeta; si no, se avisa al usuario.
    """
    if not os.environ.get("JAVA_HOME"):
        print(
            "AVISO: la variable de entorno JAVA_HOME no está definida. "
            "Spark necesita un JDK (el proyecto se ha probado con el 11): define JAVA_HOME apuntando a su "
            "directorio de instalación (ver .env.example) antes de ejecutar este script.",
            file=sys.stderr,
        )

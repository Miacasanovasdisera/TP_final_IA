# %% 1. Acerca del dataset y del análisis
"""Análisis exploratorio de CAMUS para estimar edad a partir de ecocardiogramas.

Adaptado de las etapas de análisis del proyecto de aprobación crediticia:
adquisición, validación de estructura, tipos, faltantes, estadísticas,
distribuciones, correlaciones y revisión de valores extremos.

Instalación, una sola vez, desde la terminal de VS Code:
    python -m pip install pandas numpy matplotlib nibabel

Ejecución:
    python "ModeloRegresion v0.py"

Lee los metadatos del dataset ya descomprimido en CAMUS_public/.
Solo carga cuatro imágenes de ejemplo; el resto del análisis usa archivos .cfg.

Fuente: https://www.creatis.insa-lyon.fr/Challenge/camus/
CAMUS fue creado para segmentación; aquí exploramos sus datos para regresión.
No se entrena ningún modelo en esta etapa.
"""

from pathlib import Path
import math
import warnings

try:
    import numpy as np
    import pandas as pd
except ImportError as error:
    raise SystemExit(
        "Faltan librerías. Ejecutá en la terminal:\n"
        "python -m pip install pandas numpy matplotlib nibabel"
    ) from error


# %% 2. Configuración
BASE = Path(__file__).resolve().parent
RUTA_DATASET = BASE / "CAMUS_public" / "CAMUS_public" / "database_nifti"
CARPETA_RESULTADOS = BASE / "resultados_eda"
MOSTRAR_GRAFICOS = True  # False: guardar gráficos sin abrir ventanas.
PACIENTES_ESPERADOS = 500

# Diccionario de las principales variables de df (una fila por paciente).
DICCIONARIO = {
    "Paciente": "Carpeta que identifica al paciente; no es una variable predictora.",
    "Edad": "Edad cronológica en años; variable objetivo continua (Age).",
    "Sexo": "Sexo registrado en el dataset (Sex); variable categórica.",
    "FE_2CH": "Fracción de eyección (%) declarada en Info_2CH.cfg (EF).",
    "FE_4CH": "Fracción de eyección (%) declarada en Info_4CH.cfg (EF).",
    "Calidad_2CH": "Calidad de imagen de la vista de dos cámaras (ImageQuality).",
    "Calidad_4CH": "Calidad de imagen de la vista de cuatro cámaras.",
    "Fotogramas_2CH": "Cantidad de fotogramas declarada en la vista 2CH (NbFrame).",
    "Fotogramas_4CH": "Cantidad de fotogramas declarada en la vista 4CH.",
    "FPS_2CH": "Frecuencia de adquisición de la vista 2CH (FrameRate).",
    "FPS_4CH": "Frecuencia de adquisición de la vista 4CH.",
    "Archivos_imagen": "Cantidad de archivos de imágenes, sin máscaras de segmentación.",
    "Archivos_mascara": "Cantidad de archivos de segmentación (_gt).",
}


# %% 3. Leer los metadatos locales y armar una fila por paciente
def leer_cfg(ruta):
    """Los metadatos de CAMUS usan líneas del tipo 'Age: 65'."""
    datos = {}
    if not ruta.exists():
        return datos
    for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
        clave, separador, valor = linea.partition(":")
        if separador:
            datos[clave.strip()] = valor.strip()
    return datos


def construir_dataframe(ruta):
    """Reunir ambas vistas por paciente y registrar los problemas encontrados."""
    filas, incidencias = [], []

    def registrar(paciente, campo, detalle):
        incidencias.append({"Paciente": paciente, "Campo": campo, "Detalle": detalle})

    def numero(valor, paciente, campo):
        if valor is None or str(valor).strip().lower() in {"", "nan", "none", "na"}:
            return np.nan
        convertido = pd.to_numeric(str(valor).replace(",", "."), errors="coerce")
        if not np.isfinite(convertido):
            registrar(paciente, campo, f"Valor no numérico o no finito: {valor!r}")
            return np.nan
        return float(convertido)

    for carpeta in sorted(ruta.glob("patient*")):
        if not carpeta.is_dir():
            continue
        paciente = carpeta.name
        fila = {"Paciente": paciente}
        metadatos = {}
        imagenes = list(carpeta.glob("*.nii.gz"))
        fila["Archivos_imagen"] = sum("_gt" not in a.name for a in imagenes)
        fila["Archivos_mascara"] = len(imagenes) - fila["Archivos_imagen"]
        for vista in ("2CH", "4CH"):
            archivo_cfg = carpeta / f"Info_{vista}.cfg"
            datos = leer_cfg(archivo_cfg)
            metadatos[vista] = datos
            if not archivo_cfg.exists():
                registrar(paciente, f"Info_{vista}", "Falta el archivo de metadatos.")
            for destino, origen in (("FE", "EF"), ("Fotogramas", "NbFrame"), ("FPS", "FrameRate")):
                fila[f"{destino}_{vista}"] = numero(datos.get(origen), paciente, f"{destino}_{vista}")
            fila[f"Calidad_{vista}"] = datos.get("ImageQuality") or pd.NA
            if not any(f"_{vista}_" in a.name for a in imagenes if "_gt" not in a.name):
                registrar(paciente, vista, "No se encontraron archivos de imagen de esta vista.")

        # Edad y sexo son datos del paciente. No promediamos valores discrepantes.
        for campo, destino in (("Age", "Edad"), ("Sex", "Sexo")):
            valores = []
            for vista in ("2CH", "4CH"):
                valor = metadatos[vista].get(campo)
                if campo == "Age":
                    valor = numero(valor, paciente, f"Edad_{vista}")
                else:
                    valor = valor.strip().upper() if valor else pd.NA
                if pd.notna(valor):
                    valores.append(valor)
            unicos = list(dict.fromkeys(valores))
            fila[destino] = unicos[0] if len(unicos) == 1 else (np.nan if campo == "Age" else pd.NA)
            if len(unicos) > 1:
                registrar(paciente, destino, f"Discrepancia entre vistas: {unicos}; revisar antes de usar.")
            elif not unicos:
                registrar(paciente, destino, "No disponible en los metadatos.")
        filas.append(fila)

    if not filas:
        raise ValueError(f"No se encontraron carpetas de pacientes en: {ruta}")
    columnas = list(DICCIONARIO)
    df = pd.DataFrame(filas)[columnas]
    problemas = pd.DataFrame(incidencias, columns=["Paciente", "Campo", "Detalle"])
    return df, problemas


# %% 4. Validación de estructura y Data Wrangling
def preparar_datos(df):
    # Como en el notebook, conservamos el DataFrame original.
    df2 = df.copy()
    for columna in ("Sexo", "Calidad_2CH", "Calidad_4CH"):
        df2[columna] = df2[columna].astype("category")
    # La edad permanece numérica: es el objetivo de regresión.
    # No imputamos edades, eliminamos extremos ni balanceamos registros aquí.
    return df2


def analizar_tabla(df, df2, problemas, salida):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\nDICCIONARIO DE DATOS")
    for variable, descripcion in DICCIONARIO.items():
        print(f"  {variable}: {descripcion}")
    print("\nPRIMEROS REGISTROS\n", df.head().to_string(index=False))
    print("\nÚLTIMOS REGISTROS\n", df.tail().to_string(index=False))
    print("\nDIMENSIÓN (pacientes, variables):", df.shape)
    print("COLUMNAS:", df.columns.tolist())
    print("\nTIPOS Y VALORES NO NULOS")
    df2.info()
    pacientes = df2["Paciente"].nunique()
    print(f"\nPacientes encontrados: {pacientes}; esperados: {PACIENTES_ESPERADOS}")
    if pacientes != PACIENTES_ESPERADOS:
        warnings.warn("La copia analizada no contiene exactamente 500 pacientes. Revisá su estructura.")
    print("Identificadores duplicados:", df2["Paciente"].duplicated().sum())
    # Coincidir en metadatos no implica que dos ecocardiogramas sean duplicados.
    print("Filas con metadatos idénticos (sin ID):", df2.drop(columns="Paciente").duplicated().sum())
    print("Este conteo no verifica imágenes duplicadas ni justifica eliminar pacientes.")

    faltantes = pd.DataFrame({"Cantidad": df2.isna().sum(), "Porcentaje": df2.isna().mean() * 100})
    print("\nDATOS FALTANTES\n", faltantes.to_string())
    descriptivas = df2.describe(include="all")
    print("\nESTADÍSTICAS DESCRIPTIVAS\n", descriptivas.to_string())
    print("\nEDAD POR SEXO\n", df2.groupby("Sexo", observed=True)["Edad"].describe().to_string())
    print("\nFRECUENCIAS CATEGÓRICAS")
    for columna in ("Sexo", "Calidad_2CH", "Calidad_4CH"):
        print(f"\n{columna}\n{df2[columna].value_counts(dropna=False).to_string()}")
    print("\nCALIDAD 2CH VS. 4CH\n", pd.crosstab(df2["Calidad_2CH"], df2["Calidad_4CH"]).to_string())

    # Señalamos valores fuera de dominio y extremos por IQR; no los borramos.
    validaciones = []
    for columna in df2.select_dtypes(include="number").columns:
        valores = df2[columna].dropna()
        q1, q3 = valores.quantile([0.25, 0.75])
        iqr = q3 - q1
        extremos = (df2[columna] < q1 - 1.5 * iqr) | (df2[columna] > q3 + 1.5 * iqr)
        for paciente in df2.loc[extremos, "Paciente"]:
            validaciones.append({"Paciente": paciente, "Campo": columna, "Detalle": "Extremo por IQR; revisar, no implica error."})
        if columna == "Edad":
            fuera = df2[columna].le(0) | df2[columna].gt(120)
        elif columna.startswith("FE_"):
            fuera = df2[columna].lt(0) | df2[columna].gt(100)
        elif columna.startswith(("Fotogramas_", "FPS_")):
            fuera = df2[columna].le(0)
        else:
            continue
        for paciente in df2.loc[fuera, "Paciente"]:
            validaciones.append({"Paciente": paciente, "Campo": columna, "Detalle": "Valor fuera del rango esperado; verificar en la fuente."})
    revisiones = pd.DataFrame(validaciones, columns=problemas.columns)
    problemas = pd.concat([problemas, revisiones], ignore_index=True)
    print(f"\nObservaciones para revisar: {len(problemas)} (ver observaciones.csv).")
    if df2["Edad"].isna().all():
        warnings.warn("Esta copia no incluye edades utilizables. No permite entrenar la regresión sin conseguir esas etiquetas.")

    # Excluimos ID y conteos de archivos de la correlación. No codificamos sexo
    # ni calidad con números arbitrarios. Los parámetros de adquisición no
    # deben interpretarse como características biológicas.
    variables = ["Edad", "FE_2CH", "FE_4CH", "Fotogramas_2CH", "Fotogramas_4CH", "FPS_2CH", "FPS_4CH"]
    correlacion = df2[variables].corr(method="pearson", min_periods=3)
    print("\nCORRELACIONES DE PEARSON\n", correlacion.round(3).to_string())
    print("Correlación exploratoria no implica causalidad ni rendimiento predictivo.")

    df2.to_csv(salida / "pacientes_camus.csv", index=False, encoding="utf-8-sig")
    faltantes.to_csv(salida / "datos_faltantes.csv", encoding="utf-8-sig")
    descriptivas.to_csv(salida / "estadisticas_descriptivas.csv", encoding="utf-8-sig")
    correlacion.to_csv(salida / "correlaciones.csv", encoding="utf-8-sig")
    problemas.to_csv(salida / "observaciones.csv", index=False, encoding="utf-8-sig")
    pd.Series(DICCIONARIO, name="Descripción").rename_axis("Variable").to_csv(
        salida / "diccionario_variables.csv", encoding="utf-8-sig"
    )
    return correlacion


# %% 5. Exploratory Data Analysis: gráficos
def graficar_datos(df2, correlacion, salida):
    import matplotlib.pyplot as plt
    from pandas.plotting import scatter_matrix

    plt.style.use("ggplot")

    def guardar(figura, nombre):
        figura.tight_layout()
        figura.savefig(salida / nombre, dpi=160, bbox_inches="tight")
        if not MOSTRAR_GRAFICOS:
            plt.close(figura)

    # Equivalente a missingno.matrix sin agregar una dependencia.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(df2.isna().to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(df2.columns)), df2.columns, rotation=70, ha="right")
    ax.set(ylabel="Pacientes", title="Datos faltantes: negro = faltante; blanco = disponible")
    guardar(fig, "01_matriz_faltantes.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, columna in zip(axes, ["Sexo", "Calidad_2CH", "Calidad_4CH"]):
        conteos = df2[columna].astype("string").fillna("Sin dato").value_counts()
        conteos.plot.bar(ax=ax, rot=20, color="steelblue")
        ax.set(title=columna, ylabel="Pacientes")
    guardar(fig, "02_frecuencias_categoricas.png")

    edades = df2["Edad"].dropna()
    if not edades.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(edades, bins="auto", edgecolor="white", color="steelblue")
        axes[0].set(title="Distribución de la edad objetivo", xlabel="Edad (años)", ylabel="Pacientes")
        axes[1].boxplot(edades.to_numpy(), orientation="horizontal")
        axes[1].set(title="Edad: mediana, dispersión y extremos", xlabel="Edad (años)")
        guardar(fig, "03_distribucion_edad.png")

        # Los intervalos se usan solo para visualizar cobertura; y sigue continua.
        maximo = max(10, (math.floor(edades.max() / 10) + 1) * 10)
        bordes = np.arange(0, maximo + 1, 10)
        grupos = pd.cut(df2["Edad"], bordes, right=False)
        conteos = grupos.value_counts(sort=False)
        conteos.rename_axis("Rango de edad").to_csv(salida / "distribucion_edades.csv", encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(11, 4))
        conteos.plot.bar(ax=ax, rot=40, color="steelblue")
        ax.set(title="Cobertura por intervalos de edad", xlabel="Edad (años)", ylabel="Pacientes")
        guardar(fig, "04_cobertura_edades.png")

    grupos_sexo = [(str(sexo), grupo["Edad"].dropna().to_numpy())
                   for sexo, grupo in df2.groupby("Sexo", observed=True) if grupo["Edad"].notna().any()]
    if grupos_sexo:
        etiquetas, muestras = zip(*grupos_sexo)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].boxplot(muestras, tick_labels=etiquetas)
        axes[0].set(title="Edad por sexo: boxplot", xlabel="Sexo", ylabel="Edad (años)")
        # La estimación de densidad requiere más de un valor y variación.
        utilizables = [(e, m) for e, m in grupos_sexo if len(m) > 1 and np.ptp(m) > 0]
        if utilizables:
            axes[1].violinplot([m for _, m in utilizables], showmedians=True)
            axes[1].set_xticks(range(1, len(utilizables) + 1), [e for e, _ in utilizables])
        axes[1].set(title="Edad por sexo: violinplot", xlabel="Sexo", ylabel="Edad (años)")
        guardar(fig, "05_edad_por_sexo.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vista in zip(axes, ("2CH", "4CH")):
        ax.scatter(df2["Edad"], df2[f"FE_{vista}"], alpha=0.5, s=20)
        ax.set(title=f"Edad y fracción de eyección ({vista})", xlabel="Edad (años)", ylabel="FE (%)")
    guardar(fig, "06_edad_fraccion_eyeccion.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    matriz = np.ma.masked_invalid(correlacion.to_numpy(dtype=float))
    imagen = ax.imshow(matriz, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlacion)), correlacion.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(correlacion)), correlacion.index)
    for i in range(len(correlacion)):
        for j in range(len(correlacion)):
            valor = correlacion.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}" if pd.notna(valor) else "N/D", ha="center", va="center", fontsize=8)
    ax.set_title("Correlaciones: metadatos clínicos y de adquisición")
    fig.colorbar(imagen, ax=ax, shrink=0.8)
    guardar(fig, "07_correlaciones.png")

    # Equivalente al pairplot del notebook. Se omiten columnas constantes/vacías
    # y se usan las mismas filas completas en todos los paneles.
    columnas = [c for c in ("Edad", "FE_2CH", "FE_4CH") if df2[c].nunique() > 1]
    completos = df2[columnas].dropna()
    if len(columnas) >= 2 and len(completos) >= 3:
        ejes = scatter_matrix(completos, figsize=(10, 10), diagonal="hist", alpha=0.4)
        fig = ejes[0, 0].figure
        fig.suptitle(f"Relaciones entre variables: {len(completos)} pacientes con datos completos", y=1.02)
        guardar(fig, "08_relaciones_variables.png")


# %% 6. Inspección visual de algunos ecocardiogramas
def mostrar_ejemplos(ruta, salida):
    """Leer hasta cuatro imágenes ED NIfTI; nunca usar las máscaras como entrada."""
    import matplotlib.pyplot as plt
    try:
        import nibabel as nib
    except ImportError:
        print("\nPara ver imágenes NIfTI instalá: python -m pip install nibabel")
        return
    candidatos = sorted(ruta.glob("patient*/*_ED.nii.gz"))[:4]
    if not candidatos:
        print("\nNo hay imágenes ED NIfTI para la vista previa. Los metadatos sí se analizaron.")
        return
    fig, axes = plt.subplots(1, min(4, len(candidatos)), figsize=(14, 4), squeeze=False)
    for ax, archivo in zip(axes.flat, candidatos[:4]):
        try:
            imagen = nib.load(archivo)
            datos = np.asanyarray(imagen.dataobj)
            # Un archivo ED contiene un fotograma; quitamos dimensiones de tamaño 1.
            datos = np.squeeze(datos)
            if datos.ndim != 2:
                raise ValueError(f"Se esperaba una imagen ED 2D y se obtuvo {datos.shape}.")
            ax.imshow(datos.T, cmap="gray", origin="lower", aspect="equal")
            ax.set_title(archivo.name.replace(".nii.gz", ""), fontsize=8)
            print(f"Imagen de ejemplo: {archivo}; dimensiones={datos.shape}; dtype={datos.dtype}")
        except (ValueError, OSError, EOFError) as error:
            warnings.warn(f"No se pudo visualizar {archivo}: {error}")
        ax.axis("off")
    fig.suptitle("Ejemplos ED sin máscaras; orientación según la matriz almacenada")
    fig.tight_layout()
    fig.savefig(salida / "09_ejemplos_ecocardiogramas.png", dpi=160, bbox_inches="tight")
    if not MOSTRAR_GRAFICOS:
        plt.close(fig)


# %% 7. Ejecutar el análisis completo
def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df, problemas = construir_dataframe(RUTA_DATASET)
    df2 = preparar_datos(df)
    correlacion = analizar_tabla(df, df2, problemas, CARPETA_RESULTADOS)
    graficar_datos(df2, correlacion, CARPETA_RESULTADOS)
    mostrar_ejemplos(RUTA_DATASET, CARPETA_RESULTADOS)
    print(f"\nAnálisis terminado. Tablas y gráficos en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICOS:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()

r"""Embedding explícito y prueba de memorización, SOLO con pacientes de train.

Ejecutar:
    .\.venv\Scripts\python.exe "Diagnostico embedding.py"

1) CNN original desde cero, sin aumentos, sobremuestreo ni dropout.
2) Exportar sus embeddings de 64 dimensiones, sin anexar etiquetas como features.
3) Congelar esos embeddings y ajustar un MLP nuevo a los mismos 16 pacientes.

No se leen imágenes ni se calculan errores de validación/test. La partición
se toma del primer entrenamiento existente para conservar exactamente sus IDs.
Los resultados miden MEMORIZACIÓN, no capacidad de predecir pacientes nuevos.
"""

from pathlib import Path
from datetime import datetime
import importlib.util
import json
import time

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cnn_camus", BASE / "ModeloRegresion CNN.py")
cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cnn)
np, pd, keras, plt = cnn.np, cnn.pd, cnn.keras, cnn.plt
from sklearn.decomposition import PCA

# %% Configuración del diagnóstico
PARTICIONES = BASE / "resultados_cnn" / "entrenamiento_20261112_231222_995627" / "particiones.csv"
N_PACIENTES = 16
MAX_EPOCAS = 1000
UMBRAL_MAE = 1.0  # Criterio operativo del diagnóstico, no objetivo clínico.


def entrenar_para_memorizar(modelo, X, y, salida, nombre):
    """Sin validación ni early stopping por validación; aprender el mismo lote."""
    umbral_parada = 0.75 * UMBRAL_MAE
    class Progreso(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.mejor_mae, self.pesos = np.inf, None
            self.inicio = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            mae = float(logs["mae"])
            if not np.isfinite(mae):
                raise RuntimeError("MAE no finito durante el diagnóstico.")
            if mae < self.mejor_mae:
                self.mejor_mae, self.pesos = mae, self.model.get_weights()
            if epoch == 0 or (epoch + 1) % 50 == 0 or mae <= umbral_parada:
                print(f"{nombre}: época {epoch + 1}, MAE={mae:.3f} años, "
                      f"{time.monotonic() - self.inicio:.0f} s", flush=True)
            if mae <= umbral_parada:
                self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.pesos is not None:
                self.model.set_weights(self.pesos)

    historial = modelo.fit(
        X, y, batch_size=len(y), epochs=MAX_EPOCAS, shuffle=False, verbose=0,
        callbacks=[Progreso(), keras.callbacks.TerminateOnNaN(),
                   keras.callbacks.CSVLogger(str(salida / f"historial_{nombre}.csv"))],
    )
    pred = cnn.makePredictions(modelo, X, y)
    modelo.save(salida / f"modelo_{nombre}.keras")
    return historial, pred


def main():
    keras.utils.set_random_seed(cnn.SEMILLA)
    cnn.torch.set_num_threads(4)
    plt.style.use("ggplot")
    if not PARTICIONES.is_file():
        raise FileNotFoundError(f"Indicá la ruta de particiones.csv en PARTICIONES: {PARTICIONES}")
    tabla = pd.read_csv(PARTICIONES)
    if not tabla["Paciente"].is_unique:
        raise ValueError("Hay IDs duplicados en las particiones.")
    train = tabla.loc[tabla["Particion"].eq("train")].sort_values(["Edad", "Paciente"])
    if len(train) < N_PACIENTES:
        raise ValueError("No hay suficientes pacientes de entrenamiento.")
    # Distribuir los 16 ejemplos a lo largo de las edades disponibles EN TRAIN.
    indices = np.round(np.linspace(0, len(train) - 1, N_PACIENTES)).astype(int)
    pacientes = train.iloc[indices].reset_index(drop=True)
    assert set(pacientes.Paciente).isdisjoint(set(tabla.loc[tabla.Particion.ne("train"), "Paciente"]))
    assert pacientes.Paciente.is_unique
    salida = cnn.CARPETA_RESULTADOS / f"memorizacion_{datetime.now():%Y%m%d_%H%M%S_%f}"
    salida.mkdir(parents=True, exist_ok=False)
    pacientes.to_csv(salida / "pacientes_solo_train.csv", index=False)
    config = {
        "particiones_origen": str(PARTICIONES), "n_pacientes": len(pacientes),
        "seleccion": "edades distribuidas dentro de train", "epocas_maximas": MAX_EPOCAS,
        "umbral_mae_anios": UMBRAL_MAE, "dim_embedding": 64,
        "aumentos": False, "sobremuestreo": False, "dropout": False,
        "pesos_preentrenados": False, "evaluacion": "mismos pacientes usados para ajustar",
        "validez": "diagnóstico de memorización; no estima generalización",
        "semilla": cnn.SEMILLA,
    }
    (salida / "configuracion.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Diagnóstico: {len(pacientes)} pacientes SOLO DE TRAIN. Resultados: {salida}", flush=True)
    X = cnn.prep_pixels(pacientes)
    y = pacientes.Edad.to_numpy(dtype=np.float32)
    modelo = cnn.myModel(X.shape[1:], float(y.mean()), aumentos=False, dropout=0)
    assert not any(isinstance(c, (keras.layers.Dropout, keras.layers.RandomRotation,
                                  keras.layers.RandomTranslation)) for c in modelo.layers)
    extractor = cnn.crear_extractor_embeddings(modelo)
    Z_inicial = np.asarray(extractor.predict(X, verbose=0))
    np.save(salida / "embeddings_antes_entrenar.npy", Z_inicial)
    hist_cnn, pred_cnn = entrenar_para_memorizar(modelo, X, y, salida, "cnn")
    Z = cnn.guardar_embeddings(modelo, X, pacientes, salida / "embedding")

    # %% MLP sobre embeddings congelados (no vuelve a modificar el extractor).
    # Se ajusta el escalado SOLO con los 16 pacientes de esta prueba.
    media, escala = Z.mean(axis=0), Z.std(axis=0)
    escala = np.where(escala > 1e-8, escala, 1.0)
    Z_escalado = ((Z - media) / escala).astype(np.float32)
    np.savez(salida / "escalado_embedding.npz", media=media, escala=escala)
    keras.utils.set_random_seed(cnn.SEMILLA)
    mlp = keras.Sequential([
        keras.Input(shape=(64,)), keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="linear", bias_initializer=keras.initializers.Constant(float(y.mean()))),
    ], name="MLP_embedding_congelado")
    mlp.compile(optimizer=keras.optimizers.Adam(0.003), loss="mse", metrics=["mae"], jit_compile=False)
    hist_mlp, pred_mlp = entrenar_para_memorizar(mlp, Z_escalado, y, salida, "mlp_embedding")
    referencia = np.full(len(y), np.median(y))
    metricas = pd.DataFrame([
        {"Modelo": "Mediana de los 16", **cnn.metricas_regresion(y, referencia)},
        {"Modelo": "CNN sin regularización", **cnn.metricas_regresion(y, pred_cnn)},
        {"Modelo": "MLP sobre embedding congelado", **cnn.metricas_regresion(y, pred_mlp)},
    ])
    metricas.to_csv(salida / "metricas_memorizacion.csv", index=False)
    resultados = pacientes[["Paciente", "Edad", "Sexo"]].copy()
    resultados["Pred_CNN"] = pred_cnn
    resultados["Pred_MLP_embedding"] = pred_mlp
    resultados.to_csv(salida / "predicciones_mismos_pacientes.csv", index=False)
    resumen = {
        "cnn_memoriza": bool(metricas.iloc[1].MAE < UMBRAL_MAE),
        "mlp_embedding_memoriza": bool(metricas.iloc[2].MAE < UMBRAL_MAE),
        "mae_cnn": float(metricas.iloc[1].MAE), "mae_mlp_embedding": float(metricas.iloc[2].MAE),
        "epocas_cnn": len(hist_cnn.history["loss"]), "epocas_mlp": len(hist_mlp.history["loss"]),
        "std_pred_cnn": float(np.std(pred_cnn)), "std_edades": float(np.std(y)),
        "features_con_variacion": int(np.sum(Z.std(axis=0) > 1e-8)),
        "rango_matriz_embedding_centrada": int(np.linalg.matrix_rank(Z - Z.mean(axis=0))),
        "interpretacion": "Solo ajuste a los mismos 16 pacientes; no hay validación ni test en este resultado.",
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    # %% Gráficos del diagnóstico: no etiquetarlos como validación.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, nombre in zip(axes, [hist_cnn, hist_mlp], ["CNN", "MLP sobre embedding"]):
        ax.plot(np.arange(1, len(hist.history["mae"]) + 1), hist.history["mae"])
        ax.axhline(UMBRAL_MAE, linestyle="--", color="black", label="Umbral diagnóstico: 1 año")
        ax.set(xlabel="Época", ylabel="MAE de entrenamiento (años)", title=nombre)
        ax.legend()
    fig.suptitle("Memorización de 16 pacientes de train — no mide generalización")
    cnn.guardar_figura(fig, salida, "01_memorizacion_curvas.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, pred_cnn, label="CNN", marker="o")
    ax.scatter(y, pred_mlp, label="MLP sobre embedding", marker="x")
    limites = [min(y.min(), pred_cnn.min(), pred_mlp.min()) - 2,
               max(y.max(), pred_cnn.max(), pred_mlp.max()) + 2]
    ax.plot(limites, limites, "k--")
    ax.set(xlabel="Edad real (años)", ylabel="Edad predicha (años)",
           title="Mismos 16 pacientes utilizados para entrenar", xlim=limites, ylim=limites)
    ax.legend()
    cnn.guardar_figura(fig, salida, "02_memorizacion_predicciones.png")
    pca = PCA(n_components=2)
    puntos = pca.fit_transform(Z_escalado)
    pd.DataFrame({"Paciente": pacientes.Paciente, "PC1": puntos[:, 0], "PC2": puntos[:, 1]}).to_csv(
        salida / "embedding_pca_2d.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.scatter(puntos[:, 0], puntos[:, 1], c=y, cmap="viridis", s=60)
    for x, yy, edad in zip(puntos[:, 0], puntos[:, 1], y):
        ax.annotate(f"{edad:.0f}", (x, yy), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel=f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f} %)",
           ylabel=f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f} %)",
           title="Embedding de 64 valores proyectado a 2D — solo train")
    fig.colorbar(im, ax=ax, label="Edad real (solo para colorear)")
    cnn.guardar_figura(fig, salida, "03_embedding_pca.png")
    print("\nRESULTADOS DE MEMORIZACIÓN (no son de validación/test)", flush=True)
    print(metricas.round(3).to_string(index=False), flush=True)
    print(json.dumps(resumen, indent=2), flush=True)
    print(f"\nArchivos guardados en: {salida}", flush=True)


if __name__ == "__main__":
    main()

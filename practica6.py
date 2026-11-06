# Práctica 6 - Sistemas de Ayuda a la Decisión
# Clasificación de provincias según su nivel de seguridad con AHPSort II
# Manuel Villegas Aceituno

import os
import numpy as np
import pandas as pd
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

CRITERIOS = ["Crímenes", "Robos", "Vehículos", "Orden público", "Drogas"]
CLASES = ["Alto", "Medio", "Bajo"]          # nivel de seguridad (CP1, CP2, CP3)
COLORES = {"Alto": "#4caf50", "Medio": "#ffd54f", "Bajo": "#e53935"}
RI = {3: 0.58, 4: 0.9, 5: 1.12}

os.makedirs("resultados", exist_ok=True)


# ---------------------------------------------------------------------------
# Funciones del método
# ---------------------------------------------------------------------------

def pesos_ahp(A):
    """Pesos por el método de autovalores, CI y CR."""
    A = np.array(A, dtype=float)
    n = len(A)
    valores, vectores = np.linalg.eig(A)
    k = np.argmax(valores.real)
    lam = valores[k].real
    w = np.abs(vectores[:, k].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1)
    cr = ci / RI[n]
    return w, lam, ci, cr


def prioridad_local(x, puntos):
    """Función de prioridad local lineal a trozos (AHPSort II, ec. 2).
    puntos: lista de (valor, prioridad). Entre dos puntos consecutivos s_o y s_o+1:
    p = p_o + (p_o+1 - p_o) / (s_o+1 - s_o) * (x - s_o)"""
    puntos = sorted(puntos)
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]
    x = min(max(x, xs[0]), xs[-1])          # truncamiento al rango RP1-RP6
    return float(np.interp(x, xs, ys))


def prioridades(datos, funciones):
    """Tabla de prioridades locales de cada alternativa en cada criterio."""
    tabla = pd.DataFrame(index=datos.index, columns=CRITERIOS, dtype=float)
    for c in CRITERIOS:
        for a in datos.index:
            tabla.loc[a, c] = prioridad_local(datos.loc[a, c], funciones[c])
    return tabla


def clasificar(p, cp):
    """Asigna la clase del perfil central más cercano (cp ordenado de mayor a menor)."""
    if p >= cp[0]:
        return CLASES[0]
    if p < cp[-1]:
        return CLASES[-1]
    for i in range(len(cp) - 1):
        if cp[i + 1] <= p < cp[i]:
            # en caso de empate se usa la visión optimista (clase superior)
            return CLASES[i] if (cp[i] - p) <= (p - cp[i + 1]) else CLASES[i + 1]


def resolver(datos, funciones, valores_cp, w, titulo):
    """Calcula prioridades de los perfiles centrales y de las alternativas y clasifica."""
    print("\n" + "=" * 70 + "\n" + titulo + "\n" + "=" * 70)

    cp_local = pd.DataFrame(index=["CP1 (Alto)", "CP2 (Medio)", "CP3 (Bajo)"], columns=CRITERIOS, dtype=float)
    for i, fila in enumerate(cp_local.index):
        for c in CRITERIOS:
            cp_local.loc[fila, c] = prioridad_local(valores_cp[c][i], funciones[c])
    cp_local["Global"] = cp_local[CRITERIOS].values @ w
    print("\nPrioridades locales y globales de los perfiles centrales:")
    print(cp_local.round(3).to_string())

    loc = prioridades(datos, funciones)
    loc["Global"] = loc[CRITERIOS].values @ w
    loc["Clase"] = [clasificar(p, cp_local["Global"].values) for p in loc["Global"]]
    print("\nPrioridades locales, global y clasificación:")
    print(loc.round(3).to_string())
    return cp_local, loc


# ---------------------------------------------------------------------------
# Mapas
# ---------------------------------------------------------------------------

CODIGOS = {"álava": "01", "albacete": "02", "alicante": "03", "almería": "04", "ávila": "05", "badajoz": "06",
           "balears": "07", "baleares": "07", "barcelona": "08", "burgos": "09", "cáceres": "10", "cádiz": "11",
           "castell": "12", "ciudad real": "13", "córdoba": "14", "coruña": "15", "cuenca": "16", "girona": "17",
           "granada": "18", "guadalajara": "19", "gipuzkoa": "20", "huelva": "21", "huesca": "22", "jaén": "23",
           "león": "24", "lleida": "25", "rioja": "26", "lugo": "27", "madrid": "28", "málaga": "29",
           "murcia": "30", "navarra": "31", "ourense": "32", "asturias": "33", "palencia": "34", "palmas": "35",
           "pontevedra": "36", "salamanca": "37", "santa cruz": "38", "cantabria": "39", "segovia": "40",
           "sevilla": "41", "soria": "42", "tarragona": "43", "teruel": "44", "toledo": "45", "valencia": "46",
           "valladolid": "47", "bizkaia": "48", "zamora": "49", "zaragoza": "50"}


def codigo(nombre):
    n = nombre.lower()
    for clave, cod in CODIGOS.items():
        if clave in n:
            return cod
    return None


def dibujar_mapa(clases, fichero_shp, campo_codigo, titulo, salida, latlon):
    """Pinta cada provincia del shapefile con el color de su clase."""
    r = shapefile.Reader(fichero_shp, encoding="utf-8")
    nombres = [f[0] for f in r.fields[1:]]
    por_codigo = {codigo(a): c for a, c in clases.items()}
    fig, ax = plt.subplots(figsize=(10, 7))
    for shp, rec in zip(r.shapes(), r.records()):
        rec = dict(zip(nombres, rec))
        cod = rec[campo_codigo][4:6] if campo_codigo == "NATCODE" else rec[campo_codigo]
        clase = por_codigo.get(cod)
        color = COLORES[clase] if clase else "#dddddd"
        partes = list(shp.parts) + [len(shp.points)]
        for i in range(len(partes) - 1):
            ax.add_patch(Polygon(shp.points[partes[i]:partes[i + 1]], closed=True,
                                 facecolor=color, edgecolor="black", linewidth=0.4))
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.radians(40)) if latlon else 1)
    ax.axis("off")
    ax.set_title(titulo)
    ax.legend(handles=[Patch(facecolor=COLORES[c], edgecolor="black", label=f"Nivel {c.lower()}") for c in CLASES],
              loc="lower right")
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dibujar_funciones(funciones, valores_cp, titulo, salida):
    """Gráficas de las funciones de prioridad local (puntos rojos = perfiles centrales)."""
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, c in zip(axs.flat, CRITERIOS):
        pts = sorted(funciones[c])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        ax.plot(valores_cp[c], [prioridad_local(v, funciones[c]) for v in valores_cp[c]], "D", color="red")
        ax.set_title(c)
        ax.set_xlabel("Valor")
        ax.set_ylabel("Prioridad local")
        ax.grid(alpha=0.3)
    axs.flat[-1].axis("off")
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(salida, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def leer_andalucia():
    """Lee el resumen por provincia (columnas 3 y 4) del fichero Andalucía.xls."""
    df = pd.read_excel("datos/Andalucia.xls", header=None)
    provincias = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"]
    datos = {}
    for i, fila in df.iterrows():
        if fila[0] in provincias:
            datos[fila[0]] = [float(df.loc[i + 1 + k, 3]) for k in range(5)]
    return pd.DataFrame(datos, index=CRITERIOS).T


def leer_habitantes():
    df = pd.read_excel("datos/habitantes.xlsx", header=None, skiprows=2)
    return pd.Series(df[2].values, index=df[1].values)


# Matriz de comparación de los criterios (Tabla 4)
MATRIZ_CRITERIOS = [[1, 5, 3, 7, 3],
                    [1 / 5, 1, 3, 1 / 2, 1 / 4],
                    [1 / 3, 1 / 3, 1, 1 / 5, 1 / 5],
                    [1 / 7, 2, 5, 1, 1 / 5],
                    [1 / 3, 4, 5, 5, 1]]

# Ejercicio 1: perfiles centrales (Tabla 3) y puntos de las funciones (Tabla 6)
CP_EJ1 = {"Crímenes": [20, 100, 160], "Robos": [5000, 95000, 180000], "Vehículos": [5000, 19000, 27000],
          "Orden público": [300, 1800, 2500], "Drogas": [300, 600, 900]}
FUN_EJ1 = {
    "Crímenes": [(0, 0.336), (20, 0.277), (41.2, 0.132), (82.4, 0.086), (100, 0.058), (123.6, 0.032),
                 (160, 0.029), (164.8, 0.026), (206, 0.023)],
    "Robos": [(0, 0.32), (5000, 0.261), (50914.4, 0.139), (95000, 0.075), (101828.8, 0.074), (152743.2, 0.047),
              (180000, 0.032), (203657.6, 0.031), (254572, 0.022)],
    "Vehículos": [(0, 0.459), (5000, 0.138), (7025.4, 0.124), (14050.8, 0.069), (19000, 0.063), (21076.2, 0.048),
                  (27000, 0.04), (28101.6, 0.034), (35127, 0.026)],
    "Orden público": [(0, 0.308), (300, 0.259), (764.8, 0.149), (1529.6, 0.071), (1800, 0.066), (2294.4, 0.06),
                      (2500, 0.034), (3059.2, 0.032), (3824, 0.023)],
    "Drogas": [(0, 0.385), (300, 0.262), (321.6, 0.112), (600, 0.07), (643.2, 0.062), (900, 0.035), (964.8, 0.033),
               (1286.4, 0.023), (1608, 0.018)]}

# Ejercicio 2: perfiles centrales (Tabla 8) y puntos de las funciones (Tabla 10)
CP_EJ2 = {"Crímenes": [0.26, 0.34, 0.41], "Robos": [194.29, 215.32, 261.32], "Vehículos": [22.95, 28.44, 36.20],
          "Orden público": [2.87, 3.86, 5.32], "Drogas": [2.06, 3.23, 6.25]}
FUN_EJ2 = {
    "Crímenes": [(0, 0.34), (0.10, 0.13), (0.20, 0.09), (0.26, 0.05), (0.30, 0.03), (0.34, 0.03), (0.39, 0.03),
                 (0.41, 0.03), (0.49, 0.02)],
    "Robos": [(0, 0.32), (56.59, 0.14), (113.18, 0.07), (169.77, 0.05), (194.29, 0.04), (215.32, 0.03),
              (226.35, 0.03), (261.32, 0.03), (282.94, 0.02)],
    "Vehículos": [(0, 0.46), (9.96, 0.12), (19.93, 0.07), (22.95, 0.06), (28.44, 0.05), (29.89, 0.05),
                  (36.20, 0.04), (39.85, 0.03), (49.82, 0.03)],
    "Orden público": [(0, 0.31), (1.17, 0.15), (2.33, 0.07), (2.87, 0.07), (3.50, 0.06), (3.86, 0.05),
                      (4.66, 0.03), (5.32, 0.03), (5.83, 0.02)],
    "Drogas": [(0, 0.39), (1.36, 0.11), (2.06, 0.09), (2.73, 0.06), (3.23, 0.05), (4.09, 0.03), (5.45, 0.02),
               (6.25, 0.02), (6.81, 0.02)]}

# Ejercicios 3 y 4: prioridades de los 6 RP heredadas (Tabla 13)
PRIOR_RP = {"Crímenes": [0.336, 0.132, 0.086, 0.032, 0.026, 0.023],
            "Robos": [0.32, 0.139, 0.074, 0.047, 0.031, 0.022],
            "Vehículos": [0.459, 0.124, 0.069, 0.048, 0.034, 0.026],
            "Orden público": [0.308, 0.149, 0.071, 0.06, 0.032, 0.023],
            "Drogas": [0.385, 0.112, 0.062, 0.033, 0.023, 0.018]}


# ---------------------------------------------------------------------------
# Ejercicios
# ---------------------------------------------------------------------------

def main():
    # Pesos de los criterios
    w, lam, ci, cr = pesos_ahp(MATRIZ_CRITERIOS)
    print("Pesos de los criterios (autovalores):")
    for c, p in zip(CRITERIOS, w):
        print(f"  {c:15s} {p:.4f}")
    print(f"  lambda max = {lam:.4f}  CI = {ci:.4f}  CR = {cr:.4f}  ->",
          "válidos" if cr < 0.1 else "no válidos (CR > 0,1)")

    # ---------------- Ejercicio 1 ----------------
    datos1 = leer_andalucia()
    cp1, res1 = resolver(datos1, FUN_EJ1, CP_EJ1, w, "EJERCICIO 1 - Andalucía 2017 (datos absolutos)")
    res1.round(4).to_csv("resultados/ejercicio1.csv")
    dibujar_funciones(FUN_EJ1, CP_EJ1, "Ejercicio 1 - Funciones de prioridad local", "resultados/funciones_ej1.png")
    dibujar_mapa(res1["Clase"], "mapas/Andalucia/da03_provincia", "COD_PROV",
                 "Ejercicio 1 - Andalucía 2017 (sin normalizar)", "resultados/mapa_ej1.png", latlon=False)

    # ---------------- Ejercicio 2 ----------------
    hab = leer_habitantes()
    datos2 = datos1.div(hab.loc[datos1.index], axis=0) * 10000
    print("\nDatos de Andalucía por cada 10.000 habitantes:")
    print(datos2.round(3).to_string())
    print("RP calculados (de 0 al máximo en 5 intervalos iguales):")
    print(pd.DataFrame({c: [datos2[c].max() * k / 5 for k in range(6)] for c in CRITERIOS},
                       index=[f"RP{k + 1}" for k in range(6)]).round(2).to_string())
    cp2, res2 = resolver(datos2, FUN_EJ2, CP_EJ2, w, "EJERCICIO 2 - Andalucía 2017 (por 10.000 habitantes)")
    res2.round(4).to_csv("resultados/ejercicio2.csv")
    dibujar_funciones(FUN_EJ2, CP_EJ2, "Ejercicio 2 - Funciones de prioridad local", "resultados/funciones_ej2.png")
    dibujar_mapa(res2["Clase"], "mapas/Andalucia/da03_provincia", "COD_PROV",
                 "Ejercicio 2 - Andalucía 2017 (por 10.000 habitantes)", "resultados/mapa_ej2.png", latlon=False)

    # ---------------- Ejercicio 3 ----------------
    nac17 = pd.read_excel("datos/Datos_nacionales_2017_por_10000_ES.xlsx", index_col=0)
    # (a) comprobación de la normalización con los datos absolutos de Andalucía
    pob_implicita = datos1["Robos"] / nac17.loc[datos1.index, "Robos"] * 10000
    print("\nPoblación implícita en el fichero nacional (hechos / tasa x 10.000) frente a habitantes.xlsx:")
    print(pd.DataFrame({"Población implícita": pob_implicita.round(0),
                        "habitantes.xlsx": hab.loc[datos1.index]}).to_string())
    # (b) RP: de 0 al máximo nacional en 5 intervalos; CP: percentiles 20, 50 y 80
    rp3 = {c: [nac17[c].max() * k / 5 for k in range(6)] for c in CRITERIOS}
    cp3 = {c: [nac17[c].quantile(q) for q in (0.2, 0.5, 0.8)] for c in CRITERIOS}
    print("RP nacionales:")
    print(pd.DataFrame(rp3, index=[f"RP{k + 1}" for k in range(6)]).round(2).to_string())
    print("CP nacionales (percentiles 20, 50 y 80):")
    print(pd.DataFrame(cp3, index=["CP1", "CP2", "CP3"]).round(2).to_string())
    fun3 = {c: list(zip(rp3[c], PRIOR_RP[c])) for c in CRITERIOS}
    cp3_tab, res3 = resolver(nac17, fun3, cp3, w, "EJERCICIO 3 - España 2017 (por 10.000 habitantes)")
    res3.round(4).to_csv("resultados/ejercicio3.csv")
    dibujar_funciones(fun3, cp3, "Ejercicio 3 - Funciones de prioridad local (2017)", "resultados/funciones_ej3.png")
    dibujar_mapa(res3["Clase"], "mapas/Espana/recintos_provinciales_inspire_peninbal_etrs89", "NATCODE",
                 "Nivel de seguridad por provincias - 2017", "resultados/mapa_ej3_2017.png", latlon=True)

    # ---------------- Ejercicio 4 ----------------
    df23 = pd.read_excel("datos/Datos_nacionales_2023_por_10000_ES.xlsx", index_col=0)
    nac23 = df23[CRITERIOS]
    fuera = (nac23 > pd.Series({c: rp3[c][-1] for c in CRITERIOS})).sum()
    print("\nValores de 2023 por encima de RP6 (se truncan a RP6):")
    print(fuera.to_string())
    _, res4 = resolver(nac23, fun3, cp3, w, "EJERCICIO 4 - España 2023 (pesos, CP, RP y funciones de 2017)")
    res4.round(4).to_csv("resultados/ejercicio4.csv")
    dibujar_mapa(res4["Clase"], "mapas/Espana/recintos_provinciales_inspire_peninbal_etrs89", "NATCODE",
                 "Nivel de seguridad por provincias - 2023", "resultados/mapa_ej4_2023.png", latlon=True)

    # Tabla de transición por provincias y matriz de cambios
    c17 = {codigo(p): c for p, c in res3["Clase"].items()}
    trans = pd.DataFrame({"2017": [c17[codigo(p)] for p in res4.index], "2023": res4["Clase"].values},
                         index=res4.index)
    trans["Cambio"] = np.where(trans["2017"] == trans["2023"], "=",
                               np.where(trans["2017"].map(CLASES.index) > trans["2023"].map(CLASES.index),
                                        "Mejora", "Empeora"))
    print("\nTabla de transición por provincias:")
    print(trans.to_string())
    matriz = pd.crosstab(trans["2017"], trans["2023"]).reindex(index=CLASES, columns=CLASES, fill_value=0)
    print("\nMatriz de cambios entre clases (filas 2017, columnas 2023):")
    print(matriz.to_string())
    trans.to_csv("resultados/transicion.csv")
    matriz.to_csv("resultados/matriz_cambios.csv")


if __name__ == "__main__":
    main()

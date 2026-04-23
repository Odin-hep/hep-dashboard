#####################################
# === Librerías === #
#####################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import load_model

import joblib
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import ast
import pickle
import os

###########################################
# === Funciones ===
###########################################

# --- Clase del Modelo Ensamble --- #
class EnsambleParticulas:
    def __init__(self, señal, config_path="config_ensamble.csv"):
        self.señal = señal
        # Cargar configuración de pesos y threshold
        config = pd.read_csv(config_path, index_col=0).loc[señal]
        self.w = {
            'BDT': config['w_BDT'],
            'DNN': config['w_DNN'],
            'GNB': config['w_GNB']
        }
        self.thr = config['threshold']
        
        # Cargar modelos
        self.bdt = xgb.XGBClassifier()
        self.bdt.load_model(f"BDT_{señal}.json")
        self.dnn = load_model(f"DNN_{señal}.h5")
        self.gnb = joblib.load(f"GNB_{señal}.pkl")

    def predict_proba(self, X_raw, X_scaled, X_pt):
        """Calcula la probabilidad combinada ponderada"""
        p_bdt = self.bdt.predict_proba(X_raw)[:, 1]
        p_dnn = self.dnn.predict(X_scaled, verbose=0).flatten()
        p_gnb = self.gnb.predict_proba(X_pt)[:, 1]
        
        p_ens = (self.w['BDT'] * p_bdt) + \
                (self.w['DNN'] * p_dnn) + \
                (self.w['GNB'] * p_gnb)
        return p_ens

    def predict(self, X_raw, X_scaled, X_pt):
        """Devuelve la clasificación binaria (0 o 1) usando el thr óptimo"""
        proba = self.predict_proba(X_raw, X_scaled, X_pt)
        return (proba >= self.thr).astype(int)

# ==========================================================
# CLASE PARA EL MODELO DE ENSAMBLE
# ==========================================================
class HEPWeightedEnsemble:
    """
    Representa el modelo de ensamble ponderado para identificación de partículas.
    Combina las salidas de BDT, DNN y GNB.
    """
    def __init__(self, bdt, dnn, gnb, weights):
        self.bdt = bdt
        self.dnn = dnn
        self.gnb = gnb
        self.w = weights  # Diccionario con w_BDT, w_DNN, w_GNB

    def predict_proba(self, X_data, X_scaled):
        # Obtenemos probabilidades individuales
        p_bdt = self.bdt.predict_proba(X_data)[:, 1]
        p_dnn = self.dnn.predict(X_scaled, verbose=0).flatten()
        p_gnb = self.gnb.predict_proba(X_data)[:, 1]
        
        # Cálculo del ensamble basado en la configuración del CSV
        p_ens = (self.w['w_BDT'] * p_bdt) + \
                (self.w['w_DNN'] * p_dnn) + \
                (self.w['w_GNB'] * p_gnb)
        return p_ens

# Función de limpieza
def filtro(df):
    mascara = (
        (df['pT'] > 0) &
        (df['pAbs'] > 0) &
        (df['dedx'] > 0) &
        (df['y'].abs() < 0.9)
    )
    return df[mascara]

def generar_master_inferencia(señal, X_data, X_scaled, y_true, modelos_cargados, thresholds, pesos_ensamble):
    """
    Genera el DataFrame maestro usando los nombres exactos de tus archivos CSV.
    """
    print(f"--- Procesando Inferencia: {señal} ---")
    
    df_m = pd.DataFrame()
    df_m['pT'] = X_data['pT'].values
    df_m['dedx'] = X_data['dedx'].values
    df_m['y_true'] = y_true.values
    
    # 1. Inferencia Individual
    p_bdt = modelos_cargados['BDT'].predict_proba(X_data)[:, 1]
    p_dnn = modelos_cargados['DNN'].predict(X_scaled, verbose=0).flatten()
    p_gnb = modelos_cargados['GNB'].predict_proba(X_data)[:, 1]
    
    # 2. Ensamble (Asegúrate que config_ensamble.csv use w_BDT, w_DNN, w_GNB)
    p_ens = (pesos_ensamble['w_BDT'] * p_bdt) + \
            (pesos_ensamble['w_DNN'] * p_dnn) + \
            (pesos_ensamble['w_GNB'] * p_gnb)
    
    # 3. Guardar Scores y Binarizar
    # Usamos 'ENSAMBLE' para que coincida con tu columna del CSV
    nombres_modelos = ['BDT', 'DNN', 'GNB', 'ENSAMBLE']
    probas = [p_bdt, p_dnn, p_gnb, p_ens]
    
    for name, proba in zip(nombres_modelos, probas):
        # Guardamos en el DataFrame con nombres consistentes
        df_m[f'proba_{name}'] = proba
        df_m[f'pred_{name}'] = (proba >= thresholds[name]).astype(int)
        
    return df_m


def aplicar_discretizacion(df, edges):
    """
    Asigna cada evento a un bin de pT y calcula centros de forma vectorizada.
    """
    df_bin = df.copy()
    
    # 1. Asignación de bines
    df_bin['pT_bin'] = pd.cut(df_bin['pT'], bins=edges, include_lowest=True)
    
    # 2. Validación de NaNs (Eventos fuera de rango)
    n_nans = df_bin['pT_bin'].isna().sum()
    if n_nans > 0:
        print(f"¡Atención! {n_nans} eventos quedaron fuera de los bines definidos.")

    # 3. Cálculo vectorial del centro (Más rápido que .apply)
    # Convertimos los intervalos a sus puntos medios directamente
    centros = {intervalo: intervalo.mid for intervalo in df_bin['pT_bin'].unique() if pd.notnull(intervalo)}
    df_bin['pT_center'] = df_bin['pT_bin'].map(centros).astype(float)
    
    return df_bin



def agrupar_metricas_base(df, modelos=['BDT', 'DNN', 'GNB', 'ENSAMBLE']):
    """
    Agrupa por bin de pT y calcula TP, FP y FN para cada modelo de forma robusta.
    """
    resultados_por_modelo = []

    for mod in modelos:
        col_pred = f'pred_{mod}'
        
        # 1. Creamos un DataFrame temporal con las condiciones calculadas
        # Esto le da nombre a las columnas automáticamente
        df_temp = pd.DataFrame({
            'pT_bin': df['pT_bin'],
            'TP': ((df['y_true'] == 1) & (df[col_pred] == 1)).astype(int),
            'FP': ((df['y_true'] == 0) & (df[col_pred] == 1)).astype(int),
            'FN': ((df['y_true'] == 1) & (df[col_pred] == 0)).astype(int)
        })
        
        # 2. Agrupamos y sumamos
        # observed=False mantiene todos los bines aunque no tengan eventos
        df_res = df_temp.groupby('pT_bin', observed=False).sum().reset_index()
        
        # 3. Añadimos la etiqueta del modelo
        df_res['modelo'] = mod
        
        resultados_por_modelo.append(df_res)

    # Concatenamos los resultados de los 4 modelos
    df_final = pd.concat(resultados_por_modelo, ignore_index=True)
    
    return df_final[['pT_bin', 'modelo', 'TP', 'FP', 'FN']]



def calcular_metricas_fisicas(df_conteos):
    """
    Calcula Precision (Pureza), Recall (Eficiencia) y Significancia por bin.
    Maneja divisiones por cero devolviendo 0.0 en bines vacíos o sin aciertos.
    """
    # Trabajar sobre una copia para no alterar el DataFrame original
    df = df_conteos.copy()
    
    # Denominadores
    den_precision = df['TP'] + df['FP']
    den_recall = df['TP'] + df['FN']
    
    # Precision (P = TP / (TP + FP))
    # np.where(condicion, valor_si_verdadero, valor_si_falso)
    df['precision'] = np.where(den_precision > 0, 
                               df['TP'] / den_precision, 
                               0.0)
    
    # Recall o Eficiencia (ε = TP / (TP + FN))
    df['recall'] = np.where(den_recall > 0, 
                            df['TP'] / den_recall, 
                            0.0)
    
    # Significancia (S = TP / sqrt(TP + FP))
    df['significancia'] = np.where(den_precision > 0, 
                                   df['TP'] / np.sqrt(den_precision), 
                                   0.0)
    
    # Organizar columnas para una lectura limpia
    columnas_finales = ['pT_bin', 'modelo', 'TP', 'FP', 'FN', 'precision', 'recall', 'significancia']
    
    return df[columnas_finales]



def construir_factores_correccion(df_metricas):
    """
    Calcula el factor de corrección (precision / recall) por bin y modelo.
    """
    df = df_metricas.copy()
    
    # Calcular factor con manejo seguro de división por cero
    df['correction_factor'] = np.where(
        df['recall'] > 0,
        df['precision'] / df['recall'],
        0.0
    )
    
    # Filtrar y ordenar las columnas solicitadas
    columnas_salida = [
        'pT_bin', 'modelo', 'precision', 
        'recall', 'significancia', 'correction_factor'
    ]
    
    return df[columnas_salida]


def aplicar_correccion_conteos(df_conteos, df_factores):
    """
    Cruza los conteos de un nuevo dataset con los factores de calibración
    y calcula el número de partículas corregido (N_corr).
    """
    # 1. Merge (Left join para asegurar que no se pierdan bines del dataset de conteos)
    df_merged = pd.merge(
        df_conteos,
        df_factores[['pT_bin', 'modelo', 'correction_factor']],
        on=['pT_bin', 'modelo'],
        how='left'
    )
    
    # 2. Manejo de bines huérfanos (si un bin/modelo no existía en la tabla de factores)
    df_merged['correction_factor'] = df_merged['correction_factor'].fillna(0.0)
    
    # 3. Aplicar la corrección
    df_merged['N_corr'] = df_merged['N_clasif'] * df_merged['correction_factor']
    
    # 4. Formatear la salida
    columnas_salida = [
        'pT_bin', 'modelo', 'N_clasif', 
        'correction_factor', 'N_corr'
    ]
    
    return df_merged[columnas_salida]



def ejecutar_pipeline_hep(df_master_val, df_master_new, bin_edges, modelos, señal):
    """
    Orquesta el flujo completo de calibración (Fase 1) y aplicación (Fase 2).
    Genera un DataFrame Tidy optimizado para visualización en Dashboards.
    """
    # ==========================================
    # FASE 1: CALIBRACIÓN con la Construcción de Factores
    # ==========================================
    
    # 1. Discretización o Binning
    df_val_bin = aplicar_discretizacion(df_master_val, bin_edges)
    
    # 2. Reducción (Agrupar TP, FP, FN)
    df_val_conteos = agrupar_metricas_base(df_val_bin, modelos)
    
    # 3. Métricas Físicas (Pureza, Eficiencia, Significancia)
    df_val_metricas = calcular_metricas_fisicas(df_val_conteos)
    
    # 4. Factor de Corrección
    df_factores = construir_factores_correccion(df_val_metricas)
    
    
    # ==========================================
    # FASE 2: APLICACIÓN con la Corrección del Nuevo Dataset
    # ==========================================
    
    # 1. Discretización usando los MISMOS bines
    df_new_bin = aplicar_discretizacion(df_master_new, bin_edges)
    
    # 2. Reducción a conteos del nuevo dataset
    df_new_conteos = agrupar_metricas_base(df_new_bin, modelos)
    
    # (Aseguramos que N_clasif exista en caso de que la función base retorne TP/FP)
    if 'N_clasif' not in df_new_conteos.columns:
        df_new_conteos['N_clasif'] = df_new_conteos['TP'] + df_new_conteos['FP']
        
    # 3. Cruce y aplicación matemática de la corrección
    df_aplicado = aplicar_correccion_conteos(df_new_conteos, df_factores)
    
    
    # ==========================================
    # ENSAMBLADO FINAL
    # ==========================================
    
    # df_aplicado ya tiene N_clasif, correction_factor y N_corr.
    # Recuperamos las métricas de validación de df_factores para tenerlas de referencia
    cols_metricas = ['pT_bin', 'modelo', 'precision', 'recall', 'significancia']
    
    df_final = pd.merge(
        df_aplicado,
        df_factores[cols_metricas],
        on=['pT_bin', 'modelo'],
        how='left'
    )
    
    # Insertar la columna de la señal al principio
    df_final.insert(0, 'señal', señal)
    
    # Orden estricto solicitado
    columnas_ordenadas = [
        'señal', 'modelo', 'pT_bin', 
        'N_clasif', 'correction_factor', 'N_corr', 
        'precision', 'recall', 'significancia'
    ]
    
    return df_final[columnas_ordenadas]


def cargar_y_preparar_datos(path_root, señales, columnas):
    """Carga desde ROOT, filtra y organiza en un diccionario por señal."""
    with uproot.open(path_root) as file:
        # Usamos el nombre correcto que arrojó el error: 'particulas'
        df_raw = file["particulas"].arrays(columnas, library="pd")
    
    print(f"Eventos originales: {len(df_raw)}")
    df_filtrado = filtro(df_raw)
    print(f"Eventos tras filtro físico: {len(df_filtrado)}")
    
    diccionario_masters = {}
    for s in señales:
        temp_df = df_filtrado.copy()
        temp_df['y_true'] = temp_df[f'label{s}']
        diccionario_masters[s] = temp_df
    
    return diccionario_masters

# Evaluador Método Tradicional
def evaluar_metodo_tradicional(df_raw_test, df_metricas_ml, bin_edges, col_dedx='dedx'):
    """
    Calcula la precisión del método de cortes 1D en dE/dx igualando la eficiencia 
    de los modelos ML mediante cuantiles (Vectorizado, sin loops de barrido).
    """
    # 1. Asegurar que el raw data tiene los bines correctos
    df_raw = df_raw_test.copy()
    if 'pT_bin' not in df_raw.columns:
        df_raw['pT_bin'] = pd.cut(df_raw['pT'], bins=bin_edges)
        
    resultados_tradicional = []
    
    # 2. Agrupar eventos por bin para acceso rápido (O(1) en lookup)
    grupos_bins = dict(tuple(df_raw.groupby('pT_bin', observed=False)))

    # 3. Iterar sobre las métricas generadas por el pipeline ML
    for idx, row in df_metricas_ml.iterrows():
        bin_actual = row['pT_bin']
        modelo = row['modelo']
        target_recall = row['recall']
        precision_ml = row['precision']
        
        # Valores por defecto en caso de bines vacíos o recall 0
        precision_trad = np.nan
        corte_optimo = np.nan
        
        if bin_actual in grupos_bins and target_recall > 0:
            df_bin = grupos_bins[bin_actual]
            
            # Separar señal y fondo en este bin
            sig = df_bin[df_bin['y_true'] == 1][col_dedx].values
            bkg = df_bin[df_bin['y_true'] == 0][col_dedx].values
            
            if len(sig) > 0 and len(bkg) > 0:
                # ==========================================
                # DECISIÓN AUTOMÁTICA DE DIRECCIÓN DEL CORTE
                # ==========================================
                if np.median(sig) > np.median(bkg):
                    # Señal a la derecha -> Conservar valores > threshold
                    corte_optimo = np.quantile(sig, max(0, 1 - target_recall))
                    tp_trad = np.sum(sig >= corte_optimo)
                    fp_trad = np.sum(bkg >= corte_optimo)
                else:
                    # Señal a la izquierda -> Conservar valores < threshold
                    corte_optimo = np.quantile(sig, min(1, target_recall))
                    tp_trad = np.sum(sig <= corte_optimo)
                    fp_trad = np.sum(bkg <= corte_optimo)
                
                # Calcular precision del método tradicional
                if (tp_trad + fp_trad) > 0:
                    precision_trad = tp_trad / (tp_trad + fp_trad)
                else:
                    precision_trad = 0.0

        resultados_tradicional.append({
            'pT_bin': bin_actual,
            'modelo': modelo,
            'recall_objetivo': target_recall,
            'precision_modelo': precision_ml,
            'precision_trad': precision_trad,
            'corte_dedx': corte_optimo
        })
        
    return pd.DataFrame(resultados_tradicional)


# =========================================================
# 1. BACKEND LÓGICO (Cálculos de HEP)
# =========================================================

def compute_dashboard_metrics(df_inference, bin_edges):
    """Calcula métricas de ML agrupadas por pT_bin."""
    df = df_inference.copy()
    df['pT_bin_idx'] = np.digitize(df['pT'], bin_edges) - 1
    df = df[(df['pT_bin_idx'] >= 0) & (df['pT_bin_idx'] < len(bin_edges)-1)]
    
    metrics_list = []
    modelos = ['BDT', 'DNN', 'GNB', 'ENSAMBLE']
    
    # Verificación de seguridad para columnas
    for mod in modelos:
        col_pred = f'{mod}_pred'
        if col_pred not in df.columns:
            # Si no existe la columna, saltamos este modelo o lanzamos error descriptivo
            continue
            
        grouped = df.groupby('pT_bin_idx', observed=True)
        
        tp = grouped.apply(lambda x: ((x[col_pred] == 1) & (x['y_true'] == 1)).sum())
        fp = grouped.apply(lambda x: ((x[col_pred] == 1) & (x['y_true'] == 0)).sum())
        fn = grouped.apply(lambda x: ((x[col_pred] == 0) & (x['y_true'] == 1)).sum())
        
        res = pd.DataFrame({
            'recall': tp / (tp + fn),
            'precision': tp / (tp + fp),
            'modelo': mod,
            'pT_bin_idx': tp.index,
            'pT_center': [(bin_edges[i] + bin_edges[i+1])/2 for i in tp.index]
        })
        metrics_list.append(res)
        
    return pd.concat(metrics_list).reset_index(drop=True) if metrics_list else pd.DataFrame()

def compute_traditional_baseline(df_inference, df_ml_metrics, bin_edges, col_dedx='dedx'):
    """Calcula el baseline tradicional igualando el Recall de ML."""
    if df_ml_metrics.empty: return df_ml_metrics
    
    df_raw = df_inference.copy()
    df_raw['pT_bin_idx'] = np.digitize(df_raw['pT'], bin_edges) - 1
    grupos_raw = dict(tuple(df_raw.groupby('pT_bin_idx', observed=True)))
    
    resultados_tradicional = []
    for _, row in df_ml_metrics.iterrows():
        b_idx = int(row['pT_bin_idx'])
        target_recall = row['recall']
        p_trad, corte = np.nan, np.nan
        
        if b_idx in grupos_raw and target_recall > 0:
            df_bin = grupos_raw[b_idx]
            sig = df_bin[df_bin['y_true'] == 1][col_dedx].values
            bkg = df_bin[df_bin['y_true'] == 0][col_dedx].values
            
            if len(sig) > 0 and len(bkg) > 0:
                if np.median(sig) > np.median(bkg):
                    corte = np.quantile(sig, max(0, 1 - target_recall))
                    tp_t, fp_t = np.sum(sig >= corte), np.sum(bkg >= corte)
                else:
                    corte = np.quantile(sig, min(1, target_recall))
                    tp_t, fp_t = np.sum(sig <= corte), np.sum(bkg <= corte)
                p_trad = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0

        resultados_tradicional.append({
            'pT_bin_idx': b_idx, 'modelo': row['modelo'], 
            'precision_trad': p_trad, 'corte_dedx': corte
        })
    
    return pd.merge(df_ml_metrics, pd.DataFrame(resultados_tradicional), on=['pT_bin_idx', 'modelo'])

def run_dashboard_pipeline(df_master, bin_edges):
    df_ml = compute_dashboard_metrics(df_master, bin_edges)
    return compute_traditional_baseline(df_master, df_ml, bin_edges)

##########################################
# === RUTAS ===
#########################################

# Ruta base (misma carpeta que el notebook)
path = '.'

# Conjuntos de datos
path_train = path + '/train_particulas.root'
path_valtest = path + '/val_test_particulas.root'

# Conjuntos de aplicación
path_CR1 = path + '/SoftQCD_CR1.root'
path_CR0 = path + '/SoftQCD_CR0.root'


#####################################
# === Preprocesamiento y Carga === #
#####################################

features = ['pT', 'y', 'dedx']
columnas = ['id','pT','pAbs','eta','y','bg','dedx', 'labelPion','labelKaon','labelProton']

print("Cargando datos...")
# Train
file_train = uproot.open(path_train)
df_train = file_train['particulas;1'].arrays(columnas, library='pd')

# Validation/Test
file_valtest = uproot.open(path_valtest)
df_valtest = file_valtest['particulas;1'].arrays(columnas, library='pd')

print("Aplicando filtro físico...")
df_train = filtro(df_train)
df_valtest = filtro(df_valtest)

print("Dividiendo conjuntos...")
df_val, df_test = train_test_split(df_valtest, test_size=0.5, random_state=123)

print("Normalizando variables...")
scaler = StandardScaler()

X_train = df_train[features]
X_val = df_val[features]
X_test = df_test[features]

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("¡Datos listos para entrenar!")

columnas_root = ['pT', 'pAbs', 'y', 'dedx', 'labelPion', 'labelKaon', 'labelProton']
mis_señales = ["Pion", "Kaon", "Proton"]
masters_test = cargar_y_preparar_datos(path_valtest, mis_señales, columnas_root)

# Verificación
for s in masters_test:
    print(f"{s}: {len(masters_test[s])} eventos listos (dedx presente: {'dedx' in masters_test[s].columns})")


# ==========================================================
# CARGA DE DATOS Y CONFIGURACIÓN
# ==========================================================
print("Cargando archivos de configuración...")
df_config = pd.read_csv('config_ensamble.csv', index_col=0)
df_thr_ind = pd.read_csv('thresholds.csv', index_col=0)

señales = ["Pion", "Kaon", "Proton"]
modelos_por_señal = {}
parametros_por_señal = {}

for s in señales:
    print(f" -> Construyendo pipeline para {s}...")
    
    # 1. Cargar modelos base desde disco
    m_bdt = xgb.XGBClassifier()
    m_bdt.load_model(f"BDT_{s}.json")
    m_dnn = load_model(f"DNN_{s}.h5")
    m_gnb = joblib.load(f"GNB_{s}.pkl")
    
    # 2. Extraer pesos del config_ensamble.csv
    pesos = {
        'w_BDT': df_config.loc[s, 'w_BDT'],
        'w_DNN': df_config.loc[s, 'w_DNN'],
        'w_GNB': df_config.loc[s, 'w_GNB']
    }
    
    # 3. Modelo de Ensamble
    modelo_ensamble = HEPWeightedEnsemble(m_bdt, m_dnn, m_gnb, pesos)
    
    # 4. Guardar en el diccionario (Ahora incluye el ENSAMBLE como modelo)
    modelos_por_señal[s] = {
        'BDT': m_bdt,
        'DNN': m_dnn,
        'GNB': m_gnb,
        'ENSAMBLE': modelo_ensamble  # <--- Aquí está "incluido" como modelo
    }
    
    # 5. Mapear umbrales (Individuales + Ensamble optimizado)
    parametros_por_señal[s] = {
        'thr': {
            'BDT': df_thr_ind.loc[s, 'BDT'],
            'DNN': df_thr_ind.loc[s, 'DNN'],
            'GNB': df_thr_ind.loc[s, 'GNB'],
            'ENSAMBLE': df_config.loc[s, 'threshold']
        },
        'weights': pesos
    }

print("\n¡Configuración completada! El Ensamble ahora es un objeto de primer nivel.")


# --- FASE DE CALIBRACIÓN (Validación) ---
features = ['pT', 'y', 'dedx']

X_train = df_train[features]
X_val = df_val[features]
X_test = df_test[features]

# Cargar masters_val
with open("masters_val.pkl", "rb") as f:
    masters_val = pickle.load(f)

# Cargar masters_test
with open("masters_test.pkl", "rb") as f:
    masters_test = pickle.load(f)

print("masters cargados correctamente")



# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
app = dash.Dash(__name__)

# Definimos los modelos que tienes en tu pipeline
modelos_lista = ['BDT', 'DNN', 'GNB', 'ENSAMBLE']

# Estilos para las pestañas (visibilidad y distinción)
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '12px',
    'fontWeight': 'bold',
    'backgroundColor': '#e9ecef',
    'color': '#495057',
    'border': '1px solid #ced4da',
    'borderRadius': '5px 5px 0px 0px',
    'marginRight': '2px'
}

tab_selected_style = {
    'borderTop': '4px solid #007bff',
    'borderBottom': '1px solid white',
    'backgroundColor': '#ffffff',
    'color': '#007bff',
    'padding': '12px',
    'fontWeight': 'bold',
    'borderRadius': '5px 5px 0px 0px',
    'boxShadow': '0px -2px 5px rgba(0,0,0,0.1)'
}

# ==========================================
# 2. FRONTEND
# ==========================================
app.layout = html.Div([
    html.Div([
        html.H2("HEP Analysis Dashboard: ML vs Tradicional", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'sans-serif'}),
        html.Hr()
    ]),

    html.Div([
        # Columna Izquierda: Controles
        html.Div([
            html.B("Seleccionar Partícula:"),
            dcc.Dropdown(
                id='particle-selector',
                options=[{'label': s, 'value': s} for s in ["Pion", "Kaon", "Proton"]],
                value='Pion',
                clearable=False,
                style={'marginBottom': '15px'}
            ),
            
            html.B("Seleccionar Modelo (ML vs Tradicional):"),
            dcc.Dropdown(
                id='model-selector',
                options=[{'label': m, 'value': m} for m in modelos_lista],
                value='ENSAMBLE',
                clearable=False,
                style={'marginBottom': '20px'}
            ),
            
            html.B("Rango de pT (GeV/c):"),
            dcc.RangeSlider(
                id='pt-range-slider',
                min=0, max=20, step=0.5,
                value=[3, 10],
                marks={i: str(i) for i in range(0, 21, 2)}
            ),
            html.Br(),
            
            html.B("Número de Bines (Regulares):"),
            dcc.Input(
                id='n-bins-input',
                type='number',
                value=15,
                min=0, max=50, #aquí cambié min=2 a 0
                step=1, # Esto lo agregué
                debounce=True, # Esto lo agregué
                style={'width': '100%', 'padding': '5px', 'marginTop': '5px', 'marginBottom': '15px'}
            ),

            html.B("Bines Personalizados (Opcional):"),
            dcc.Input(
                id='custom-bins-input',
                type='text',
                placeholder='Ej: [3, 3.2, 4, 5, 8, 10.5]',
                style={'width': '100%', 'padding': '5px', 'marginTop': '5px'}
            ),
            
            html.Div(id='bin-info-display', style={'marginTop': '20px', 'fontSize': '12px', 'color': 'gray'})
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

        # Columna Derecha: Gráficas
        html.Div([
            dcc.Tabs(id='tabs-selector', value='Efficiency', children=[
                dcc.Tab(label='Eficiencia (Recall)', value='Efficiency', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Pureza (Precisión)', value='Purity', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='ML vs Tradicional', value='Comparison', style=tab_style, selected_style=tab_selected_style)
            ]),
            dcc.Graph(id='main-graph', style={'height': '600px'})
        ], style={'width': '70%', 'display': 'inline-block', 'paddingLeft': '30px'})
    ], style={'display': 'flex'})
])

# ==========================================
# 3.  ESTILO ROOT 
# ==========================================
def add_step_trace(fig, name, df_sub, y_col, line_dash='solid', color=None):
    """Convierte el DataFrame en un escalón tipo ROOT (TH1)."""
    x_edges = [bin.left for bin in df_sub['pT_bin']]
    x_edges.append(df_sub['pT_bin'].iloc[-1].right)
    
    y_vals = df_sub[y_col].tolist()
    y_vals.append(y_vals[-1])
    
    fig.add_trace(go.Scatter(
        x=x_edges,
        y=y_vals,
        mode='lines',
        line_shape='hv', 
        name=name,
        line=dict(dash=line_dash, width=2, color=color),
        connectgaps=True
    ))

# ==========================================
# 4. BACKEND: CALLBACK PRINCIPAL
# ==========================================
@app.callback(
    [Output('main-graph', 'figure'),
     Output('bin-info-display', 'children')],
    [Input('particle-selector', 'value'),
     Input('model-selector', 'value'),
     Input('pt-range-slider', 'value'),
     Input('n-bins-input', 'value'),
     Input('custom-bins-input', 'value'),
     Input('tabs-selector', 'value')]
)
def update_dashboard(particle, selected_model, pt_range, n_bins, custom_bins_input, tab):
    # ==========================================
    # PROTECCIÓN CONTRA None
    # ==========================================
    if n_bins is None or n_bins < 2:
        n_bins = 15

    if pt_range is None or len(pt_range) != 2:
        pt_range = [0, 20]

    # Crea bordes de bines dinámicos o personalizados
    bines_personalizados_usados = False

    if custom_bins_input:
        try:
            parsed = ast.literal_eval(custom_bins_input)

            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                new_edges = np.array(parsed)
                bin_width_str = "Personalizado"
                bines_personalizados_usados = True
            else:
                raise ValueError("Formato inválido")

        except:
            new_edges = np.linspace(pt_range[0], pt_range[1], int(n_bins) + 1)
            bin_width_str = f"{(new_edges[1] - new_edges[0]):.2f} GeV/c"

    else:
        new_edges = np.linspace(pt_range[0], pt_range[1], int(n_bins) + 1)
        bin_width_str = f"{(new_edges[1] - new_edges[0]):.2f} GeV/c"
    
    # B. Ejecutar Fuente de Verdad (Pipeline ML)
    df_ml = ejecutar_pipeline_hep(
        df_master_val=masters_val[particle],
        df_master_new=masters_test[particle],
        bin_edges=new_edges,
        modelos=modelos_lista,
        señal=particle
    )
    
    fig = go.Figure()
    
    # C. Lógica de Graficación por Tab
    if tab == 'Efficiency':
        for mod in df_ml['modelo'].unique():
            df_sub = df_ml[df_ml['modelo'] == mod]
            add_step_trace(fig, name=mod, df_sub=df_sub, y_col='recall')
        title_y = "Eficiencia (Recall)"

    elif tab == 'Purity':
        for mod in df_ml['modelo'].unique():
            df_sub = df_ml[df_ml['modelo'] == mod]
            add_step_trace(fig, name=mod, df_sub=df_sub, y_col='precision')
        title_y = "Pureza (Precisión)"

    elif tab == 'Comparison':
        # Ejecutar Pipeline Tradicional
        df_trad = evaluar_metodo_tradicional(
            df_raw_test=masters_test[particle],
            df_metricas_ml=df_ml,
            bin_edges=new_edges
        )
        
        # Filtrar solo el modelo seleccionado para comparar
        df_mod = df_trad[df_trad['modelo'] == selected_model]
        
        # Graficar SOLO el modelo seleccionado vs su equivalente tradicional
        add_step_trace(fig, f"{selected_model} (ML)", df_mod, 'precision_modelo', color='blue')
        add_step_trace(fig, f"{selected_model} (Tradicional)", df_mod, 'precision_trad', line_dash='dash', color='red')
        
        title_y = "Comparación de Pureza"

    # D. Estética Final
    fig.update_layout(
        title=f"Resultados para {particle} (Ancho de bin: {bin_width_str})",
        xaxis_title='pT (GeV/c)',
        yaxis_title=title_y,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if bines_personalizados_usados:
        info_texto = f"Bines personalizados usados: {len(new_edges) - 1} bines."
    else:
        info_texto = f"Bines calculados: {n_bins}. Rango: {pt_range[0]} - {pt_range[1]} GeV/c"
    
    return fig, info_texto

# ==========================================
# 5. INVOCACIÓN DEL SERVIDOR
# ==========================================
#if __name__ == '__main__':
#    app.run(debug=True, port=8050)

server = app.server  # IMPORTANTE para deploy

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
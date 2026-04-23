import numpy as np
import pandas as pd
import ast
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import os

# =========================================================
# 1. CARGA DE DATOS PREPROCESADOS (Ligero y Rápido)
# =========================================================
# En tu computadora local o en un script separado (offline), debes procesar
# los .root, pasarles el .predict() de tus modelos y guardar un DataFrame con:
# ['pT', 'dedx', 'y_true', 'pred_BDT', 'pred_DNN', 'pred_GNB', 'pred_ENSAMBLE']
# Aquí simulamos la carga de esos datos ya digeridos.

def cargar_datos_produccion():
    """Carga los DataFrames pre-calculados (formato parquet recomendado)"""
    masters_test = {}
    señales = ["Pion", "Kaon", "Proton"]
    
    for s in señales:
        ruta_archivo = f"data/{s.lower()}_data.parquet"
        # Si no tienes parquet aún, puedes usar read_csv(f"data/{s}.csv")
        if os.path.exists(ruta_archivo):
            masters_test[s] = pd.read_parquet(ruta_archivo)
        else:
            # Fallback temporal para evitar caídas si falta un archivo
            masters_test[s] = pd.DataFrame(columns=[
                'pT', 'dedx', 'y_true', 'pred_BDT', 'pred_DNN', 'pred_GNB', 'pred_ENSAMBLE'
            ])
            print(f"Advertencia: No se encontró {ruta_archivo}")
            
    return masters_test

#masters_test = cargar_datos_produccion()
# Cargar masters_test
with open("masters_test.pkl", "rb") as f:
    masters_test = pickle.load(f)

print("masters cargados correctamente")

modelos_lista = ['BDT', 'DNN', 'GNB', 'ENSAMBLE']

# =========================================================
# 2. BACKEND LÓGICO Y MÉTRICAS (Física)
# =========================================================

def compute_dashboard_metrics(df_inference, bin_edges):
    """Calcula métricas de ML agrupadas por pT_bin de forma vectorizada."""
    if df_inference.empty: return pd.DataFrame()
    
    df = df_inference.copy()
    # Asignación rápida de bines
    df['pT_bin_idx'] = np.digitize(df['pT'], bin_edges) - 1
    # Filtrar eventos fuera del rango visualizado
    df = df[(df['pT_bin_idx'] >= 0) & (df['pT_bin_idx'] < len(bin_edges)-1)]
    
    metrics_list = []
    
    for mod in modelos_lista:
        col_pred = f'pred_{mod}'
        if col_pred not in df.columns:
            continue
            
        grouped = df.groupby('pT_bin_idx', observed=True)
        
        # Cálculos de TP, FP, FN
        tp = grouped.apply(lambda x: ((x[col_pred] == 1) & (x['y_true'] == 1)).sum())
        fp = grouped.apply(lambda x: ((x[col_pred] == 1) & (x['y_true'] == 0)).sum())
        fn = grouped.apply(lambda x: ((x[col_pred] == 0) & (x['y_true'] == 1)).sum())
        
        # Manejo seguro de división por cero
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        
        res = pd.DataFrame({
            'recall': recall,
            'precision': precision,
            'modelo': mod,
            'pT_bin_idx': tp.index,
            'pT_center': [(bin_edges[i] + bin_edges[i+1])/2 for i in tp.index]
        })
        metrics_list.append(res)
        
    return pd.concat(metrics_list).reset_index(drop=True) if metrics_list else pd.DataFrame()

def compute_traditional_baseline(df_inference, df_ml_metrics, bin_edges, col_dedx='dedx'):
    """Calcula el baseline tradicional (cortes en dE/dx) igualando el Recall de ML."""
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
            'pT_bin_idx': b_idx, 
            'modelo': row['modelo'], 
            'precision_trad': p_trad, 
            'corte_dedx': corte
        })
    
    return pd.merge(df_ml_metrics, pd.DataFrame(resultados_tradicional), on=['pT_bin_idx', 'modelo'])

# =========================================================
# 3. INTERFAZ GRÁFICA (DASH)
# =========================================================
app = dash.Dash(__name__)
server = app.server  # EXPOSICIÓN OBLIGATORIA PARA RENDER/GUNICORN

tab_style = {
    'borderBottom': '1px solid #d6d6d6', 'padding': '12px', 'fontWeight': 'bold',
    'backgroundColor': '#e9ecef', 'color': '#495057', 'border': '1px solid #ced4da',
    'borderRadius': '5px 5px 0px 0px', 'marginRight': '2px'
}

tab_selected_style = {
    'borderTop': '4px solid #007bff', 'borderBottom': '1px solid white',
    'backgroundColor': '#ffffff', 'color': '#007bff', 'padding': '12px',
    'fontWeight': 'bold', 'borderRadius': '5px 5px 0px 0px',
    'boxShadow': '0px -2px 5px rgba(0,0,0,0.1)'
}

app.layout = html.Div([
    html.Div([
        html.H2("HEP Analysis Dashboard: ML vs Tradicional", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'sans-serif'}),
        html.Hr()
    ]),

    html.Div([
        # Controles
        html.Div([
            html.B("Seleccionar Partícula:"),
            dcc.Dropdown(id='particle-selector', options=[{'label': s, 'value': s} for s in ["Pion", "Kaon", "Proton"]], value='Pion', clearable=False, style={'marginBottom': '15px'}),
            
            html.B("Seleccionar Modelo (Comparación):"),
            dcc.Dropdown(id='model-selector', options=[{'label': m, 'value': m} for m in modelos_lista], value='ENSAMBLE', clearable=False, style={'marginBottom': '20px'}),
            
            html.B("Rango de pT (GeV/c):"),
            dcc.RangeSlider(id='pt-range-slider', min=0, max=20, step=0.5, value=[3, 10], marks={i: str(i) for i in range(0, 21, 2)}),
            html.Br(),
            
            html.B("Número de Bines:"),
            dcc.Input(id='n-bins-input', type='number', value=15, min=2, max=50, step=1, debounce=True, style={'width': '100%', 'padding': '5px', 'margin': '5px 0 15px 0'}),

            html.B("Bines Personalizados (Opcional):"),
            dcc.Input(id='custom-bins-input', type='text', placeholder='Ej: [3, 4.5, 6, 10]', style={'width': '100%', 'padding': '5px', 'marginTop': '5px'}),
            
            html.Div(id='bin-info-display', style={'marginTop': '20px', 'fontSize': '12px', 'color': 'gray'})
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),

        # Gráficas
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

# =========================================================
# 4. FUNCIONES GRÁFICAS Y CALLBACKS
# =========================================================

def add_step_trace(fig, name, df_sub, y_col, line_dash='solid', color=None):
    """Convierte el DataFrame en un escalón tipo ROOT (TH1)."""
    if df_sub.empty: return
    # Necesitamos el ancho del bin para el trazo correcto
    x_vals = []
    y_vals = []
    for _, row in df_sub.iterrows():
        center = row['pT_center']
        # Usamos un ancho simulado basado en el centro si no tenemos el left/right explícito
        # Como optimización, asumo que pT_bin_idx te puede dar los bordes exactos,
        # pero para Plotly line_shape='hv' requiere los puntos x e y correlativos.
        x_vals.append(center) 
        y_vals.append(row[y_col])
        
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='lines+markers', line_shape='hv', 
        name=name, line=dict(dash=line_dash, width=2, color=color), connectgaps=True
    ))

@app.callback(
    [Output('main-graph', 'figure'), Output('bin-info-display', 'children')],
    [Input('particle-selector', 'value'), Input('model-selector', 'value'),
     Input('pt-range-slider', 'value'), Input('n-bins-input', 'value'),
     Input('custom-bins-input', 'value'), Input('tabs-selector', 'value')]
)
def update_dashboard(particle, selected_model, pt_range, n_bins, custom_bins_input, tab):
    # Protecciones
    n_bins = n_bins if (n_bins is not None and n_bins >= 2) else 15
    pt_range = pt_range if (pt_range is not None and len(pt_range) == 2) else [0, 20]

    bines_personalizados = False
    if custom_bins_input:
        try:
            parsed = ast.literal_eval(custom_bins_input)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                new_edges = np.array(parsed)
                bin_width_str = "Personalizado"
                bines_personalizados = True
            else: raise ValueError
        except:
            new_edges = np.linspace(pt_range[0], pt_range[1], int(n_bins) + 1)
            bin_width_str = f"{(new_edges[1] - new_edges[0]):.2f} GeV/c"
    else:
        new_edges = np.linspace(pt_range[0], pt_range[1], int(n_bins) + 1)
        bin_width_str = f"{(new_edges[1] - new_edges[0]):.2f} GeV/c"
    
    # Extraer el DataFrame preprocesado
    df_raw = masters_test.get(particle, pd.DataFrame())
    
    fig = go.Figure()
    title_y = "Métrica"
    
    if not df_raw.empty:
        df_ml = compute_dashboard_metrics(df_raw, new_edges)
        
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
            df_trad = compute_traditional_baseline(df_raw, df_ml, new_edges)
            if not df_trad.empty:
                df_mod = df_trad[df_trad['modelo'] == selected_model]
                add_step_trace(fig, f"{selected_model} (ML)", df_mod, 'precision', color='blue')
                add_step_trace(fig, f"{selected_model} (Tradicional)", df_mod, 'precision_trad', line_dash='dash', color='red')
            title_y = "Comparación de Pureza"

    fig.update_layout(
        title=f"Resultados para {particle} (Ancho de bin: {bin_width_str})",
        xaxis_title='pT (GeV/c)', yaxis_title=title_y, template='plotly_white',
        hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    info_texto = f"Bines usados: {len(new_edges) - 1}." if bines_personalizados else f"Bines: {n_bins}. Rango: {pt_range[0]} - {pt_range[1]} GeV/c"
    return fig, info_texto

server = app.server  # IMPORTANTE para deploy

if __name__ == '__main__':
    #app.run(debug=False, host='0.0.0.0', port=8050)
    #app.run(debug=True, host='127.0.0.1', port=8050)
    app.run(
            debug=not is_prod, 
            host='0.0.0.0' if is_prod else '127.0.0.1', 
            port=port
        )

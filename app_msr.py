
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import itertools
import io
import requests

warnings.filterwarnings('ignore')

st.set_page_config(page_title="MSR Pro Statgraphics", page_icon="📈", layout="wide")

st.title("📈 Metodología de Superficie de Respuesta (MSR)")
st.markdown("Diseño y análisis de experimentos (DOE) con enfoque en optimización de procesos.")
st.markdown("---")

def obtener_datos(tab_key):
    return st.radio(
        "Método de carga de datos:",
        ["✏️ Manual (Editar tabla)", "📁 Cargar CSV", "🔗 Cargar desde URL"],
        horizontal=True, key=f"metodo_{tab_key}"
    )

def procesar_archivo(archivo):
    try:
        return pd.read_csv(io.StringIO(archivo.getvalue().decode('utf-8')))
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        return None

def procesar_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"❌ Error al cargar desde URL: {e}")
        return None

# ============================================================
# PESTAÑAS
# ============================================================
tab_fase1, tab_fase2 = st.tabs(["🚀 Fase 1: Ascenso Rápido (1er Orden)", "🎯 Fase 2: Optimización (2do Orden / DCC)"])

# ==============================================================================
# FASE 1: DISEÑO FACTORIAL Y ASCENSO RÁPIDO
# ==============================================================================
with tab_fase1:
    st.header("1. Modelo de Primer Orden y Trayectoria")
    
    with st.expander("📌 Generador de Diseño Factorial 2^k", expanded=False):
        col1, col2 = st.columns(2)
        with col1: k_f1 = st.number_input("N° Factores (k):", 2, 5, 2, key="k_f1")
        with col2: n_centros_f1 = st.number_input("Puntos Centrales (0,0):", 2, 10, 5, key="nc_f1")
        if st.button("Generar Diseño 2^k"):
            niveles = [-1, 1]
            df = pd.DataFrame(list(itertools.product(niveles, repeat=k_f1)), columns=[f"X{i+1}" for i in range(k_f1)])
            centros = pd.DataFrame([[0]*k_f1]*n_centros_f1, columns=df.columns)
            df = pd.concat([df, centros], ignore_index=True)
            df.insert(0, "Run", range(1, len(df)+1))
            st.session_state['df_f1'] = df
            st.rerun()

    metodo_f1 = obtener_datos("f1")
    df_f1 = None
    
    if metodo_f1 == "✏️ Manual (Editar tabla)":
        if 'df_f1' in st.session_state:
            st.subheader("📋 Tabla del Diseño")
            y_col_f1 = st.text_input("Variable Respuesta (Y):", value="Y", key="y_f1")
            df_temp = st.session_state['df_f1'].copy()
            if y_col_f1 not in df_temp.columns: df_temp[y_col_f1] = np.nan
            df_f1 = st.data_editor(df_temp, num_rows="fixed", use_container_width=True, key="edit_f1")
    elif metodo_f1 == "📁 Cargar CSV":
        archivo = st.file_uploader("Sube tu CSV:", type="csv", key="csv_f1")
        if archivo: df_f1 = procesar_archivo(archivo)
    elif metodo_f1 == "🔗 Cargar desde URL":
        url = st.text_input("URL del CSV:", key="url_f1")
        if st.button("Cargar URL") and url: df_f1 = procesar_url(url)

    if df_f1 is not None:
        if metodo_f1 != "✏️ Manual (Editar tabla)": 
            st.dataframe(df_f1)
        
        # Mejora en la detección: X1, X2 o las primeras columnas numéricas
        cols = df_f1.columns.tolist()
        posibles_f = [c for c in cols if str(c).startswith('X')]
        if not posibles_f:
            # Si no empiezan con X, tomamos las primeras numéricas excepto 'Run'
            posibles_f = [c for c in cols if c not in ['Run', 'Bloque'] and df_f1[c].dtype in ['float64', 'int64']]
            
        if len(posibles_f) > 0:
            # Selección de cuántos factores usar (por defecto todos los detectados menos el último)
            n_f = st.number_input("¿Cuántos Factores (X) deseas usar en el modelo?", min_value=1, max_value=len(posibles_f), value=min(len(posibles_f), 3))
            factores = posibles_f[:n_f]
            
            posibles_y = [c for c in cols if c not in factores and c not in ['Run', 'Bloque']]
            y_sel = st.selectbox("Selecciona la Respuesta (Y) a analizar:", posibles_y, index=0) if posibles_y else None
            
            if st.button("Ejecutar Análisis 1er Orden", type="primary") and y_sel:
                if df_f1[y_sel].isna().any(): 
                    st.error("Faltan datos en la columna Y seleccionada.")
                else:
                    try:
                        X = sm.add_constant(df_f1[factores])
                        Y = df_f1[y_sel]
                        modelo = sm.OLS(Y, X).fit()
                        
                        st.markdown("### ANOVA del Modelo de 1er Orden")
                        
                        mask_centros = (df_f1[factores] == 0).all(axis=1)
                        df_centros = df_f1[mask_centros]
                        if len(df_centros) > 1:
                            ss_pure = ((df_centros[y_sel] - df_centros[y_sel].mean())**2).sum()
                            gl_pure = len(df_centros) - 1
                            ms_pure = ss_pure / gl_pure if gl_pure > 0 else 0
                        else: ss_pure, gl_pure, ms_pure = 0, 0, 0
                        
                        ss_res = modelo.ssr
                        gl_res = modelo.df_resid
                        ss_lof = max(0, ss_res - ss_pure)
                        gl_lof = gl_res - gl_pure
                        ms_lof = ss_lof / gl_lof if gl_lof > 0 else 0
                        
                        anova_dict = {"Fuente": [], "Suma de Cuadrados": [], "Gl": [], "Cuadrado Medio": [], "Razón-F": [], "Valor-P": []}
                        
                        for f in factores:
                            r2_f = sm.OLS(Y, sm.add_constant(df_f1[[f]])).fit().rsquared
                            ss_f = r2_f * ((Y - Y.mean())**2).sum()
                            ms_f = ss_f
                            f_val = ms_f / modelo.mse_resid if modelo.mse_resid > 0 else 0
                            p_val = 1 - stats.f.cdf(f_val, 1, modelo.df_resid)
                            anova_dict["Fuente"].append(f)
                            anova_dict["Suma de Cuadrados"].append(ss_f)
                            anova_dict["Gl"].append(1)
                            anova_dict["Cuadrado Medio"].append(ms_f)
                            anova_dict["Razón-F"].append(f_val)
                            anova_dict["Valor-P"].append(p_val)
                        
                        f_lof = ms_lof / ms_pure if ms_pure > 0 else 0
                        p_lof = 1 - stats.f.cdf(f_lof, gl_lof, gl_pure) if gl_pure > 0 and ms_pure > 0 else np.nan
                        
                        anova_dict["Fuente"].extend(["Falta de ajuste", "Error puro", "Total (corr.)"])
                        anova_dict["Suma de Cuadrados"].extend([ss_lof, ss_pure, ((Y - Y.mean())**2).sum()])
                        anova_dict["Gl"].extend([gl_lof, gl_pure, len(Y)-1])
                        anova_dict["Cuadrado Medio"].extend([ms_lof, ms_pure, np.nan])
                        anova_dict["Razón-F"].extend([f_lof, np.nan, np.nan])
                        anova_dict["Valor-P"].extend([p_lof, np.nan, np.nan])
                        
                        st.dataframe(pd.DataFrame(anova_dict).style.format(precision=4, na_rep="-"), use_container_width=True)
                        st.write(f"**R² = {modelo.rsquared*100:.2f}%** | **R² ajustado = {modelo.rsquared_adj*100:.2f}%**")

                        st.markdown("---")
                        st.subheader("🚀 Ascenso Más Rápido")
                        
                        beta = modelo.params.drop('const') 
                        max_beta = max([abs(b) for b in beta])
                        max_idx = [abs(b) for b in beta].index(max_beta)
                        
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            st.write(f"Mayor efecto: **{factores[max_idx]}** (β = {beta.iloc[max_idx]:.4f})")
                            paso_base = st.number_input("Tamaño de paso (Δ):", value=5.0)
                        with col_b2:
                            trayectoria = []
                            for step in range(1, 6):
                                row = {"Paso": step}
                                for i, f in enumerate(factores):
                                    delta = paso_base if i == max_idx else (beta.iloc[i] / beta.iloc[max_idx]) * paso_base
                                    row[f"Δ {f}"] = delta * step
                                trayectoria.append(row)
                            st.dataframe(pd.DataFrame(trayectoria).style.format(precision=3), hide_index=True)
                    except Exception as e:
                        st.error(f"Error en los cálculos: {e}")
        else:
            st.warning("No se detectaron factores válidos en la tabla.")

# ==============================================================================
# FASE 2: DISEÑO CENTRAL COMPUESTO Y OPTIMIZACIÓN
# ==============================================================================
with tab_fase2:
    st.header("2. Modelo de Segundo Orden (DCC)")
    
    with st.expander("📌 Generador de Diseño Central Compuesto (DCC)", expanded=True):
        col1, col2 = st.columns(2)
        with col1: k_f2 = st.number_input("N° Factores (k):", 2, 5, 2, key="k_f2")
        with col2: n_centros_f2 = st.number_input("Puntos Centrales (0,0):", 2, 10, 5, key="nc_f2")
        if st.button("Generar Diseño DCC"):
            df_fact = pd.DataFrame(list(itertools.product([-1, 1], repeat=k_f2)), columns=[f"X{i+1}" for i in range(k_f2)])
            alpha = (2**k_f2)**0.25 
            axiales = []
            for i in range(k_f2):
                for val in [-alpha, alpha]:
                    p = [0]*k_f2
                    p[i] = val
                    axiales.append(p)
            df_ax = pd.DataFrame(axiales, columns=[f"X{i+1}" for i in range(k_f2)])
            df_cen = pd.DataFrame([[0]*k_f2] * n_centros_f2, columns=[f"X{i+1}" for i in range(k_f2)])
            df = pd.concat([df_fact, df_ax, df_cen], ignore_index=True)
            df.insert(0, "Run", range(1, len(df)+1))
            st.session_state['df_f2'] = df
            st.session_state['label_f2'] = f"DCC (α={alpha:.3f}) + {n_centros_f2} centros"
            st.rerun()

    metodo_f2 = obtener_datos("f2")
    df_f2 = None
    
    if metodo_f2 == "✏️ Manual (Editar tabla)":
        if 'df_f2' in st.session_state:
            st.subheader(f"Tabla del Diseño: {st.session_state['label_f2']}")
            y_col_f2 = st.text_input("Variable Respuesta (Y):", value="Y", key="y_f2")
            df_temp = st.session_state['df_f2'].copy()
            if y_col_f2 not in df_temp.columns: df_temp[y_col_f2] = np.nan
            df_f2 = st.data_editor(df_temp, num_rows="fixed", use_container_width=True, key="edit_f2")
    elif metodo_f2 == "📁 Cargar CSV":
        archivo = st.file_uploader("Sube tu CSV:", type="csv", key="csv_f2")
        if archivo: df_f2 = procesar_archivo(archivo)
    elif metodo_f2 == "🔗 Cargar desde URL":
        url = st.text_input("URL del CSV:", key="url_f2")
        if st.button("Cargar URL") and url: df_f2 = procesar_url(url)

    if df_f2 is not None:
        if metodo_f2 != "✏️ Manual (Editar tabla)": 
            st.dataframe(df_f2)
            
        cols = df_f2.columns.tolist()
        posibles_f = [c for c in cols if str(c).startswith('X')]
        if not posibles_f:
            posibles_f = [c for c in cols if c not in ['Run', 'Bloque'] and df_f2[c].dtype in ['float64', 'int64']]
            
        if len(posibles_f) > 0:
            n_f = st.number_input("¿Cuántos Factores (X) deseas usar en el modelo?", min_value=1, max_value=len(posibles_f), value=min(len(posibles_f), 3), key="nf_f2")
            factor_cols_f2 = posibles_f[:n_f]
            
            posibles_y_f2 = [c for c in cols if c not in factor_cols_f2 and c not in ['Run', 'Bloque']]
            y_sel_f2 = st.selectbox("Selecciona la Respuesta (Y) a analizar:", posibles_y_f2, index=0) if posibles_y_f2 else None
            
            if st.button("Ejecutar Análisis 2do Orden", type="primary") and y_sel_f2:
                if df_f2[y_sel_f2].isna().any(): 
                    st.error("Faltan datos en la columna Y seleccionada.")
                else:
                    try:
                        terms = list(factor_cols_f2)
                        for a, b in itertools.combinations(factor_cols_f2, 2):
                            terms.append(f"{a}*{b}")
                        for a in factor_cols_f2:
                            terms.append(f"I({a}**2)")
                        
                        formula = f"Q('{y_sel_f2}') ~ " + " + ".join(terms)
                        modelo_cuad = ols(formula, data=df_f2).fit()
                        
                        mask_centros = (df_f2[factor_cols_f2] == 0).all(axis=1)
                        df_centros = df_f2[mask_centros]
                        if len(df_centros) > 1:
                            ss_pure = ((df_centros[y_sel_f2] - df_centros[y_sel_f2].mean())**2).sum()
                            gl_pure = len(df_centros) - 1
                            ms_pure = ss_pure / gl_pure if gl_pure > 0 else 0
                        else: ss_pure, gl_pure, ms_pure = 0, 0, 0
                        
                        anova_table = sm.stats.anova_lm(modelo_cuad, typ=1)
                        ss_res = anova_table.loc['Residual', 'sum_sq']
                        gl_res = anova_table.loc['Residual', 'df']
                        
                        ss_lof = max(0, ss_res - ss_pure)
                        gl_lof = gl_res - gl_pure
                        ms_lof = ss_lof / gl_lof if gl_lof > 0 else 0
                        
                        st.markdown("### Análisis de Varianza para Respuesta")
                        anova_dict = {"Fuente": [], "Suma de Cuadrados": [], "Gl": [], "Cuadrado Medio": [], "Razón-F": [], "Valor-P": []}
                        
                        for term in anova_table.index:
                            if term == 'Residual': continue
                            clean_name = term.replace("Q('"+y_sel_f2+"')", "").replace(":", "*").replace("I(", "").replace(" ** 2)", "^2").strip()
                            ss = anova_table.loc[term, 'sum_sq']
                            df_val = anova_table.loc[term, 'df']
                            ms = ss / df_val
                            f_val = ms / ms_pure if ms_pure > 0 else np.nan
                            p_val = 1 - stats.f.cdf(f_val, df_val, gl_pure) if ms_pure > 0 else np.nan
                            
                            anova_dict["Fuente"].append(clean_name)
                            anova_dict["Suma de Cuadrados"].append(ss)
                            anova_dict["Gl"].append(int(df_val))
                            anova_dict["Cuadrado Medio"].append(ms)
                            anova_dict["Razón-F"].append(f_val)
                            anova_dict["Valor-P"].append(p_val)
                        
                        f_lof = ms_lof / ms_pure if ms_pure > 0 else np.nan
                        p_lof = 1 - stats.f.cdf(f_lof, gl_lof, gl_pure) if gl_pure > 0 and not np.isnan(f_lof) else np.nan
                        
                        anova_dict["Fuente"].extend(["Falta de ajuste", "Error puro", "Total (corr.)"])
                        anova_dict["Suma de Cuadrados"].extend([ss_lof, ss_pure, anova_table['sum_sq'].sum()])
                        anova_dict["Gl"].extend([int(gl_lof), int(gl_pure), int(anova_table['df'].sum())])
                        anova_dict["Cuadrado Medio"].extend([ms_lof, ms_pure, np.nan])
                        anova_dict["Razón-F"].extend([f_lof, np.nan, np.nan])
                        anova_dict["Valor-P"].extend([p_lof, np.nan, np.nan])
                        
                        st.dataframe(pd.DataFrame(anova_dict).style.format(precision=4, na_rep="-"), use_container_width=True)
                        st.write(f"**R² = {modelo_cuad.rsquared*100:.2f}%** | **R² ajustado = {modelo_cuad.rsquared_adj*100:.2f}%** | **Error estándar est. = {np.sqrt(modelo_cuad.mse_resid):.4f}**")

                        st.markdown("---")
                        st.subheader("🎯 Punto Estacionario / Óptimo")
                        
                        betas = modelo_cuad.params
                        b_linear = np.array([betas.get(f, 0) for f in factor_cols_f2])
                        
                        A = np.zeros((len(factor_cols_f2), len(factor_cols_f2)))
                        for i, f1 in enumerate(factor_cols_f2):
                            A[i, i] = 2 * betas.get(f"I({f1} ** 2)", 0)
                            for j, f2 in enumerate(factor_cols_f2):
                                if i < j:
                                    val = betas.get(f"{f1}:{f2}", betas.get(f"{f2}:{f1}", 0))
                                    A[i, j] = val
                                    A[j, i] = val
                                    
                        try:
                            sol = np.linalg.solve(A, -b_linear)
                            
                            df_opt_clean = pd.DataFrame({'Intercept': [1]})
                            for f, val in zip(factor_cols_f2, sol):
                                df_opt_clean[f] = [val]
                                df_opt_clean[f"I({f} ** 2)"] = [val**2]
                            for a, b in itertools.combinations(factor_cols_f2, 2):
                                df_opt_clean[f"{a}:{b}"] = [df_opt_clean[a][0] * df_opt_clean[b][0]]
                            
                            y_opt = modelo_cuad.predict(df_opt_clean)[0]
                            
                            opt_str = ", ".join([f"{f} = {val:.4f}" for f, val in zip(factor_cols_f2, sol)])
                            st.success(f"**Coordenadas del Óptimo:** {opt_str}")
                            st.success(f"**Respuesta Máxima Predicha:** {y_opt:.4f}")
                            
                            eigvals = np.linalg.eigvals(A)
                            if np.all(eigvals < 0): st.info("📍 El punto es un **Máximo**.")
                            elif np.all(eigvals > 0): st.info("📍 El punto es un **Mínimo**.")
                            else: st.info("📍 El punto es una **Silla**.")
                            
                        except np.linalg.LinAlgError:
                            st.warning("⚠️ Matriz singular. No se pudo calcular el punto estacionario.")

                        if len(factor_cols_f2) == 2:
                            st.markdown("---")
                            st.subheader("🗺️ Superficie de Respuesta y Curvas de Contorno")
                            x1, x2 = factor_cols_f2
                            x1_range = np.linspace(df_f2[x1].min(), df_f2[x1].max(), 50)
                            x2_range = np.linspace(df_f2[x2].min(), df_f2[x2].max(), 50)
                            XX1, XX2 = np.meshgrid(x1_range, x2_range)
                            
                            grid = pd.DataFrame({
                                'Intercept': np.ones(XX1.size), 
                                x1: XX1.ravel(), 
                                x2: XX2.ravel(), 
                                f'I({x1} ** 2)': XX1.ravel()**2, 
                                f'I({x2} ** 2)': XX2.ravel()**2, 
                                f'{x1}:{x2}': XX1.ravel() * XX2.ravel()
                            })
                            
                            Z = modelo_cuad.predict(grid).values.reshape(XX1.shape)
                            
                            tab_c, tab_s = st.tabs(["Curvas de Contorno", "Superficie 3D"])
                            with tab_c:
                                fig_c = go.Figure(data=go.Contour(x=x1_range, y=x2_range, z=Z, colorscale="Viridis", contours=dict(showlabels=True)))
                                fig_c.update_layout(xaxis_title=x1, yaxis_title=x2)
                                st.plotly_chart(fig_c, use_container_width=True)
                            with tab_s:
                                fig_s = go.Figure(data=[go.Surface(x=x1_range, y=x2_range, z=Z, colorscale="Viridis")])
                                fig_s.update_layout(scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=y_sel_f2))
                                st.plotly_chart(fig_s, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error en los cálculos: {e}")
        else:
            st.warning("No se detectaron factores válidos en la tabla.")


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

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(page_title="MSR Pro Statgraphics", page_icon="📈", layout="wide")

st.title("📈 Metodología de Superficie de Respuesta (MSR)")
st.markdown("Diseño y análisis de experimentos (DOE) con enfoque en optimización de procesos, al estilo **Statgraphics**.")
st.markdown("---")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
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
# GENERADORES DE DISEÑOS (ESTILO STATGRAPHICS)
# ============================================================
def factorial_3_2():
    runs = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
    return pd.DataFrame(runs, columns=["X1", "X2"]), "3^2 Factorial (9 runs)"

def factorial_3_2_blocks():
    base = factorial_3_2()[0]
    base["Bloque"] = [1,1,1, 2,2,2, 3,3,3]
    return base, "3^2 Factorial en 3 bloques (9 runs)"

def ccd_2_2_star(n_centros=5):
    cube = [(i, j) for i in (-1, 1) for j in (-1, 1)]
    star = [(-1.414, 0), (1.414, 0), (0, -1.414), (0, 1.414)]
    center = [(0, 0)] * n_centros
    df = pd.DataFrame(cube + star + center, columns=["X1", "X2"])
    df["Bloque"] = [1,1,1,1, 2,2,2,2] + [3]*n_centros
    return df, f"Central Composite 2^2 + star ({10+n_centros-1} runs)"

def ccd_blocked(n_centros=5):
    return ccd_2_2_star(n_centros)

def user_specified(n_runs=10, n_factors=2):
    cols = [f"X{i+1}" for i in range(n_factors)]
    df = pd.DataFrame(np.nan, index=range(n_runs), columns=cols)
    df.insert(0, "Run", range(1, n_runs+1))
    return df, "User-specified Design"

DESIGN_BUILDERS = {
    "3-level factorial design: 3^2": lambda: factorial_3_2(),
    "3-level factorial design: 3^2 in 3 blocks": lambda: factorial_3_2_blocks(),
    "Central composite design: 2^2 + star": lambda: ccd_2_2_star(st.session_state.get('n_centros_dcc', 5)),
    "Central composite blocked cube-star": lambda: ccd_blocked(st.session_state.get('n_centros_dcc', 5)),
    "User-specified design": lambda: user_specified(10, 2),
}

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
        with col1: k_f1 = st.number_input("N° Factores (k):", 2, 4, 2, key="k_f1")
        with col2: n_centros_f1 = st.number_input("Puntos Centrales:", 2, 10, 5, key="nc_f1")
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
        if metodo_f1 != "✏️ Manual (Editar tabla)": st.dataframe(df_f1)
        cols_no_y = [c for c in df_f1.columns if c not in ['Run', 'Y', 'Rendimiento', 'Respuesta']]
        factores = [c for c in cols_no_y if df_f1[c].dtype in ['float64', 'int64']]
        posibles_y = [c for c in df_f1.columns if c not in factores and c != 'Run']
        y_sel = st.selectbox("Columna de Respuesta (Y):", posibles_y, index=0) if posibles_y else None
        
        if st.button("Ejecutar Análisis 1er Orden", type="primary") and y_sel:
            if df_f1[y_sel].isna().any(): st.error("Faltan datos.")
            else:
                # ANOVA Completo para Fase 1
                X = sm.add_constant(df_f1[factores])
                Y = df_f1[y_sel]
                modelo = sm.OLS(Y, X).fit()
                
                st.markdown("### ANOVA del Modelo de 1er Orden")
                
                # Calcular Error Puro (puntos centrales)
                mask_centros = (df_f1[factores] == 0).all(axis=1)
                df_centros = df_f1[mask_centros]
                if len(df_centros) > 1:
                    ss_pure = ((df_centros[y_sel] - df_centros[y_sel].mean())**2).sum()
                    gl_pure = len(df_centros) - 1
                    ms_pure = ss_pure / gl_pure if gl_pure > 0 else 0
                else: ss_pure, gl_pure, ms_pure = 0, 0, 0
                
                # Falta de Ajuste
                ss_res = modelo.ssr
                gl_res = modelo.df_resid
                ss_lof = max(0, ss_res - ss_pure)
                gl_lof = gl_res - gl_pure
                ms_lof = ss_lof / gl_lof if gl_lof > 0 else 0
                
                anova_dict = {"Fuente": [], "Suma de Cuadrados": [], "Gl": [], "Cuadrado Medio": [], "Razón-F": [], "Valor-P": []}
                
                for f in factores:
                    # Regresión simple para obtener la suma de cuadrados secuencial
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
                beta = modelo.params[1:]
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

# ==============================================================================
# FASE 2: DISEÑO CENTRAL COMPUESTO Y OPTIMIZACIÓN (CON INTERFAZ STATGRAPHICS)
# ==============================================================================
with tab_fase2:
    st.header("2. Modelo de Segundo Orden (DCC)")
    
    # 1. SELECCIÓN DEL DISEÑO (INTERFAZ ORIGINAL)
    with st.expander("📌 1. Selección del Diseño Experimental (Response Surface)", expanded=True):
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            design_name = st.selectbox(
                "Selecciona el diseño:",
                list(DESIGN_BUILDERS.keys()),
                index=2,
                key="design_select"
            )
        with col_d2:
            n_centros_dcc = st.number_input("Puntos Centrales DCC:", 1, 10, 5, key="n_centros_dcc")
            n_factors = st.number_input("N° Factores (X):", 2, 4, 2, step=1)

        if st.button("🎲 Generar Tabla del Diseño", type="primary", key="btn_gen_design"):
            builder = DESIGN_BUILDERS[design_name]
            df_dis, label = builder()
            
            # Renombrar columnas X1, X2... a nombres personalizados
            factor_cols = [c for c in df_dis.columns if c.startswith("X")]
            for i, c in enumerate(factor_cols):
                new_name = st.session_state.get(f"name_{c}", c)
                df_dis.rename(columns={c: new_name}, inplace=True)

            st.session_state["df_f2"] = df_dis
            st.session_state["factor_cols_f2"] = factor_cols
            st.session_state["design_label"] = label
            st.rerun()

    if "df_f2" in st.session_state:
        df_msr = st.session_state["df_f2"].copy()
        factor_cols = st.session_state["factor_cols_f2"]

        st.subheader(f"📋 Tabla del Diseño — {st.session_state['design_label']}")
        
        # 2. CARGA DE DATOS
        metodo_f2 = obtener_datos("f2")
        df_f2 = None
        
        if metodo_f2 == "✏️ Manual (Editar tabla)":
            y_col_f2 = st.text_input("Variable Respuesta (Y):", value="Y", key="y_f2")
            if y_col_f2 not in df_msr.columns: df_msr[y_col_f2] = np.nan
            df_f2 = st.data_editor(df_msr, num_rows="fixed", use_container_width=True, key="edit_f2")
            
        elif metodo_f2 == "📁 Cargar CSV":
            archivo = st.file_uploader("Sube tu CSV:", type="csv", key="csv_f2")
            if archivo: df_f2 = procesar_archivo(archivo)
        elif metodo_f2 == "🔗 Cargar desde URL":
            url = st.text_input("URL del CSV:", key="url_f2")
            if st.button("Cargar URL") and url: df_f2 = procesar_url(url)

        if df_f2 is not None:
            if metodo_f2 != "✏️ Manual (Editar tabla)": st.dataframe(df_f2)
            
            posibles_y_f2 = [c for c in df_f2.columns if c not in factor_cols and c not in ['Run', 'Bloque']]
            y_sel_f2 = st.selectbox("Columna de Respuesta (Y):", posibles_y_f2, index=0) if posibles_y_f2 else None
            
            include_blocks = "Bloque" in df_f2.columns
            use_blocks = st.checkbox("Incluir Bloques en el modelo", value=include_blocks)

            # 3. EJECUCIÓN DEL ANÁLISIS (TABLA ANOVA ESTILO IMAGEN 2)
            if st.button("Ejecutar Análisis 2do Orden", type="primary") and y_sel_f2:
                if df_f2[y_sel_f2].isna().any(): st.error("Faltan datos.")
                else:
                    # Construcción de términos para el modelo cuadrático
                    terms = list(factor_cols)
                    for a, b in itertools.combinations(factor_cols, 2):
                        terms.append(f"{a}*{b}")
                    for a in factor_cols:
                        terms.append(f"I({a}**2)")
                    if use_blocks:
                        terms.append("C(Bloque)")
                    
                    formula = f"Q('{y_sel_f2}') ~ " + " + ".join(terms)
                    modelo_cuad = ols(formula, data=df_f2).fit()
                    
                    # Calcular Error Puro (Puntos centrales)
                    mask_centros = (df_f2[factor_cols] == 0).all(axis=1)
                    df_centros = df_f2[mask_centros]
                    
                    if len(df_centros) > 1:
                        ss_pure = ((df_centros[y_sel_f2] - df_centros[y_sel_f2].mean())**2).sum()
                        gl_pure = len(df_centros) - 1
                        ms_pure = ss_pure / gl_pure if gl_pure > 0 else 0
                    else: ss_pure, gl_pure, ms_pure = 0, 0, 0
                    
                    # Extraer ANOVA base
                    anova_table = sm.stats.anova_lm(modelo_cuad, typ=1)
                    ss_res = anova_table.loc['Residual', 'sum_sq']
                    gl_res = anova_table.loc['Residual', 'df']
                    
                    # Calcular Falta de Ajuste
                    ss_lof = max(0, ss_res - ss_pure)
                    gl_lof = gl_res - gl_pure
                    ms_lof = ss_lof / gl_lof if gl_lof > 0 else 0
                    
                    # Construir tabla ANOVA final
                    st.markdown("### Análisis de Varianza para Respuesta")
                    anova_dict = {"Fuente": [], "Suma de Cuadrados": [], "Gl": [], "Cuadrado Medio": [], "Razón-F": [], "Valor-P": []}
                    
                    for term in anova_table.index:
                        if term == 'Residual': continue
                        
                        # Limpiar nombres (ej. I(X1 ** 2) -> X1^2)
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
                    
                    # Añadir filas finales
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

                    # 4. PUNTO ESTACIONARIO Y GRÁFICOS (Solo para 2 factores)
                    if len(factor_cols) == 2:
                        st.markdown("---")
                        st.subheader("🎯 Punto Estacionario / Óptimo")
                        
                        x1, x2 = factor_cols
                        # Añadir variables manualmente para predecir
                        X_dict = {'Intercept': np.ones(len(df_f2)), x1: df_f2[x1], x2: df_f2[x2], f'{x1}^2': df_f2[x1]**2, f'{x2}^2': df_f2[x2]**2, f'{x1}*{x2}': df_f2[x1]*df_f2[x2]}
                        modelo_opt = sm.OLS(df_f2[y_sel_f2], pd.DataFrame(X_dict)).fit()
                        
                        b1, b2 = modelo_opt.params[x1], modelo_opt.params[x2]
                        b11, b22 = modelo_opt.params[f'{x1}^2'], modelo_opt.params[f'{x2}^2']
                        b12 = modelo_opt.params[f'{x1}*{x2}']
                        
                        A = np.array([[2*b11, b12], [b12, 2*b22]])
                        B = np.array([-b1, -b2])
                        
                        try:
                            sol = np.linalg.solve(A, B)
                            X_opt = pd.DataFrame({'Intercept': [1], x1: [sol[0]], x2: [sol[1]], f'{x1}^2': [sol[0]**2], f'{x2}^2': [sol[1]**2], f'{x1}*{x2}': [sol[0]*sol[1]]})
                            y_opt = modelo_opt.predict(X_opt)[0]
                            st.success(f"**Coordenadas del Óptimo:** {x1} = {sol[0]:.4f}, {x2} = {sol[1]:.4f}")
                            st.success(f"**Respuesta Máxima Predicha:** {y_opt:.4f}")
                        except: st.warning("Matriz singular.")
                            
                        st.markdown("---")
                        st.subheader("🗺️ Superficie de Respuesta y Curvas de Contorno")
                        x1_range = np.linspace(df_f2[x1].min(), df_f2[x1].max(), 50)
                        x2_range = np.linspace(df_f2[x2].min(), df_f2[x2].max(), 50)
                        XX1, XX2 = np.meshgrid(x1_range, x2_range)
                        grid = pd.DataFrame({'Intercept': np.ones(XX1.size), x1: XX1.ravel(), x2: XX2.ravel(), f'{x1}^2': XX1.ravel()**2, f'{x2}^2': XX2.ravel()**2, f'{x1}*{x2}': XX1.ravel()*XX2.ravel()})
                        Z = modelo_opt.predict(grid).values.reshape(XX1.shape)
                        
                        tab_c, tab_s = st.tabs(["Curvas de Contorno", "Superficie 3D"])
                        with tab_c:
                            fig_c = go.Figure(data=go.Contour(x=x1_range, y=x2_range, z=Z, colorscale="Viridis", contours=dict(showlabels=True)))
                            st.plotly_chart(fig_c, use_container_width=True)
                        with tab_s:
                            fig_s = go.Figure(data=[go.Surface(x=x1_range, y=x2_range, z=Z, colorscale="Viridis")])
                            st.plotly_chart(fig_s, use_container_width=True)

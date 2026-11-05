
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

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="📈 Metodología de Superficie de Respuesta",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Metodología de Superficie de Respuesta (MSR)")
st.markdown("Diseño y análisis de experimentos (DOE) con enfoque en optimización de procesos, al estilo **Statgraphics**.")
st.markdown("---")

# ============================================================
# 1. GENERADORES DE DISEÑOS EXPERIMENTALES
# ============================================================
def factorial_3_2():
    runs = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
    return pd.DataFrame(runs, columns=["X1", "X2"]), "3^2 Factorial (9 runs)"

def factorial_3_2_blocks():
    base = factorial_3_2()[0]
    base["Bloque"] = [1,1,1, 2,2,2, 3,3,3]
    return base, "3^2 Factorial en 3 bloques (9 runs)"

def ccd_2_2_star():
    cube = [(i, j) for i in (-1, 1) for j in (-1, 1)]
    star = [(-1.414, 0), (1.414, 0), (0, -1.414), (0, 1.414)]
    center = [(0, 0)]
    df = pd.DataFrame(cube + star + center, columns=["X1", "X2"])
    df["Bloque"] = [1,1,1,1, 2,2,2,2, 3]
    return df, "Central Composite 2^2 + star (10 runs)"

def ccd_blocked():
    return ccd_2_2_star()

def user_specified(n_runs=10, n_factors=2):
    cols = [f"X{i+1}" for i in range(n_factors)]
    df = pd.DataFrame(np.nan, index=range(n_runs), columns=cols)
    df.insert(0, "Run", range(1, n_runs+1))
    return df, "User-specified Design"

DESIGN_BUILDERS = {
    "3-level factorial design: 3^2": factorial_3_2,
    "3-level factorial design: 3^2 in 3 blocks": factorial_3_2_blocks,
    "Central composite design: 2^2 + star": ccd_2_2_star,
    "Central composite blocked cube-star": ccd_blocked,
    "User-specified design": user_specified,
}

# ============================================================
# 2. INTERFAZ: SELECCIÓN DEL DISEÑO
# ============================================================
with st.expander("📌 1. Selección del Diseño Experimental", expanded=True):
    st.markdown("**Response Surface Design Selection**")
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        design_name = st.selectbox(
            "Selecciona el diseño:",
            list(DESIGN_BUILDERS.keys()),
            index=2,
            key="design_select"
        )
    with col_d2:
        n_factors = st.number_input("N° Factores (X):", 1, 5, 2, 1)

    st.info(
        f"**Diseño seleccionado:** {design_name}  \n"
        f"**Factores:** {n_factors}  \n"
        f"**Corridas estimadas:** depende del diseño seleccionado"
    )

    if st.button("🎲 Generar Tabla del Diseño", type="primary", key="btn_gen_design"):
        builder = DESIGN_BUILDERS[design_name]
        if design_name == "User-specified design":
            df_dis, label = builder(n_runs=10, n_factors=n_factors)
        else:
            df_dis, label = builder()

        factor_cols = [c for c in df_dis.columns if c.startswith("X")]
        for i, c in enumerate(factor_cols):
            new_name = st.session_state.get(f"name_{c}", c)
            df_dis.rename(columns={c: new_name}, inplace=True)

        st.session_state["df_msr"] = df_dis
        st.session_state["factor_cols_msr"] = factor_cols
        st.session_state["design_label"] = label
        st.rerun()

# ============================================================
# 3. EDICIÓN DE DATOS Y DEFINICIÓN DEL MODELO
# ============================================================
if "df_msr" in st.session_state:
    df_msr = st.session_state["df_msr"].copy()
    factor_cols = st.session_state["factor_cols_msr"]

    st.markdown("---")
    st.subheader(f"📋 Tabla del Diseño — {st.session_state['design_label']}")
    st.dataframe(df_msr, use_container_width=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        resp_name = st.text_input("Nombre de la respuesta (Y):", value="Y")
    with col_r2:
        model_order = st.selectbox(
            "Modelo de superficie:",
            ["Lineal (1er orden)", "Interacción (1er orden + 2FI)", "Cuadrático (2do orden)"]
        )
    with col_r3:
        include_blocks = "Bloque" in df_msr.columns
        use_blocks = st.checkbox("Incluir Bloques en el modelo", value=include_blocks)

    if resp_name not in df_msr.columns:
        df_msr[resp_name] = np.nan

    st.markdown("**✏️ Ingresa los valores de la respuesta (Y):**")
    df_edit = st.data_editor(
        df_msr,
        num_rows="fixed",
        use_container_width=True,
        key="editor_msr"
    )

    # ============================================================
    # 4. ANÁLISIS MSR
    # ============================================================
    if st.button("🚀 Ejecutar Análisis MSR", type="primary"):
        y_vals = df_edit[resp_name].values.astype(float)
        if np.isnan(y_vals).any():
            st.error("❌ Faltan datos en la respuesta Y. Completa la tabla antes de analizar.")
        else:
            terms = list(factor_cols)
            if model_order in ("Interacción (1er orden + 2FI)", "Cuadrático (2do orden)"):
                for a, b in itertools.combinations(factor_cols, 2):
                    terms.append(f"{a}*{b}")
            if model_order == "Cuadrático (2do orden)":
                for a in factor_cols:
                    terms.append(f"I({a}**2)")
            if use_blocks and "Bloque" in df_edit.columns:
                terms.append("C(Bloque)")

            formula = f"{resp_name} ~ " + " + ".join(terms)

            st.markdown("---")
            st.subheader("📊 ANOVA del Modelo de Superficie")
            try:
                modelo = ols(formula, data=df_edit).fit()
                anova = sm.stats.anova_lm(modelo, typ=1)
                if "mean_sq" not in anova.columns:
                    anova["mean_sq"] = anova["sum_sq"] / anova["df"]
                anova.rename(columns={
                    "sum_sq": "Sum Sq", "df": "DF", "mean_sq": "Mean Sq",
                    "F": "F-Ratio", "PR(>F)": "P-Value"
                }, inplace=True)
                st.dataframe(anova.style.format({
                    "Sum Sq": "{:.4f}", "Mean Sq": "{:.4f}",
                    "F-Ratio": "{:.4f}", "P-Value": "{:.6f}"
                }, na_rep="None"))
                st.write(
                    f"**R² = {modelo.rsquared:.6f}** | "
                    f"**R² ajustado = {modelo.rsquared_adj:.6f}**"
                )
            except Exception as e:
                st.error(f"❌ Error en ANOVA: {e}")
                modelo = None

            if modelo is not None:
                st.markdown("---")
                st.subheader("🧮 Coeficientes de Regresión")
                coefs = pd.DataFrame({
                    "Término": modelo.params.index,
                    "Coeficiente": modelo.params.values,
                    "Error Std": modelo.bse.values,
                    "t": modelo.tvalues.values,
                    "P>|t|": modelo.pvalues.values,
                })
                st.dataframe(coefs.style.format({
                    "Coeficiente": "{:.4f}", "Error Std": "{:.4f}",
                    "t": "{:.4f}", "P>|t|": "{:.6f}"
                }))

                if len(factor_cols) == 2:
                    st.markdown("---")
                    st.subheader("🗺️ Superficie de Respuesta y Curvas de Contorno")
                    x1, x2 = factor_cols
                    x1_range = np.linspace(df_edit[x1].min(), df_edit[x1].max(), 50)
                    x2_range = np.linspace(df_edit[x2].min(), df_edit[x2].max(), 50)
                    XX1, XX2 = np.meshgrid(x1_range, x2_range)

                    grid = pd.DataFrame({x1: XX1.ravel(), x2: XX2.ravel()})
                    if use_blocks and "Bloque" in df_edit.columns:
                        grid["Bloque"] = df_edit["Bloque"].mode()[0]
                    Z = modelo.predict(grid).values.reshape(XX1.shape)

                    tab1, tab2 = st.tabs(["🗺️ Contorno", "⛰️ Superficie 3D"])
                    with tab1:
                        fig_c = go.Figure(data=go.Contour(
                            x=x1_range, y=x2_range, z=Z,
                            contours=dict(showlabels=True),
                            colorscale="Viridis"
                        ))
                        fig_c.update_layout(xaxis_title=x1, yaxis_title=x2, title="Curvas de Contorno")
                        st.plotly_chart(fig_c, use_container_width=True)
                    with tab2:
                        fig_s = go.Figure(data=[go.Surface(
                            x=x1_range, y=x2_range, z=Z, colorscale="Viridis"
                        )])
                        fig_s.update_layout(
                            scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=resp_name),
                            title="Superficie de Respuesta 3D"
                        )
                        st.plotly_chart(fig_s, use_container_width=True)

                if model_order == "Cuadrático (2do orden)" and len(factor_cols) == 2:
                    st.markdown("---")
                    st.subheader("🎯 Punto Estacionario / Óptimo")
                    try:
                        x1, x2 = factor_cols
                        p = modelo.params
                        b0, b1, b2 = p.get("Intercept", 0), p.get(x1, 0), p.get(x2, 0)
                        b11 = p.get(f"I({x1} ** 2)", 0)
                        b22 = p.get(f"I({x2} ** 2)", 0)
                        b12 = p.get(f"{x1}:{x2}", p.get(f"{x2}:{x1}", 0))

                        A = np.array([[2*b11, b12], [b12, 2*b22]])
                        B = np.array([-b1, -b2])
                        sol = np.linalg.solve(A, B)
                        y_opt = modelo.predict(pd.DataFrame({x1: [sol[0]], x2: [sol[1]]})).values[0]

                        st.success(
                            f"**Punto estacionario:** {x1} = {sol[0]:.4f}, {x2} = {sol[1]:.4f}  \n"
                            f"**Respuesta predicha:** {resp_name} = {y_opt:.4f}"
                        )
                    except np.linalg.LinAlgError:
                        st.warning("⚠️ Matriz singular. No se puede calcular el punto estacionario.")
                    except Exception as e:
                        st.warning(f"No se pudo calcular el óptimo: {e}")

                st.markdown("---")
                st.subheader("🔮 Predicciones del Modelo")
                pred_df = df_edit[factor_cols].copy()
                pred_df["Y_pred"] = modelo.predict(df_edit)
                pred_df["Residuo"] = y_vals - pred_df["Y_pred"]
                st.dataframe(pred_df.style.format("{:.4f}"), use_container_width=True)

                fig_res = px.scatter(
                    pred_df, x="Y_pred", y="Residuo",
                    title="Residuos vs Valores Ajustados",
                    labels={"Y_pred": "Valor Ajustado", "Residuo": "Residuo"}
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)

                csv_out = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Descargar resultados (CSV)",
                    data=csv_out,
                    file_name="resultados_msr.csv",
                    mime="text/csv"
                )

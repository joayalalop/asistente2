import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import torch

# 0. OPTIMIZACIÓN DE HARDWARE PARA NUBE GRATUITA (Evita que Linux congele la CPU)
torch.set_num_threads(1)

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Asistente CPC - ENESEM (V3.1 Cloud)", layout="wide"
)


# 2. FUNCIÓN DE NORMALIZACIÓN LINGÜÍSTICA
def normalizar_glosa(texto):
    """Limpia puntuación, espacios y conectores ('Y', 'DE', 'EN') para un Match Exacto flexible."""
    if not texto or pd.isna(texto):
        return ""
    t = str(texto).upper().strip()
    t = re.sub(r"[,\.\-/()#]", " ", t)
    t = re.sub(
        r"\b(Y|E|DE|DEL|LA|EL|LOS|LAS|EN|CON|PARA|POR|AL|UN|UNA)\b", " ", t
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t


# 3. FUNCIONES CON CACHÉ (Modelos ligeros y carga optimizada)
@st.cache_resource
def cargar_modelo():
    # MODELO OPTIMIZADO PARA NUBE: Consume mitad de RAM y es 3x más rápido
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data
def cargar_catalogo_oficial():
    """Carga cpc.xlsx en memoria para consultar descripciones técnicas."""
    try:
        df_cpc = pd.read_excel("cpc.xlsx", sheet_name=0)
        col_cod = [
            c for c in df_cpc.columns if "CODIGO" in str(c).upper()
        ][0]
        col_desc = [
            c
            for c in df_cpc.columns
            if "DESCRIP" in str(c).upper() or "CPC" in str(c).upper()
        ][-1]

        mapa = {}
        for _, row in df_cpc.iterrows():
            c_raw = str(row[col_cod]).replace(".0", "").strip()
            desc = str(row[col_desc]).strip()
            if c_raw and desc and desc != "nan" and desc != "":
                mapa[c_raw] = desc
                if len(c_raw) == 8:
                    mapa[c_raw.zfill(9)] = desc
                if len(c_raw) == 3:
                    mapa[c_raw.zfill(4)] = desc
        return mapa
    except Exception as e:
        return {}


@st.cache_data
def cargar_y_desagregar_datos():
    """Carga los Excel limpios y separa las glosas históricas para evitar dilución semántica."""
    df_com = pd.read_excel("diccionario_cpc_comercio_limpio.xlsx").fillna("")
    df_ser = pd.read_excel("diccionario_cpc_servicios_limpio.xlsx").fillna("")
    df_mp = pd.read_excel(
        "diccionario_cpc_materias_primas_limpio.xlsx"
    ).fillna("")
    df_prod = pd.read_excel("diccionario_cpc_productos_limpio.xlsx").fillna("")

    def desagregar_modulo(df, es_comercio=False):
        filas_exp = []
        for _, row in df.iterrows():
            cod = row["CODIGO_CPC"]
            ciiu = row["CIIU_ASOCIADO"]
            historial_completo = row["EJEMPLOS_REALES_LIMPIOS"]
            glosas = str(historial_completo).split(" || ")
            for g in glosas:
                g_clean = g.strip()
                if g_clean:
                    if es_comercio:
                        tipo = row["TIPO_COMERCIO"]
                        filas_exp.append({
                            "CODIGO_CPC": cod,
                            "CIIU_ASOCIADO": ciiu,
                            "TIPO_COMERCIO": tipo,
                            "EJEMPLOS_REALES_LIMPIOS": historial_completo,
                            "GLOSA_INDIVIDUAL": g_clean,
                            "TEXTO_A_VECTORIZAR": f"COMERCIO {tipo}: {g_clean}",
                        })
                    else:
                        filas_exp.append({
                            "CODIGO_CPC": cod,
                            "CIIU_ASOCIADO": ciiu,
                            "EJEMPLOS_REALES_LIMPIOS": historial_completo,
                            "GLOSA_INDIVIDUAL": g_clean,
                            "TEXTO_A_VECTORIZAR": g_clean,
                        })
        return pd.DataFrame(filas_exp)

    df_com_exp = desagregar_modulo(df_com, es_comercio=True)
    df_ser_exp = desagregar_modulo(df_ser)
    df_mp_exp = desagregar_modulo(df_mp)
    df_prod_exp = desagregar_modulo(df_prod)

    dict_ex_com = {}
    for _, row in df_com.iterrows():
        glosas = str(row["EJEMPLOS_REALES_LIMPIOS"]).split(" || ")
        for g in glosas:
            g_norm = normalizar_glosa(g)
            if g_norm:
                dict_ex_com[(g_norm, row["TIPO_COMERCIO"])] = (
                    row["CODIGO_CPC"],
                    row["EJEMPLOS_REALES_LIMPIOS"],
                )

    def crear_dict_ex(df):
        d = {}
        for _, row in df.iterrows():
            glosas = str(row["EJEMPLOS_REALES_LIMPIOS"]).split(" || ")
            for g in glosas:
                g_norm = normalizar_glosa(g)
                if g_norm:
                    d[g_norm] = (
                        row["CODIGO_CPC"],
                        row["EJEMPLOS_REALES_LIMPIOS"],
                    )
        return d

    return (
        df_com_exp,
        df_ser_exp,
        df_mp_exp,
        df_prod_exp,
        dict_ex_com,
        crear_dict_ex(df_ser),
        crear_dict_ex(df_mp),
        crear_dict_ex(df_prod),
    )


# CÁLCULO BAJO DEMANDA (Lazy Loading): Solo calcula el bloque cuando se busca por primera vez
@st.cache_data(show_spinner=False)
def obtener_vectores(_modelo, df_subconjunto):
    return _modelo.encode(
        df_subconjunto["TEXTO_A_VECTORIZAR"].tolist(), show_progress_bar=False
    )


def get_desc_oficial(codigo_hybrid, largo_codigo, mapa_cpc):
    cod_str = str(codigo_hybrid).zfill(largo_codigo)
    cpc_puro = cod_str[-4:] if largo_codigo == 8 else cod_str[-9:]

    if cpc_puro in mapa_cpc:
        return mapa_cpc[cpc_puro]
    if largo_codigo == 13:
        if cpc_puro[:7] in mapa_cpc:
            return mapa_cpc[cpc_puro[:7]] + " (Nivel agrupado)"
        if cpc_puro[:5] in mapa_cpc:
            return mapa_cpc[cpc_puro[:5]] + " (Nivel agrupado)"
    elif largo_codigo == 8:
        if cpc_puro[:3] in mapa_cpc:
            return mapa_cpc[cpc_puro[:3]] + " (Nivel agrupado)"
    return "Definición técnica según Clasificador Central de Productos (CPC 2.0)"


# 4. INTERFAZ VISUAL PRINCIPAL
st.title("🛡️ Motor de Codificación Asistida CPC")
st.subheader("Versión 3.1 Cloud — Arquitectura Lider en Eficiencia")
st.markdown("Herramienta NLP de la Encuesta Estructural Empresarial - ENESEM")

# Arranque ligero (Ya no precalcula vectores masivos de golpe)
with st.spinner("Inicializando motor de Inteligencia Artificial..."):
    modelo = cargar_modelo()
    mapa_oficial = cargar_catalogo_oficial()
    (
        df_comercio,
        df_servicios,
        df_mat_primas,
        df_productos,
        dict_ex_com,
        dict_ex_ser,
        dict_ex_mp,
        dict_ex_prod,
    ) = cargar_y_desagregar_datos()

st.write("---")

with st.sidebar:
    st.header("🏢 CIIU de la Empresa")
    ciiu_empresa = (
        st.text_input(
            "CIIU de la empresa (opcional):",
            max_chars=4,
            placeholder="Ej: 4630",
        )
        .strip()
        .zfill(4)
    )
    if (
        ciiu_empresa
        and ciiu_empresa != "0000"
        and not ciiu_empresa.isdigit()
    ):
        st.error("El código CIIU debe contener 4 números.")


# 5. LÓGICA DE BÚsqueda Y DEDUPLICACIÓN
def mostrar_sugerencias(
    df_modulo, glosa_input, largo_codigo, es_comercio=False
):
    # Aquí se ejecuta la vectorización bajo demanda sin saturar la RAM inicial
    with st.spinner("Calculando similitud semántica en tiempo real..."):
        vectores_modulo = obtener_vectores(modelo, df_modulo)
        vector_consulta = modelo.encode([glosa_input])
        similitudes = cosine_similarity(vector_consulta, vectores_modulo)[0]

    df_res = pd.DataFrame({
        "similitud": similitudes,
        "CODIGO_CPC": df_modulo["CODIGO_CPC"].values,
        "CIIU_ASOCIADO": df_modulo["CIIU_ASOCIADO"].values,
        "EJEMPLOS_REALES_LIMPIOS": df_modulo["EJEMPLOS_REALES_LIMPIOS"].values,
        "GLOSA_MATCH": df_modulo["GLOSA_INDIVIDUAL"].values,
        "TIPO_COMERCIO": (
            df_modulo["TIPO_COMERCIO"].values
            if es_comercio
            else [None] * len(df_modulo)
        ),
    })

    df_res = df_res.sort_values(by="similitud", ascending=False)
    if es_comercio:
        top_3 = df_res.drop_duplicates(
            subset=["CODIGO_CPC", "TIPO_COMERCIO"]
        ).head(3)
    else:
        top_3 = df_res.drop_duplicates(subset=["CODIGO_CPC"]).head(3)

    st.markdown("### Top 3 Sugerencias Semánticas (Rankeadas por IA)")

    for i, (_, row) in enumerate(top_3.iterrows()):
        codigo = str(row["CODIGO_CPC"]).zfill(largo_codigo)
        ejemplos = row["EJEMPLOS_REALES_LIMPIOS"]
        ciiu_asociado = str(row["CIIU_ASOCIADO"]).zfill(4)
        confianza = row["similitud"] * 100
        glosa_ganadora = row["GLOSA_MATCH"]
        desc_oficial = get_desc_oficial(codigo, largo_codigo, mapa_oficial)

        if confianza >= 85:
            color, banda = "green", "🟢 AUTOMATIZAR (Banda Verde)"
        elif confianza >= 50:
            color, banda = "orange", "🟡 REVISAR - Asistido (Banda Amarilla)"
        else:
            color, banda = "red", "🔴 MANUAL (Banda Roja)"

        alerta_ciiu = ""
        if ciiu_empresa and ciiu_empresa != "0000":
            if ciiu_empresa == ciiu_asociado:
                alerta_ciiu = " | ⭐ **COINCIDE CON EL CIIU DE LA EMPRESA**"
            else:
                alerta_ciiu = (
                    f" | ⚠️ *CIIU del código sugerido ({ciiu_asociado})"
                    " difiere del de la empresa*"
                )

        with st.expander(
            f"Opción {i + 1}: Código CPC {codigo} (Confianza: {confianza:.2f}%)"
        ):
            st.markdown(
                f"**Código Final Sugerido:** `{codigo}` (Sub-frase más"
                f" cercana: *'{glosa_ganadora}'*)"
            )
            st.markdown(
                f"**CIIU de Origen del Código:** `{ciiu_asociado}`{alerta_ciiu}"
            )
            st.markdown(
                f"**Nivel de Certidumbre:** :{color}[{confianza:.2f}%] ->"
                f" **Acción sugerida:** {banda}"
            )
            st.markdown("---")
            st.markdown(
                f"📖 **Descripción Oficial (CPC 2.0):** \n> *{desc_oficial}*"
            )
            st.markdown(
                f"🏢 **Historial completo de Glosas (2020-2024):** \n*{ejemplos}*"
            )


# 6. PESTAÑAS DE LA INTERFAZ
tab_com, tab_ser, tab_mp, tab_prod = st.tabs([
    "🏪 Comercio",
    "🛠️ Servicios",
    "🪵 Materias Primas",
    "📦 Productos",
])

with tab_com:
    col_t1, col_t2 = st.columns([2, 5])
    with col_t1:
        tipo_comercio = st.radio(
            "Tipo de Venta:", ["POR MAYOR", "POR MENOR"], key="com_tipo"
        )
    with col_t2:
        glosa_comercio = st.text_input(
            "Mercancía a codificar:",
            placeholder="Ej: Útiles escolares / Atún en conserva",
            key="com_txt",
        )

    if st.button(
        "Buscar Código CPC - Comercio", type="primary", key="btn_com"
    ):
        if glosa_comercio:
            glosa_norm = normalizar_glosa(glosa_comercio)
            if (glosa_norm, tipo_comercio) in dict_ex_com:
                codigo_ex, ejemplos_ex = dict_ex_com[(
                    glosa_norm,
                    tipo_comercio,
                )]
                desc_ex = get_desc_oficial(codigo_ex, 8, mapa_oficial)
                st.success(
                    "🎯 MATCH EXACTO HISTÓRICO ENCONTRADO EN " + tipo_comercio
                )
                st.metric(
                    label="Confianza", value="100%", delta="Asignación Directa"
                )
                st.info(
                    f"**Código Asignado (8 dígitos):** `{str(codigo_ex).zfill(8)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning("Buscando aproximaciones mediante NLP...")
                idx_filt = df_comercio[
                    df_comercio["TIPO_COMERCIO"] == tipo_comercio
                ].index.tolist()
                df_sub = df_comercio.loc[idx_filt].reset_index(drop=True)
                mostrar_sugerencias(
                    df_sub, glosa_comercio, largo_codigo=8, es_comercio=True
                )
        else:
            st.error("Por favor, ingrese una descripción.")

with tab_ser:
    glosa_servicios = st.text_input(
        "Servicio prestado por la empresa:",
        placeholder=(
            "Ej: Transporte de carga por carretera / Hospedaje en hotel"
        ),
        key="ser_txt",
    )
    if st.button(
        "Buscar Código CPC - Servicios", type="primary", key="btn_ser"
    ):
        if glosa_servicios:
            glosa_norm = normalizar_glosa(glosa_servicios)
            if glosa_norm in dict_ex_ser:
                codigo_ex, ejemplos_ex = dict_ex_ser[glosa_norm]
                desc_ex = get_desc_oficial(codigo_ex, 8, mapa_oficial)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (8 dígitos):** `{str(codigo_ex).zfill(8)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en la base de"
                    " Servicios..."
                )
                mostrar_sugerencias(df_servicios, glosa_servicios, largo_codigo=8)
        else:
            st.error("Por favor, ingrese una descripción.")

with tab_mp:
    glosa_mp = st.text_input(
        "Materia Prima:",
        placeholder=(
            "Ej: Planchas de tol galvanizado / Harina de trigo industrial"
        ),
        key="mp_txt",
    )
    if st.button(
        "Buscar Código CPC - Materias Primas", type="primary", key="btn_mp"
    ):
        if glosa_mp:
            glosa_norm = normalizar_glosa(glosa_mp)
            if glosa_norm in dict_ex_mp:
                codigo_ex, ejemplos_ex = dict_ex_mp[glosa_norm]
                desc_ex = get_desc_oficial(codigo_ex, 13, mapa_oficial)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(codigo_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en glosas"
                    " individuales..."
                )
                mostrar_sugerencias(
                    df_mat_primas, glosa_mp, largo_codigo=13
                )
        else:
            st.error("Por favor, ingrese una descripción.")

with tab_prod:
    glosa_prod = st.text_input(
        "Producto fabricado:",
        placeholder="Ej: Bloques de hormigón / Aceite refinado de palma",
        key="prod_txt",
    )
    if st.button(
        "Buscar Código CPC - Productos", type="primary", key="btn_prod"
    ):
        if glosa_prod:
            glosa_norm = normalizar_glosa(glosa_prod)
            if glosa_norm in dict_ex_prod:
                codigo_ex, ejemplos_ex = dict_ex_prod[glosa_norm]
                desc_ex = get_desc_oficial(codigo_ex, 13, mapa_oficial)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(codigo_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en glosas"
                    " individuales..."
                )
                mostrar_sugerencias(
                    df_productos, glosa_prod, largo_codigo=13
                )
        else:
            st.error("Por favor, ingrese una descripción.")

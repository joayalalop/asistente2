import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import torch

# 0. OPTIMIZACIÓN DE HARDWARE: Limita los hilos de CPU para que Linux no sature el contenedor
torch.set_num_threads(1)

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Asistente CPC - ENESEM (V3.2 Cloud)", layout="wide"
)


# 2. FUNCIÓN DE NORMALIZACIÓN LINGÜÍSTICA
def normalizar_glosa(texto):
    """Limpia puntuación y conectores ('Y', 'DE', 'EN') para asegurar Match Exacto."""
    if not texto or pd.isna(texto):
        return ""
    t = str(texto).upper().strip()
    t = re.sub(r"[,\.\-/()#]", " ", t)
    t = re.sub(
        r"\b(Y|E|DE|DEL|LA|EL|LOS|LAS|EN|CON|PARA|POR|AL|UN|UNA)\b", " ", t
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t


# 3. CARGA DE MODELO Y CATÁLOGO OFICIAL (En caché permanente)
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    # Modelo compacto MiniLM: Mitad de RAM y 3x más rápido en servidores de nube
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def cargar_catalogo_oficial():
    """Carga cpc.xlsx en memoria únicamente para consultar descripciones oficiales."""
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


# 4. CARGA MODULAR Y DESAGREGADA INDIVIDUAL (Cero colapso de RAM en el arranque)
@st.cache_data(show_spinner=False)
def cargar_modulo_desagregado(nombre_archivo, es_comercio=False):
    """Carga SOLO el archivo Excel que el crítico está consultando en ese momento."""
    df = pd.read_excel(nombre_archivo).fillna("")

    filas_exp = []
    dict_exacto = {}

    for _, row in df.iterrows():
        cod = row["CODIGO_CPC"]
        ciiu = row["CIIU_ASOCIADO"]
        historial_completo = row["EJEMPLOS_REALES_LIMPIOS"]

        glosas = str(historial_completo).split(" || ")
        for g in glosas:
            g_clean = g.strip()
            g_norm = normalizar_glosa(g_clean)

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
                    if g_norm:
                        dict_exacto[(g_norm, tipo)] = (cod, historial_completo)
                else:
                    filas_exp.append({
                        "CODIGO_CPC": cod,
                        "CIIU_ASOCIADO": ciiu,
                        "EJEMPLOS_REALES_LIMPIOS": historial_completo,
                        "GLOSA_INDIVIDUAL": g_clean,
                        "TEXTO_A_VECTORIZAR": g_clean,
                    })
                    if g_norm:
                        dict_exacto[g_norm] = (cod, historial_completo)

    df_expanded = pd.DataFrame(filas_exp)
    return df_expanded, dict_exacto


@st.cache_data(show_spinner=False)
def obtener_vectores(_modelo, df_subconjunto):
    """Vectorización matemática bajo demanda."""
    return _modelo.encode(
        df_subconjunto["TEXTO_A_VECTORIZAR"].tolist(), show_progress_bar=False
    )


# FUNCIÓN TRADUCTORA A DESCRIPCIÓN OFICIAL
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


# 5. INTERFAZ VISUAL PRINCIPAL (Arranque instantáneo en 0.5 segundos)
st.title("🛡️ Motor de Codificación Asistida CPC")
st.subheader("Versión 3.2 Cloud — Arquitectura Modular Segura")
st.markdown("Herramienta NLP de la Encuesta Estructural Empresarial - ENESEM")
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


# 6. LÓGICA DE EVALUACIÓN Y DEDUPLICACIÓN
def procesar_y_mostrar(
    df_modulo, glosa_input, largo_codigo, es_comercio=False
):
    with st.spinner("Consultando motor semántico IA y vectorizando..."):
        modelo = cargar_modelo()
        mapa_oficial = cargar_catalogo_oficial()
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


# 7. PESTAÑAS DE LA INTERFAZ
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
            with st.spinner("Cargando módulo de Comercio..."):
                df_mod, dict_ex = cargar_modulo_desagregado(
                    "diccionario_cpc_comercio_limpio.xlsx", es_comercio=True
                )
                mapa_of = cargar_catalogo_oficial()

            glosa_norm = normalizar_glosa(glosa_comercio)
            if (glosa_norm, tipo_comercio) in dict_ex:
                cod_ex, hist_ex = dict_ex[(glosa_norm, tipo_comercio)]
                desc_ex = get_desc_oficial(cod_ex, 8, mapa_of)
                st.success(
                    "🎯 MATCH EXACTO HISTÓRICO ENCONTRADO EN " + tipo_comercio
                )
                st.metric(
                    label="Confianza", value="100%", delta="Asignación Directa"
                )
                st.info(
                    f"**Código Asignado (8 dígitos):** `{str(cod_ex).zfill(8)}` \n\n"
                    f" 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n 🏢"
                    f" **Historial de Glosas:** {hist_ex}"
                )
            else:
                st.warning("Buscando aproximaciones mediante NLP...")
                idx_filt = df_mod[
                    df_mod["TIPO_COMERCIO"] == tipo_comercio
                ].index.tolist()
                procesar_y_mostrar(
                    df_mod.loc[idx_filt].reset_index(drop=True),
                    glosa_comercio,
                    largo_codigo=8,
                    es_comercio=True,
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
            with st.spinner("Cargando módulo de Servicios..."):
                df_mod, dict_ex = cargar_modulo_desagregado(
                    "diccionario_cpc_servicios_limpio.xlsx"
                )
                mapa_of = cargar_catalogo_oficial()

            glosa_norm = normalizar_glosa(glosa_servicios)
            if glosa_norm in dict_ex:
                cod_ex, hist_ex = dict_ex[glosa_norm]
                desc_ex = get_desc_oficial(cod_ex, 8, mapa_of)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (8 dígitos):** `{str(cod_ex).zfill(8)}` \n\n"
                    f" 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n 🏢"
                    f" **Historial de Glosas:** {hist_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en Servicios..."
                )
                procesar_y_mostrar(df_mod, glosa_servicios, largo_codigo=8)
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
            with st.spinner("Cargando módulo de Materias Primas..."):
                df_mod, dict_ex = cargar_modulo_desagregado(
                    "diccionario_cpc_materias_primas_limpio.xlsx"
                )
                mapa_of = cargar_catalogo_oficial()

            glosa_norm = normalizar_glosa(glosa_mp)
            if glosa_norm in dict_ex:
                cod_ex, hist_ex = dict_ex[glosa_norm]
                desc_ex = get_desc_oficial(cod_ex, 13, mapa_of)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(cod_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {hist_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en Materias Primas..."
                )
                procesar_y_mostrar(df_mod, glosa_mp, largo_codigo=13)
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
            with st.spinner("Cargando módulo de Productos..."):
                df_mod, dict_ex = cargar_modulo_desagregado(
                    "diccionario_cpc_productos_limpio.xlsx"
                )
                mapa_of = cargar_catalogo_oficial()

            glosa_norm = normalizar_glosa(glosa_prod)
            if glosa_norm in dict_ex:
                cod_ex, hist_ex = dict_ex[glosa_norm]
                desc_ex = get_desc_oficial(cod_ex, 13, mapa_of)
                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(cod_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {hist_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en Productos..."
                )
                procesar_y_mostrar(df_mod, glosa_prod, largo_codigo=13)
        else:
            st.error("Por favor, ingrese una descripción.")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Asistente CPC - ENESEM", layout="wide"
)


# 2. FUNCIONES CON CACHÉ (Optimización de memoria y velocidad)
@st.cache_resource
def cargar_modelo():
    # Modelo multilingüe optimizado para emparejamiento de textos técnicos
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


@st.cache_data
def cargar_catalogo_oficial():
    """Carga cpc.xlsx en memoria para consultar descripciones técnicas al instante."""
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
                # Soporte por si faltan ceros a la izquierda en Excel
                if len(c_raw) == 8:
                    mapa[c_raw.zfill(9)] = desc
                if len(c_raw) == 3:
                    mapa[c_raw.zfill(4)] = desc
        return mapa
    except Exception as e:
        st.warning(
            f"⚠️ No se pudo cargar cpc.xlsx para descripciones oficiales: {e}"
        )
        return {}


@st.cache_data
def cargar_datos_cpc():
    # Carga de los 4 diccionarios híbridos generados estrictamente en Excel
    df_comercio = pd.read_excel("diccionario_cpc_comercio_limpio.xlsx")
    df_servicios = pd.read_excel("diccionario_cpc_servicios_limpio.xlsx")
    df_mat_primas = pd.read_excel(
        "diccionario_cpc_materias_primas_limpio.xlsx"
    )
    df_productos = pd.read_excel("diccionario_cpc_productos_limpio.xlsx")

    # Rellenar vacíos por seguridad
    for df in [df_comercio, df_servicios, df_mat_primas, df_productos]:
        df["TEXTO_A_VECTORIZAR"] = df["TEXTO_A_VECTORIZAR"].fillna("")
        df["EJEMPLOS_REALES_LIMPIOS"] = df[
            "EJEMPLOS_REALES_LIMPIOS"
        ].fillna("")

    # Construcción de diccionarios para Match Exacto rápido mapeando (Glosa -> Código)
    dict_exacto_comercio = {}
    for _, row in df_comercio.iterrows():
        glosas = str(row["EJEMPLOS_REALES_LIMPIOS"]).split(" || ")
        for g in glosas:
            if g.strip():
                dict_exacto_comercio[(g.strip(), row["TIPO_COMERCIO"])] = (
                    row["CODIGO_CPC"],
                    row["EJEMPLOS_REALES_LIMPIOS"],
                )

    def crear_dict_exacto(df):
        d = {}
        for _, row in df.iterrows():
            glosas = str(row["EJEMPLOS_REALES_LIMPIOS"]).split(" || ")
            for g in glosas:
                if g.strip():
                    d[g.strip()] = (
                        row["CODIGO_CPC"],
                        row["EJEMPLOS_REALES_LIMPIOS"],
                    )
        return d

    return (
        df_comercio,
        df_servicios,
        df_mat_primas,
        df_productos,
        dict_exacto_comercio,
        crear_dict_exacto(df_servicios),
        crear_dict_exacto(df_mat_primas),
        crear_dict_exacto(df_productos),
    )


@st.cache_data
def calcular_vectores_cpc(_modelo, df):
    return _modelo.encode(
        df["TEXTO_A_VECTORIZAR"].tolist(), show_progress_bar=False
    )


# FUNCIÓN TRADUCTORA DE CÓDIGO A DESCRIPCIÓN OFICIAL
def get_desc_oficial(codigo_hybrid, largo_codigo, mapa_cpc):
    """Extrae la cola CPC del código institucional y devuelve su texto legal."""
    cod_str = str(codigo_hybrid).zfill(largo_codigo)
    # En 8 dígitos extraemos los últimos 4 (CPC); en 13 extraemos los últimos 9
    cpc_puro = cod_str[-4:] if largo_codigo == 8 else cod_str[-9:]

    if cpc_puro in mapa_cpc:
        return mapa_cpc[cpc_puro]

    # Respaldos jerárquicos (por si el catálogo oficial está agrupado a un nivel superior)
    if largo_codigo == 13:
        if cpc_puro[:7] in mapa_cpc:
            return mapa_cpc[cpc_puro[:7]] + " (Nivel agrupado)"
        if cpc_puro[:5] in mapa_cpc:
            return mapa_cpc[cpc_puro[:5]] + " (Nivel agrupado)"
    elif largo_codigo == 8:
        if cpc_puro[:3] in mapa_cpc:
            return mapa_cpc[cpc_puro[:3]] + " (Nivel agrupado)"

    return "Definición técnica según Clasificador Central de Productos (CPC 2.0)"


# 3. INTERFAZ VISUAL PRINCIPAL
st.title("Motor de Codificación Asistida CPC")
st.subheader("Versión 2.1 (Con Catálogo Oficial INEC Integrado)")
st.markdown("Herramienta NLP de la Encuesta Estructural Empresarial - ENESEM")

# Inicialización y barra de carga invisible en ejecuciones posteriores
with st.spinner(
    "Iniciando motor NLP y vinculando Catálogo Oficial en memoria caché..."
):
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
    ) = cargar_datos_cpc()

    vectores_comercio = calcular_vectores_cpc(modelo, df_comercio)
    vectores_servicios = calcular_vectores_cpc(modelo, df_servicios)
    vectores_mat_primas = calcular_vectores_cpc(modelo, df_mat_primas)
    vectores_productos = calcular_vectores_cpc(modelo, df_productos)

st.write("---")

# Formulario lateral de datos generales de la empresa
with st.sidebar:
    st.header("CIIU de la Empresa")
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

# Creación de pestañas para aislar el universo de búsqueda
tab_comercio, tab_servicios, tab_mat_primas, tab_productos = st.tabs([
    "🏪 Comercio",
    "🛠️ Servicios",
    "🪵 Materias Primas",
    "📦 Productos",
])


# Función genérica para renderizar y evaluar las sugerencias de la IA
def mostrar_sugerencias(
    df_modulo,
    vectores_modulo,
    glosa_input,
    indices_top,
    similitudes,
    largo_codigo,
):
    st.markdown("### Top 3 Sugerencias del Modelo")
    for i, idx in enumerate(indices_top):
        codigo = str(df_modulo.iloc[idx]["CODIGO_CPC"]).zfill(largo_codigo)
        ejemplos = df_modulo.iloc[idx]["EJEMPLOS_REALES_LIMPIOS"]
        ciiu_asociado = str(df_modulo.iloc[idx]["CIIU_ASOCIADO"]).zfill(4)
        confianza = similitudes[idx] * 100

        # Obtener descripción oficial desde memoria
        desc_oficial = get_desc_oficial(codigo, largo_codigo, mapa_oficial)

        # Semáforo de asignación automatizada
        if confianza >= 85:
            color, banda = "green", "🟢 AUTOMATIZAR (Banda Verde)"
        elif confianza >= 50:
            color, banda = "orange", "🟡 REVISAR - Asistido (Banda Amarilla)"
        else:
            color, banda = "red", "🔴 MANUAL (Banda Roja)"

        # Validación inteligente cruzada con el CIIU de la empresa
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
            st.markdown(f"**Código Final Sugerido:** `{codigo}`")
            st.markdown(
                f"**CIIU de Origen del Código:** `{ciiu_asociado}`{alerta_ciiu}"
            )
            st.markdown(
                f"**Nivel de Certidumbre:** :{color}[{confianza:.2f}%] ->"
                f" **Acción sugerida:** {banda}"
            )
            st.markdown("---")
            # AQUÍ SE MUESTRA LA DESCRIPCIÓN OFICIAL DEL INEC
            st.markdown(
                f"📖 **Descripción Oficial (CPC 2.0):** \n> *{desc_oficial}*"
            )
            st.markdown(
                f"🏢 **Glosas históricas asociadas (2020-2024):** \n*{ejemplos}*"
            )


# ==============================================================================
# PESTAÑA 1: COMERCIO
# ==============================================================================
with tab_comercio:
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
            glosa_limpia = glosa_comercio.upper().strip()

            if (glosa_limpia, tipo_comercio) in dict_ex_com:
                codigo_ex, ejemplos_ex = dict_ex_com[(
                    glosa_limpia,
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
                indices_filtrados = df_comercio[
                    df_comercio["TIPO_COMERCIO"] == tipo_comercio
                ].index.tolist()
                df_filtrado = df_comercio.loc[indices_filtrados].reset_index(
                    drop=True
                )
                vectores_filtrados = vectores_comercio[indices_filtrados]

                vector_consulta = modelo.encode([glosa_comercio])
                similitudes = cosine_similarity(
                    vector_consulta, vectores_filtrados
                )[0]
                indices_top = np.argsort(similitudes)[::-1][:3]

                mostrar_sugerencias(
                    df_filtrado,
                    vectores_filtrados,
                    glosa_comercio,
                    indices_top,
                    similitudes,
                    largo_codigo=8,
                )
        else:
            st.error(
                "Por favor, ingrese una descripción para realizar la búsqueda."
            )

# ==============================================================================
# PESTAÑA 2: SERVICIOS
# ==============================================================================
with tab_servicios:
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
            glosa_limpia = glosa_servicios.upper().strip()

            if glosa_limpia in dict_ex_ser:
                codigo_ex, ejemplos_ex = dict_ex_ser[glosa_limpia]
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
                vector_consulta = modelo.encode([glosa_servicios])
                similitudes = cosine_similarity(
                    vector_consulta, vectores_servicios
                )[0]
                indices_top = np.argsort(similitudes)[::-1][:3]

                mostrar_sugerencias(
                    df_servicios,
                    vectores_servicios,
                    glosa_servicios,
                    indices_top,
                    similitudes,
                    largo_codigo=8,
                )
        else:
            st.error("Por favor, ingrese una descripción.")

# ==============================================================================
# PESTAÑA 3: MATERIAS PRIMAS
# ==============================================================================
with tab_mat_primas:
    glosa_mp = st.text_input(
        "Materia Prima:",
        placeholder=(
            "Ej: Planchas de acero galvanizado / Harina de trigo industrial"
        ),
        key="mp_txt",
    )
    if st.button(
        "Buscar Código CPC - Materias Primas", type="primary", key="btn_mp"
    ):
        if glosa_mp:
            glosa_limpia = glosa_mp.upper().strip()

            if glosa_limpia in dict_ex_mp:
                codigo_ex, ejemplos_ex = dict_ex_mp[glosa_limpia]
                desc_ex = get_desc_oficial(codigo_ex, 13, mapa_oficial)

                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(codigo_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en la base de Materias"
                    " Primas..."
                )
                vector_consulta = modelo.encode([glosa_mp])
                similitudes = cosine_similarity(
                    vector_consulta, vectores_mat_primas
                )[0]
                indices_top = np.argsort(similitudes)[::-1][:3]

                mostrar_sugerencias(
                    df_mat_primas,
                    vectores_mat_primas,
                    glosa_mp,
                    indices_top,
                    similitudes,
                    largo_codigo=13,
                )
        else:
            st.error("Por favor, ingrese una descripción.")

# ==============================================================================
# PESTAÑA 4: PRODUCTOS
# ==============================================================================
with tab_productos:
    glosa_prod = st.text_input(
        "Producto fabricado:",
        placeholder="Ej: Bloques de hormigón / Aceite refinado de palma",
        key="prod_txt",
    )
    if st.button(
        "Buscar Código CPC - Productos", type="primary", key="btn_prod"
    ):
        if glosa_prod:
            glosa_limpia = glosa_prod.upper().strip()

            if glosa_limpia in dict_ex_prod:
                codigo_ex, ejemplos_ex = dict_ex_prod[glosa_limpia]
                desc_ex = get_desc_oficial(codigo_ex, 13, mapa_oficial)

                st.success("🎯 MATCH EXACTO HISTÓRICO ENCONTRADO")
                st.info(
                    f"**Código Asignado (13 dígitos):** `{str(codigo_ex).zfill(13)}`"
                    f" \n\n 📖 **Descripción Oficial (CPC 2.0):** {desc_ex} \n\n"
                    f" 🏢 **Historial de Glosas:** {ejemplos_ex}"
                )
            else:
                st.warning(
                    "Buscando aproximaciones semánticas en la base de Productos"
                    " Fabricados..."
                )
                vector_consulta = modelo.encode([glosa_prod])
                similitudes = cosine_similarity(
                    vector_consulta, vectores_productos
                )[0]
                indices_top = np.argsort(similitudes)[::-1][:3]

                mostrar_sugerencias(
                    df_productos,
                    vectores_productos,
                    glosa_prod,
                    indices_top,
                    similitudes,
                    largo_codigo=13,
                )
        else:
            st.error("Por favor, ingrese una descripción.")

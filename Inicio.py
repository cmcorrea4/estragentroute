import streamlit as st
import requests
import json
import base64
from datetime import datetime, timedelta
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico de gestión energética--ESTRA",
    page_icon="🏭",
    layout="wide"
)

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
ENDPOINTS = {
    "summary":    "https://energy-api-628964750053.us-east1.run.app/test-summary",
    "moldes":     "https://energy-api-628964750053.us-east1.run.app/test-mold",
    "referencias":"https://energy-api-628964750053.us-east1.run.app/test-reference",
    "linea_base": "https://energy-api-628964750053.us-east1.run.app/test-baseline",
}

ENDPOINT_LABELS = {
    "summary":    "📊 Resumen General",
    "moldes":     "🔩 Moldes",
    "referencias":"🏷️ Referencias",
    "linea_base": "📐 Línea Base",
}

MODEL = "gpt-4-turbo"

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def get_week_start(date):
    return date - timedelta(days=date.weekday())

def get_week_end(date):
    return date + timedelta(days=6 - date.weekday())

def get_auth_header(username, password):
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "User-Agent": "StreamlitApp/1.0",
        "Accept": "application/json"
    }

@st.cache_data(ttl=300)
def consultar_endpoint(endpoint_key, username, password, date_start=None, date_end=None):
    try:
        params = {}
        if date_start:
            params["dateStart"] = date_start
        if date_end:
            params["dateEnd"] = date_end

        response = requests.get(
            ENDPOINTS[endpoint_key],
            headers=get_auth_header(username, password),
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error HTTP {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return None, "Timeout: el servidor tardó demasiado en responder"
    except requests.exceptions.ConnectionError:
        return None, "Error de conexión al servidor"
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"


def cargar_json_por_intencion(intencion):
    cache_key = f"json_{intencion}"
    if cache_key in st.session_state:
        return st.session_state[cache_key], None

    username   = st.session_state.get("api_username", "")
    password   = st.session_state.get("api_password", "")
    date_start = st.session_state.get("date_start")
    date_end   = st.session_state.get("date_end")

    datos_json, error = consultar_endpoint(
        intencion, username, password,
        date_start.strftime("%Y-%m-%d") if date_start else None,
        date_end.strftime("%Y-%m-%d")   if date_end   else None
    )

    if error:
        return None, error

    st.session_state[cache_key] = datos_json
    return datos_json, None


# ─────────────────────────────────────────────
# ROUTER + ANÁLISIS CON OPENAI DIRECTO
# ─────────────────────────────────────────────

ROUTER_PROMPT = """
Eres un clasificador de intención para una app de análisis energético industrial.
Lee la pregunta del usuario y responde EXCLUSIVAMENTE con una de estas cuatro palabras:

- summary      → preguntas generales, consumo global, producción total, resumen, periodos
- moldes       → preguntas sobre moldes, SECn por molde, productividad de moldes, tiempos de paro por molde
- referencias  → preguntas sobre referencias, productos, SKU, códigos de producto
- linea_base   → preguntas sobre línea base, baseline, metas energéticas, benchmarks

Responde SOLO con la palabra clave, sin explicación, sin puntos, sin mayúsculas.
"""

ANALISIS_PROMPT = """
Eres un analista experto en gestión energética industrial para la empresa ESTRA.
Se te proporcionará un JSON con datos energéticos reales del sistema.
Analiza el JSON y responde la pregunta del usuario de forma clara y técnica.

Reglas:
- Responde SIEMPRE en español
- Usa lenguaje técnico adecuado para ingenieros
- Incluye unidades en los números cuando sea posible
- Sé conciso pero completo
- Si necesitas ordenar o comparar valores, hazlo directamente desde el JSON
- No inventes datos que no estén en el JSON
"""

def clasificar_intencion(pregunta, client):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user",   "content": pregunta}
            ]
        )
        intencion = resp.choices[0].message.content.strip().lower()
        return intencion if intencion in ENDPOINTS else "summary"
    except Exception:
        return "summary"


def analizar_con_gpt(pregunta, datos_json, client):
    json_str = json.dumps(datos_json, ensure_ascii=False, indent=2)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": ANALISIS_PROMPT},
                {"role": "user",   "content": f"JSON de datos:\n```json\n{json_str}\n```\n\nPregunta: {pregunta}"}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error al analizar: {str(e)}"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.title("🏭 Diagnóstico de gestión energética--ESTRA")
st.markdown("**Obtén datos del sistema energético y analízalos con IA**")

with st.sidebar:
    st.header("⚙️ Panel de Control")

    st.subheader("🔌 Credenciales del Endpoint")

    if "json_summary" not in st.session_state:
        api_username = st.text_input("👤 Usuario:", placeholder="Ingresa tu usuario")
        api_password = st.text_input("🔒 Contraseña:", type="password", placeholder="Ingresa tu contraseña")
        endpoint_configured = bool(api_username and api_password)
        if endpoint_configured:
            st.success("✅ Credenciales configuradas")
        else:
            st.warning("⚠️ Ingresa usuario y contraseña")
    else:
        api_username = st.session_state.get("api_username", "")
        api_password = st.session_state.get("api_password", "")
        endpoint_configured = True
        st.success("✅ Sesión activa")

    st.markdown("---")

    st.subheader("📅 Filtro de Fechas")
    filter_type = st.radio("Tipo de filtro:", ["Por semana", "Por rango de fechas"])

    if filter_type == "Por semana":
        today = datetime.now().date()
        selected_week = st.date_input(
            "Selecciona una fecha (se usará su semana completa):",
            value=st.session_state.get("selected_week", today)
        )
        date_start = get_week_start(selected_week)
        date_end   = get_week_end(selected_week)
        st.info(f"📅 Semana del **{date_start.strftime('%d/%m/%Y')}** al **{date_end.strftime('%d/%m/%Y')}**")
        dates_valid = True
    else:
        default_start = st.session_state.get("date_start", datetime(2024, 1, 1).date())
        default_end   = st.session_state.get("date_end",   datetime.now().date())
        date_start = st.date_input("Fecha de inicio:", value=default_start)
        date_end   = st.date_input("Fecha de fin:",    value=default_end)
        if date_start > date_end:
            st.error("⚠️ La fecha de inicio debe ser anterior a la fecha de fin")
            dates_valid = False
        else:
            dates_valid = True
            st.info(f"📊 Rango: {(date_end - date_start).days + 1} días")

    st.markdown("---")

    st.subheader("🤖 OpenAI API Key")

    if "openai_api_key" not in st.session_state:
        openai_api_key = st.text_input("🔑 API Key:", type="password", placeholder="sk-...")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.success("✅ API Key configurada")
        else:
            st.warning("⚠️ Ingresa tu API Key de OpenAI")
    else:
        st.success("✅ API Key configurada")
        if st.button("🔄 Cambiar API Key"):
            del st.session_state.openai_api_key
            st.rerun()

    st.markdown("---")

    if st.button("🔌 Obtener Datos del Sistema", use_container_width=True,
                 disabled=not (endpoint_configured and dates_valid)):
        with st.spinner("Consultando endpoint de energía..."):
            datos_json, error = consultar_endpoint(
                "summary", api_username, api_password,
                date_start.strftime("%Y-%m-%d"),
                date_end.strftime("%Y-%m-%d")
            )
            if datos_json is not None:
                st.session_state.json_summary   = datos_json
                st.session_state.api_username   = api_username
                st.session_state.api_password   = api_password
                st.session_state.date_start     = date_start
                st.session_state.date_end       = date_end
                st.session_state.filter_type    = filter_type
                if filter_type == "Por semana":
                    st.session_state.selected_week = selected_week
                for k in ["json_moldes", "json_referencias", "json_linea_base"]:
                    if k in st.session_state:
                        del st.session_state[k]
                consultar_endpoint.clear()
                st.success("✅ Datos cargados correctamente")
                st.rerun()
            else:
                st.error(f"❌ {error}")

    if "json_summary" in st.session_state:
        st.success("🟢 Datos listos")
        if "date_start" in st.session_state:
            st.info(f"📅 {st.session_state.date_start.strftime('%d/%m/%Y')} → {st.session_state.date_end.strftime('%d/%m/%Y')}")
        st.markdown("**Endpoints en caché:**")
        for k, label in ENDPOINT_LABELS.items():
            if f"json_{k}" in st.session_state:
                st.success(f"  {label} ✅")
    else:
        st.warning("🔴 Sin datos del sistema")


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────

if "json_summary" not in st.session_state:
    st.info("👆 Configura las credenciales, selecciona el rango de fechas y haz clic en 'Obtener Datos del Sistema'")
    st.markdown("---")
    st.subheader("ℹ️ Cómo funciona")
    st.markdown("""
    | Pregunta sobre... | Endpoint consultado |
    |---|---|
    | Moldes, SECn por molde, productividad | 🔩 `/test-mold` |
    | Referencias, productos, SKU | 🏷️ `/test-reference` |
    | Línea base, benchmarks, metas | 📐 `/test-baseline` |
    | General, resumen, consumo total | 📊 `/test-summary` |

    El **router inteligente** usa GPT para detectar la intención, consulta el endpoint correcto
    solo cuando se necesita, y GPT analiza el JSON directamente sin conversión intermedia.
    Sin LangChain, sin DataFrames, sin errores de parseo.
    """)

else:
    datos_json_summary = st.session_state.json_summary

    st.success("✅ Datos del sistema energético cargados")

    if "date_start" in st.session_state:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.info(f"📅 Desde: **{st.session_state.date_start.strftime('%d/%m/%Y')}**")
        with col2:
            st.info(f"📅 Hasta: **{st.session_state.date_end.strftime('%d/%m/%Y')}**")
        with col3:
            dias = (st.session_state.date_end - st.session_state.date_start).days + 1
            st.metric("📊 Días", dias)

    st.header("📊 Datos del Resumen General")
    with st.expander("🗂️ Ver JSON del endpoint `/test-summary`", expanded=False):
        st.json(datos_json_summary)

    # ─────────────────────────────────────────────
    # ANÁLISIS IA
    # ─────────────────────────────────────────────
    st.header("🤖 Análisis IA con Router Inteligente")

    st.markdown("""
    | Tu pregunta menciona... | Endpoint que se usa |
    |---|---|
    | moldes, SECn, productividad de molde | 🔩 `/test-mold` |
    | referencia, producto, SKU | 🏷️ `/test-reference` |
    | línea base, baseline, benchmark | 📐 `/test-baseline` |
    | general, resumen, consumo total | 📊 `/test-summary` |
    """)

    if "openai_api_key" not in st.session_state:
        st.warning("⚠️ Configura tu API Key de OpenAI en la barra lateral.")
    else:
        client = OpenAI(api_key=st.session_state.openai_api_key)

        st.subheader("💡 Ejemplos de preguntas:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔩 Moldes:**
            - ¿Qué moldes tienen la mayor productividad efectiva?
            - ¿Cuál molde tiene mayor SECn?
            - ¿En qué fechas se trabajó el molde 15252?

            **🏷️ Referencias:**
            - ¿Qué referencias tienen mayor consumo energético?
            - ¿Cuáles son los productos con mayor tiempo de paro?
            """)
        with col2:
            st.markdown("""
            **📐 Línea Base:**
            - ¿Cuál es la línea base de consumo energético?
            - ¿Qué referencias están por encima del benchmark?

            **📊 General:**
            - ¿Qué información contiene el dataset?
            - ¿Cuál es el consumo total del periodo?
            """)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.subheader("❓ Consulta los datos con IA")
        user_question = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Qué moldes tienen mayor SECn?",
            key="user_input"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Analizar", type="primary")
        with col2:
            if st.button("🗑️ Limpiar historial"):
                st.session_state.chat_history = []
                st.rerun()

        if ask_button and user_question:

            # 1. Router: clasificar intención
            with st.spinner("🔍 Detectando tipo de consulta..."):
                intencion = clasificar_intencion(user_question, client)

            endpoint_label = ENDPOINT_LABELS[intencion]
            st.info(f"🎯 Router → **{endpoint_label}** (`{ENDPOINTS[intencion]}`)")

            # 2. Cargar JSON bajo demanda
            if intencion == "summary":
                datos_para_analisis = datos_json_summary
                carga_error = None
            else:
                with st.spinner(f"📡 Consultando {endpoint_label}..."):
                    datos_para_analisis, carga_error = cargar_json_por_intencion(intencion)

            if carga_error:
                st.error(f"❌ Error al cargar {endpoint_label}: {carga_error}")

            elif datos_para_analisis is not None:

                # 3. Mostrar JSON que va a analizar GPT
                with st.expander(f"🗂️ JSON recibido de {endpoint_label} (que analiza GPT)", expanded=False):
                    st.json(datos_para_analisis)

                # 4. Analizar directamente sobre el JSON
                with st.spinner("🤖 GPT analizando el JSON..."):
                    respuesta = analizar_con_gpt(user_question, datos_para_analisis, client)

                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer":   respuesta,
                    "endpoint": endpoint_label,
                    "json":     datos_para_analisis
                })
                st.rerun()

        # Historial
        if st.session_state.chat_history:
            st.subheader("💬 Análisis Realizados")

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                label = f"❓ {chat['question'][:60]}..." if len(chat['question']) > 60 else f"❓ {chat['question']}"
                with st.expander(label, expanded=(i == 0)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write("**Pregunta:**")
                        st.write(chat["question"])
                    with col2:
                        st.caption(f"Fuente: {chat['endpoint']}")

                    st.write("**Respuesta:**")
                    st.write(chat["answer"])

                    with st.expander("🗂️ Ver JSON analizado", expanded=False):
                        st.json(chat["json"])
                    st.divider()

    st.markdown("---")
    if st.button("🔄 Actualizar Todos los Datos", use_container_width=True):
        consultar_endpoint.clear()
        for key in ["json_summary", "json_moldes", "json_referencias", "json_linea_base",
                    "chat_history", "api_username", "api_password"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    🏭 ESTRA - Sistema Integrado de Análisis Energético con IA | Powered by SUME & SOSPOL
    </div>
    """,
    unsafe_allow_html=True
)

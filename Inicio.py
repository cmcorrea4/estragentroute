import streamlit as st
import requests
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
# PALETA DE COLORES
# ─────────────────────────────────────────────
C_NARANJA  = "#f97316"
C_AZUL     = "#3b82f6"
C_VERDE    = "#22c55e"
C_FONDO    = "#0f172a"
C_PANEL    = "#1e293b"
C_BORDE    = "#334155"
C_TEXTO    = "#f1f5f9"

def aplicar_estilo(fig, ax):
    """Aplica la paleta oscura ESTRA a cualquier figura matplotlib."""
    fig.patch.set_facecolor(C_FONDO)
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXTO, labelsize=10)
    ax.xaxis.label.set_color(C_TEXTO)
    ax.yaxis.label.set_color(C_TEXTO)
    ax.title.set_color(C_NARANJA)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDE)
    ax.grid(True, color=C_BORDE, linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────
# UTILIDADES GENERALES
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
# ROUTER + ANÁLISIS CON OPENAI
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
# EXTRACCIÓN DE DATOS PARA GRÁFICAS
# ─────────────────────────────────────────────

def extraer_pcld(moldes_json):
    pcld   = moldes_json.get("PCLD", {})
    points = pcld.get("dataPoints", [])
    reg    = pcld.get("regression", {})
    if not points:
        return None, None, None, None
    x = [p.get("Suma de Producción Conforme [kg]", 0) for p in points]
    y = [p.get("Suma de Consumo [kWh]", 0)            for p in points]
    return x, y, reg.get("a"), reg.get("b")


def extraer_pcdd(moldes_json):
    pcdd   = moldes_json.get("PCDD", {})
    points = pcdd.get("dataPoints", [])
    curva  = pcdd.get("curva", [])
    if not points:
        return None, None, None, None
    x_pts   = [p.get("Promedio de Productividad Efectiva [kg/h]", 0) for p in points]
    y_pts   = [p.get("SECn (costo energetico) (Molde)", 0)           for p in points]
    x_curva = [c.get("flujo", 0)        for c in curva]
    y_curva = [c.get("SECnesperado", 0) for c in curva]
    return x_pts, y_pts, x_curva, y_curva


def extraer_curva(moldes_json):
    curva = moldes_json.get("PCDD", {}).get("curva", [])
    if not curva:
        return None, None
    return [c.get("flujo", 0) for c in curva], [c.get("SECnesperado", 0) for c in curva]


# ─────────────────────────────────────────────
# GRÁFICOS MATPLOTLIB
# ─────────────────────────────────────────────

def grafico_pcld(x, y, a, b):
    fig, ax = plt.subplots(figsize=(10, 5))
    aplicar_estilo(fig, ax)

    ax.scatter(x, y, color=C_NARANJA, s=80, zorder=5,
               edgecolors="white", linewidths=0.8, label="Datos reales")

    if a is not None and b is not None:
        x_arr = np.array(x)
        x_reg = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_reg = a * x_reg + b
        ax.plot(x_reg, y_reg, color=C_AZUL, linewidth=2,
                linestyle="--", label=f"Regresión: y = {a:.4f}·x + {b:.2f}")

    ax.set_xlabel("Producción Conforme [kg]", fontsize=11)
    ax.set_ylabel("Consumo [kWh]", fontsize=11)
    ax.set_title("PCLD — Producción Conforme vs Consumo Energético", fontsize=13, pad=12)
    ax.legend(facecolor=C_PANEL, edgecolor=C_BORDE, labelcolor=C_TEXTO, fontsize=10)
    fig.tight_layout()
    return fig


def grafico_pcdd(x_pts, y_pts, x_curva, y_curva):
    fig, ax = plt.subplots(figsize=(10, 5))
    aplicar_estilo(fig, ax)

    if x_curva and y_curva:
        ax.plot(x_curva, y_curva, color=C_VERDE, linewidth=2.5,
                label="Curva SECn esperado", zorder=3)

    ax.scatter(x_pts, y_pts, color=C_NARANJA, s=90, zorder=5,
               edgecolors="white", linewidths=0.8, marker="D",
               label="Moldes (datos reales)")

    ax.set_xlabel("Productividad Efectiva [kg/h]", fontsize=11)
    ax.set_ylabel("SECn — Costo Energético [kWh/kg]", fontsize=11)
    ax.set_title("PCDD — SECn vs Productividad Efectiva por Molde", fontsize=13, pad=12)
    ax.legend(facecolor=C_PANEL, edgecolor=C_BORDE, labelcolor=C_TEXTO, fontsize=10)
    fig.tight_layout()
    return fig


def grafico_curva(x, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    aplicar_estilo(fig, ax)

    ax.fill_between(x, y, alpha=0.12, color=C_VERDE)
    ax.plot(x, y, color=C_VERDE, linewidth=2.5, label="SECn esperado")
    ax.scatter(x, y, color=C_VERDE, s=45, zorder=5,
               edgecolors="white", linewidths=0.7)

    ax.set_xlabel("Flujo [kg/h]", fontsize=11)
    ax.set_ylabel("SECn Esperado [kWh/kg]", fontsize=11)
    ax.set_title("Curva de Referencia — SECn Esperado vs Flujo", fontsize=13, pad=12)
    ax.legend(facecolor=C_PANEL, edgecolor=C_BORDE, labelcolor=C_TEXTO, fontsize=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.title("🏭 Diagnóstico de gestión energética — ESTRA")
st.markdown("**Obtén datos del sistema energético y analízalos con IA**")

with st.sidebar:
    st.header("⚙️ Panel de Control")

    # ── Credenciales ──────────────────────────────────────────────────────
    st.subheader("🔌 Credenciales del Endpoint")

    if "json_summary" not in st.session_state:
        api_username = st.text_input("👤 Usuario:", placeholder="Ingresa tu usuario")
        api_password = st.text_input("🔒 Contraseña:", type="password",
                                     placeholder="Ingresa tu contraseña")
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

    # ── Fechas ────────────────────────────────────────────────────────────
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

    # ── API Key OpenAI ────────────────────────────────────────────────────
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

    # ── Botón obtener datos ───────────────────────────────────────────────
    if st.button("🔌 Obtener Datos del Sistema", use_container_width=True,
                 disabled=not (endpoint_configured and dates_valid)):
        with st.spinner("Consultando endpoint de energía..."):
            datos_json, error = consultar_endpoint(
                "summary", api_username, api_password,
                date_start.strftime("%Y-%m-%d"),
                date_end.strftime("%Y-%m-%d")
            )
            if datos_json is not None:
                st.session_state.json_summary  = datos_json
                st.session_state.api_username  = api_username
                st.session_state.api_password  = api_password
                st.session_state.date_start    = date_start
                st.session_state.date_end      = date_end
                st.session_state.filter_type   = filter_type
                if filter_type == "Por semana":
                    st.session_state.selected_week = selected_week
                for k in ["json_moldes", "json_referencias", "json_linea_base",
                          "json_moldes_viz"]:
                    if k in st.session_state:
                        del st.session_state[k]
                consultar_endpoint.clear()
                st.success("✅ Datos cargados correctamente")
                st.rerun()
            else:
                st.error(f"❌ {error}")

    # Estado de caché
    if "json_summary" in st.session_state:
        st.success("🟢 Datos listos")
        if "date_start" in st.session_state:
            st.info(
                f"📅 {st.session_state.date_start.strftime('%d/%m/%Y')} → "
                f"{st.session_state.date_end.strftime('%d/%m/%Y')}"
            )
        st.markdown("**Endpoints en caché:**")
        for k, label in ENDPOINT_LABELS.items():
            if f"json_{k}" in st.session_state:
                st.success(f"  {label} ✅")
    else:
        st.warning("🔴 Sin datos del sistema")

    st.markdown("---")
    if st.button("🔄 Actualizar Todos los Datos", use_container_width=True):
        consultar_endpoint.clear()
        for key in ["json_summary", "json_moldes", "json_referencias", "json_linea_base",
                    "json_moldes_viz", "chat_history", "api_username", "api_password"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ─────────────────────────────────────────────

if "json_summary" not in st.session_state:
    st.info("👆 Configura las credenciales, selecciona el rango de fechas y haz clic en "
            "'Obtener Datos del Sistema'")
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
    tab_ia, tab_viz = st.tabs(["🤖 Análisis IA", "📊 Visualizaciones"])

    # ════════════════════════════════════════════════════════════════════════
    # PESTAÑA 1 — ANÁLISIS IA
    # ════════════════════════════════════════════════════════════════════════
    with tab_ia:
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
                with st.spinner("🔍 Detectando tipo de consulta..."):
                    intencion = clasificar_intencion(user_question, client)

                endpoint_label = ENDPOINT_LABELS[intencion]
                st.info(f"🎯 Router → **{endpoint_label}** (`{ENDPOINTS[intencion]}`)")

                if intencion == "summary":
                    datos_para_analisis = datos_json_summary
                    carga_error = None
                else:
                    with st.spinner(f"📡 Consultando {endpoint_label}..."):
                        datos_para_analisis, carga_error = cargar_json_por_intencion(intencion)

                if carga_error:
                    st.error(f"❌ Error al cargar {endpoint_label}: {carga_error}")
                elif datos_para_analisis is not None:
                    with st.expander(
                        f"🗂️ JSON recibido de {endpoint_label} (que analiza GPT)",
                        expanded=False
                    ):
                        st.json(datos_para_analisis)

                    with st.spinner("🤖 GPT analizando el JSON..."):
                        respuesta = analizar_con_gpt(user_question, datos_para_analisis, client)

                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer":   respuesta,
                        "endpoint": endpoint_label,
                        "json":     datos_para_analisis
                    })
                    st.rerun()

            if st.session_state.chat_history:
                st.subheader("💬 Análisis Realizados")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    label = (
                        f"❓ {chat['question'][:60]}..."
                        if len(chat['question']) > 60
                        else f"❓ {chat['question']}"
                    )
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

    # ════════════════════════════════════════════════════════════════════════
    # PESTAÑA 2 — VISUALIZACIONES
    # ════════════════════════════════════════════════════════════════════════
    with tab_viz:
        st.header("📊 Visualización de Gráficas Energéticas")
        st.markdown(
            "Selecciona el tipo de gráfico para consultar el endpoint `/test-mold` "
            "y visualizar los datos de forma interactiva."
        )

        username   = st.session_state.get("api_username", "")
        password   = st.session_state.get("api_password", "")
        date_start = st.session_state.get("date_start")
        date_end   = st.session_state.get("date_end")

        # ── Selector ─────────────────────────────────────────────────────
        tipo_grafico = st.radio(
            "🎛️ Selecciona el tipo de visualización:",
            ["PCLD — Producción vs Consumo",
             "PCDD — SECn vs Productividad",
             "Curva — SECn Esperado vs Flujo"],
            horizontal=True,
        )
        clave_map = {
            "PCLD — Producción vs Consumo":   "PCLD",
            "PCDD — SECn vs Productividad":   "PCDD",
            "Curva — SECn Esperado vs Flujo": "Curva",
        }
        clave = clave_map[tipo_grafico]

        # ── Botón cargar datos ────────────────────────────────────────────
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            refrescar = st.button("🔄 Cargar / Actualizar datos", type="primary",
                                  key="btn_viz_refresh")

        if refrescar or "json_moldes_viz" not in st.session_state:
            with st.spinner("📡 Consultando `/test-mold`..."):
                datos, error = consultar_endpoint(
                    "moldes", username, password,
                    date_start.strftime("%Y-%m-%d") if date_start else None,
                    date_end.strftime("%Y-%m-%d")   if date_end   else None,
                )
            if error:
                st.error(f"❌ Error al consultar el endpoint: {error}")
                st.stop()
            st.session_state["json_moldes_viz"] = datos

        moldes_json = st.session_state.get("json_moldes_viz", {})

        with col_status:
            if moldes_json:
                st.success("✅ Datos de moldes listos")
            else:
                st.error("❌ No se obtuvieron datos del endpoint")
                st.stop()

        with st.expander("🗂️ Ver JSON crudo de `/test-mold`", expanded=False):
            st.json(moldes_json)

        st.divider()

        # ── PCLD ─────────────────────────────────────────────────────────
        if clave == "PCLD":
            x, y, a, b = extraer_pcld(moldes_json)
            if x is None:
                st.error("❌ No se encontró la sección 'PCLD' en la respuesta del endpoint.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📦 Total producción", f"{sum(x):,.1f} kg")
                c2.metric("⚡ Total consumo",    f"{sum(y):,.2f} kWh")
                c3.metric("📊 SECn medio",
                          f"{sum(y)/sum(x):.4f} kWh/kg" if sum(x) else "—")
                if a is not None:
                    c4.metric("📈 Pendiente reg.", f"{a:.4f} kWh/kg")

                fig = grafico_pcld(x, y, a, b)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("📋 Ver datos tabulados (PCLD)", expanded=False):
                    df = pd.DataFrame({
                        "Producción Conforme [kg]": x,
                        "Consumo [kWh]": y,
                    })
                    df["SECn puntual [kWh/kg]"] = [
                        yi / xi if xi else None for xi, yi in zip(x, y)
                    ]
                    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

        # ── PCDD ─────────────────────────────────────────────────────────
        elif clave == "PCDD":
            x_pts, y_pts, x_curva, y_curva = extraer_pcdd(moldes_json)
            if x_pts is None:
                st.error("❌ No se encontró la sección 'PCDD' en la respuesta del endpoint.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔩 Nº moldes",         len(x_pts))
                c2.metric("⚡ SECn mínimo",        f"{min(y_pts):.4f}")
                c3.metric("⚡ SECn máximo",        f"{max(y_pts):.4f}")
                c4.metric("🏃 Prod. máx. [kg/h]", f"{max(x_pts):.2f}")

                fig = grafico_pcdd(x_pts, y_pts, x_curva, y_curva)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("📋 Ver datos tabulados (PCDD)", expanded=False):
                    df = pd.DataFrame({
                        "Productividad Efectiva [kg/h]": x_pts,
                        "SECn real [kWh/kg]":            y_pts,
                    })
                    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

        # ── Curva ─────────────────────────────────────────────────────────
        elif clave == "Curva":
            x, y = extraer_curva(moldes_json)
            if x is None:
                st.error("❌ No se encontró la sección 'curva' en la respuesta del endpoint.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("🔢 Puntos de curva",    len(x))
                c2.metric("📉 SECn mín. esperado", f"{min(y):.4f}")
                c3.metric("📈 Flujo máximo",        f"{max(x):.2f} kg/h")

                fig = grafico_curva(x, y)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("📋 Ver datos tabulados (Curva)", expanded=False):
                    df = pd.DataFrame({
                        "Flujo [kg/h]":           x,
                        "SECn Esperado [kWh/kg]": y,
                    })
                    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    🏭 ESTRA — Sistema Integrado de Análisis Energético con IA | Powered by SUME & SOSPOL
    </div>
    """,
    unsafe_allow_html=True
)


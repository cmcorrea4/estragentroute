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
    page_title="Diagnóstico ESTRA - Agente IA",
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

MODEL = "gpt-4o-mini-2024-07-18" #gpt-4-turbo

# ─────────────────────────────────────────────
# PALETA DE COLORES
# ─────────────────────────────────────────────
C_NARANJA = "#f97316"
C_AZUL    = "#3b82f6"
C_VERDE   = "#22c55e"
C_FONDO   = "#0f172a"
C_PANEL   = "#1e293b"
C_BORDE   = "#334155"
C_TEXTO   = "#f1f5f9"

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
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"


# ─────────────────────────────────────────────
# HERRAMIENTAS OPENAI (FUNCTION CALLING)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
Eres un analista experto en gestión energética industrial para la empresa ESTRA.
Tu objetivo es responder las preguntas del usuario sobre los consumos y parámetros energéticos.
Tienes a tu disposición 4 herramientas (endpoints) que devuelven un JSON con datos reales:
1. obtener_resumen_general: para consumo total, resumen general, indicadores de planta.
2. obtener_datos_moldes: para indicadores de moldes (SECn, productividad, paradas).
3. obtener_datos_referencias: para indicadores productivos por referencia o SKU.
4. obtener_linea_base: para cálculos de línea base, benchmarks y metas.

Reglas:
- Llama a la herramienta o herramientas correspondientes para analizar y responder cada solicitud.
- Puedes usar múltiples llamadas concurrentes si la pregunta abarca diferentes tópicos.
- Una vez recibidos los datos de las herramientas, analízalos para formular la respuesta en español.
- Ocupa un lenguaje técnico pero claro para los ingenieros de planta.
- Incluye unidades de medida pertinentes (como kWh, kg, %, etc.).
- No inventes datos bajo ninguna circunstancia. Si no vienen en el JSON, infórmalo.
"""

TIPO_FECHA = {
    "type": "string",
    "description": "Fecha en formato 'YYYY-MM-DD'"
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "obtener_resumen_general",
            "description": "Devuelve un JSON detallado con el resumen general a nivel de cada orden de producción en el rango de fechas. Úsala para consultar datos como: ID de orden, máquina (cceId), tiempos totales y productivos (pdnTotalTime, pdnEffectiveTime), tiempos y porcentajes de parada (totalStopTime, stopTimePercentage), consumo bruto, producción total/conforme/rechazos (totalProduction, noComplaintProduction), productividades (realProductivity, effectiveProductivity), demanda estable, indicadores SEC desglosados (secN, secG, secS, secB) y todas las brechas específicas de cada orden procesada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dateStart": TIPO_FECHA,
                    "dateEnd": TIPO_FECHA
                },
                "required": ["dateStart", "dateEnd"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_datos_moldes",
            "description": "Devuelve un JSON con el desempeño energético y productivo desglosado por moldes. Úsala para analizar un molde específico o comparar métricas entre moldes: Consumo en kWh, Producción Conforme en kg, Productividad Efectiva en kg/h, Demanda estable en kW, indicadores SEC (SECn, SECg, SECs, SECb), análisis de Brechas (producción, calidad, proceso + tecnología), nivel de desempeño, nivel de eficiencia y área asociada al molde.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dateStart": TIPO_FECHA,
                    "dateEnd": TIPO_FECHA
                },
                "required": ["dateStart", "dateEnd"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_datos_referencias",
            "description": "Devuelve un JSON con el desempeño energético y productivo desglosado por referencias/productos (SKUs). Úsala para conocer métricas de una referencia específica o compararlas: Consumo (kWh), Producción conforme (kg), Productividad efectiva (kg/h), Demanda estable (kW), todos los indicadores SEC (SECn, SECg, SECs, SECb), análisis de Brechas (producción, calidad, proceso+tecnología), Desempeño general y nivel de Eficiencia por área.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dateStart": TIPO_FECHA,
                    "dateEnd": TIPO_FECHA
                },
                "required": ["dateStart", "dateEnd"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "obtener_linea_base",
            "description": "Devuelve datos del modelo de línea base (pendiente e intercepto), consumo actual, y un análisis CUSUM (Suma Acumulada) semanal de energía (kWh) y costos (dinero). Úsala cuando el usuario pregunte por ahorros, sobrecostos de energía, diferencias entre energía esperada vs consumida, análisis de tendencia (CUSUM) o parámetros del modelo (pendiente/intercepto).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dateStart": TIPO_FECHA,
                    "dateEnd": TIPO_FECHA
                },
                "required": ["dateStart", "dateEnd"],
                "additionalProperties": False
            }
        }
    }
]

def map_tool_to_endpoint(tool_name):
    mapping = {
        "obtener_resumen_general":   "summary",
        "obtener_datos_moldes":      "moldes",
        "obtener_datos_referencias": "referencias",
        "obtener_linea_base":        "linea_base",
    }
    return mapping.get(tool_name)

def ejecutar_herramienta(tool_name, arguments_dict, username, password):
    date_start   = arguments_dict.get("dateStart")
    date_end     = arguments_dict.get("dateEnd")
    endpoint_key = map_tool_to_endpoint(tool_name)
    if not endpoint_key:
        return {"error": f"Herramienta desconocida: {tool_name}"}
    datos, error = consultar_endpoint(endpoint_key, username, password, date_start, date_end)
    if error:
        return {"error": error}
    return datos

def consultar_agente(pregunta, client, username, password, date_start_str, date_end_str):
    contexto_fechas = (
        f"Fechas por defecto en interfaz: desde {date_start_str} hasta {date_end_str}. "
        "Úsalas si el usuario no especifica fechas."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + contexto_fechas},
        {"role": "user",   "content": pregunta}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content, []

    messages.append(response_message)
    herramientas_utilizadas = []

    for tool_call in tool_calls:
        func_name = tool_call.function.name
        try:
            func_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            func_args = {}

        resultado = ejecutar_herramienta(func_name, func_args, username, password)

        herramientas_utilizadas.append({
            "nombre":     func_name,
            "argumentos": func_args,
            "json":       resultado
        })

        messages.append({
            "tool_call_id": tool_call.id,
            "role":         "tool",
            "name":         func_name,
            "content":      json.dumps(resultado, ensure_ascii=False)
        })

    second_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    return second_response.choices[0].message.content, herramientas_utilizadas


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
# ESTILO MATPLOTLIB
# ─────────────────────────────────────────────

def aplicar_estilo(fig, ax):
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
st.title("🏭 Asistente de gestión energética — ESTRA")
st.markdown("**Powered by SUME & SOSPOL**")

with st.sidebar:
    st.header("⚙️ Configuración")

    # ── OpenAI API Key ────────────────────────────────────────────────────
    if "openai_api_key" not in st.session_state:
        api_key = st.text_input("🔑 OpenAI API Key:", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
            st.success("API Key guardada en sesión.")
    else:
        st.success("✅ OpenAI API Key configurada.")
        if st.button("🔄 Cambiar API Key"):
            del st.session_state.openai_api_key
            st.rerun()

    st.markdown("---")

    # ── Credenciales ESTRA ────────────────────────────────────────────────
    st.subheader("🔌 Credenciales de API (ESTRA)")
    api_username = st.session_state.get("api_username", "")
    api_password = st.session_state.get("api_password", "")

    if not (api_username and api_password):
        api_username = st.text_input("👤 Usuario", value=api_username)
        api_password = st.text_input("🔒 Contraseña", type="password", value=api_password)
        if st.button("Guardar Credenciales") and api_username and api_password:
            st.session_state.api_username = api_username
            st.session_state.api_password = api_password
            st.success("✅ Credenciales guardadas.")
            st.rerun()
    else:
        st.success("✅ Credenciales configuradas.")
        if st.button("🗑️ Cambiar Credenciales"):
            st.session_state.api_username = ""
            st.session_state.api_password = ""
            st.rerun()

    st.markdown("---")

    # ── Rango de fechas ───────────────────────────────────────────────────
    st.subheader("📅 Fechas de contexto")
    date_start = st.date_input("Inicio", value=datetime.today() - timedelta(days=7))
    date_end   = st.date_input("Fin",    value=datetime.today())

    st.markdown("---")

    # ── Limpiar caché de moldes si cambian fechas ─────────────────────────
    if st.button("🔄 Limpiar caché de datos", use_container_width=True):
        consultar_endpoint.clear()
        for k in ["json_moldes_viz"]:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Caché limpiado.")


# ─────────────────────────────────────────────
# GUARDIA: credenciales requeridas
# ─────────────────────────────────────────────
if not st.session_state.get("openai_api_key"):
    st.warning("⚠️ Configura tu OpenAI API Key en la barra lateral.")
    st.stop()

if not st.session_state.get("api_username"):
    st.warning("⚠️ Configura tus credenciales para los Endpoints en la barra lateral.")
    st.stop()

client = OpenAI(api_key=st.session_state.openai_api_key)

# ─────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────
tab_agente, tab_viz = st.tabs(["🤖 Agente IA", "📊 Visualizaciones"])


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — AGENTE IA  (código original intacto)
# ════════════════════════════════════════════════════════════════════════════
with tab_agente:

    with st.expander("💡 Ejemplos de preguntas para probar la inteligencia del Agente"):
        st.write("""
        - ¿Cuál molde tiene mayor SECn?
        - ¿Qué referencias tienen mayor consumo energético?
        - ¿Cuál es la línea base de consumo energético?
        - ¿Hay sobrecosto de energía respecto a la línea base?
        """)

    if "chat_history_agent" not in st.session_state:
        st.session_state.chat_history_agent = []

    st.subheader("🤖 Agente Interactivo")

    with st.form("chat_form"):
        user_question = st.text_area(
            "Pregúntale al Agente Energético...",
            placeholder="Escribe tu consulta..."
        )
        enviado = st.form_submit_button("Analizar con Agente 🚀")

    if enviado and user_question.strip():
        with st.spinner("🧠 Pensando y decidiendo las herramientas requeridas..."):
            username  = st.session_state.api_username
            password  = st.session_state.api_password
            ds_str    = date_start.strftime("%Y-%m-%d")
            de_str    = date_end.strftime("%Y-%m-%d")

            respuesta, tools_used = consultar_agente(
                pregunta=user_question,
                client=client,
                username=username,
                password=password,
                date_start_str=ds_str,
                date_end_str=de_str
            )

            st.session_state.chat_history_agent.append({
                "pregunta":   user_question,
                "respuesta":  respuesta,
                "tools_used": tools_used
            })

    if st.session_state.chat_history_agent:
        st.markdown("### 💬 Historial de Conversación")
        for chat in reversed(st.session_state.chat_history_agent):
            st.chat_message("user").write(chat["pregunta"])
            with st.chat_message("assistant"):
                st.write(chat["respuesta"])
                tu = chat.get("tools_used", [])
                if tu:
                    with st.expander(f"🛠️ Herramientas ejecutadas ({len(tu)})"):
                        for t in tu:
                            st.write(f"**Función:** `{t['nombre']}`")
                            st.write("**Argumentos:**", t["argumentos"])
                            st.write("**JSON recuperado:**")
                            st.json(t["json"])
            st.markdown("---")

        if st.button("🗑️ Limpiar Historial"):
            st.session_state.chat_history_agent = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — VISUALIZACIONES PCLD / PCDD / CURVA
# ════════════════════════════════════════════════════════════════════════════
with tab_viz:
    st.header("📊 Visualización de Gráficas Energéticas")
    st.markdown(
        "Consulta el endpoint `/test-mold` y visualiza los datos del periodo "
        "seleccionado en la barra lateral."
    )

    username   = st.session_state.get("api_username", "")
    password   = st.session_state.get("api_password", "")
    ds_str_viz = date_start.strftime("%Y-%m-%d")
    de_str_viz = date_end.strftime("%Y-%m-%d")

    # ── Selector de gráfico ───────────────────────────────────────────────
    tipo_grafico = st.radio(
        "🎛️ Tipo de visualización:",
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

    # ── Botón de carga ────────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        refrescar = st.button("🔄 Cargar / Actualizar", type="primary",
                              key="btn_viz_refresh")

    if refrescar or "json_moldes_viz" not in st.session_state:
        with st.spinner("📡 Consultando `/test-mold`..."):
            datos, error = consultar_endpoint(
                "moldes", username, password, ds_str_viz, de_str_viz
            )
        if error:
            st.error(f"❌ {error}")
            st.stop()
        st.session_state["json_moldes_viz"] = datos

    moldes_json = st.session_state.get("json_moldes_viz", {})

    with col_status:
        if moldes_json:
            st.success(f"✅ Datos listos  ·  {ds_str_viz} → {de_str_viz}")
        else:
            st.error("❌ Sin datos del endpoint")
            st.stop()

    with st.expander("🗂️ Ver JSON crudo de `/test-mold`", expanded=False):
        st.json(moldes_json)

    st.divider()

    # ── PCLD ─────────────────────────────────────────────────────────────
    if clave == "PCLD":
        x, y, a, b = extraer_pcld(moldes_json)
        if x is None:
            st.error("❌ No se encontró la sección 'PCLD' en el JSON.")
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

            with st.expander("📋 Datos tabulados (PCLD)", expanded=False):
                df = pd.DataFrame({
                    "Producción Conforme [kg]": x,
                    "Consumo [kWh]":            y,
                })
                df["SECn puntual [kWh/kg]"] = [
                    yi / xi if xi else None for xi, yi in zip(x, y)
                ]
                st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

    # ── PCDD ─────────────────────────────────────────────────────────────
    elif clave == "PCDD":
        x_pts, y_pts, x_curva, y_curva = extraer_pcdd(moldes_json)
        if x_pts is None:
            st.error("❌ No se encontró la sección 'PCDD' en el JSON.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔩 Nº moldes",         len(x_pts))
            c2.metric("⚡ SECn mínimo",        f"{min(y_pts):.4f}")
            c3.metric("⚡ SECn máximo",        f"{max(y_pts):.4f}")
            c4.metric("🏃 Prod. máx. [kg/h]", f"{max(x_pts):.2f}")

            fig = grafico_pcdd(x_pts, y_pts, x_curva, y_curva)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Datos tabulados (PCDD)", expanded=False):
                df = pd.DataFrame({
                    "Productividad Efectiva [kg/h]": x_pts,
                    "SECn real [kWh/kg]":            y_pts,
                })
                st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

    # ── Curva ─────────────────────────────────────────────────────────────
    elif clave == "Curva":
        x, y = extraer_curva(moldes_json)
        if x is None:
            st.error("❌ No se encontró la sección 'curva' en el JSON.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("🔢 Puntos de curva",    len(x))
            c2.metric("📉 SECn mín. esperado", f"{min(y):.4f}")
            c3.metric("📈 Flujo máximo",        f"{max(x):.2f} kg/h")

            fig = grafico_curva(x, y)
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("📋 Datos tabulados (Curva)", expanded=False):
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


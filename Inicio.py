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

# Definición para Strict Type en schemas si se requiere
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
    if tool_name == "obtener_resumen_general": return "summary"
    if tool_name == "obtener_datos_moldes": return "moldes"
    if tool_name == "obtener_datos_referencias": return "referencias"
    if tool_name == "obtener_linea_base": return "linea_base"
    return None

def ejecutar_herramienta(tool_name, arguments_dict, username, password):
    date_start = arguments_dict.get("dateStart")
    date_end = arguments_dict.get("dateEnd")
    endpoint_key = map_tool_to_endpoint(tool_name)

    if not endpoint_key:
        return {"error": f"Herramienta desconocida: {tool_name}"}

    datos, error = consultar_endpoint(endpoint_key, username, password, date_start, date_end)
    if error:
        return {"error": error}
    return datos

def consultar_agente(pregunta, client, username, password, date_start_str, date_end_str):
    # Damos al sistema contexto de las fechas que están activas en UI
    contexto_fechas = f"Fechas por defecto en interfaz: desde {date_start_str} hasta {date_end_str}. Úsalas si el usuario no especifica fechas."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + contexto_fechas},
        {"role": "user", "content": pregunta}
    ]
    
    # 1. Llamada inicial permitiendo a GPT decidir si usa herramientas
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Si no usa herramientas, devuelve respuesta directa
    if not tool_calls:
        return response_message.content, []
    
    # Si decide llamar herramientas
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
            "nombre": func_name,
            "argumentos": func_args,
            "json": resultado
        })
        
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": json.dumps(resultado, ensure_ascii=False)
        })
    
    # 2. Segunda llamada a GPT incluyendo los resultados de las herramientas
    second_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )
    
    return second_response.choices[0].message.content, herramientas_utilizadas

# ─────────────────────────────────────────────
# INTERFAZ (UI) STREAMLIT
# ─────────────────────────────────────────────
st.title("🏭 Diagnóstico de gestión energética--ESTRA")
st.markdown("**Powered by OpenAI Function Calling (Native)**")

with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Api Key
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
    
    # Credenciales API ESTRA
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
    
    # Rango de fechas
    st.subheader("📅 Fechas de contexto rápido")
    date_start = st.date_input("Inicio", value=datetime.today() - timedelta(days=7))
    date_end = st.date_input("Fin", value=datetime.today())

# Main content
if not st.session_state.get("openai_api_key"):
    st.warning("⚠️ Debes configurar tu OpenAI API Key en la barra lateral antes de consultar.")
elif not st.session_state.get("api_username"):
    st.warning("⚠️ Debes configurar tus credenciales para los Endpoints en la barra lateral.")
else:
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    # Recomendaciones / Ejemplos
    with st.expander("💡 Ejemplos de preguntas para probar la inteligencia del Agente"):
        st.write("""
        - "¿Cuál es el resumen y el consumo de todas las métricas en este periodo?" *(Llamará a obtener_resumen_general)*
        - "Compárame el SECn de los moldes versus lo que dice cada referencia." *(Podría llamar a moldes y referencias en paralelo)*
        - "¿Cuáles son los moldes con mayor tiempo de parada comparándolos con las metas de línea base?" *(Llamaría a obtener_datos_moldes y obtener_linea_base)*
        """)
        
    if "chat_history_agent" not in st.session_state:
        st.session_state.chat_history_agent = []

    st.subheader("🤖 Agente Interactivo")
    
    with st.form("chat_form"):
        user_question = st.text_area("Pregúntale al Agente Energético...", placeholder="Escribe tu consulta...")
        enviado = st.form_submit_button("Analizar con Agente 🚀")
        
    if enviado and user_question.strip():
        with st.spinner("🧠 Pensando y decidiendo las herramientas requeridas..."):
            username = st.session_state.api_username
            password = st.session_state.api_password
            ds_str = date_start.strftime("%Y-%m-%d")
            de_str = date_end.strftime("%Y-%m-%d")
            
            respuesta, tools_used = consultar_agente(
                pregunta=user_question,
                client=client,
                username=username,
                password=password,
                date_start_str=ds_str,
                date_end_str=de_str
            )
            
            # Guardar en historial
            st.session_state.chat_history_agent.append({
                "pregunta": user_question,
                "respuesta": respuesta,
                "tools_used": tools_used
            })

    if st.session_state.chat_history_agent:
        st.markdown("### 💬 Historial de Conversación")
        for idx, chat in enumerate(reversed(st.session_state.chat_history_agent)):
            st.chat_message("user").write(chat["pregunta"])
            with st.chat_message("assistant"):
                st.write(chat["respuesta"])
                
                # Desglose de herramientas usadas por si el usuario quiere auditar
                tu = chat.get("tools_used", [])
                if tu:
                    with st.expander(f"🛠️ Herramientas ejecutadas bajo el capó ({len(tu)})"):
                        for t in tu:
                            st.write(f"**Función:** `{t['nombre']}`")
                            st.write("**Argumentos que decidió pasar GPT:**", t['argumentos'])
                            st.write("**Respuesta JSON recuperada:**")
                            st.json(t["json"])
            st.markdown("---")
        
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.chat_history_agent = []
            st.rerun()


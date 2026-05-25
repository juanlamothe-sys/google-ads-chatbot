import streamlit as st
import pandas as pd
from openai import OpenAI
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from datetime import date
import io
import json
import re

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(
    page_title="📊 Google Ads Chat",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads - Chat con tus datos")
st.markdown(
    "Haz preguntas en lenguaje natural sobre tus campanias, "
    "costes, conversiones, productos..."
)

# --- SIDEBAR: CREDENCIALES ---
with st.sidebar:
    st.header("🔑 Credenciales")

    st.subheader("🤖 LLM (Groq)")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Gratis en https://console.groq.com/keys"
    )

    st.subheader("📊 Google Ads")
    developer_token = st.text_input(
        "Developer Token",
        type="password"
    )
    client_id = st.text_input(
        "Client ID",
        type="password"
    )
    client_secret = st.text_input(
        "Client Secret",
        type="password"
    )
    refresh_token = st.text_input(
        "Refresh Token",
        type="password"
    )
    customer_id = st.text_input(
        "Customer ID (sin guiones)"
    )
    login_customer_id = st.text_input(
        "Login Customer ID (MCC)"
    )

    st.divider()
    st.markdown(
        "🔒 Las credenciales **no se guardan**. "
        "Solo se usan durante tu sesion."
    )

# --- VALIDAR CREDENCIALES ---
all_creds = all([
    groq_key, developer_token, client_id,
    client_secret, refresh_token, customer_id
])


def get_groq_client():
    """Crea el cliente de Groq (compatible con OpenAI)."""
    return OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1"
    )


def get_google_ads_client():
    """Crea el cliente de Google Ads."""
    config = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "use_proto_plus": True,
    }
    if login_customer_id:
        config["login_customer_id"] = login_customer_id
    return GoogleAdsClient.load_from_dict(config)


def flatten_dict(d, parent_key="", sep="_"):
    """Aplana un diccionario anidado."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(
                flatten_dict(v, new_key, sep=sep).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


def run_gaql_query(query_text):
    """Ejecuta una query GAQL y devuelve un DataFrame."""
    client = get_google_ads_client()
    ga_service = client.get_service("GoogleAdsService")

    try:
        stream = ga_service.search_stream(
            customer_id=customer_id,
            query=query_text
        )

        rows = []
        for batch in stream:
            for row in batch.results:
                row_json = type(row).to_json(row)
                row_data = json.loads(row_json)
                row_dict = flatten_dict(row_data)
                rows.append(row_dict)

        if rows:
            return pd.DataFrame(rows), None
        else:
            return pd.DataFrame(), "La query no devolvio resultados."

    except GoogleAdsException as ex:
        errors = []
        for error in ex.failure.errors:
            errors.append(error.message)
        return pd.DataFrame(), " | ".join(errors)
    except Exception as e:
        return pd.DataFrame(), str(e)


SYSTEM_PROMPT = (
    "Eres un experto en Google Ads Query Language (GAQL). "
    "Tu trabajo es convertir preguntas en lenguaje natural "
    "a queries GAQL validas.\n\n"
    "REGLAS IMPORTANTES:\n"
    "- Devuelve SOLO la query GAQL, sin explicaciones ni markdown.\n"
    "- No uses comillas de bloque ni backticks.\n"
    "- Usa SOLO recursos, metricas y segmentos validos "
    "de Google Ads API v19.\n"
    "- Para costes usa metrics.cost_micros "
    "(divide entre 1000000 para euros).\n"
    "- Para fechas usa segments.date y WHERE "
    "con formato YYYY-MM-DD.\n"
    "- Para campanias activas: campaign.status = 'ENABLED'\n"
    "- Para Shopping: "
    "campaign.advertising_channel_type = 'SHOPPING'\n"
    "- Para PMAX: "
    "campaign.advertising_channel_type = 'PERFORMANCE_MAX'\n"
    "- Para productos usa shopping_performance_view "
    "como recurso FROM.\n"
    "- Segmentos de tiempo validos: LAST_7_DAYS, LAST_30_DAYS, "
    "THIS_MONTH, LAST_MONTH, TODAY, YESTERDAY.\n"
    "- Si piden datos por dia, incluye segments.date en SELECT.\n"
    "- Si piden datos por campania, incluye campaign.name "
    "en SELECT.\n"
    "- ORDER BY solo puede usar campos que esten en SELECT.\n\n"
    "RECURSOS PRINCIPALES:\n"
    "- campaign: campanias (id, name, status, "
    "advertising_channel_type)\n"
    "- ad_group: grupos de anuncios\n"
    "- ad_group_ad: anuncios\n"
    "- shopping_performance_view: datos de productos "
    "Shopping/PMAX\n"
    "- keyword_view: keywords\n"
    "- segments.date: fecha\n"
    "- segments.product_title: titulo de producto\n\n"
    "METRICAS COMUNES:\n"
    "- metrics.impressions, metrics.clicks, metrics.cost_micros\n"
    "- metrics.conversions, metrics.conversions_value\n"
    "- metrics.ctr, metrics.average_cpc\n"
    "- metrics.all_conversions, metrics.all_conversions_value\n\n"
    "EJEMPLOS:\n"
    "Pregunta: Coste por campania este mes\n"
    "Query: SELECT campaign.name, metrics.cost_micros "
    "FROM campaign WHERE segments.date DURING THIS_MONTH "
    "AND campaign.status = 'ENABLED'\n\n"
    "Pregunta: Top 10 productos con mas clics ultimos 7 dias\n"
    "Query: SELECT segments.product_title, metrics.clicks, "
    "metrics.impressions, metrics.cost_micros "
    "FROM shopping_performance_view "
    "WHERE segments.date DURING LAST_7_DAYS "
    "ORDER BY metrics.clicks DESC LIMIT 10\n\n"
    "Pregunta: Gasto diario ultimos 30 dias\n"
    "Query: SELECT segments.date, metrics.cost_micros "
    "FROM campaign WHERE segments.date DURING LAST_30_DAYS "
    "AND campaign.status = 'ENABLED' "
    "ORDER BY segments.date DESC"
)


def ask_llm_for_gaql(user_question, error_feedback=None):
    """Pide al LLM que genere una query GAQL."""
    client = get_groq_client()

    user_msg = "Pregunta del usuario: " + user_question
    if error_feedback:
        user_msg = (
            user_msg
            + "\n\nLa query anterior dio este error: "
            + error_feedback
            + "\nPor favor corrige la query."
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
        max_tokens=500
    )

    gaql = response.choices[0].message.content.strip()
    gaql = gaql.replace("```sql", "")
    gaql = gaql.replace("```gaql", "")
    gaql = gaql.replace("```", "")
    gaql = gaql.strip()
    return gaql


def ask_llm_to_explain(user_question, df):
    """Pide al LLM que explique los resultados."""
    client = get_groq_client()

    if len(df) > 50:
        data_str = df.head(50).to_string(index=False)
        data_str = (
            data_str
            + "\n... (mostrando 50 de "
            + str(len(df))
            + " filas)"
        )
    else:
        data_str = df.to_string(index=False)

    prompt = (
        "El usuario pregunto: " + user_question + "\n\n"
        "Estos son los datos obtenidos de Google Ads:\n\n"
        + data_str + "\n\n"
        "INSTRUCCIONES:\n"
        "- Responde en espanol, de forma clara y concisa.\n"
        "- Si hay cost_micros, conviertelo a euros "
        "(divide entre 1000000) y usa el simbolo del euro.\n"
        "- Destaca los datos mas relevantes.\n"
        "- Si hay tendencias o insights interesantes, "
        "mencionalos.\n"
        "- Usa formato markdown con negritas y listas "
        "si es util.\n"
        "- No repitas toda la tabla, resume los puntos clave.\n"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Eres un analista de Google Ads experto. Respondes siempre en espanol."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


# --- HISTORIAL DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "dataframe" in message:
            st.dataframe(
                message["dataframe"],
                use_container_width=True
            )
        if "gaql" in message:
            with st.expander("🔍 Query GAQL ejecutada"):
                st.code(message["gaql"], language="sql")

# --- CHAT ---
if not all_creds:
    st.info(
        "👈 Introduce tus credenciales en el sidebar para empezar."
    )
    st.markdown(
        """
        ### 💡 Ejemplos de preguntas:

        | Pregunta | Que obtendras |
        |---|---|
        | *Cuanto gaste ayer?* | Coste total |
        | *Top 10 productos con mas clics esta semana* | Ranking Shopping/PMAX |
        | *Gasto diario ultimos 30 dias* | Tabla coste/dia |
        | *Que campanias estan activas?* | Lista de campanias |
        | *Conversiones del ultimo mes por campania* | Rendimiento |
        | *Productos con mas impresiones en PMAX* | Top productos |

        ### 🔑 Credenciales necesarias:
        1. **Groq API Key** → gratis en https://console.groq.com/keys
        2. **Google Ads** → Developer Token, Client ID/Secret, Refresh Token
        3. **Customer IDs** → Account ID + MCC ID (sin guiones)
        """
    )
else:
    user_input = st.chat_input(
        "Pregunta lo que quieras sobre tus campanias..."
    )

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            with st.spinner("🤔 Generando query..."):
                gaql = ask_llm_for_gaql(user_input)

            st.caption("Ejecutando consulta en Google Ads...")

            with st.spinner("📊 Consultando Google Ads..."):
                df, error = run_gaql_query(gaql)

            if error:
                with st.spinner("⚠️ Corrigiendo query..."):
                    gaql = ask_llm_for_gaql(
                        user_input, error
                    )
                    df, error = run_gaql_query(gaql)

            if error:
                response_text = (
                    "❌ No pude obtener los datos. "
                    "Error: " + error + "\n\n"
                    "Prueba a reformular la pregunta."
                )
                st.markdown(response_text)
                with st.expander("🔍 Query GAQL intentada"):
                    st.code(gaql, language="sql")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "gaql": gaql
                })

            elif df.empty:
                response_text = (
                    "La consulta no devolvio resultados. "
                    "Puede que no haya datos para ese "
                    "periodo o filtro."
                )
                st.markdown(response_text)
                with st.expander("🔍 Query GAQL ejecutada"):
                    st.code(gaql, language="sql")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "gaql": gaql
                })

            else:
                st.dataframe(
                    df, use_container_width=True
                )

                with st.expander("🔍 Query GAQL ejecutada"):
                    st.code(gaql, language="sql")

                with st.spinner("💡 Analizando..."):
                    explanation = ask_llm_to_explain(
                        user_input, df
                    )
                st.markdown(explanation)

                buffer = io.BytesIO()
                df.to_excel(
                    buffer,
                    index=False,
                    sheet_name="Google Ads"
                )
                st.download_button(
                    "📥 Descargar Excel",
                    data=buffer.getvalue(),
                    file_name="google_ads_data.xlsx",
                    mime=(
                        "application/"
                        "vnd.openxmlformats-"
                        "officedocument."
                        "spreadsheetml.sheet"
                    )
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": explanation,
                    "dataframe": df,
                    "gaql": gaql
                })

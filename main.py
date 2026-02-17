import streamlit as st
import os
import functools
from typing import Annotated, Literal, TypedDict

# LangChain & LangGraph
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Conchita AI (2.5 Flash)", page_icon="⚡", layout="wide")

# --- 2. INTERFAZ VISUAL ---
st.title("⚡ Conchita: News Writer Agent")
st.markdown("### Redacción de noticias con IA de última generación")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    
    # Campo para API KEYS
    google_key_input = st.text_input("Google API Key", type="password")
    tavily_key_input = st.text_input("Tavily API Key", type="password")
    
    st.divider()
    
    # SELECCIÓN DE MODELO
    st.subheader("Versión de Gemini")
    model_name = st.text_input(
        "Nombre del Modelo", 
        value="gemini-2.5-flash", 
        help="Si el 2.5 falla, prueba con 'gemini-2.0-flash-exp' o 'gemini-1.5-flash'."
    )
    
    if google_key_input: os.environ["GOOGLE_API_KEY"] = google_key_input
    if tavily_key_input: os.environ["TAVILY_API_KEY"] = tavily_key_input

# --- ÁREA PRINCIPAL ---
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("¿Sobre qué quieres escribir hoy?", placeholder="Ej: Impacto de la IA en la educación 2026")
with col2:
    st.write("") 
    st.write("") 
    generate_btn = st.button("Generar Artículo", type="primary", use_container_width=True)

# --- 3. LÓGICA DE AGENTES ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Forzamos que el mensaje tenga un nombre para identificarlo luego
    result.name = name
    return {"messages": [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    
    # --- CORRECCIÓN 1: Uso seguro de getattr ---
    # Esto evita el error si last_message es HumanMessage
    tool_calls = getattr(last_message, "tool_calls", [])
    
    if tool_calls:
        return "tools"
    return "outliner"

# --- 4. INICIALIZACIÓN DEL GRAFO ---
def init_graph(selected_model):
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        return None, "⚠️ Faltan las API Keys en la barra lateral."

    try:
        llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.5)
        # Prueba de conexión rápida
        llm.invoke("Hello") 
    except Exception as e:
        if "404" in str(e):
            return None, f"❌ El modelo '{selected_model}' no fue encontrado (Error 404). Verifica acceso o usa 'gemini-1.5-flash'."
        return None, f"Error de conexión: {str(e)}"

    tools = [TavilySearchResults(max_results=5)]
    
    # Prompts
    search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.
    NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node."""

    outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline for the article."""

    writer_template = """Your job is to write an article, do it in this format:
    TITLE: <title>
    BODY: <body>
    NOTE: Do not copy the outline. You need to write the article with the info provided by the outline."""

    # Agentes
    search
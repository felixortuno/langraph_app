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
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Conchita AI (2.5 Flash)", page_icon="⚡", layout="wide")

# --- 2. INTERFAZ VISUAL ---
st.title("⚡ Conchita: News Writer Agent")
st.markdown("### Redacción de noticias con IA de última generación")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Campo para API KEYS
    google_key_input = st.text_input("Google API Key", type="password")
    tavily_key_input = st.text_input("Tavily API Key", type="password")
    
    st.divider()
    
    # SELECCIÓN DE MODELO
    st.subheader("Versión de Gemini")
    model_name = st.text_input(
        "Nombre del Modelo", 
        value="gemini-2.5-flash", 
        help="Si falla, prueba 'gemini-1.5-flash'."
    )
    
    # Guardamos las claves en el entorno al momento
    if google_key_input: os.environ["GOOGLE_API_KEY"] = google_key_input
    if tavily_key_input: os.environ["TAVILY_API_KEY"] = tavily_key_input

# --- ÁREA PRINCIPAL CON FORMULARIO (ESTO ARREGLA EL BOTÓN) ---

# Usamos st.form para 'congelar' la entrada hasta que pulses el botón
with st.form("my_form"):
    col1, col2 = st.columns([4, 1])
    with col1:
        topic = st.text_input("¿Sobre qué quieres escribir hoy?", placeholder="Ej: Avances en medicina 2025")
    with col2:
        st.write("") 
        st.write("") 
        # Este es el botón especial de formulario
        submitted = st.form_submit_button("Generar Artículo", type="primary", use_container_width=True)

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
    result.name = name # Etiquetamos el mensaje
    return {"messages": [result]}

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    
    # USO SEGURO: Verificamos si existe tool_calls sin romper el código
    tool_calls = getattr(last_message, "tool_calls", [])
    
    if tool_calls:
        return "tools"
    return "outliner"

# --- 4. INICIALIZACIÓN DEL GRAFO ---
def init_graph(selected_model):
    # Verificación estricta de claves
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        return None, "FALTAN_CLAVES"

    try:
        llm = ChatGoogleGenerativeAI(model=selected_model, temperature=0.5)
        llm.invoke("Hello") # Test de conexión
    except Exception as e:
        return None, f"Error conectando con {selected_model}: {str(e)}"

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
    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    # Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("search", functools.partial(agent_node, agent=search_agent, name="Search Agent"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("outliner", functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent"))
    workflow.add_node("writer", functools.partial(agent_node, agent=writer_agent, name="Writer Agent"))

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(), None

# --- 5. EJECUCIÓN (SOLO SI SE PULSA EL BOTÓN DEL FORMULARIO) ---

if submitted:
    # 1. Validación rápida
    if not topic:
        st.warning("⚠️ Escribe un tema primero.")
    else:
        st.info("✅ Botón presionado. Iniciando proceso...") # Feedback visual inmediato
        
        # 2. Inicializar Grafo
        graph, error_msg = init_graph(model_name)
        
        if error_msg == "FALTAN_CLAVES":
            st.error("⛔ ALTO: Necesitas poner la Google API Key y la Tavily API Key en el menú de la izquierda.")
        elif error_msg:
            st.error(f"❌ {error_msg}")
        else:
            # 3. Ejecutar
            status_box = st.status(f"🚀 Trabajando en: {topic}", expanded=True)
            try:
                input_data = {"messages": [HumanMessage(content=topic)]}
                final_content = ""
                
                for event in graph.stream(input_data, stream_mode="values"):
                    current_messages = event.get("messages", [])
                    if current_messages:
                        last_msg = current_messages[-1]
                        
                        tool_calls = getattr(last_msg, "tool_calls", [])
                        
                        if hasattr(last_msg, "name") and last_msg.name:
                            status_box.write(f"🤖 **{last_msg.name}**: Tarea completada.")
                        elif tool_calls:
                            status_box.write("🔎 **Buscando información en internet...**")
                            
                        final_content = last_msg.content

                status_box.update(label="✅ ¡Artículo Terminado!", state="complete", expanded=False)
                
                st.divider()
                st.subheader("📰 Tu Noticia:")
                st.markdown(final_content)

            except Exception as e:
                status_box.update(label="❌ Algo falló", state="error")
                st.error(f"Error técnico: {str(e)}")
import streamlit as st
import os
import sys

# --- 1. INTERFAZ PRIMERO (Para evitar pantalla negra) ---
st.set_page_config(page_title="Conchita AI (2.5)", page_icon="⚡", layout="wide")

st.title("⚡ Conchita: News Writer Agent")
st.markdown("### Redacción con Gemini 2.5 Flash")

# --- 2. BARRA LATERAL ---
with st.sidebar:
    st.header("⚙️ Configuración")
    # Credenciales
    google_key_input = st.text_input("Google API Key", type="password")
    tavily_key_input = st.text_input("Tavily API Key", type="password")
    
    st.divider()
    
    # Selector de modelo (Editable por si 2.5 falla)
    model_name = st.text_input("Modelo", value="gemini-2.5-flash")
    
    if google_key_input: os.environ["GOOGLE_API_KEY"] = google_key_input
    if tavily_key_input: os.environ["TAVILY_API_KEY"] = tavily_key_input

# --- 3. ÁREA DE BÚSQUEDA ---
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("Tema del artículo", placeholder="Ej: Futuro de la IA en 2025")
with col2:
    st.write("") 
    st.write("") 
    generate_btn = st.button("Generar", type="primary", use_container_width=True)

# --- 4. CARGA DIFERIDA DE LIBRERÍAS (Para atrapar errores) ---
if generate_btn:
    if not topic:
        st.warning("⚠️ Escribe un tema primero.")
        st.stop()
    
    if not google_key_input or not tavily_key_input:
        st.error("❌ Faltan las API Keys en la barra lateral.")
        st.stop()

    # Aquí empieza la magia (dentro de un spinner para ver si carga)
    with st.spinner("🔄 Cargando librerías y conectando a Gemini..."):
        try:
            # Importamos AQUÍ para que si fallan, no rompan la app al inicio
            import functools
            from typing import Annotated, Literal, TypedDict
            from langgraph.graph import END, StateGraph
            from langgraph.graph.message import add_messages
            from langgraph.prebuilt import ToolNode
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_community.tools.tavily_search import TavilySearchResults
            from langchain_core.messages import HumanMessage, BaseMessage
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            # --- DEFINICIÓN DE AGENTES ---
            class AgentState(TypedDict):
                messages: Annotated[list[BaseMessage], add_messages]

            def create_agent(llm, tools, system_message: str):
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_message),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                prompt = prompt.partial(system_message=system_message)
                if tools: return prompt | llm.bind_tools(tools)
                return prompt | llm

            def agent_node(state, agent, name):
                result = agent.invoke(state)
                return {"messages": [result]}

            def should_search(state) -> Literal["tools", "outliner"]:
                if state['messages'][-1].tool_calls: return "tools"
                return "outliner"

            # INICIALIZACIÓN
            try:
                # Intentamos conectar con el modelo elegido (2.5)
                llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5)
                # Prueba de vida
                llm.invoke("Hi")
            except Exception as e:
                st.error(f"❌ Error con el modelo '{model_name}': {str(e)}")
                st.info("💡 Intenta cambiar el nombre del modelo en la barra lateral a 'gemini-1.5-flash' o 'gemini-2.0-flash-exp'.")
                st.stop()

            # GRAFO
            tools = [TavilySearchResults(max_results=5)]
            search_agent = create_agent(llm, tools, "Search for relevant news. Do not write yet.")
            outliner_agent = create_agent(llm, [], "Create an outline based on the news.")
            writer_agent = create_agent(llm, [], "Write the article based on the outline. Format: TITLE / BODY.")

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
            
            app = workflow.compile()
            
        except ImportError as e:
            st.error(f"❌ Error crítico de instalación: {str(e)}")
            st.info("Asegúrate de que tu requirements.txt es correcto y reinicia la app.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
            st.stop()

    # --- 5. EJECUCIÓN VISUAL ---
    status_box = st.status(f"🚀 Trabajando con {model_name}...", expanded=True)
    try:
        final_content = ""
        for event in app.stream({"messages": [HumanMessage(content=topic)]}, stream_mode="values"):
            msg = event.get("messages", [])[-1]
            if hasattr(msg, "name") and msg.name:
                status_box.write(f"✅ {msg.name}")
            elif msg.tool_calls:
                status_box.write("🔎 Buscando en Google/Tavily...")
            final_content = msg.content
        
        status_box.update(label="¡Completado!", state="complete", expanded=False)
        st.markdown("---")
        st.markdown(final_content)
        
    except Exception as e:
        status_box.update(label="Falló", state="error")
        st.error(f"Error durante la generación: {e}")
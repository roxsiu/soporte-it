"""
IT-Copilot: Agente Inteligente de Soporte TI Nivel 1
=====================================================
Sistema de resolución autónoma de tickets mediante
arquitectura RAG y agente con herramientas (LangChain).

Autores: [Nombres del equipo]
Curso:   Proyecto Integrador - Diploma AI Engineer
Fecha:   Junio 2026
"""

import streamlit as st
import os
import json
import uuid
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

DIRECTORIO_DATOS = "data"
DIRECTORIO_CHROMA = "chroma_db"
DIRECTORIO_TICKETS = "tickets"
MODELO_LLM = "gpt-4o-mini"
MODELO_EMBEDDINGS = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# Crear directorios necesarios si no existen
Path(DIRECTORIO_TICKETS).mkdir(exist_ok=True)


# ============================================================
# PROMPT DEL SISTEMA
# ============================================================

SYSTEM_PROMPT = """Eres Soporte-IT, el asistente virtual de soporte técnico de Nivel 1 \
de NovaTech Solutions S.A.C.

Se te proporcionará DOCUMENTACIÓN RELEVANTE junto con la consulta del usuario. \
Usa esa documentación para responder.

NO asumas detalles que el usuario no haya mencionado. \
Responde solo con lo que el usuario dijo y lo que dicen los documentos.

INSTRUCCIONES:
1. Lee la documentación proporcionada.
2. Si la documentación contiene la solución:
   - Escribe en tu respuesta los pasos COMPLETOS y NUMERADOS de la solución.
   - Luego usa 'registrar_ticket' con estado 'resuelto'.
   - Luego usa 'enviar_email' con la solución en el cuerpo del correo.
3. Si la documentación NO contiene la solución o el problema coincide con \
   criterios de escalamiento:
   - Explica al usuario que su caso requiere atención especializada.
   - Usa 'escalar_ticket' para derivar el caso.

REGLAS:
- Tu respuesta SIEMPRE debe incluir los pasos detallados de la solución.
- NUNCA digas solo "te envié un correo con los pasos". MUESTRA los pasos.
- NUNCA inventes soluciones que no estén en la documentación.
- Responde en español, tono profesional y amable.

FORMATO DE RESPUESTA CUANDO HAY SOLUCIÓN:
- Saludo breve
- Pasos numerados de la solución
- Nota adicional si aplica
- Mensaje de cierre

FORMATO CUANDO SE ESCALA:
- Reconoce el problema del usuario
- Explica brevemente por qué requiere atención especializada
- Informa que un técnico se comunicará pronto
"""


# ============================================================
# PIPELINE RAG: INGESTA Y VECTORIZACIÓN
# ============================================================

@st.cache_resource(show_spinner="Procesando documentos de soporte...")
def construir_base_vectorial() -> Chroma:
    """
    Carga los PDFs del directorio de datos, los segmenta en chunks
    y los almacena como vectores en ChromaDB.

    Returns:
        Chroma: Base de datos vectorial lista para búsquedas.
    """
    if not os.path.exists(DIRECTORIO_DATOS):
        st.error(f"No se encontró el directorio '{DIRECTORIO_DATOS}/'.")
        st.stop()

    archivos_pdf = list(Path(DIRECTORIO_DATOS).glob("*.pdf"))
    if not archivos_pdf:
        st.error(f"No hay archivos PDF en '{DIRECTORIO_DATOS}/'.")
        st.stop()

    # Cargar todos los PDFs del directorio
    loader = PyPDFDirectoryLoader(DIRECTORIO_DATOS)
    documentos = loader.load()

    # Segmentar en chunks con solapamiento
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)

    # Crear embeddings y almacenar en ChromaDB
    embeddings = OpenAIEmbeddings(model=MODELO_EMBEDDINGS)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_CHROMA
    )

    return vectorstore


# ============================================================
# HERRAMIENTAS DEL AGENTE
# ============================================================

def crear_herramientas() -> list:
    """
    Define las herramientas que el agente puede usar de forma autónoma.
    La búsqueda RAG se realiza antes de llamar al agente, por lo que
    aquí solo van las herramientas de acción.

    Returns:
        Lista de herramientas configuradas.
    """

    @tool
    def registrar_ticket(
        resumen: str,
        estado: str,
        email_usuario: str,
        categoria: str
    ) -> str:
        """Registra un ticket de soporte en el sistema.
        Usa esta herramienta después de resolver o escalar una consulta.

        Args:
            resumen: Descripción breve del problema reportado.
            estado: Estado del ticket, debe ser 'resuelto' o 'escalado'.
            email_usuario: Correo electrónico del usuario que reporta.
            categoria: Categoría del problema (ej: contraseñas, vpn, correo, impresoras, wifi, microsoft365).
        """
        ticket = {
            "id": f"TK-{uuid.uuid4().hex[:8].upper()}",
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email_usuario": email_usuario,
            "resumen": resumen,
            "categoria": categoria,
            "estado": estado
        }

        archivo = os.path.join(DIRECTORIO_TICKETS, "tickets.json")
        tickets_existentes = []
        if os.path.exists(archivo):
            with open(archivo, "r", encoding="utf-8") as f:
                tickets_existentes = json.load(f)

        tickets_existentes.append(ticket)
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(tickets_existentes, f, ensure_ascii=False, indent=2)

        if "tickets" not in st.session_state:
            st.session_state.tickets = []
        st.session_state.tickets.append(ticket)

        return f"Ticket {ticket['id']} registrado exitosamente con estado: {estado}."

    @tool
    def enviar_email(
        email_usuario: str,
        asunto: str,
        cuerpo: str
    ) -> str:
        """Envía un correo electrónico al usuario con la solución a su problema.
        Usa esta herramienta después de proporcionar una solución exitosa.

        Args:
            email_usuario: Dirección de correo del destinatario.
            asunto: Asunto del correo.
            cuerpo: Contenido del correo con la solución detallada.
        """
        email_registro = {
            "id": f"EM-{uuid.uuid4().hex[:8].upper()}",
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "para": email_usuario,
            "asunto": asunto,
            "cuerpo": cuerpo,
            "estado": "enviado"
        }

        archivo = os.path.join(DIRECTORIO_TICKETS, "emails.json")
        emails_existentes = []
        if os.path.exists(archivo):
            with open(archivo, "r", encoding="utf-8") as f:
                emails_existentes = json.load(f)

        emails_existentes.append(email_registro)
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(emails_existentes, f, ensure_ascii=False, indent=2)

        return f"Correo enviado exitosamente a {email_usuario} con asunto: '{asunto}'."

    @tool
    def escalar_ticket(
        resumen: str,
        email_usuario: str,
        razon: str
    ) -> str:
        """Escala un ticket al equipo de soporte de Nivel 2 cuando el problema
        excede las capacidades de resolución de Nivel 1.

        Args:
            resumen: Descripción del problema que se escala.
            email_usuario: Correo del usuario afectado.
            razon: Motivo específico por el cual se escala.
        """
        ticket = {
            "id": f"TK-{uuid.uuid4().hex[:8].upper()}",
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email_usuario": email_usuario,
            "resumen": resumen,
            "estado": "escalado",
            "razon_escalamiento": razon,
            "asignado_a": "Equipo de Soporte Nivel 2"
        }

        archivo = os.path.join(DIRECTORIO_TICKETS, "tickets.json")
        tickets_existentes = []
        if os.path.exists(archivo):
            with open(archivo, "r", encoding="utf-8") as f:
                tickets_existentes = json.load(f)

        tickets_existentes.append(ticket)
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(tickets_existentes, f, ensure_ascii=False, indent=2)

        if "tickets" not in st.session_state:
            st.session_state.tickets = []
        st.session_state.tickets.append(ticket)

        return (
            f"Ticket {ticket['id']} escalado al Nivel 2. "
            f"Razón: {razon}. "
            f"Un técnico especializado contactará a {email_usuario} en breve."
        )

    return [registrar_ticket, enviar_email, escalar_ticket]


# ============================================================
# CREACIÓN DEL AGENTE
# ============================================================

@st.cache_resource(show_spinner="Configurando agente de soporte...")
def crear_agente(_vectorstore: Chroma) -> AgentExecutor:
    """
    Configura el agente con el LLM, las herramientas y el prompt del sistema.

    Args:
        _vectorstore: Base de datos vectorial (prefijo _ para que Streamlit no la hashee).

    Returns:
        AgentExecutor listo para procesar consultas.
    """
    herramientas = crear_herramientas()

    llm = ChatOpenAI(
        model=MODELO_LLM,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agente = create_tool_calling_agent(llm, herramientas, prompt)

    return AgentExecutor(
        agent=agente,
        tools=herramientas,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True
    )


# ============================================================
# FUNCIONES AUXILIARES DE LA INTERFAZ
# ============================================================

def cargar_tickets_guardados() -> list:
    """Carga los tickets guardados en disco."""
    archivo = os.path.join(DIRECTORIO_TICKETS, "tickets.json")
    if os.path.exists(archivo):
        with open(archivo, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def mostrar_sidebar():
    """Renderiza la barra lateral con información del usuario y tickets."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/technical-support.png",
            width=80
        )
        st.title("Soporte-IT")
        st.caption("Soporte TI Nivel 1 · NovaTech Solutions")

        st.divider()

        # Campo para email del usuario
        email = st.text_input(
            "Tu correo corporativo",
            placeholder="nombre@novatech.com",
            key="email_usuario"
        )

        st.divider()

        # Mostrar tickets de la sesión
        st.subheader("Tickets de esta sesión")
        tickets = st.session_state.get("tickets", [])

        if not tickets:
            st.info("Aún no hay tickets registrados.")
        else:
            for tk in reversed(tickets):
                estado = tk.get("estado", "desconocido")
                if estado == "resuelto":
                    icono = "✅"
                    color = "green"
                elif estado == "escalado":
                    icono = "🔴"
                    color = "red"
                else:
                    icono = "⚪"
                    color = "gray"

                with st.expander(f"{icono} {tk['id']} - {estado.upper()}"):
                    st.write(f"**Fecha:** {tk['fecha']}")
                    st.write(f"**Usuario:** {tk['email_usuario']}")
                    st.write(f"**Resumen:** {tk['resumen']}")
                    if "razon_escalamiento" in tk:
                        st.write(f"**Razón de escalamiento:** {tk['razon_escalamiento']}")

        st.divider()

        # Botón para limpiar conversación
        if st.button("Nueva conversación", use_container_width=True):
            st.session_state.mensajes = []
            st.rerun()
        
        # Botón para cerrar conversación
        if st.button("Cerrar sesión", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.session_state.sesion_cerrada = True
            st.rerun()


        # Info de documentos cargados
        with st.expander("📄 Documentos cargados"):
            archivos = list(Path(DIRECTORIO_DATOS).glob("*.pdf"))
            if archivos:
                for archivo in sorted(archivos):
                    st.write(f"• {archivo.name}")
                st.caption(f"Total: {len(archivos)} manuales")
            else:
                st.warning("No hay PDFs en /data")

    return email


def obtener_historial_chat() -> list:
    """Convierte el historial de mensajes a formato LangChain."""
    historial = []
    for msg in st.session_state.get("mensajes", []):
        if msg["role"] == "user":
            historial.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            historial.append(AIMessage(content=msg["content"]))
    return historial

def mostrar_dashboard():
    """Muestra métricas y gráficos basados en los tickets registrados."""
    archivo = os.path.join(DIRECTORIO_TICKETS, "tickets.json")

    if not os.path.exists(archivo):
        st.info("Aún no hay datos. Usa el chat para generar tickets.")
        return

    with open(archivo, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    if not tickets:
        st.info("Aún no hay datos. Usa el chat para generar tickets.")
        return

    df = pd.DataFrame(tickets)

    # --- Métricas principales ---
    total = len(df)
    resueltos = len(df[df["estado"] == "resuelto"])
    escalados = len(df[df["estado"] == "escalado"])
    tasa = round((resueltos / total) * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tickets", total)
    col2.metric("Resueltos", resueltos, delta=f"{tasa}%")
    col3.metric("Escalados", escalados)
    col4.metric("Tasa de resolución", f"{tasa}%")

    st.divider()

    # --- Gráficos ---
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("Tickets por estado")
        estado_counts = df["estado"].value_counts()
        st.bar_chart(estado_counts)

    with col_der:
        st.subheader("Tickets por categoría")
        if "categoria" in df.columns:
            cat_counts = df["categoria"].value_counts()
            st.bar_chart(cat_counts)
        else:
            st.info("Sin datos de categoría aún.")

    st.divider()

    # --- Tabla de tickets ---
    st.subheader("Historial de tickets")
    columnas_mostrar = ["id", "fecha", "email_usuario", "resumen", "estado"]
    columnas_disponibles = [c for c in columnas_mostrar if c in df.columns]
    st.dataframe(
        df[columnas_disponibles].sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================

def main():
    st.set_page_config(
        page_title="Soporte-IT · NovaTech Solutions",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "No se encontró la API key de OpenAI. "
            "Crea un archivo `.env` con tu clave:\n\n"
            "`OPENAI_API_KEY=sk-tu-clave-aqui`"
        )
        st.stop()

    # Inicializar session state
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    if "tickets" not in st.session_state:
        st.session_state.tickets = cargar_tickets_guardados()

    # Verificar si se cerró sesión
    if st.session_state.get("sesion_cerrada"):
        st.title("👋 Sesión finalizada")
        st.info("Gracias por usar Soporte-IT. Cierra esta pestaña o recarga la página para volver a empezar.")
        st.stop()

    # Sidebar
    email_usuario = mostrar_sidebar()

    # Construir RAG y Agente
    vectorstore = construir_base_vectorial()
    agente = crear_agente(vectorstore)

    # Encabezado del chat
    st.title("🛠️ Soporte-IT")
    st.caption("Asistente de soporte técnico Nivel 1 — NovaTech Solutions S.A.C.")

    # Pestañas: Chat y Dashboard
    tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

    with tab_dashboard:
        mostrar_dashboard()

    with tab_chat:
    # Mostrar historial de mensajes
        for msg in st.session_state.mensajes:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Mensaje de bienvenida si no hay historial
        if not st.session_state.mensajes:
            with st.chat_message("assistant"):
                bienvenida = (
                    "¡Hola! Soy **Soporte-IT**, tu asistente de soporte técnico. "
                    "Puedo ayudarte con:\n\n"
                    "- 🔑 Contraseñas y desbloqueo de cuentas\n"
                    "- 🌐 Configuración de VPN\n"
                    "- 📧 Problemas con el correo electrónico\n"
                    "- 🖨️ Impresoras y periféricos\n"
                    "- 📶 Conexión Wi-Fi y red corporativa\n"
                    "- 💼 Microsoft 365 (Teams, OneDrive, SharePoint)\n\n"
                    "Antes de empezar, asegúrate de escribir tu correo corporativo "
                    "en el panel izquierdo. ¿En qué puedo ayudarte?"
                )
                st.markdown(bienvenida)

        # Input del usuario
        if pregunta := st.chat_input("Describe tu problema técnico..."):

            # Validar que el email esté configurado
            if not email_usuario or "@" not in email_usuario:
                st.warning(
                    "Por favor, ingresa tu correo corporativo en el panel "
                    "izquierdo antes de continuar."
                )
                st.stop()

            # Agregar mensaje del usuario al historial
            st.session_state.mensajes.append({
                "role": "user",
                "content": pregunta
            })
            with st.chat_message("user"):
                st.markdown(pregunta)

            # Procesar con el agente
            with st.chat_message("assistant"):
               with st.spinner("Buscando en la documentación..."):
                    try:
                        # Buscar en documentos (RAG)
                        docs = vectorstore.similarity_search(pregunta, k=4)
                        contexto = "\n\n---\n\n".join(
                            [doc.page_content for doc in docs]
                        )

                        consulta_enriquecida = (
                            f"[DOCUMENTACIÓN RELEVANTE:]\n{contexto}\n\n"
                            f"[CONSULTA DEL USUARIO:]\n{pregunta}"
                        )

                        # Generar respuesta con el LLM
                        llm = ChatOpenAI(model=MODELO_LLM, temperature=0.0)
                        historial = obtener_historial_chat()

                        mensajes_llm = [
                            ("system", SYSTEM_PROMPT),
                        ]
                        for msg in historial:
                            if isinstance(msg, HumanMessage):
                                mensajes_llm.append(("human", msg.content))
                            elif isinstance(msg, AIMessage):
                                mensajes_llm.append(("assistant", msg.content))
                        mensajes_llm.append(("human", consulta_enriquecida))

                        respuesta = llm.invoke(mensajes_llm)
                        texto_respuesta = respuesta.content
                        st.markdown(texto_respuesta)

                        # Clasificar si fue resuelto o escalado
                        clasificacion = llm.invoke([
                            ("system", "Responde SOLO con la palabra 'resuelto' o 'escalado'."),
                            ("human",
                             f"Según esta respuesta de soporte TI, "
                             f"¿el problema fue resuelto o escalado?\n\n"
                             f"{texto_respuesta}")
                        ])
                        estado = clasificacion.content.strip().lower()

                        if "escal" in estado:
                            escalar_resultado = crear_herramientas()[2].invoke({
                                "resumen": pregunta,
                                "email_usuario": email_usuario,
                                "razon": "Requiere atención de Nivel 2"
                            })
                        else:
                            crear_herramientas()[0].invoke({
                                "resumen": pregunta,
                                "estado": "resuelto",
                                "email_usuario": email_usuario,
                                "categoria": "soporte general"
                            })
                            crear_herramientas()[1].invoke({
                                "email_usuario": email_usuario,
                                "asunto": f"Solución: {pregunta[:50]}",
                                "cuerpo": texto_respuesta
                            })

                    except Exception as e:
                        texto_respuesta = (
                            "Lo siento, ocurrió un error al procesar tu consulta. "
                            "Por favor, intenta de nuevo o contacta a la mesa de "
                            f"ayuda al interno 5000.\n\n*Error: {str(e)}*"
                        )
                        st.error(texto_respuesta)

            # Guardar respuesta en historial
            st.session_state.mensajes.append({
                "role": "assistant",
                "content": texto_respuesta
            })

            # Rerun para actualizar sidebar con nuevos tickets
            st.rerun()


if __name__ == "__main__":
    main()

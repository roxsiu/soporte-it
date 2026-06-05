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
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

SYSTEM_PROMPT = """Eres IT-Copilot, el asistente virtual de soporte técnico de Nivel 1 \
de NovaTech Solutions S.A.C. Tu rol es ayudar a los colaboradores a resolver \
problemas técnicos comunes usando EXCLUSIVAMENTE la documentación oficial de la empresa.

REGLAS ESTRICTAS:
1. SIEMPRE usa la herramienta 'buscar_en_documentos' PRIMERO para buscar información \
   relevante antes de responder cualquier consulta técnica.
2. Responde ÚNICAMENTE con información encontrada en los documentos. NUNCA inventes \
   soluciones, comandos o procedimientos que no estén en la documentación.
3. Si encuentras la solución, responde con pasos claros y numerados, luego:
   - Usa 'registrar_ticket' con estado 'resuelto'
   - Usa 'enviar_email' para notificar al usuario
4. Si el problema coincide con un CRITERIO DE ESCALAMIENTO descrito en los documentos, \
   o si NO encuentras información relevante:
   - Informa al usuario que su caso será atendido por un técnico especializado
   - Usa 'escalar_ticket' para derivar el caso
5. Si el usuario pregunta algo que NO es de soporte técnico (clima, recetas, etc.), \
   responde amablemente que solo puedes ayudar con temas de soporte TI.
6. Responde siempre en español.
7. Sé profesional pero amable. Usa un tono de servicio al cliente.

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

def crear_herramientas(vectorstore: Chroma) -> list:
    """
    Define las herramientas que el agente puede usar de forma autónoma.

    Args:
        vectorstore: Base de datos vectorial para búsquedas RAG.

    Returns:
        Lista de herramientas configuradas.
    """

    @tool
    def buscar_en_documentos(consulta: str) -> str:
        """Busca información relevante en los manuales de soporte TI de NovaTech Solutions.
        Usa esta herramienta SIEMPRE como primer paso para responder consultas técnicas.
        Devuelve los fragmentos de documentación más relevantes.

        Args:
            consulta: La pregunta o problema técnico del usuario.
        """
        resultados = vectorstore.similarity_search(consulta, k=TOP_K)
        if not resultados:
            return "NO SE ENCONTRÓ INFORMACIÓN RELEVANTE en los documentos de soporte."

        fragmentos = []
        for i, doc in enumerate(resultados, 1):
            fuente = doc.metadata.get("source", "Desconocido")
            nombre_archivo = Path(fuente).stem if fuente else "Desconocido"
            fragmentos.append(
                f"[Fragmento {i} - Fuente: {nombre_archivo}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(fragmentos)

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

        # Guardar en archivo JSON
        archivo = os.path.join(DIRECTORIO_TICKETS, "tickets.json")
        tickets_existentes = []
        if os.path.exists(archivo):
            with open(archivo, "r", encoding="utf-8") as f:
                tickets_existentes = json.load(f)

        tickets_existentes.append(ticket)
        with open(archivo, "w", encoding="utf-8") as f:
            json.dump(tickets_existentes, f, ensure_ascii=False, indent=2)

        # Guardar en session_state para mostrar en sidebar
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

        # Guardar registro del email
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
        Usa esta herramienta cuando el problema requiera intervención de un técnico
        especializado o coincida con criterios de escalamiento de los manuales.

        Args:
            resumen: Descripción del problema que se escala.
            email_usuario: Correo del usuario afectado.
            razon: Motivo específico por el cual se escala (ej: requiere acceso administrativo, falla de hardware, incidente de seguridad).
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

    return [buscar_en_documentos, registrar_ticket, enviar_email, escalar_ticket]


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
    herramientas = crear_herramientas(_vectorstore)

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
        st.title("IT-Copilot")
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


# ============================================================
# APLICACIÓN PRINCIPAL
# ============================================================

def main():
    st.set_page_config(
        page_title="IT-Copilot · NovaTech Solutions",
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

    # Sidebar
    email_usuario = mostrar_sidebar()

    # Construir RAG y Agente
    vectorstore = construir_base_vectorial()
    agente = crear_agente(vectorstore)

    # Encabezado del chat
    st.title("🛠️ IT-Copilot")
    st.caption(
        "Asistente de soporte técnico Nivel 1 — NovaTech Solutions S.A.C."
    )

    # Mostrar historial de mensajes
    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Mensaje de bienvenida si no hay historial
    if not st.session_state.mensajes:
        with st.chat_message("assistant"):
            bienvenida = (
                "¡Hola! Soy **IT-Copilot**, tu asistente de soporte técnico. "
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
                    # Inyectar email del usuario en la consulta para que
                    # el agente lo use al registrar tickets y enviar emails
                    consulta_enriquecida = (
                        f"[Email del usuario: {email_usuario}]\n\n"
                        f"Consulta: {pregunta}"
                    )

                    respuesta = agente.invoke({
                        "input": consulta_enriquecida,
                        "chat_history": obtener_historial_chat()
                    })

                    texto_respuesta = respuesta.get("output", "")
                    st.markdown(texto_respuesta)

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

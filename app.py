"""
IT-Copilot: Agente Inteligente de Soporte TI Nivel 1
=====================================================
Sistema de resolución autónoma de tickets mediante
arquitectura RAG y agente con herramientas (LangChain).

Autores: Roxana Siu Loo - Sebastián Rodríguez
Curso:   Proyecto Integrador - Diploma AI Engineer
Fecha:   Julio 2026
"""

import streamlit as st
import os
import json
import uuid
import time
import re
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
from openai import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    BadRequestError,
)

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
DOMINIO_EMPRESA = "novatech.com"

# Costos por 1,000 tokens (USD) - GPT-4o-mini
COSTO_INPUT_POR_1K = 0.00015
COSTO_OUTPUT_POR_1K = 0.0006

# Crear directorios necesarios
Path(DIRECTORIO_TICKETS).mkdir(exist_ok=True)


# ============================================================
# PROMPT DEL SISTEMA
# ============================================================

SYSTEM_PROMPT = """Eres IT-Copilot, el asistente virtual de soporte técnico de Nivel 1 \
de NovaTech Solutions S.A.C.

Se te proporcionará DOCUMENTACIÓN RELEVANTE junto con la consulta del usuario. \
Usa esa documentación para responder.

INSTRUCCIONES:
1. Lee la documentación proporcionada.
2. Si la documentación contiene la solución:
   - Escribe en tu respuesta los pasos COMPLETOS y NUMERADOS de la solución.
   - Incluye las URLs, nombres de sistemas y detalles específicos que \
     aparezcan en la documentación.
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
- NO asumas detalles que el usuario no haya mencionado. Responde solo con \
  lo que el usuario dijo y lo que dicen los documentos.
- Si el usuario pregunta algo que NO es de soporte TI, responde ÚNICAMENTE \
  que solo puedes ayudar con temas de soporte técnico de NovaTech. NO \
  proporciones la respuesta a la pregunta bajo ninguna circunstancia, ni \
  siquiera como comentario adicional o "por si acaso".    \
- Responde en español, tono profesional y amable.
- Si el usuario mezcla una pregunta técnica con una no técnica, \
  responde SOLO la parte técnica. IGNORA completamente la parte no técnica, \
  no la menciones ni la respondas.

FORMATO DE RESPUESTA CUANDO HAY SOLUCIÓN:
- Saludo breve
- Pasos numerados de la solución
- Nota adicional si aplica
- Mensaje de cierre
- NUNCA menciones escalamiento ni técnicos si estás dando una solución.
  Si tienes pasos para dar, el caso ES resuelto, no escalado.

FORMATO CUANDO SE ESCALA:
- Reconoce el problema del usuario
- Explica brevemente por qué requiere atención especializada
- Informa que un técnico se comunicará pronto
- NUNCA des pasos de solución si vas a escalar. Solo escala.
"""


# ============================================================
# PROTECCIÓN CONTRA PROMPT INJECTION
# ============================================================

PATRONES_INJECTION = [
    r"ignora\s+(tus|las)\s+instrucciones",
    r"ignore\s+(your|all)\s+instructions",
    r"olvida\s+(tus|las)\s+(reglas|instrucciones)",
    r"forget\s+(your|all)\s+(rules|instructions)",
    r"actua\s+como\s+si",
    r"act\s+as\s+if",
    r"pretend\s+(you|to\s+be)",
    r"nuevo\s+rol",
    r"new\s+role",
    r"eres\s+ahora",
    r"you\s+are\s+now",
    r"system\s*prompt",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"modo\s+desarrollador",
]


def detectar_prompt_injection(texto: str) -> bool:
    """
    Analiza el texto del usuario para detectar intentos de
    manipulación del prompt (prompt injection).

    Args:
        texto: Mensaje del usuario.

    Returns:
        True si se detecta un intento de injection.
    """
    texto_lower = texto.lower()
    for patron in PATRONES_INJECTION:
        if re.search(patron, texto_lower):
            return True
    return False


# ============================================================
# VALIDACIÓN DE CORREO CORPORATIVO
# ============================================================

def validar_email_corporativo(email: str) -> bool:
    """
    Verifica que el correo pertenezca al dominio corporativo.

    Args:
        email: Dirección de correo a validar.

    Returns:
        True si el email es válido y del dominio correcto.
    """
    if not email or "@" not in email:
        return False
    dominio = email.split("@")[1].lower().strip()
    return dominio == DOMINIO_EMPRESA


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

    loader = PyPDFDirectoryLoader(DIRECTORIO_DATOS)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documentos)

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
    Define las herramientas de acción del agente.
    La búsqueda RAG se realiza antes de llamar al agente,
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

        # Campo para email del usuario con validación visual
        email = st.text_input(
            "Tu correo corporativo",
            placeholder="nombre@novatech.com",
            key="email_usuario"
        )

        # Mostrar estado de validación del email
        if email:
            if validar_email_corporativo(email):
                st.success("✓ Correo válido")
            else:
                st.error(f"✗ Solo se permiten correos @{DOMINIO_EMPRESA}")

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
                elif estado == "escalado":
                    icono = "🔴"
                else:
                    icono = "⚪"

                with st.expander(f"{icono} {tk['id']} - {estado.upper()}"):
                    st.write(f"**Fecha:** {tk['fecha']}")
                    st.write(f"**Usuario:** {tk['email_usuario']}")
                    st.write(f"**Resumen:** {tk['resumen']}")
                    if "razon_escalamiento" in tk:
                        st.write(
                            f"**Razón de escalamiento:** "
                            f"{tk['razon_escalamiento']}"
                        )

        st.divider()

        # Botón para limpiar conversación
        if st.button("Nueva conversación", use_container_width=True):
            st.session_state.mensajes = []
            st.session_state.metricas = []
            st.rerun()

        # Botón para cerrar sesión
        if st.button(
            "Cerrar sesión",
            use_container_width=True,
            type="secondary"
        ):
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


# ============================================================
# DASHBOARD
# ============================================================

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

    st.divider()

    # --- Métricas de rendimiento (costo y latencia) ---
    st.subheader("Rendimiento del sistema")
    metricas = st.session_state.get("metricas", [])

    if metricas:
        df_metricas = pd.DataFrame(metricas)

        col_m1, col_m2, col_m3 = st.columns(3)

        tokens_total_in = df_metricas["tokens_input"].sum()
        tokens_total_out = df_metricas["tokens_output"].sum()
        costo_total = df_metricas["costo_usd"].sum()
        latencia_prom = df_metricas["latencia_seg"].mean()

        col_m1.metric(
            "Tokens usados",
            f"{tokens_total_in + tokens_total_out:,}",
            help=f"Input: {tokens_total_in:,} | Output: {tokens_total_out:,}"
        )
        col_m2.metric(
            "Costo acumulado",
            f"${costo_total:.4f} USD"
        )
        col_m3.metric(
            "Latencia promedio",
            f"{latencia_prom:.1f} seg"
        )

        st.dataframe(
            df_metricas[["consulta", "tokens_input", "tokens_output",
                         "costo_usd", "latencia_seg"]].rename(
                columns={
                    "consulta": "Consulta",
                    "tokens_input": "Tokens entrada",
                    "tokens_output": "Tokens salida",
                    "costo_usd": "Costo (USD)",
                    "latencia_seg": "Latencia (seg)"
                }
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Las métricas de costo y latencia aparecerán aquí "
            "después de realizar consultas en el chat."
        )


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
    if "metricas" not in st.session_state:
        st.session_state.metricas = []

    # Verificar si se cerró sesión
    if st.session_state.get("sesion_cerrada"):
        st.title("👋 Sesión finalizada")
        st.info(
            "Gracias por usar IT-Copilot. Cierra esta pestaña "
            "o recarga la página para volver a empezar."
        )
        st.stop()

    # Sidebar
    email_usuario = mostrar_sidebar()

    # Construir RAG
    vectorstore = construir_base_vectorial()

    # Encabezado
    st.title("🛠️ IT-Copilot")
    st.caption(
        "Asistente de soporte técnico Nivel 1 — NovaTech Solutions S.A.C."
    )

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
                    "¡Hola! Soy **IT-Copilot**, tu asistente de soporte "
                    "técnico. Puedo ayudarte con:\n\n"
                    "- 🔑 Contraseñas y desbloqueo de cuentas\n"
                    "- 🌐 Configuración de VPN\n"
                    "- 📧 Problemas con el correo electrónico\n"
                    "- 🖨️ Impresoras y periféricos\n"
                    "- 📶 Conexión Wi-Fi y red corporativa\n"
                    "- 💼 Microsoft 365 (Teams, OneDrive, SharePoint)\n\n"
                    "Antes de empezar, asegúrate de escribir tu correo "
                    "corporativo (@novatech.com) en el panel izquierdo. "
                    "¿En qué puedo ayudarte?"
                )
                st.markdown(bienvenida)

        # Input del usuario
        if pregunta := st.chat_input("Describe tu problema técnico..."):

            # Validar que el email esté configurado y sea corporativo
            if not email_usuario:
                st.warning(
                    "Por favor, ingresa tu correo corporativo en el "
                    "panel izquierdo antes de continuar."
                )
                st.stop()

            if not validar_email_corporativo(email_usuario):
                st.warning(
                    f"Solo se permiten correos del dominio "
                    f"@{DOMINIO_EMPRESA}. Verifica tu correo en el "
                    f"panel izquierdo."
                )
                st.stop()

            # Verificar prompt injection
            if detectar_prompt_injection(pregunta):
                st.session_state.mensajes.append({
                    "role": "user",
                    "content": pregunta
                })
                with st.chat_message("user"):
                    st.markdown(pregunta)

                respuesta_seguridad = (
                    "⚠️ He detectado que tu mensaje contiene "
                    "instrucciones que podrían intentar modificar mi "
                    "comportamiento. Por seguridad, no puedo procesar "
                    "esta solicitud.\n\n"
                    "Si tienes un problema técnico legítimo, por favor "
                    "descríbelo normalmente y con gusto te ayudaré."
                )
                with st.chat_message("assistant"):
                    st.markdown(respuesta_seguridad)
                st.session_state.mensajes.append({
                    "role": "assistant",
                    "content": respuesta_seguridad
                })
                st.rerun()

            # Agregar mensaje del usuario al historial
            st.session_state.mensajes.append({
                "role": "user",
                "content": pregunta
            })
            with st.chat_message("user"):
                st.markdown(pregunta)

            # Procesar consulta
            with st.chat_message("assistant"):
                with st.spinner("Buscando en la documentación..."):
                    try:
                        # Medir latencia
                        inicio = time.time()

                        # Buscar en documentos (RAG)
                        docs = vectorstore.similarity_search(
                            pregunta, k=TOP_K
                        )
                        contexto = "\n\n---\n\n".join(
                            [doc.page_content for doc in docs]
                        )

                        consulta_enriquecida = (
                            f"[DOCUMENTACIÓN RELEVANTE:]\n{contexto}"
                            f"\n\n[CONSULTA DEL USUARIO:]\n{pregunta}"
                        )

                        # Generar respuesta con el LLM
                        llm = ChatOpenAI(
                            model=MODELO_LLM,
                            temperature=0.0
                        )
                        historial = obtener_historial_chat()

                        mensajes_llm = [("system", SYSTEM_PROMPT)]
                        for msg in historial:
                            if isinstance(msg, HumanMessage):
                                mensajes_llm.append(
                                    ("human", msg.content)
                                )
                            elif isinstance(msg, AIMessage):
                                mensajes_llm.append(
                                    ("assistant", msg.content)
                                )
                        mensajes_llm.append(
                            ("human", consulta_enriquecida)
                        )

                        respuesta = llm.invoke(mensajes_llm)
                        texto_respuesta = respuesta.content

                        # Calcular latencia
                        latencia = round(time.time() - inicio, 2)

                        # Obtener uso de tokens
                        tokens_in = respuesta.response_metadata.get(
                            "token_usage", {}
                        ).get("prompt_tokens", 0)
                        tokens_out = respuesta.response_metadata.get(
                            "token_usage", {}
                        ).get("completion_tokens", 0)
                        costo = round(
                            (tokens_in / 1000) * COSTO_INPUT_POR_1K
                            + (tokens_out / 1000) * COSTO_OUTPUT_POR_1K,
                            6
                        )

                        # Guardar métricas
                        st.session_state.metricas.append({
                            "consulta": pregunta[:60],
                            "tokens_input": tokens_in,
                            "tokens_output": tokens_out,
                            "costo_usd": costo,
                            "latencia_seg": latencia
                        })


                        # Clasificar si fue resuelto o escalado
                        # Si la respuesta tiene pasos numerados, es resuelto
                        tiene_pasos = any(
                            f"{i}." in texto_respuesta
                            for i in range(1, 10)
                        )
                        frases_escalamiento = [
                            "atención especializada",
                            "técnico especializado",
                            "equipo de nivel 2",
                            "escalar tu caso",
                            "derivar tu caso"
                        ]
                        es_escalamiento_puro = any(
                            frase in texto_respuesta.lower()
                            for frase in frases_escalamiento
                        ) and not tiene_pasos

                        if tiene_pasos:
                            estado = "resuelto"
                        elif es_escalamiento_puro:
                            estado = "escalado"
                        else:
                            estado = "resuelto"

                        # Agregar métricas al texto de respuesta
                        linea_metricas = (
                            f"\n\n---\n"
                            f"⚡ {latencia}s · "
                            f"📊 {tokens_in + tokens_out} tokens · "
                            f"💰 ${costo:.4f} USD"
                        )
                        texto_respuesta += linea_metricas

                        st.markdown(texto_respuesta)

                        herramientas = crear_herramientas()

                        if "escal" in estado:
                            herramientas[2].invoke({
                                "resumen": pregunta,
                                "email_usuario": email_usuario,
                                "razon": "Requiere atención de Nivel 2"
                            })
                        else:
                            herramientas[0].invoke({
                                "resumen": pregunta,
                                "estado": "resuelto",
                                "email_usuario": email_usuario,
                                "categoria": "soporte general"
                            })
                            herramientas[1].invoke({
                                "email_usuario": email_usuario,
                                "asunto": f"Solución: {pregunta[:50]}",
                                "cuerpo": texto_respuesta
                            })

                    except AuthenticationError:
                        texto_respuesta = (
                            "🔑 **Error de autenticación.** La API key "
                            "de OpenAI no es válida o ha expirado. "
                            "Verifica tu archivo `.env`."
                        )
                        st.error(texto_respuesta)

                    except RateLimitError:
                        texto_respuesta = (
                            "⏳ **Límite de uso alcanzado.** Se agotaron "
                            "los créditos de la API de OpenAI o se "
                            "excedió el límite de solicitudes. Espera "
                            "unos minutos e intenta de nuevo."
                        )
                        st.error(texto_respuesta)

                    except APITimeoutError:
                        texto_respuesta = (
                            "⌛ **Tiempo de espera agotado.** El "
                            "servidor de OpenAI tardó demasiado en "
                            "responder. Intenta de nuevo."
                        )
                        st.error(texto_respuesta)

                    except APIConnectionError:
                        texto_respuesta = (
                            "🌐 **Error de conexión.** No se pudo "
                            "conectar con el servidor de OpenAI. "
                            "Verifica tu conexión a internet."
                        )
                        st.error(texto_respuesta)

                    except BadRequestError as e:
                        texto_respuesta = (
                            "❌ **Error en la solicitud.** La consulta "
                            "no pudo ser procesada por el modelo. "
                            f"Detalle: {str(e)}"
                        )
                        st.error(texto_respuesta)

                    except Exception as e:
                        texto_respuesta = (
                            "Lo siento, ocurrió un error inesperado al "
                            "procesar tu consulta. Por favor, intenta "
                            "de nuevo o contacta a la mesa de ayuda al "
                            f"interno 5000.\n\n*Error: {str(e)}*"
                        )
                        st.error(texto_respuesta)

            # Guardar respuesta en historial
            st.session_state.mensajes.append({
                "role": "assistant",
                "content": texto_respuesta
            })

            st.rerun()


if __name__ == "__main__":
    main()

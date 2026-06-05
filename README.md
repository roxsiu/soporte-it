# 🛠️ Soporte-IT

**Agente Inteligente de Soporte TI Nivel 1 mediante RAG**

Sistema de resolución autónoma de tickets de soporte técnico para NovaTech Solutions S.A.C. Utiliza una arquitectura de Generación Aumentada por Recuperación (RAG) combinada con un agente inteligente capaz de buscar soluciones, registrar tickets, enviar notificaciones y escalar casos complejos.

## Arquitectura

```
Usuario → Streamlit Chat → Agente LangChain → Herramientas
                                   │
                           ┌───────┼───────┐
                           │       │       │
                      Buscar en  Registrar  Escalar a
                      documentos  ticket    Nivel 2
                      (RAG)       + Email
                           │
                    ChromaDB ← PDFs de Soporte
```

## Requisitos

- Python 3.10 o superior
- Clave de API de OpenAI

## Instalación

### En GitHub Codespaces

1. Abre el repositorio en Codespaces.

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura tu API key de OpenAI:
```bash
cp .env.example .env
```
Abre el archivo `.env` y reemplaza `sk-tu-clave-aqui` con tu clave real.

4. Verifica que los manuales PDF estén en la carpeta `data/`:
```bash
ls data/
```
Deberías ver los 6 manuales de soporte (MAN-TI-001 a MAN-TI-006).

5. Ejecuta la aplicación:
```bash
streamlit run app.py
```

### En local

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/it-copilot.git
cd it-copilot
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# o bien: venv\Scripts\activate # Windows
```

3. Sigue los pasos 2 a 5 de la sección anterior.

## Estructura del Proyecto

```
it-copilot/
├── app.py              ← Aplicación principal (Streamlit + RAG + Agente)
├── requirements.txt    ← Dependencias de Python
├── .env.example        ← Plantilla para la API key
├── .gitignore          ← Archivos excluidos de Git
├── README.md           ← Este archivo
├── data/               ← Manuales de soporte en PDF (entrada del RAG)
│   ├── MAN-TI-001_Contrasenas_y_Cuentas.pdf
│   ├── MAN-TI-002_VPN_GlobalProtect.pdf
│   ├── MAN-TI-003_Correo_Electronico.pdf
│   ├── MAN-TI-004_Impresoras_y_Perifericos.pdf
│   ├── MAN-TI-005_WiFi_y_Red_Corporativa.pdf
│   └── MAN-TI-006_Microsoft_365.pdf
├── chroma_db/          ← Base vectorial (se genera automáticamente, no se sube a Git)
└── tickets/            ← Tickets registrados (se genera automáticamente, no se sube a Git)
```

## Uso

1. Escribe tu correo corporativo en el panel izquierdo.
2. Describe tu problema técnico en el chat.
3. IT-Copilot buscará en los manuales y te dará una solución paso a paso.
4. El ticket se registrará automáticamente y podrás verlo en el panel izquierdo.

## Dashboard

La pestaña **📊 Dashboard** muestra en tiempo real:
- Métricas de tickets: total, resueltos, escalados y tasa de resolución.
- Gráfico de tickets por estado.
- Gráfico de tickets por categoría.
- Historial completo de tickets en tabla.

Los datos se generan automáticamente con cada interacción en el chat y se almacenan en `tickets/tickets.json`.   

## Tecnologías

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit |
| Orquestación | LangChain |
| Base Vectorial | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |

## Bloques del Diploma Integrados

1. **Chatbots con LLM + RAG:** Pipeline de recuperación y generación de respuestas basado en documentación.
2. **Agentes Inteligentes:** Agente con herramientas autónomas para registro de tickets, notificación por email y escalamiento.

## Autores

- Roxana Siu Loo
- Sebastian Rodriguez

Proyecto Integrador · Diploma AI Engineer · DMC Institute · 2026

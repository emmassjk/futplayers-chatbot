# ⚽ FutPlayers.cl — Chatbot de Ventas con IA

**ISY0101 · Ingeniería de Soluciones con IA · DuocUC 2025**

Chatbot de ventas para FutPlayers.cl, tienda chilena de poleras de fútbol.
Implementa una arquitectura completa con LLM + RAG + Agente con Memoria.

---

## 🏗️ Arquitectura

```
Usuario
  │
  ▼
Streamlit (app.py)
  │
  ▼
FutPlayersAgent (src/4_agente.py)
  ├── Retriever RAG ──► ChromaDB ──► catalogo.csv + politicas.txt
  ├── Prompt Engine ──► System prompt + Few-Shot + CoT
  ├── LLM ────────────► GitHub Models API (gpt-4o-mini)
  └── Memoria ────────► ConversationBufferMemory (sesión)
```

## 📁 Estructura del Proyecto

```
futplayers-chatbot/
├── app.py                  # Interfaz Streamlit (chatbot)
├── requirements.txt        # Dependencias
├── .env.example            # Variables de entorno (plantilla)
├── data/
│   ├── catalogo.csv        # 12 productos con precios y stock
│   └── politicas.txt       # Políticas de envío, pago, devoluciones
└── src/
    ├── 1_conexion_llm.py   # Conexión LangChain + GitHub Models (IL1.1)
    ├── 2_prompts.py        # Prompt Engineering: zero-shot, few-shot, CoT (IL1.2)
    ├── 3_rag_pipeline.py   # Pipeline RAG: carga, chunking, embeddings, ChromaDB (IL1.3)
    ├── 4_agente.py         # Agente con memoria ConversationBufferMemory (RA2/IL2.1-2.2)
    └── 5_evaluacion.py     # Métricas RAG: precision, recall, faithfulness, relevancy (IL1.4)
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/futplayers-chatbot.git
cd futplayers-chatbot
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env y agrega tu GITHUB_TOKEN
```

Obtén tu token en: https://github.com/settings/tokens (sin permisos especiales necesarios)

### 4. Construir el índice RAG (solo la primera vez)

```bash
python src/3_rag_pipeline.py
```

Verás: `✅ Pipeline RAG listo.`

### 5. Lanzar el chatbot

```bash
streamlit run app.py
```

Se abrirá en http://localhost:8501

---

## 🧪 Pruebas individuales por módulo

```bash
# Probar conexión al LLM
python src/1_conexion_llm.py

# Verificar el vector store y retriever
python src/3_rag_pipeline.py

# Probar el agente en consola (sin UI)
python src/4_agente.py

# Ejecutar evaluación de métricas RAG
python src/5_evaluacion.py
```

---

## 📊 Métricas de Evaluación RAG

| Métrica | Descripción |
|---|---|
| Context Precision | % de docs recuperados que son relevantes |
| Context Recall | % de información necesaria efectivamente recuperada |
| Faithfulness | La respuesta se basa solo en el contexto real |
| Answer Relevancy | La respuesta responde directamente la pregunta |

Para ejecutar la evaluación completa desde la UI, usa el botón **"▶ Ejecutar Evaluación"** en el sidebar.

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|---|---|
| LLM | GitHub Models API (gpt-4o-mini) |
| Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | text-embedding-3-small |
| Memoria | ConversationBufferMemory |
| UI | Streamlit |

---

## ⚠️ Notas Importantes

- **No subas tu `.env`** al repositorio (está en `.gitignore`)
- El directorio `chroma_db/` se genera automáticamente; no es necesario subirlo
- Si actualizas `catalogo.csv`, vuelve a ejecutar `python src/3_rag_pipeline.py`

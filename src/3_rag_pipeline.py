"""
src/3_rag_pipeline.py
=====================
Pipeline RAG completo para FutPlayers.cl
Fuentes: catalogo.csv  |  politicas.txt  |  PDFs en data/pdfs/

MEJORA: SmartRetriever — detecta consultas por equipo/categoría y
        devuelve TODOS los productos del equipo desde el CSV directamente,
        evitando el límite k del vector store.
"""

import os
import csv
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PDFS_DIR = DATA_DIR / "pdfs"
DB_DIR   = BASE_DIR / "chroma_db"

CATALOGO_PATH  = DATA_DIR / "catalogo.csv"
POLITICAS_PATH = DATA_DIR / "politicas.txt"


# ─────────────────────────────────────────────────────────────
# MAPEO DE ALIAS DE EQUIPOS (para detectar en la pregunta)
# ─────────────────────────────────────────────────────────────

ALIAS_EQUIPOS = {
    "Universidad de Chile": [
        "universidad de chile", "la u", "la u de chile", "u de chile",
        "azul azul", "la u ", " la u", "u chile",
    ],
    "Colo-Colo": [
        "colo-colo", "colo colo", "el colo", "los albos", "cacique",
    ],
    "Universidad Católica": [
        "universidad católica", "universidad catolica", "la católica",
        "la catolica", "cruzados", "católica", "catolica",
    ],
    "Real Madrid": ["real madrid", "madrid"],
    "Barcelona": ["barcelona", "barça", "barca", "fcb"],
    "Manchester United": ["manchester united", "man united", "man utd", "united"],
    "Manchester City": ["manchester city", "man city", "city"],
    "PSG": ["psg", "paris saint germain", "paris saint-germain"],
    "Bayern Múnich": ["bayern", "bayern munich", "münchen"],
    "Chelsea": ["chelsea"],
    "Arsenal": ["arsenal"],
    "Sevilla": ["sevilla"],
    "AC Milan": ["ac milan", "milan"],
    "Santos": ["santos"],
    "River Plate": ["river plate", "river"],
}

# Palabras que indican categoría retro
PALABRAS_RETRO = ["retro", "antigua", "antiguo", "clásica", "clasica", "histórica",
                  "historica", "vintage", "throwback"]

# Palabras que indican categoría actual
PALABRAS_ACTUAL = ["actual", "nueva", "nuevo", "2026", "temporada actual", "última",
                   "ultima", "moderna"]

# Palabras que indican "muéstrame todo de X"
PALABRAS_TODO = ["todo", "todos", "todas", "completo", "completa", "catálogo",
                 "catalogo", "qué tienen", "que tienen", "opciones", "modelos",
                 "disponible", "disponibles", "tienen de", "tienes de"]


# ─────────────────────────────────────────────────────────────
# 1. CARGADOR DE CATÁLOGO CSV
# ─────────────────────────────────────────────────────────────

def cargar_catalogo() -> list[Document]:
    documentos = []
    with open(CATALOGO_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            precio     = int(fila["precio_clp"])
            precio_fmt = f"${precio:,}".replace(",", ".")

            stock_por_talla = (
                f"S:{fila.get('stock_S','?')} "
                f"M:{fila.get('stock_M','?')} "
                f"L:{fila.get('stock_L','?')} "
                f"XL:{fila.get('stock_XL','?')}"
            )

            jugador  = fila.get("jugador", "").strip()
            tipo     = fila.get("tipo", "").strip()
            color    = fila.get("color", "").strip()
            producto = fila.get("producto", "Camiseta").strip()
            temporada = fila.get("temporada", "").strip()

            # Marcar explícitamente si es retro o actual en el texto
            es_retro = "Camiseta Retro" in producto or (
                temporada.isdigit() and int(temporada) < 2026
            )
            categoria = "RETRO" if es_retro else "ACTUAL 2026"

            texto = (
                f"Producto: {producto} {fila['equipo']} {jugador} {color} {tipo} | "
                f"Categoría: {categoria} | "
                f"Liga: {fila['liga']} ({fila['pais']}) | "
                f"Tallas disponibles: {fila['tallas_disponibles']} | "
                f"Stock por talla — {stock_por_talla} | "
                f"Precio: {precio_fmt} CLP | "
                f"Temporada/Año: {temporada}"
            ).replace("  ", " ").strip()

            doc = Document(
                page_content=texto,
                metadata={
                    "fuente":    "catalogo",
                    "equipo":    fila["equipo"],
                    "jugador":   jugador,
                    "precio":    precio,
                    "color":     color,
                    "temporada": temporada,
                    "categoria": categoria,
                    "producto":  producto,
                }
            )
            documentos.append(doc)

    print(f"✅ Catálogo cargado: {len(documentos)} productos")
    return documentos


# ─────────────────────────────────────────────────────────────
# 2. CARGADOR DE POLÍTICAS
# ─────────────────────────────────────────────────────────────

def cargar_politicas() -> list[Document]:
    with open(POLITICAS_PATH, encoding="utf-8") as f:
        contenido = f.read()
    doc = Document(page_content=contenido, metadata={"fuente": "politicas"})
    print(f"✅ Políticas cargadas: {len(contenido)} caracteres")
    return [doc]


# ─────────────────────────────────────────────────────────────
# 3. CARGADOR DE PDFs
# ─────────────────────────────────────────────────────────────

def cargar_pdfs() -> list[Document]:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = list(PDFS_DIR.glob("*.pdf"))

    if not pdfs:
        print("ℹ️  No hay PDFs en data/pdfs/ — se omite esta fuente.")
        return []

    try:
        from pypdf import PdfReader
    except ImportError:
        print("⚠️  Instala pypdf:  pip install pypdf")
        return []

    documentos = []
    for pdf_path in pdfs:
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                texto = (page.extract_text() or "").strip()
                if not texto:
                    continue
                doc = Document(
                    page_content=texto,
                    metadata={"fuente": "pdf", "archivo": pdf_path.name, "pagina": i + 1}
                )
                documentos.append(doc)
            print(f"✅ PDF procesado: {pdf_path.name} ({len(reader.pages)} pág.)")
        except Exception as e:
            print(f"⚠️  Error con {pdf_path.name}: {e}")

    print(f"✅ Total documentos desde PDFs: {len(documentos)}")
    return documentos


# ─────────────────────────────────────────────────────────────
# 4. CHUNKING / EMBEDDINGS / VECTOR STORE
# ─────────────────────────────────────────────────────────────

def dividir_documentos(documentos: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    print(f"✅ Chunking: {len(chunks)} chunks generados")
    return chunks


def crear_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com"),
        api_key=os.getenv("GITHUB_TOKEN"),
        model="text-embedding-3-small",
    )


def construir_vector_store(chunks: list[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    import shutil
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("🔄 Vector store anterior eliminado — reconstruyendo...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="futplayers_catalogo",
    )
    print(f"✅ Vector store creado: {len(chunks)} chunks en {DB_DIR}")
    return vector_store


def cargar_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    if not DB_DIR.exists():
        raise FileNotFoundError(
            "Vector store no encontrado.\n"
            "Ejecuta primero:  python src/3_rag_pipeline.py"
        )
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="futplayers_catalogo",
    )


# ─────────────────────────────────────────────────────────────
# 5. SMART RETRIEVER — detecta equipo/categoría y usa CSV directo
# ─────────────────────────────────────────────────────────────

class SmartRetriever:
    """
    Retriever inteligente:
    - Si detecta un equipo + categoría (retro/actual/todo) en la pregunta,
      lee TODOS los productos de ese equipo directamente del CSV filtrado.
    - Si no, usa el vector store con k=12 para mayor cobertura.
    """

    def __init__(self, vector_store: Chroma, todos_los_docs: list[Document]):
        self.vector_store  = vector_store
        self.todos_los_docs = [d for d in todos_los_docs if d.metadata.get("fuente") == "catalogo"]
        self.retriever_base = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 12},
        )

    def _detectar_equipo(self, pregunta: str) -> str | None:
        p = pregunta.lower()
        for equipo, alias in ALIAS_EQUIPOS.items():
            for a in alias:
                if a in p:
                    return equipo
        return None

    def _detectar_categoria(self, pregunta: str) -> str | None:
        """Retorna 'RETRO', 'ACTUAL 2026', o None (= mostrar todo)."""
        p = pregunta.lower()
        if any(w in p for w in PALABRAS_RETRO):
            return "RETRO"
        if any(w in p for w in PALABRAS_ACTUAL):
            return "ACTUAL 2026"
        return None

    def _es_consulta_de_equipo(self, pregunta: str) -> bool:
        """True si la pregunta pide el listado de un equipo completo."""
        p = pregunta.lower()
        return any(w in p for w in PALABRAS_TODO) or self._detectar_equipo(pregunta) is not None

    def invoke(self, pregunta: str) -> list[Document]:
        equipo    = self._detectar_equipo(pregunta)
        categoria = self._detectar_categoria(pregunta)

        if equipo:
            # Filtra directamente desde el CSV en memoria
            resultado = [
                d for d in self.todos_los_docs
                if d.metadata.get("equipo") == equipo
            ]
            if categoria:
                resultado = [
                    d for d in resultado
                    if d.metadata.get("categoria") == categoria
                ]
            if resultado:
                print(f"🎯 SmartRetriever: {len(resultado)} productos de '{equipo}'"
                      f"{' [' + categoria + ']' if categoria else ''}")
                return resultado

        # Fallback al vector store con k=12
        docs = self.retriever_base.invoke(pregunta)
        print(f"🔍 VectorStore fallback: {len(docs)} chunks recuperados")
        return docs


# ─────────────────────────────────────────────────────────────
# 6. INICIALIZACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────

def inicializar_rag():
    print("\n🚀 Inicializando pipeline RAG de FutPlayers.cl...\n")
    catalogo  = cargar_catalogo()
    politicas = cargar_politicas()
    pdfs      = cargar_pdfs()
    todos     = catalogo + politicas + pdfs

    chunks       = dividir_documentos(todos)
    embeddings   = crear_embeddings()
    vector_store = construir_vector_store(chunks, embeddings)
    retriever    = SmartRetriever(vector_store, todos)

    print("\n✅ Pipeline RAG listo.\n")
    return retriever, embeddings


def obtener_retriever():
    catalogo     = cargar_catalogo()
    politicas    = cargar_politicas()
    todos        = catalogo + politicas
    embeddings   = crear_embeddings()
    vector_store = cargar_vector_store(embeddings)
    return SmartRetriever(vector_store, todos)


if __name__ == "__main__":
    retriever, _ = inicializar_rag()
    pruebas = [
        "camisetas retro de la U",
        "camiseta Colo-Colo negra Correa talla M",
        "todo lo que tienen de Universidad de Chile",
        "Real Madrid Mbappé 2026",
        "buzos de Colo-Colo",
    ]
    for q in pruebas:
        print(f"\n📋 Consulta: '{q}'")
        for i, doc in enumerate(retriever.invoke(q), 1):
            print(f"  [{i}] {doc.page_content[:120]}...")
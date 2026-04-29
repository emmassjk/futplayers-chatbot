"""
src/3_rag_pipeline.py
=====================
IL1.3 - Pipeline RAG completo para FutPlayers.cl
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR   = BASE_DIR / "chroma_db"

CATALOGO_PATH  = DATA_DIR / "catalogo.csv"
POLITICAS_PATH = DATA_DIR / "politicas.txt"


def cargar_catalogo() -> list[Document]:
    documentos = []
    with open(CATALOGO_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            texto = (
                f"Producto: Polera {fila['equipo']} | "
                f"Liga: {fila['liga']} ({fila['pais']}) | "
                f"Color: {fila['color_principal']} | "
                f"Tallas disponibles: {fila['tallas_disponibles']} | "
                f"Precio: ${fila['precio_clp']} CLP | "
                f"Stock: {fila['stock']} unidades | "
                f"Temporada: {fila['temporada']}"
            )
            doc = Document(
                page_content=texto,
                metadata={
                    "fuente": "catalogo",
                    "equipo": fila["equipo"],
                    "precio": int(fila["precio_clp"]),
                    "stock":  int(fila["stock"]),
                }
            )
            documentos.append(doc)
    print(f"✅ Catálogo cargado: {len(documentos)} productos")
    return documentos


def cargar_politicas() -> list[Document]:
    with open(POLITICAS_PATH, encoding="utf-8") as f:
        contenido = f.read()
    doc = Document(page_content=contenido, metadata={"fuente": "politicas"})
    print(f"✅ Políticas cargadas: {len(contenido)} caracteres")
    return [doc]


def dividir_documentos(documentos: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    print(f"✅ Chunking completado: {len(chunks)} chunks generados")
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
        print("🔄 Vector store anterior eliminado, reconstruyendo...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="futplayers_catalogo",
    )
    print(f"✅ Vector store creado con {len(chunks)} chunks en {DB_DIR}")
    return vector_store


def cargar_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    if not DB_DIR.exists():
        raise FileNotFoundError(
            "Vector store no encontrado. Ejecuta primero: python src/3_rag_pipeline.py"
        )
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="futplayers_catalogo",
    )


def crear_retriever(vector_store: Chroma):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


def inicializar_rag():
    print("\n🚀 Inicializando pipeline RAG de FutPlayers.cl...\n")
    docs_catalogo  = cargar_catalogo()
    docs_politicas = cargar_politicas()
    chunks         = dividir_documentos(docs_catalogo + docs_politicas)
    embeddings     = crear_embeddings()
    vector_store   = construir_vector_store(chunks, embeddings)
    retriever      = crear_retriever(vector_store)
    print("\n✅ Pipeline RAG listo.\n")
    return retriever, embeddings


def obtener_retriever():
    embeddings   = crear_embeddings()
    vector_store = cargar_vector_store(embeddings)
    return crear_retriever(vector_store)


if __name__ == "__main__":
    retriever, _ = inicializar_rag()
    print("\n📋 Prueba de recuperación:")
    print("Consulta: 'polera Barcelona talla XL'\n")
    resultados = retriever.invoke("polera Barcelona talla XL")
    for i, doc in enumerate(resultados, 1):
        print(f"  [{i}] {doc.page_content[:120]}...")
        print(f"       Fuente: {doc.metadata.get('fuente')}\n")
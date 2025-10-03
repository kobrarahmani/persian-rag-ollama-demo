import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


import subprocess
import requests
import streamlit as st


import os
os.environ["OLLAMA_GPU_DRIVER"] = "cpu"


def diagnose_ollama():
    """Comprehensive Ollama diagnosis"""
    st.header("🔧 Ollama Diagnosis")
    
    # Check if Ollama process is running
    try:
        result = subprocess.run(["tasklist", "/fi", "imagename eq ollama.exe"], 
                              capture_output=True, text=True)
        if "ollama.exe" in result.stdout:
            st.success("✅ Ollama process is running")
        else:
            st.error("❌ Ollama process NOT found")
            st.info("Start Ollama with: `ollama serve`")
            return False
    except:
        st.warning("⚠️ Could not check Ollama process")
    
    # Check API connectivity
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            st.success("✅ Ollama API is responding")
            models = response.json().get("models", [])
            if models:
                st.write("📋 Loaded models:")
                for model in models:
                    st.write(f"  - {model['name']}")
            else:
                st.warning("⚠️ No models loaded")
            return True
        else:
            st.error(f"❌ Ollama API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Ollama API")
        st.info("Make sure Ollama is running: `ollama serve`")
        return False
    except Exception as e:
        st.error(f"❌ Ollama check failed: {str(e)}")
        return False

def restart_ollama():
    """Restart Ollama service"""
    try:
        # Kill existing Ollama processes
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                      capture_output=True)
        
        # Start Ollama
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        st.success("🔄 Ollama restarted! Waiting for it to start...")
        time.sleep(5)  # Wait for Ollama to start
        return True
    except Exception as e:
        st.error(f"❌ Failed to restart Ollama: {str(e)}")
        return False
    

system_prompt = """
# شما یک دستیار هوش مصنوعی متخصص در تحلیل قوانین، مقررات و اسناد حقوقی هستید.
# وظیفه‌ی شما این است که ابتدا پرسش کاربر را در <<صورت محاوره‌ای یا غیررسمی بودن>>، به فارسی رسمی و استاندارد بازنویسی کنید. 
# سپس فقط بر اساس «متن زمینه» ارائه‌شده، پاسخ کوتاه، دقیق و شفاف بدهید.

# دستورالعمل‌ها:
# 1. فقط بر اساس اطلاعات موجود در متن زمینه پاسخ دهید.
# 2.پاسخ باید کوتاه، دقیق و شفاف باشد. از توضیحات طولانی خودداری کنید.
# 3. اگر پرسش شامل چند بخش است، هر بخش را به صورت جداگانه و شفاف پاسخ دهید.
# 4. در پایان هر پاسخ، محل دقیق استناد را ذکر کنید (مانند: ماده، بند، تبصره، صفحه یا شماره بخش).
# 5. اگر پاسخ در متن موجود نیست، صریحاً بگویید که اطلاعات کافی برای پاسخ وجود ندارد.
# 6. زبان پاسخ باید ساده و رسمی باشد و از اصطلاحات حقوقی به شکل دقیق استفاده شود.
# 7. پاسخ باید از راست به چپ و بدون غلط املایی باشد.
# 8. پاسخ باید با توجه به قوانین نگارشی فارسی درست باشد.
# 9. از فرضیات یا اطلاعات عمومی استفاده نکنید؛ فقط بر اساس متن زمینه پاسخ دهید.
# 10. اطلاعات مربوط به ماده، بند، تبصره یا صفحه فقط باید در بخش «استناد» ذکر شود و نباید در متن پاسخ تکرار شود.
# 11. قبل از نوشتن «استناد»، مطمئن شو که محل دقیق آن در متن زمینه وجود دارد. اگر مطمئن نیستی، بنویس: «استناد دقیق مشخص نیست».
# 12. از دانش قبلی یا اطلاعات عمومی استفاده نکنید.
# 13. اگر عددی هم استفاده کردید به صورت استاندارد فارسی استفاده کنید.

# پرسش: تعریف دانشچو چیست؟ 
# فردی است که در یکی از دوره های آموزش عالی برابر ضوابط معین پذیرفته شده، ثبت نام کرده و مشغول به تحصیل است.

# قالب پاسخ:
# -پاسخ به صورت مختصر و دقیق و شفاف 
# - در انتها با برچسب «استناد:» محل ذکر شده در متن را بیاورید.

# مثال قالب:
# پاسخ: [متن کوتاه پاسخ]
#  یا ماده یا تبصره X، بند Y، صفحه Z: استناد"""




def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name


    try:     
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "?", "!", " ", "","تبصره","ماده","بند"],
        )
        return text_splitter.split_documents(docs)
    
    finally:
        # This ensures the file gets deleted even if there's an error
        try:
            os.unlink(temp_file_path)
        except PermissionError:
            # If file is still in use, just skip deletion
            pass


import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

def get_ollama_embedding_function():
    """Creates and returns an Ollama embedding function configuration.
    
    Returns:
        OllamaEmbeddingFunction: Configured embedding function for Persian model
    """
    return OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="persian-embed-hedariAI-q4:latest",
    )

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the persian-hedariAI-q8:latest model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = get_ollama_embedding_function()
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "ip"},
    )



def add_to_vector_collection(documents, collection_name):
    """Add documents to the vector collection in batches.
    
    Args:
        documents: List of documents to add
        collection_name: Name of the collection to add documents to
    """
  
    # Use the existing collection
    collection = get_vector_collection()
    
    # Prepare documents in smaller batches
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        
        # Extract content and metadata
        documents_batch = [doc.page_content for doc in batch_docs]
        metadatas_batch = [doc.metadata for doc in batch_docs]
        ids_batch = [f"{collection_name}_{i + j}" for j in range(len(batch_docs))]
        
        try:
            # Upsert with retry logic
            collection.upsert(
                documents=documents_batch,
                metadatas=metadatas_batch,
                ids=ids_batch
            )
            print(f"Successfully upserted batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
            # Continue with next batch instead of failing completely
            continue


import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(2),  # Only retry twice
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(httpx.ReadTimeout)
)
def query_collection(prompt: str, n_results: int = 2, timeout = 60 ):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    try:
        collection = get_vector_collection()
        results = collection.query(query_texts=[prompt], n_results=n_results)
        return results
    except httpx.ReadTimeout:
        st.error("⏰ Ollama embedding generation timed out. Server might be busy.")
        return None
    except Exception as e:
        st.error(f"❌ Query error: {str(e)}")
        return None

def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(
        model="gemma3:1b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


# def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
#     """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

#     Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
#     their relevance to the query prompt. Returns the concatenated text of the top 3 most
#     relevant documents along with their indices.

#     Args:
#         documents: List of document strings to be re-ranked.

#     Returns:
#         tuple: A tuple containing:
#             - relevant_text (str): Concatenated text from the top 3 ranked documents
#             - relevant_text_ids (list[int]): List of indices for the top ranked documents

#     Raises:
#         ValueError: If documents list is empty
#         RuntimeError: If cross-encoder model fails to load or rank documents
#     """
#     relevant_text = ""
#     relevant_text_ids = []

#     encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     ranks = encoder_model.rank(prompt, documents, top_k=3)
#     for rank in ranks:
#         relevant_text += documents[rank["corpus_id"]]
#         relevant_text_ids.append(rank["corpus_id"])

#     return relevant_text, relevant_text_ids


if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="RAG Question Answer")

        if st.button("Diagnose Ollama"):
            diagnose_ollama()
            
        if st.button("Restart Ollama"):
            restart_ollama()

        uploaded_file = st.file_uploader(
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=False
        )

        process = st.button(
            "⚡️ Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        # relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
        response = call_llm(context=context, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)

        # with st.expander("See most relevant document ids"):
        #     st.write(relevant_text_ids)
        #     st.write(relevant_text)
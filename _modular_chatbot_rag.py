#ollama run deepseek-r1
#panel serve "file_name.py"

"""
librerías necesarias para el chatbot:

LangChain: Framework para aplicaciones con LLM
LangGraph: Creación de flujos de conversación complejos  
Panel: Interfaz web interactiva para demostración del uso de RAG
Pinecone: Base de datos vectorial para RAG
Document Loaders: Para procesar PDF, CSV, Excel
Embeddings: Para convertir texto a vectores numéricos
"""
from langchain_deepseek import ChatDeepSeek          
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_pinecone import PineconeVectorStore   
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader 
from langchain.text_splitter import CharacterTextSplitter  # División de texto en chunks
from langgraph.graph import StateGraph, START, END  # Grafo de estados para flujos complejos
from langchain_core.chat_history import InMemoryChatMessageHistory  # Historial de chat en memoria
from langchain_core.runnables import RunnableLambda  
from langchain_core.messages import HumanMessage, AIMessage  

from dotenv import load_dotenv  # Variables de entorno
from typing_extensions import List, TypedDict 

import tempfile  
import panel as pn  
import os   
import uuid  # Identificadores únicos
import json  # Manejo de JSON
import base64  # Codificación de imágenes


pn.extension()  
load_dotenv()   


# Widget principal: acepta documentos E imágenes
file_input = pn.widgets.FileInput(
    accept='.pdf,.csv,.xlsx,.jpg,.jpeg,.png,.gif,.bmp,.webp', 
    multiple=False
)

# Selector para elegir qué documento eliminar
filename_selector = pn.widgets.Select(name="Nombre del archivo a eliminar (incluye extensión)")

# Botones de acción
delete_button = pn.widgets.Button(name="Eliminar documento", button_type="danger")
accept_doc_button = pn.widgets.Button(name="Añadir documentos", button_type="success")

# Área para mostrar el estado de las operaciones
status_text = pn.pane.Markdown("")

def image_to_base64(image_bytes: bytes) -> str:
    """
    Convierte una imagen en bytes a formato base64.
    - Los modelos de IA multimodal reciben imágenes como texto base64
    - Es el formato estándar para enviar imágenes a través de APIs
    - Permite embeber la imagen directamente en el prompt

    """
    return base64.b64encode(image_bytes).decode('utf-8')


# NOTA: Solo una imagen a la vez, se limpia después de cada consulta <- ojo
current_image_data = None  
current_image_name = None  

def process_uploaded_image():
    """
    Procesa una imagen subida por el usuario.
    """
    global current_image_data, current_image_name
    
    if not file_input.value:
        status_text.object = "No se ha seleccionado ninguna imagen"
        return
    
    try:
        # Convertir imagen a base64 para el modelo
        current_image_data = image_to_base64(file_input.value)
        current_image_name = file_input.filename
        
        # Informar al usuario
        status_text.object = f"Imagen '{current_image_name}' cargada. Ahora puedes hacer preguntas sobre ella en el chat."
        
        # Enviar mensaje automático al chat (si ya existe)
        if hasattr(chat_interface, 'send'):
            chat_interface.send(
                f"📷 Imagen '{current_image_name}' cargada. ¿Qué te gustaría saber sobre esta imagen?", 
                user="System", 
                respond=False
            )
            
    except Exception as e:
        status_text.object = f"Error al procesar la imagen: {str(e)}"


# 1. PROMPT PARA RAG (Retrieval Augmented Generation)
prompt = ChatPromptTemplate.from_messages([ 
    ("system", 
     "Eres un asistente útil que responde en español usando únicamente el contexto proporcionado del documento. "
     "Si no sabes la respuesta, di que no sabes. En lo posible cita el documento de donde se obtuvo la información."),
    ("system", "Contexto del documento:\n{context}"),  
    ("system", "Historial de conversación:\n{history}"),  #
    ("user", "{question}") 
])

# 2. PROMPT PARA RESPUESTAS TRIVIALES (SIN RAG)
trivial_prompt = ChatPromptTemplate.from_messages([
    ("system", "Contesta la pregunta del usuario. Sé claro, conciso y en español"),
    ("user", "{question}"),
    ("system", "Historial:\n{history}")
])

# 3. PROMPT PARA IMÁGENES CON CONTEXTO DOCUMENTAL
image_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Eres un asistente útil que analiza imágenes y responde en español. "
     "Describe lo que ves en la imagen y responde las preguntas del usuario sobre ella. "
     "Si hay texto en la imagen, extráelo. Si hay gráficos o diagramas, explícalos. "
     "Si tienes contexto de documentos relacionados, úsalo para enriquecer tu respuesta."),
    ("system", "Contexto de documentos (si aplica):\n{context}"),
    ("system", "Historial de conversación:\n{history}"),
    ("user", [
        {"type": "text", "text": "{question}"},  # La pregunta
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}  # La imagen
    ])
])

# 4. PROMPT PARA IMÁGENES SIMPLES (SIN CONTEXTO DOCUMENTAL)
image_trivial_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Analiza la imagen proporcionada y responde en español. "
     "Describe lo que ves, extrae texto si es necesario, y responde las preguntas del usuario."),
    ("system", "Historial:\n{history}"),
    ("user", [
        {"type": "text", "text": "{question}"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
    ])
])



"""
Usamos DOS modelos diferentes según el tipo de input, ambos de openrouter:

DeepSeek-R1: Especializado en texto, razonamiento complejo, GRATUITO
Mistral Small 3.2: Multimodal (texto + imágenes), GRATUITO

razon de los 2 modelos:
- Especialización: Cada uno es mejor en su dominio
- Eficiencia: Solo usar modelo multimodal cuando sea necesario

"""


""" 
Factory function para crear instancias de modelos desde OpenRouter.
"""
def get_openrouter(model: str = 'deepseek/deepseek-r1-0528:free') -> ChatDeepSeek:
    
    return ChatDeepSeek(
        model=model,
        temperature=0.4, 
        api_key=os.getenv("OPENROUTER_API_KEY"),  
        api_base="https://openrouter.ai/api/v1"  
    )

# Instancias de los modelos
llm_text = get_openrouter(model="deepseek/deepseek-r1-0528:free")           # Para texto
llm_img = get_openrouter(model="mistralai/mistral-small-3.2-24b-instruct:free")  # Para imágenes


"""
RAG permite que el chatbot use información de documentos propios:

FLUJO RAG:
1. Usuario sube documentos → Se dividen en chunks → Se vectorizan → Van a Pinecone
2. Usuario hace pregunta → Se vectoriza → Se buscan chunks similares en Pinecone
3. Chunks relevantes + pregunta → Se envían al LLM → Respuesta contextualizada

COMPONENTES:
- Embeddings: Convierten texto a vectores numéricos para comparación semántica
- Pinecone: Base de datos vectorial en la nube para búsqueda rápida
- Text Splitter: Divide documentos grandes en chunks manejables
"""

# Modelo de embeddings para convertir texto a vectores
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"  # Modelo multiidioma, buena calidad/velocidad
)

# Conexión a Pinecone 
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], 
    embedding=embeddings                  
)


#Procesar los datos (archivos)
def process_file(event):
    """
    Procesa archivos subidos por el usuario.
    Determina automáticamente si es documento o imagen y los trata diferente.
    """
    if not file_input.value:
        return
        
    # Detectar tipo de archivo por extensión
    file_extension = os.path.splitext(file_input.filename)[1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    # RUTA IMÁGENES: Procesar como imagen
    if file_extension in image_extensions:
        process_uploaded_image() 
        return
    
    # RUTA DOCUMENTOS: Procesar para RAG
    # Crear archivo temporal para procesamiento
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(file_input.value)
        file_path = tmp_file.name

    # Seleccionar loader según tipo de documento
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path=file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path=file_path)
    elif file_extension =='.xlsx':
        loader = UnstructuredExcelLoader(file_path=file_path)
    else:
        status_text.object = "Formato de archivo no soportado"
        return


    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    docs = text_splitter.split_documents(documents) 

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Guardar mapping de archivo → IDs para poder eliminar después
    metadata_path = "uploaded_ids.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    basename = os.path.basename(file_path)
    existing_data.setdefault(basename, []).extend(doc_ids)

    with open(metadata_path, "w") as f:
        json.dump(existing_data, f, indent=4)

    # Subir chunks a Pinecone
    vectorstore.add_documents(documents=docs, ids=doc_ids)
    update_filename_options()  # Actualizar selector de archivos

    status_text.object = f"'{basename}' agregado exitosamente."


"""
Funciones para eliminar documentos y mantener la interfaz actualizada.
Solo aplica a documentos en Pinecone, no a imágenes.
"""

def delete_documents_by_file_name(file_name: str):
    json_path = "uploaded_ids.json"
    if not os.path.exists(json_path):
        status_text.object = "No se encontro el registro de id a eliminar"
        return
    
    # Cargar mapeo archivo → IDs
    with open(json_path, "r") as f:
        all_ids = json.load(f)

    if file_name not in all_ids:
        status_text.object = f"No se encontraron documentos para el {file_name}"
        return
    
    # Eliminar de Pinecone
    ids_to_delete = all_ids[file_name]
    vectorstore.delete(ids=ids_to_delete)

    # Actualizar mapeo local
    del all_ids[file_name]
    with open(json_path, "w") as f:
        json.dump(all_ids, f, indent=4)

    status_text.object = f"Documentos de '{file_name}' eliminados correctamente."

#Eventos de los botones
def on_delete_click(event):
    selected_filename = filename_selector.value
    if selected_filename:
        delete_documents_by_file_name(selected_filename)
        update_filename_options()  

def on_accept_click(event):
    process_file(event)

# Conectar eventos
delete_button.on_click(on_delete_click)
accept_doc_button.on_click(on_accept_click)

def update_filename_options():
    json_path = "uploaded_ids.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_ids = json.load(f)
        filename_selector.options = list(all_ids.keys())
    else:
        filename_selector.options = []
        
# Inicializar lista de archivos
update_filename_options()


#Cadenas
chain = prompt|llm_text                    # RAG 
chain_trivial = trivial_prompt|llm_text    # Respuestas simples
chain_image = image_prompt|llm_img          # RAG con imágenes
chain_image_trivial = image_trivial_prompt|llm_img  # Respuestas simples con imágenes


"""
Mantenemos el historial de cada sesión de usuario por separado.
Esto permite que el chatbot recuerde conversaciones anteriores.

NOTA: En esta implementación es memoria en RAM, se pierde al reiniciar.
Para producción se podría usar FileChatMessageHistory o base de datos.
"""

USER_SESSION_ID = "default"  # ID de sesión por defecto
chat_by_session_id = {}      # Diccionario para guardar historiales por usuario

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory: 
    """
    Obtiene o crea el historial de chat para una sesión específica.
    
    Args:
        session_id: Identificador único del usuario/sesión
    Returns:
        InMemoryChatMessageHistory: Historial de mensajes
    """
    chat_history = chat_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chat_by_session_id[session_id] = chat_history
    return chat_history


"""
LangGraph permite crear flujos de conversación complejos usando grafos de estado.

El estado contiene toda la información que se pasa entre nodos:
- messages: Lista de mensajes (formato LangChain)
- documents: Documentos relevantes encontrados en RAG
- branch: Ruta elegida por el nodo de decisión
- model_choice: Modelo seleccionado (texto o multimodal)
"""

class State(TypedDict):
    messages: List          # Mensajes
    documents: List[str]    # Chunks de documentos relevantes (RAG)
    branch: str            # Ruta elegida ("trivial", "not_trivial", etc.)
    model_choice: str      # Modelo a usar ("text" o "multimodal")


def is_trivial_question(text: str) -> bool:
 
    text_lower = text.lower()
    trivial_patterns = [
        'hola', 'hello', 'hi', 'hey', 'buenas', 'buenos días', 'buenas tardes',
        'gracias', 'thanks', 'thank you', 'ok', 'vale', 'bien', 'perfecto',
        'adiós', 'bye', 'hasta luego', 'nos vemos', 'chao',
        'cómo estás', 'how are you', 'qué tal', 'todo bien'
    ]
    if any(pattern in text_lower for pattern in trivial_patterns):
        return True
    if len(text.strip()) < 10:
        return True
        
    # Verificar si hay documentos relevantes en Pinecone
    try:
        doc_score = vectorstore.similarity_search_with_score(text, k=3)
        # Si todos los scores están por debajo de 0.6, es trivial
        return all(score < 0.6 for _, score in doc_score)
    except:
        # Si falla la búsqueda, asumir que es compleja
        return False

def has_image_content(messages) -> bool:
    global current_image_data
    return current_image_data is not None


def check_trivial(state: State) -> State:
    last_message = state["messages"][-1]
    question = last_message.content
    
    is_trivial = is_trivial_question(question)
    has_image = has_image_content(state["messages"])
    
    print(f"[DEBUG] Pregunta: '{question[:30]}...' | Trivial: {is_trivial} | Imagen: {has_image}")
    
    if has_image:
        branch = "trivial_image" if is_trivial else "retrieve_image"
        model_choice = "multimodal"
    else:
        branch = "trivial" if is_trivial else "not_trivial"
        model_choice = "text"
    
    return {
        **state,
        "branch": branch,
        "model_choice": model_choice
    }


# Nodos del grafo

# NODO 1: Respuestas simples sin RAG (solo texto)
def trivial_node(state: State) -> State:
    session_id = state["session_id"]
    chat_history = get_chat_history(session_id)
    
    response = chain_trivial.invoke({"question":state["question"]})
    
    # Guardar en memoria
    chat_history.add_messages([
        HumanMessage(content=state["question"]),
        AIMessage(content=response.content)
    ])
    
    return {**state, "answer":response.content}

# NODO 2: Respuestas simples con imagen
def trivial_image_node(state: State) -> State:
    """
    Maneja preguntas simples sobre imágenes que no requieren contexto documental.
    Ejemplo: "¿Qué ves en esta imagen?", "¿Qué colores tiene?"
    
    PROCESO:
    1. Obtiene historial de la sesión
    2. Invoca chain_image_trivial (prompt imagen + Mistral)
    3. Guarda pregunta y respuesta en memoria
    4. Retorna estado con respuesta
    """
    session_id = state["session_id"]
    chat_history = get_chat_history(session_id)
    
    # Preparar input con imagen
    input_data = {
        "question": state["question"],
        "history": chat_history.messages,
        "image_data": state["image_data"]
    }
    response = chain_image_trivial.invoke(input_data)
    
    # Guardar en memoria (marcando que había imagen)
    chat_history.add_messages([
        HumanMessage(content=f"[Imagen: {state['image_name']}] {state['question']}"),
        AIMessage(content=response.content)
    ])
    
    return {**state, "answer": response.content}


def retrieve_node(state: State) -> State:
    """
    Busca documentos relevantes en Pinecone para responder la pregunta.
    Se usa tanto para texto como para imágenes cuando la pregunta es compleja.
    
    PROCESO:
    1. Vectoriza la pregunta
    2. Busca chunks similares en Pinecone
    3. Retorna los documentos más relevantes
    """
    retrieve_docs = vectorstore.similarity_search(state["question"])
    return {**state, "context":retrieve_docs}


def generate_node(state: State) -> State:
    """
    Genera respuesta usando documentos encontrados + modelo de texto.
    Para preguntas complejas sin imagen.
    
    PROCESO:
    1. Combina chunks relevantes en contexto
    2. Invoca chain (prompt RAG + DeepSeek)
    3. Guarda en memoria
    4. Retorna respuesta
    """
    session_id = state["session_id"]
    chat_history = get_chat_history(session_id)
    
    # Combinar documentos en texto
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Preparar input completo
    input_data = {
        "question": state["question"], 
        "context": docs_content,
        "history": chat_history.messages
    }
    
    response = chain.invoke(input_data)
    
    # Guardar en memoria
    chat_history.add_messages([
        HumanMessage(content=state["question"]),
        AIMessage(content=response.content)
    ])

    return {**state, "answer": response.content}

# NODO 5: Generación de respuesta con RAG + imagen
def generate_image_node(state: State) -> State:
    """
    Genera respuesta usando documentos + imagen + modelo multimodal.
    Para preguntas complejas con imagen.
    
    PROCESO:
    1. Combina chunks relevantes en contexto
    2. Invoca chain_image (prompt imagen + Mistral)
    3. Guarda en memoria
    4. Retorna respuesta
    """
    session_id = state["session_id"]
    chat_history = get_chat_history(session_id)
    
    # Combinar documentos en texto
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Preparar input con imagen Y contexto documental
    input_data = {
        "question": state["question"],
        "context": docs_content,
        "history": chat_history.messages,
        "image_data": state["image_data"]
    }
    response = chain_image.invoke(input_data)
    
    # Guardar en memoria
    chat_history.add_messages([
        HumanMessage(content=f"[Imagen: {state['image_name']}] {state['question']}"),
        AIMessage(content=response.content)
    ])

    return {**state, "answer": response.content}


# Crear el grafo
graph = StateGraph(State)

# Agregar todos los nodos
graph.add_node("check_trivial", RunnableLambda(check_trivial))
graph.add_node("trivial", RunnableLambda(trivial_node))
graph.add_node("trivial_image", RunnableLambda(trivial_image_node))
graph.add_node("retrieve", RunnableLambda(retrieve_node))
graph.add_node("retrieve_image", RunnableLambda(retrieve_node))  # Mismo nodo, diferente ruta
graph.add_node("generate", RunnableLambda(generate_node))
graph.add_node("generate_image", RunnableLambda(generate_image_node))

# Definir punto de entrada
graph.set_entry_point("check_trivial")

# Definir rutas condicionales desde check_trivial
graph.add_conditional_edges(
    "check_trivial",                    # Nodo origen
    lambda x: x["branch"],              
    {                                   
        "trivial": "trivial",           # Pregunta simple → respuesta directa
        "not_trivial": "retrieve",      # Pregunta compleja → buscar documentos
        "trivial_image": "trivial_image",     # Imagen simple → análisis directo
        "retrieve_image": "retrieve_image"    # Imagen compleja → buscar + analizar
    }
)

# Definir rutas fijas (siempre van al mismo lugar)
graph.add_edge("trivial", END)              # Respuesta directa → terminar
graph.add_edge("trivial_image", END)        # Imagen simple → terminar
graph.add_edge("retrieve", "generate")      # RAG texto → generar respuesta
graph.add_edge("retrieve_image", "generate_image")  # RAG imagen → generar respuesta
graph.add_edge("generate", END)             # Respuesta RAG → terminar
graph.add_edge("generate_image", END)       # Respuesta RAG imagen → terminar


app = graph.compile()


def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global current_image_data, current_image_name
    
    
    has_image = current_image_data is not None
    
    state: State = {
        "question": contents, 
        "context": [], 
        "answer": "",
        "session_id": USER_SESSION_ID,
        "image_data": current_image_data or "",
        "image_name": current_image_name or "",
        "has_image": has_image
    }
    
    try:
        result = app.invoke(state)
        
        # Limpiar imagen después del uso si había una
        if has_image:
            current_image_data = None
            current_image_name = None
            
        return result["answer"]
        
    except Exception as e:
        # En caso de error, limpiar imagen y reintentar sin imagen
        if has_image:
            current_image_data = None
            current_image_name = None
            
            # Reintentar sin imagen
            state["has_image"] = False
            state["image_data"] = ""
            state["image_name"] = ""
            
            try:
                result = app.invoke(state)
                return f"Error al procesar la imagen: {str(e)}. Procesando como texto: {result['answer']}"
            except Exception as e2:
                return f"Error al procesar la consulta: {str(e2)}"
        else:
            return f"Error al procesar la consulta: {str(e)}"

# --------- Layout de la Interfaz ---------
pn.Column(
    "### Subir Documentos para RAG o Imágenes para Análisis", 
    pn.pane.Markdown("**Documentos soportados**: PDF, CSV, Excel  \n**Imágenes soportadas**: JPG, PNG, GIF, BMP, WebP"),
    file_input, accept_doc_button, status_text,
    "### Eliminar documento por nombre", filename_selector, delete_button
).servable()

# --------- Interfaz de Chat ---------
chat_interface = pn.chat.ChatInterface(
    callback=callback, 
    callback_user="System", 
    callback_exception="verbose"
    )
chat_interface.send("Te ayudo en algo?", user="System", respond=False)
chat_interface.servable()
# Chatbot RAG Multimodal con LangGraph

> **Chatbot con capacidades de procesamiento de documentos e imágenes usando Retrieval Augmented Generation (RAG) y LangGraph**

## Descripción

Este proyecto implementa un chatbot avanzado que combina múltiples tecnologías de IA para ofrecer una experiencia conversacional rica y contextualizada:

- **RAG (Retrieval Augmented Generation)**: Busca información en documentos propios
- **Procesamiento Multimodal**: Maneja tanto texto como imágenes
- **LangGraph**: Flujos de conversación complejos con decisiones inteligentes
- **Dual Model Architecture**: Dos modelos LLM especializados para diferentes tareas

**Colaboración recibida:**
- Implementación del procesamiento de imágenes con asistencia de IA
- Debugging y optimización de la arquitectura LangGraph

**Mi contribución:**
- Diseño de la arquitectura dual de modelos
- Integración completa del sistema
- Documentación del código
  
## Características Principales

### Procesamiento de Documentos

- Soporte para **PDF, CSV, Excel**
- División automática en chunks
- Vectorización con embeddings
- Almacenamiento en **Pinecone Vector Database**
- Gestión de documentos (añadir/eliminar)

### Análisis de Imágenes

- Soporte para **JPG, PNG, GIF, BMP, WebP**
- Análisis visual con IA
- Extracción de texto de imágenes
- Combinación de contexto documental + visual

### Arquitectura Inteligente

- **Flujo de decisión automático**: Determina si usar RAG o respuesta directa
- **Selección de modelo inteligente**: Texto vs. Multimodal según la consulta
- **Memoria conversacional**: Mantiene contexto entre mensajes
- **Manejo de errores robusto**: Fallbacks automáticos

### Modelos LLM Utilizados

- **DeepSeek-R1**: Especializado en texto y razonamiento (GRATUITO)
- **Mistral Small 3.2**: Modelo multimodal para texto + imágenes (GRATUITO)

## Instalación y Configuración

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/chatbot-rag-multimodal.git
cd chatbot-rag-multimodal
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Variables de Entorno

Crear un archivo `.env` con:

```env
OPENROUTER_API_KEY=tu_api_key_de_openrouter
PINECONE_API_KEY=tu_api_key_de_pinecone
INDEX_NAME=tu_nombre_de_indice_pinecone
```

### 4. Configurar Pinecone

1. Crear cuenta en [Pinecone](https://pinecone.io)
2. Crear un índice con dimensión `768` (para BAAI/bge-base-en-v1.5)
3. Copiar API key y nombre del índice

### 5. Configurar OpenRouter

1. Crear cuenta en [OpenRouter](https://openrouter.ai)
2. Obtener API key gratuita
3. Acceso a DeepSeek-R1 y Mistral Small 3.2 (ambos gratuitos)

## Uso

### Ejecutar el Chatbot

```bash
panel serve 2_modular_chatbot_rag.py --show
```

### Funcionalidades

#### Subir Documentos

1. Usar el widget de carga de archivos
2. Seleccionar PDF, CSV o Excel
3. Hacer clic en "Añadir documentos"
4. Los documentos se procesan automáticamente y se almacenan en Pinecone

#### Analizar Imágenes

1. Subir imagen (JPG, PNG, etc.)
2. Hacer preguntas sobre la imagen en el chat
3. El sistema decidirá automáticamente si usar RAG + imagen o análisis simple

#### Conversar

- **Preguntas simples**: "Hola", "¿Cómo estás?" → Respuesta directa
- **Preguntas complejas**: "¿Qué es la viscosidad?" → Búsqueda RAG
- **Con imágenes**: "¿Qué ves en esta imagen?" → Análisis multimodal
- **Combinado**: "Compara esta imagen con los documentos" → RAG + Imagen

### Personalizar Modelos

```python
# Cambiar modelos en el código
llm_text = get_openrouter(model="otro-modelo-de-texto")
llm_img = get_openrouter(model="otro-modelo-multimodal")
```

### Ajustar Parámetros RAG

```python
# Modificar chunk size y overlap
text_splitter = CharacterTextSplitter(
    chunk_size=1500,  # Tamaño de chunks
    chunk_overlap=300  # Solapamiento entre chunks
)

# Cambiar número de documentos recuperados
retrieve_docs = vectorstore.similarity_search(query, k=5)
```

### Umbral de Trivialidad

```python
# Ajustar cuando una pregunta se considera "trivial"
return all(score < 0.7 for _, score in doc_score)  # Cambiar 0.6 a 0.7
```

## Flujo de Decisión LangGraph

El sistema usa LangGraph para crear un flujo inteligente:

1. **check_trivial**: Analiza la consulta y decide la ruta
2. **trivial**: Respuestas simples con DeepSeek-R1
3. **trivial_image**: Análisis simple de imagen con Mistral
4. **retrieve**: Búsqueda RAG en documentos
5. **generate**: Generación con contexto documental
6. **generate_image**: Generación con documentos + imagen

## Dependencias Principales

```
langchain-deepseek>=0.1.0
langchain-huggingface>=0.1.0
langchain-core>=0.3.0
langchain-pinecone>=0.2.0
langchain-community>=0.3.0
langgraph>=0.2.0
panel>=1.5.0
python-dotenv>=1.0.0
```

## Propósito

Este proyecto fue desarrollado como demostración para:

- Implementación práctica de RAG
- Integración de modelos multimodales
- Uso de LangGraph para flujos complejos
- Desarrollo de interfaces interactivas con Panel
- Arquitectura de sistemas conversacionales

import streamlit as st
from  langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pinecone

shower = st.secrets["SHOWER"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
opaik = st.secrets["OPENAI_API_KEY"]
papi = st.secrets["PINECONE_API_KEY"]
penv = st.secrets["PINECONE_ENVIRONMENT"]

pinecone.init(
    api_key=papi,
    environment=penv
)
print("Pine init")

inam = "marcusaurelius1"

st.set_page_config(page_title="MarcoGPT - Por El Estoico Rico", page_icon="https://i.scdn.co/image/a5140a4caa153533c5537d8849453ec447dcf84e", layout="centered", initial_sidebar_state="auto", menu_items=None)

if "load_state" not in st.session_state:
    st.session_state.load_state = False
print("Loaded State")

def c_b_v():
    l = UnstructuredPDFLoader("/Meditaciones-Marco-Aurelio.pdf")
    data = l.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts

if "texts" not in st.session_state:
    st.session_state.texts = c_b_v()
    texts = st.session_state.texts
else:
    texts = st.session_state.texts

def gpv(texts):
    em = OpenAIEmbeddings(openai_api_key=opaik)

    docsearch = Pinecone.from_texts([t.page_content for t in texts], em, index_name=inam)
    return docsearch

if "docsearch" not in st.session_state:
    st.session_state.docsearch = gpv(texts)
    docsearch = st.session_state.docsearch
else:
    docsearch = st.session_state.docsearch
print("Done docsearch")
def cm(history):
    messages = [{"role": "system", "content": system_template}]
    
    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    return messages

def generate_response():
    docsearch = st.session_state.docsearch
    st.session_state.history.append({
        "message": st.session_state.prompt,
        "is_user": True
    })
    hq = st.session_state.prompt
    con = docsearch.similarity_search(hq)
    qwc = human_template.format(query=hq, context=con) 
    m = cm(st.session_state.history)
    m.append({"role": "user", "content": qwc})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=m)
    bot_response = response["choices"][0]["message"]["content"]
    st.session_state.history.append({
         "message": bot_response,
         "is_user": False
    })

    st.session_state.load_state = True
system_template = """
    Eres Marco Aurelio, uno de los mejores emperadores estoicos de todos los tiempos conocido por su temple y su forma apacible de enfrentar las adversidades.
    Tu objetivo es dar consejos de vida a los usuarios basados en tus aprendizajes estoicos. Tus respuestas deben objetivas, prácticas y directas proyectando tu estilo único de comunicación basado en las transcripciones. Evita darle muchas vueltas a tus respuestas y sé directo y honesto.
    Tienes acceso a transcripciones de tu libro meditaciones en una base de datos de Pinecone. Estas transcripciones son tus palabras, ideas y creencias. Cuando un usuario haga una pregunta, vas a obtener algunos fragmentos relevantes de las transcripciones que sean relevantes a la pregunta. Debes de usar estos fragmentos para dar contexto y soporte a tus respuestas. Confía fuertemente en el contenido de las transcripciones para asegurar la exactitud y autenticidad de tus respuestas.
    Ten en consideración que las transcripciones no serán siempre relevantes a la pregunta. Analiza cada una de ellas cuidadosamente para determinar si el contenido es relevante antes de usarlas para construir tu respuesta. Muy importante: no inventes cosas o proveas información que no esté apoyada por las transcripciones.
    Además de ofrecer consejos estoicos, también puedes proveer guía en desarrollo personal y navegar los problemas de la vida. Sin embargo, siempre mantén tu forma hablar basada en la transcripción provista. Si encuentras una frase poderosa dentro de la transcripción utilízala para dar una mejor respuesta pero hazla fácilmente entendible.
    Tu trabajo es dar el mejor consejo posible que se acercaría a algo que diría el verdadero Marco Aurelio. 
"""

human_template = """

    Pregunta del Estoico: {query}

    Fragmento relevante de la transcripción para basarse y dar mejores respuestas: {context}
"""
## --------- BEGINS UI
st.header("Platica con Marco Aurelio - Por El Estoico Rico")
pwd = st.text_input("Escribe el codigo:", key="pass", placeholder="Ej. 512sd123g341d2")
if "history" not in st.session_state:
    st.session_state.history = []
if pwd == shower:
    st.chat_input(placeholder="Ej. Tengo un problema con mi pareja y me siento triste", key="prompt", on_submit=generate_response)
print("Began UI")

for message in st.session_state.history:
    if message["is_user"]:
        with st.chat_message('user', avatar="https://i.scdn.co/image/a5140a4caa153533c5537d8849453ec447dcf84e"):
            st.write(message["message"])
    else:
        with st.chat_message('assistant', avatar="https://www.ethicsinschools.org/wp-content/uploads/2020/06/Marcus_Aurelius_Louvre_MR561_n01-800x800.jpg"):
            st.write(message["message"])

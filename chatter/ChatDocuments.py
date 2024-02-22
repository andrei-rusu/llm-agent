import os
import traceback
import time
import csv
import tempfile
import zipfile
import streamlit as st
from io import TextIOWrapper
from dotenv import find_dotenv, load_dotenv
from operator import itemgetter
from qdrant_client import QdrantClient

from langchain_core.globals import get_debug, set_debug
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import langchain_community.llms as llms
import langchain_community.embeddings as embedders
from langchain_community.vectorstores import Qdrant, FAISS
from langchain_community.document_loaders import CSVLoader, UnstructuredFileLoader, UnstructuredAPIFileLoader, UnstructuredAPIFileIOLoader, DirectoryLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import get_buffer_string, format_document
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker

from utils.html_templates import css_layout, bot_template, user_template, USER_AVATAR_URL, BOT_AVATAR_URL
from streamlit_chat import message


TEMP_FOLDER = 'uploaded'
DEFAULT_MESSAGE = "Hi, I am Agnes!😊 Ask me anything about your documents."
ERROR_MESSAGE = "Query could not be served because of an error"
UI_OPTIONS = ["streamlit", "st-chat", "chained"]
LLM_OPTIONS = ["none", "google", "huggingface"]
EMBEDDER_OPTIONS = ["google", "huggingface"]
VS_OPTIONS = ["faiss", "qdrant-remote", "qdrant-memory", "qdrant-local"]
LLM_DEFAULT = {'none': "", 'google': "gemini-pro", 'huggingface': "google/flan-t5-xxl"}
PROCESSOR_OPTIONS = ['default', 'direct', 'api']
MODE_OPTIONS = ['paged', 'single', 'elements', 'paged:', 'single:', 'elements:']
EMBEDDER_DEFAULT = {'google': "models/embedding-001", 'huggingface': "all-MiniLM-L6-v2"}
CHUNKING_OPTIONS = ['recursive', 'char', 'semantic']


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container=None):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            try:
                source = os.path.basename(doc.metadata['source'])
            except KeyError:
                source = doc.metadata.get('filename', None)
            page = doc.metadata.get('page_number', 1)
            doc_info = (f"**Document {idx} from {source}**" if source else f"**Document {idx}**") + f", Page {page}"
            self.status.write(doc_info)
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def process_text(docs, document_api=True):
    processor = st.session_state.doc_processor
    mode = st.session_state.doc_mode
    docs_message = False
    if mode[-1] == ':':
        mode = mode[:-1]
        docs_message = True
    processed_docs = [] if document_api else ""
    with tempfile.TemporaryDirectory(prefix=TEMP_FOLDER, dir='.') as temp_folder:
        for i, doc in enumerate(docs):
            if doc is not None:
                is_zipfile = zipfile.is_zipfile(doc)
                if is_zipfile:
                    zip_path = os.path.join(temp_folder, f'unzip_{i}')
                    with zipfile.ZipFile(doc, 'r') as zip_ref:
                        zip_ref.extractall(zip_path)
                doc.seek(0) # this is needed because checking for zipfile actually exhausts the stream
                if document_api:
                    if doc.type == "text/plain":
                        processed_docs.append(Document(page_content=TextIOWrapper(doc, encoding='utf-8').read()))
                    elif doc.type == "text/csv":
                        loader = CSVLoader('')
                        processed_docs += loader._CSVLoader__read_file(TextIOWrapper(doc, encoding='utf-8'))
                    elif is_zipfile:
                        loader = DirectoryLoader(zip_path)
                        processed_docs += loader.load()
                    else:
                        if processor == 'direct':
                            loader = UnstructuredAPIFileIOLoader(doc, api_key=os.environ['UNSTRUCTURED_API_KEY'],
                                                                 metadata_filename=doc.name,
                                                                 url=os.environ['UNSTRUCTURED_HOST'],
                                                                 mode=mode, strategy='fast')
                        else:
                            temp_filepath = os.path.join(temp_folder, doc.name)
                            with open(temp_filepath, "wb") as f:
                                f.write(doc.getvalue())
                            if processor == 'default':
                                loader = UnstructuredFileLoader(temp_filepath, mode=mode, strategy='fast')
                            else:
                                loader = UnstructuredAPIFileLoader(temp_filepath, api_key=os.environ['UNSTRUCTURED_API_KEY'],
                                                                   url=os.environ['UNSTRUCTURED_HOST'],
                                                                   mode=mode, strategy='fast')
                        processed_docs += loader.load()
                else:
                    processed_docs += f"Document {doc.name}:\n"
                    if doc.type == "text/plain":
                        processed_docs += TextIOWrapper(doc).read()
                    elif doc.type == "text/csv":
                        processed_docs += csv_to_string(TextIOWrapper(doc, encoding='utf-8'))
                    elif doc.type == "application/pdf":
                        processed_docs += pdfminer(doc) if processor == 'default' else pypdf(doc)
                    elif zipfile.is_zipfile(doc):
                        files = [open(doc, 'r') for doc in os.listdir(zip_path)]
                        processed_docs += process_text(files)
                        for f in files:
                            f.close()
    if docs_message:
        st.session_state.reported += ['<b>In-memory before chunking:</b>', processed_docs[-2:] if document_api else processed_docs]
    return processed_docs


def csv_to_string(doc):
    reader = csv.DictReader(doc)
    all_text = '\n'
    for row in reader:
        all_text += ' '.join(row.values()) + '\n'
    return all_text


def pdfminer(doc):
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    all_text = '\n'
    pages = extract_pages(doc)
    for page in pages:
        all_text += f'Page {page.pageid}:\n'
        for element in page:
            if isinstance(element, LTTextContainer):
                all_text += element.get_text()
    return all_text


def pypdf(doc):
    from pypdf import PdfReader
    all_text = '\n'
    pages = PdfReader(doc).pages
    for i, page in enumerate(pages):
        all_text += f'Page {i + 1}:\n'
        all_text += page.extract_text()
    return all_text


def get_chunks(text, strategy='recursive', chunk_size=1500, chunk_overlap=200):
    if strategy == 'recursive':
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == 'char':
        splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == 'semantic':
        splitter = SemanticChunker(
            get_embeddings(),
        )
    chunks = splitter.split_text(text) if isinstance(text, str) else splitter.split_documents(text)
    if st.session_state.doc_mode[-1] == ':':
        st.session_state.reported += ['<b>In-memory after chunking:</b>', chunks[-2:]]
        return None
    return chunks


def get_embeddings():
    family = st.session_state.embedder_family
    model = st.session_state.embedder_model.strip()
    if family == 'google':
        return GoogleGenerativeAIEmbeddings(model=model)
    elif family == 'huggingface':
        return embedders.HuggingFaceHubEmbeddings(model=model)
    else:
        return getattr(embedders, family)(model=model)


def update_session_vectorstore(docs=None, recreate=False):
    embeddings = get_embeddings()
    path = os.path.join(TEMP_FOLDER, os.environ['COLLECTION'])
    if docs:
        if recreate:
            process_method_name = 'from_texts' if isinstance(docs[0], str) else 'from_documents'
            if 'qdrant' in st.session_state.vs_provider:
                if 'memory' in st.session_state.vs_provider:
                    kwargs = dict(
                        location=':memory',
                    )
                elif 'local' in st.session_state.vs_provider:
                    kwargs = dict(
                        path=path,
                    )
                else:
                    kwargs = dict(
                        url=os.environ['QDRANT_HOST'],
                        prefer_grpc=True,
                        api_key=os.environ['QDRANT_API_KEY'],
                    )
                st.session_state.vs = getattr(Qdrant, process_method_name)(
                    docs,
                    embeddings,
                    collection_name=os.environ['COLLECTION'],
                    force_recreate=recreate,
                    **kwargs
                )
            elif st.session_state.vs_provider == 'faiss':
                st.session_state.vs = getattr(FAISS, process_method_name)(
                    docs,
                    embeddings,
                )
            # update chain with a new retreiver based on the new vectorstore
            st.session_state.chain.retriever = st.session_state.vs.as_retriever(search_kwargs={"k": st.session_state.top_k})
        else:
            # add new documents to the existing vectorstore
            st.session_state.chain.retriever.vectorstore.add_texts(docs)
        # some db's need to be explicitly instructed to save to disk
        if st.session_state.vs_provider == 'faiss':
            st.session_state.vs.save_local(path)
    else:
        if 'qdrant' in st.session_state.vs_provider:
            if 'memory' in st.session_state.vs_provider:
                client = QdrantClient('')
            elif 'local' in st.session_state.vs_provider:
                client = QdrantClient(path=path)
            else:
                client=QdrantClient(os.environ['QDRANT_HOST'], api_key=os.environ['QDRANT_API_KEY'], timeout=100)
            try:
                st.session_state.vs = Qdrant(client=client, collection_name=os.environ['COLLECTION'], embeddings=embeddings)
            except Exception as e:
                st.error(f"Could not load Qdrant collection because: {e}")
        else:
            try:
                st.session_state.vs = FAISS.load_local(path, embeddings=embeddings)
            except Exception as e:
                st.error(f"Could not load local FAISS collection because: {e}")


def get_general_prompt():
    template = """
        Given the following context, please give a short answer to the question below.
        
        CONTEXT:
        {context}
        
        QUESTION: 
        {question}

        CHAT HISTORY:
        {chat_history}
        
        ANSWER:
        """
    return PromptTemplate.from_template(template)


def get_question_generator_prompt():
    template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its input language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    return PromptTemplate.from_template(template)


def get_combine_docs_prompt():
    template = """Parse the ENTIRE following context (including roman literals) and answer the question below:
        {context}
        Question: {question}"""
    return PromptTemplate.from_template(template)


def _combine_documents(docs, document_prompt=None, document_separator="\n\n"):
    if document_prompt is None:
        document_prompt = PromptTemplate.from_template(template="{page_content}")
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def lcel_rag_chain(llm, memory, retriever):
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter('chat_history'),
    )

    standalone_question = {
        "standalone_question": {
            "question": itemgetter("question"),
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | get_question_generator_prompt()
        | llm
        | StrOutputParser(),
    }
    # Conditional chain that checks if chat_history is empty
    conditional_chain = RunnableBranch(
        (lambda x: len(x['chat_history']) == 0, RunnablePassthrough.assign(standalone_question=itemgetter('question'))),
        standalone_question,
    )

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": itemgetter("standalone_question"),
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    answer = {
        "rephrased_question": itemgetter("question"),
        "docs": itemgetter("docs"),
        "answer": final_inputs | get_combine_docs_prompt() | llm,
    }

    return loaded_memory | conditional_chain | retrieved_documents | answer


def update_session_chain():
    print('Updating session chain...')
    family = st.session_state.llm_family
    model = st.session_state.llm_model.strip()
    temperature = st.session_state.temperature
    try:
        if family == 'none':
            return None
        elif family == 'google':
            llm = ChatGoogleGenerativeAI(model=model, convert_system_message_to_human=True, temperature=temperature)
        elif family == 'huggingface':
            llm = llms.HuggingFaceHub(repo_id=model, model_kwargs=dict(temperature=temperature, max_length=512))
        else:
            llm = getattr(llms, family)(model=model, temperature=temperature)
        memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer',
                                          chat_memory=st.session_state.chat_history, return_messages=True)
        retriever = st.session_state.vs.as_retriever(search_kwargs={'k': st.session_state.top_k})
        if st.session_state.use_lcel:
            chain = lcel_rag_chain(llm, memory, retriever)
        else:
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=memory,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": get_combine_docs_prompt()}
            )
            chain.question_generator.prompt.template = chain.question_generator.prompt.template.replace('original', 'input')
        st.session_state.chain = chain
    except Exception as e:
        st.error(f"Could not (re)initialize chain because: {e}")


def handle_user_input(user_input):
    use_llm = st.session_state.llm_family != 'none'
    use_lcel = use_llm and st.session_state.use_lcel
    write_message(user_input, is_user=True)
    retrieval_container = write_message("<i>Typing...</i>", is_user=False, placeholder=True)
    if use_llm:
        callbacks = []
        retrieval_callback = PrintRetrievalHandler(retrieval_container)
        if not use_lcel:
            callbacks.append(retrieval_callback)
    else:
        time.sleep(1)

    try:
        if st.session_state.use_stream:
            response = st.session_state.chain.stream({'question': user_input}, {'callbacks': callbacks}) if use_llm \
                     else ({'answer': i} for i in DEFAULT_MESSAGE.split('😊'))
            final_reply = ''
            for chunk_id, chunk in enumerate(response):
                if 'rephrased_question' in chunk.keys():
                    retrieval_callback.on_retriever_start({}, chunk['rephrased_question'])
                if 'docs' in chunk.keys():
                    retrieval_callback.on_retriever_end(chunk['docs'])
                if 'answer' in chunk.keys():
                    answer = chunk['answer']
                    final_reply += answer.content if hasattr(answer, 'content') else answer
                    write_message(final_reply, is_user=False, i=-chunk_id-1)
                    if not use_llm:
                        time.sleep(1)
        else:
            response = st.session_state.chain.invoke({'question': user_input}, {'callbacks': callbacks}) if use_llm \
                       else {'answer': DEFAULT_MESSAGE}
            if 'docs' in response.keys():
                retrieval_callback.on_retriever_end(response['docs'])
            if 'answer' in response.keys():
                answer = response['answer']
                answer = answer.content if hasattr(answer, 'content') else answer
                write_message(answer, is_user=False)
    except Exception as e:
        write_message(f'{ERROR_MESSAGE}: {e}', is_user=False)
        traceback.print_exc()
    if use_lcel:
        st.session_state.chat_history.add_user_message(user_input)
        st.session_state.chat_history.add_ai_message(final_reply)
    st.session_state.response_container = None


def write_message(message_content, is_user=False, i=-1, add_to_history=False, placeholder=False):
    if i < 0:
        i = str(len(st.session_state.chat_history.messages) + abs(i) - 1)
    if placeholder:        
        i += '_phld'
    if st.session_state.chat_ui == 'chained':
        if is_user:
            st.markdown(user_template.format(msg=message_content), unsafe_allow_html=True)
        elif placeholder:
            parent = st.container()
            retrieval_container = parent.container()
            st.session_state.response_container = parent.empty()
            st.session_state.response_container.markdown(bot_template.format(msg=message_content), unsafe_allow_html=True)
            return retrieval_container
        elif st.session_state.response_container is None:
            st.markdown(bot_template.format(msg=message_content), unsafe_allow_html=True)
        else:
            st.session_state.response_container.markdown(bot_template.format(msg=message_content), unsafe_allow_html=True)
    elif st.session_state.chat_ui == 'st-chat':
        if is_user:
            message(message_content, is_user=True, key=f'{i}_user', avatar_style='adventurer', seed=25, logo=USER_AVATAR_URL)
        elif placeholder:
            parent = st.container()
            retrieval_container = parent.container()
            st.session_state.response_container = parent.empty()
            with st.session_state.response_container:
                message(message_content, is_user=False, key=f'{i}_model', logo=BOT_AVATAR_URL, allow_html=True)
            return retrieval_container
        elif st.session_state.response_container is None:
            message(message_content, is_user=False, key=f'{i}_model', logo=BOT_AVATAR_URL, allow_html=True)
        else:
            with st.session_state.response_container:
                message(message_content, is_user=False, key=f'{i}_model', logo=f'{BOT_AVATAR_URL}', allow_html=True)
    else:
        if is_user:
            st.chat_message('user', avatar=USER_AVATAR_URL).write(message_content)
        elif placeholder:
            parent = st.chat_message('model', avatar=BOT_AVATAR_URL)
            retrieval_container = parent.container()
            st.session_state.response_container = parent.empty()
            st.session_state.response_container.markdown(message_content, unsafe_allow_html=True)
            return retrieval_container
        elif st.session_state.response_container is None:
            st.chat_message('model', avatar=BOT_AVATAR_URL).write(message_content, unsafe_allow_html=True)
        else:
            st.session_state.response_container.markdown(message_content, unsafe_allow_html=True)
    if add_to_history:
        if is_user:
            st.session_state.chat_history.add_user_message(message_content)
        else:
            st.session_state.chat_history.add_ai_message(message_content)


def write_history():
    msgs = st.session_state.chat_history.messages
    write_message(DEFAULT_MESSAGE, is_user=False)
    for i, msg in enumerate(msgs):
        write_message(msg.content, is_user=(msg.type == 'human'), i=i)


def post_processing():
    if st.session_state.reported:
        for r in st.session_state.reported:
            if r:
                write_message(r, is_user=False)
        st.session_state.reported = []


def on_change_llm_family():
    st.session_state.llm_model = LLM_DEFAULT[st.session_state.llm_family]
    if st.session_state.llm_model:
        update_session_chain()


def config_sidebar():
    with st.sidebar:
        st.selectbox('Chat UI', options=UI_OPTIONS, key='chat_ui')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.selectbox('LLM family', options=LLM_OPTIONS, key='llm_family', 
                         on_change=on_change_llm_family)
        with col2:
            st.text_input('LLM model', key='llm_model', on_change=update_session_chain)
        st.slider('Temperature', min_value=0.0, max_value=1.0, step=0.1, key='temperature', on_change=update_session_chain)
        st.slider('Top K search', min_value=1, max_value=5, step=1, key='top_k', on_change=update_session_chain)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.checkbox("LCEL", key='use_lcel', on_change=update_session_chain)
        with col2:
            st.checkbox("Stream", key='use_stream')
        with col3:
            st.checkbox("Debug",
                        value=get_debug(),
                        on_change=lambda: set_debug(not get_debug()))
        docs = st.file_uploader("Upload your documents:", accept_multiple_files=True, type=["pdf", "txt", "csv", "zip"])
        col1, col2 = st.columns([1, 1])
        with col1:
            st.selectbox('Doc processor', options=PROCESSOR_OPTIONS, key='doc_processor')
            st.selectbox('Embedder family', options=EMBEDDER_OPTIONS, key='embedder_family',
                         on_change=lambda: setattr(st.session_state, 'embedder_model', EMBEDDER_DEFAULT[st.session_state.embedder_family]))
            st.selectbox('Vectorstore', options=VS_OPTIONS, key='vs_provider')
            recreate_collection = st.checkbox("Reset knowledge", value=True)
        with col2:
            st.selectbox('Doc mode', options=MODE_OPTIONS, key='doc_mode')
            st.text_input('Embedder model', key='embedder_model')
            chunking_strategy = st.selectbox('Chunking', options=CHUNKING_OPTIONS)
            document_api = st.checkbox("Use document API", value=True)

        if st.button("Upload"):
            with st.spinner("Processing..."):
                # get text from files
                text = process_text(docs, document_api=document_api)
                # get chunks
                chunks = get_chunks(text, strategy=chunking_strategy)
                # update/recreate vectorstore
                if chunks:
                    update_session_vectorstore(chunks, recreate=recreate_collection)


def config_main_panel():
    if st.session_state.chat_ui:
        st.write(css_layout, unsafe_allow_html=True)
    write_history()
    user_input = st.chat_input("Ask a question about your documents:")
    if user_input:
        handle_user_input(user_input)
    st.button("Reset chat", key='reset_chat', on_click=st.session_state.chat_history.clear)


def init():
    load_dotenv(find_dotenv())
    st.set_page_config(page_title="Chat Documents", page_icon=":books:", layout="wide")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = StreamlitChatMessageHistory()
    if 'chat_ui' not in st.session_state:
        st.session_state.chat_ui = UI_OPTIONS[0]
    if 'llm_family' not in st.session_state:
        st.session_state.llm_family = LLM_OPTIONS[1]
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = LLM_DEFAULT[st.session_state.llm_family]
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = PROCESSOR_OPTIONS[0]
    if 'doc_mode' not in st.session_state:
        st.session_state.doc_mode = MODE_OPTIONS[0]
    if 'embedder_family' not in st.session_state:
        st.session_state.embedder_family = EMBEDDER_OPTIONS[0]
    if 'embedder_model' not in st.session_state:
        st.session_state.embedder_model = EMBEDDER_DEFAULT[st.session_state.embedder_family]
    if 'vs_provider' not in st.session_state:
        st.session_state.vs_provider = VS_OPTIONS[0]
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.0
    if 'use_stream' not in st.session_state:
        st.session_state.use_stream = True
    if 'use_lcel' not in st.session_state:
        st.session_state.use_lcel = False
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 3
    if 'response_container' not in st.session_state:
        st.session_state.response_container = None
    if 'reported' not in st.session_state:
        st.session_state.reported = []
    if 'vs' not in st.session_state:
        update_session_vectorstore()
    if 'chain' not in st.session_state:
        update_session_chain()


def main():
    init()
    config_main_panel()
    config_sidebar()
    post_processing()


if __name__ == "__main__":
    main()
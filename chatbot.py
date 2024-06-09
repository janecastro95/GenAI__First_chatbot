#---- 0. Set up ----
#instalar bibliotecas
# pode instalar no terminal que for rodar o código, para não ter que chamar toda vez
#pip install pypdf2 langchain faiss-cpu openai tiktoken streamlit

import streamlit as st #library para criar UI
from PyPDF2 import PdfReader #library para ler pdfs

#langchain = tools to connect language models with external data sources
from langchain_text_splitters import RecursiveCharacterTextSplitter #para quebrar o texto
from langchain.embeddings.openai import OpenAIEmbeddings #para gerar embeddings
from langchain_community.vectorstores import FAISS #para criar um database de embeddings
from langchain.chains.question_answering import load_qa_chain #para a geração de resultados
from langchain_community.chat_models import ChatOpenAI #para chamar LLM


#---- 1. Cria uma interface em que o usuário possa subir os pdfs ----
st.header("My first Chatbot") #Nome do chatbot

with st.sidebar: #cria uma barra lateral com funcionalidade para subir pdf
    st.title("Your Documents") #título da barra lateral
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf") #salva o arquivo em "file"

#---- 2. Extrai o texto e divide os pdfs em pedaços menores ----
if file is not None: #só roda o programa se tiver algum arquivo
    pdf_reader = PdfReader(file) #lê o arquivo
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() # concatena todo texto, página a página, na variável text
    # st.write(text) #só para aparecer resultado parcial no st e ver passo a passo

    text_splitter = RecursiveCharacterTextSplitter( #quebra o texto em chunks para otimizar
        separators='\n', #divisor para um novo chunk é uma nova linha
        chunk_size=1000, #tamanho
        chunk_overlap=150, #captura 150 caracteres do chunk anterior para não ficar desconexo
        length_function=len
    )
    chunks = text_splitter.split_text(text) #aplica ao arquivo lido
    #st.write(chunks) #para aparecer no st e ver passo a passo

#---- 3. Gera embedding com base nos chunks de texto, usando serviços do OpenAi via API ----
    # precisa passar credenciais (key) do serviço OpenAI: https://platform.openai.com/api-keys
    # precisa ter saldo na conta OpenAI para usar: https://platform.openai.com/settings/organization/billing/overview
    embeddings = OpenAIEmbeddings(openai_api_key = "")

#---- 4. Armaze os embeddings ----
    vector_store = FAISS.from_texts(chunks, embeddings) #chama o serviço da OpenAI para criar embeddings e salvar em um vector database FAISS

#---- 5. Coloca um inputer para pergunta do usuário ----
    user_question = st.text_input("Type your question here")

#---- 6. Acha a melhor resposta ----
    if user_question: #se existir uma pergunta
        match = vector_store.similarity_search(user_question) #gera embedding da pergunta e faz match com o texto, via embeddings no vector_store
        #st.write(match) #só para aparecer resultado parcial no st e ver passo a passo

#---- 7. Usa LLM para gerar conteúdo escrito com a resposta ----
    #define modelo desejado e os parâmetros
    llm = ChatOpenAI(
        openai_api_key="", #credenciais
        temperature = 0, #valor inteiro = quão menor, mais conciso.
        max_tokens = 1000, #máximo de tokens, que impacta a quantidade de palavras (1000 token ~750 palavras)
        model_name = 'gpt-3.5-turbo' #nome do modelo LLM
    )

    # chama de "chain" um sequência de eventos = get the question > get relevant documents > pass to LLM > generate output
    chain = load_qa_chain(llm=llm, chain_type='stuff') #stuff = juntar tudo e passar para o LLM
    response = chain.run(input_documents = match, question = user_question)#a resposta não vai vir do ChatGPT puro, mas das informações salvas em "match"
    st.write(response)

#RUN CODE
#NO CONSOLE APARECERÁ: streamlit run C:\Users\Usuario\Desktop\AI for Beginners\chatbot.py
#RODAR NO TERMINAR (cuidado com pastas com espaço; se der erro entrar na pasta no terminal (usando cd pasta), e então streamlit run chatbot.py
#NO TERMINAL APARECERÁ CAMINHO PARA A PÁGINA CRIADA: Local URL: http://localhost:8502
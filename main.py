import os
import tempfile 
import streamlit as st 
from PIL import Image


from langchain.llms import OpenAI 
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#Setting Up the User interface: Title, subtile 

st.title("Chatbot: Interacting with your PDF Doc ")
st.subheader("Loading your PDF, Question and Response ")

# Inteface Image 
image = Image.open("PDF Chatbot.jpg")
st.image(image, use_column_width=True )

# Loading your file 
st.subheading = ('Upload your file')
users_file = st.file_uploader("", type=(['pdf', 'tsv', 'csv', 'txt', 'tab', 'xlsx', 'xls' ]))

temp_file_path = os.getcwd()
while users_file is None:
    x = 1
    
if users_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    full_file_path = os.path.join(temp_dir.name, users_file.name)
    with open(full_file_path, "wb") as temp_file:
        temp_file.write(users_file.read())
        
    st.write("The Path of Your file", temp_file_path)
    
    
    
#OpenAI APIKey
os.environ['OPENAI_API_KEY'] = ' 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'

#OPENAI LLM instance 
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()


# Create and load PDF Loader
loader = PyPDFLoader(temp_file_path)
# Split pages from pdf 
pages = loader.load_and_split()

store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')
#Converting Document to vector 
vectorstore_info = VectorStoreInfo(
    name="Pdf",
    description=" A pdf file to answer your questions",
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)


executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True

)



# Adding prompt Section 

prompt = st.text_input('Input ypu message')


if prompt:
    response = executor
    
    st.write(response)
    
    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)
    
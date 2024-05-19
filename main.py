
import getpass
import os

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# default payload
payload = {
    "model": "gpt-3.5-turbo-0125",
    "file_path": './ALL_Dialogues_in_friends-2.csv',
    "template": """You are supposed to be the person responding to the message like my partner. 
                Mimic the tone of Chandler in the context:{context} and 
                produce a answer based on the message: {message}"""
}


def process_payload(payload):
    """process the payload and return the required fields"""

    if "model" in payload and "file_path" in payload and "template" in payload:
        return payload["model"], payload["file_path"], payload["template"]
    else:
        raise ValueError("Please provide a file path in the payload")

def load_file(file_path):
    """load file from the provided path and return the loaded data"""

    if file_path.endswith('.csv'):
        loader = CSVLoader(file_path=file_path, csv_args={
            'delimiter': ',',
            'quotechar': '"',    
        })
        return loader.load()
    elif file_path.endswith('.txt'):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=0 
        )
        loader = TextLoader(file_path=file_path)
        return loader.load_and_split(
            text_splitter=text_splitter
        )
    else:
        raise ValueError("File type not supported, please upload a .csv or .txt file")        


def generate_embeddings(data, embeddings, persist_directory):
    """create embeddings"""

    vectorstore = Chroma.from_documents(
            documents=data, 
            embedding=embeddings, 
            persist_directory=persist_directory
    )
    
    return vectorstore


# generate message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from operator import itemgetter

persist_directory = "./embeddings/doc"
embeddings = OpenAIEmbeddings()
model, file_path, template = process_payload(payload)
data = load_file(file_path)
vectorstore = generate_embeddings(data, embeddings, persist_directory)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})

llm = ChatOpenAI(model=model)
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {
      
        "context":itemgetter("message") | retriever,
        "message": itemgetter("message"),

    }
    | prompt
    | llm
    | StrOutputParser()
)


while True:
    message = input(">> ")

    reply = chain.invoke({"message": message})

    print(reply)




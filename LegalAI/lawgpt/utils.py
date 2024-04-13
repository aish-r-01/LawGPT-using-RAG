# Importing Dependencies
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Faiss Index Path
FAISS_INDEX = "lawgpt/vectorstore/index.faiss/"
api_key = os.environ.get('GOOGLE_API_KEY')
# Custom prompt template
custom_prompt_template = """Use the context and answer the below question.
Context : {context}
Question : {question}
Answer 
"""

# Return the custom prompt template
def set_custom_prompt_template():
    #Set the custom prompt template for the LLMChain
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    #Load the LLM
    # API Key for Gemini
    gemini_api_key = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)
    return llm

# Return the chain
def retrieval_qa_chain(llm, prompt, db):
    #Create the Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def qa_pipeline():
    # Load the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()
    # Load the index
    db = FAISS.load_local("lawgpt/vectorstore/index.faiss/", embeddings, allow_dangerous_deserialization=True)
    # Load the LLM
    llm = load_llm()
    # Set the custom prompt template
    qa_prompt = set_custom_prompt_template()
    # Create the retrieval QA chain
    chain = retrieval_qa_chain(llm, qa_prompt, db)
    return chain
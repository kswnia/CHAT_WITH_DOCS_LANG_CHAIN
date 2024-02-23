from langchain.document_loaders import TextLoader
import textwrap
import os
import numpy as np
import faiss
os.environ["HUGGINGFACEHUB_API_TOKEN"] ="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"



loader = TextLoader("part_of_book.txt")
document = loader.load()

print(document)


def wrap_text_preserve_newlines(text, width = 110):
    
    lines= text.split('\n')

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


print(wrap_text_preserve_newlines(str(document[0])))



from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_documents(document)

print(docs)
print(len(docs))



from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import  FAISS 
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

query = "what book about design you know?"

doc = db.similarity_search(query) 

print(wrap_text_preserve_newlines(str(document[0].page_content)))



from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.8, "max_length": 512})

chain = load_qa_chain(llm, chain_type="stuff")

input_data = {
    'documents': 'docsResult' ,
    'question': "What book about design you know?"
}


queryText = "What book about design you know?"
docsResult = db.similarity_search(query)
chain.invoke(input={'input_documents': docsResult, 'question' : queryText})

input_data = {
    'documents': docsResult ,
    'question': "What book about design you know?"
}

# Then, call invoke with this 'input' argument
chain.invoke(input=input_data)

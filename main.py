from docx import Document
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

# Read .docx files and extract text
def read_docx(file_path):
    doc = Document(file_path)
    text = []
    
    for para in doc.paragraphs:
        text.append(para.text)
        
    return "\n".join(text)

data = []
folder = "documents"
for file in os.listdir(folder):
    if file.endswith(".docx"):
        text = read_docx(os.path.join(folder, file))
        data.append(text)

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)
documents = splitter.create_documents(data)

# Create embeddings and store in FAISS vector database
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
    #model_name="mixedbread-ai/mxbai-embed-large-v1"
)
# Create the vector database and save it locally
vector_db = FAISS.from_documents(
    documents,
    embedding
)
vector_db.save_local("admission_vector_db")

# Example query
query = input("Enter your question about college admission policies: ")

docs = vector_db.similarity_search(query, k=5)

# for d in docs:
#     print(d.page_content)

context = "\n\n".join([d.page_content for d in docs])
print(context)

prompt = f"""
You are an AI assistant helping students understand college admission policies.

Use ONLY the information from the context.
If there any mail id or contact number in the context, extract it and provide it in the answer more precisely.
If the answer is not in the context, say:
"I cannot find the answer in the admission policy."

Context:
{context}

Question:
{query}

Give a clear and structured answer.
"""

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

print("\nFinal Answer:\n")
print(response["message"]["content"])
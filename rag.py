import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import json
import re
import time
# Step 1: Load documents (example corpus)
documents = [
    "Prof. Shewta Dalvi is the HOD of GSS BCA",
    "Prof. Jeevan Bodas was the former HOD of GSS BCA",
    "The BCA department offers a 3-year undergraduate program",
    "The fee structure for BCA is 51,999 rs yearly",
    "GSS BCA has various events like Codeathon, Yuvatrang, Techspectra, Adios, Incite.",
    "Incite is the Fresher's Orientation Program",
    "Techspectra is a IT - Feast Event",
    "Codeathon is a 24 hours competitive coding competion",
    "Adios is Fare-well event for the Seniors's Batch",
    "Yuvatrang is a Multi Cultural Fest enevt",
    "You can Contact-Us on +91 9999-0000-9999 ",
    "Here is our web-site you can reach on 'http://gssbca.org'",
  
]

# Step 2: Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# Function to retrieve relevant documents
def retrieve_docs(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)
    most_similar_idx = similarities.argmax()
    return documents[most_similar_idx]

# Step 3: Query the RAG system
def rag_query(query):
    relevant_doc = retrieve_docs(query)
    print(f"Retrieved Document: {relevant_doc}")
    
    # Step 4: Send the relevant document and query to Ollama API with configuration
    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": f"You are a helpful AI assistant which guides new student with their queries about GSS BCA college. Use this context to answer:\n{relevant_doc}\n\nQuery: {query}",
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )
   
    # Handle the raw response text
        # Handle the raw response text
    raw_response = response.text
     # Extract all 'response' values from the JSON objects
    responses = re.findall(r'"response":\s*"([^"]+)"', raw_response)
    # Combine all parts into a complete sentence
    final_response = "".join(responses)
    print("A.I : ", end="", flush=True)
    for char in final_response:
        print(char, end="", flush=True)
        time.sleep(0.05)
    print()

# Example Query
while (True):
    user = input("User : ")
    rag_query(user)

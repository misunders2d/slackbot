import os
from dotenv import load_dotenv
load_dotenv()

import firebase_admin
from firebase_admin import firestore,  credentials
firebase_credentials = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
}
cred = credentials.Certificate(firebase_credentials)
from functools import cache
from numpy import dot, array
from numpy.linalg import norm
from typing import Final
from openai import OpenAI, NotFoundError # future development
from openai.types.beta.thread import Thread # future development


OPENAI_KEY: Final = os.getenv('KNOWLEDGE_BASE_AI_KEY')

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

client = OpenAI(api_key = OPENAI_KEY)

@cache
def fetch_data_from_db():
    stream = db.collection('knowledge').stream()
    all_texts = []
    for item in stream:
        data = item.to_dict()
        record = {
            'key':item.id,
            'embedding':data['embedding'],
            'problem':data['problem'],
            'solution':data['solution'],
            'date_created':item.create_time.date(),
            'date_modified':item.update_time.date()
            }
        all_texts.append(record)
    return all_texts

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(query_vector, stored_vector):
    return dot(query_vector, stored_vector) / (norm(query_vector) * norm(stored_vector))

def upload_texts_to_vector(texts:list):
    for text in texts:
        problem, solution = text
        embedding = get_embedding('\n\n'.join(text))
        db.collection('knowledge').add({"problem": problem, "solution": solution, "embedding": embedding})
        
def delete_record_from_vector(key: str):
    db.collection('knowledge').document(key).delete()
    
def modify_record_vector(key: str, text: tuple):
    problem, solution = text
    embedding = get_embedding('\n\n'.join(text))
    db.collection('knowledge').document(key).update({"problem": problem, "solution": solution, "embedding": embedding})

def vector_search(query:str, top_n:int=5):
    query_embedding = get_embedding(query)
    all_texts = fetch_data_from_db()    

    distances = {}
    for record in all_texts:
        embedding = array(record['embedding'])
        distance = cosine_similarity(query_embedding, embedding)
        distances[record['key']] = {
            'problem':record['problem'],
            'solution':record['solution'],
            'date_created':record['date_created'],
            'date_modified':record['date_modified'],
            'prob':distance} 

    # Sort based on distance and take the top N
    results = dict(sorted(distances.items(), key=lambda item: item[1]['prob'], reverse=True)[:top_n])
    return results

def get_response(query, search_results):
    pre_prompt = f'''
                Below is the database search for my question "{query}".
                Please summarize and structure the search to best answer my question. If there is already a structure in the search - keep it reasonably intact.
                Please drop irrelevant results from your summary, but make sure to keep all links, file references and tool mentions.
                If there is a specific answer to my question in the search results - please answer it first, and then summarize the rest.
                If there is not enough information in search results to answer user's question - let the user know about it and try to answer with your own knowledge.
                Try to answer the user in the language which he used to ask the question, if possible.
                Make sure to include the "created" date and also "modified" date if it's different from the date of creation.
                '''
    response = client.chat.completions.create(
        messages = [
            {'role':'user','content':pre_prompt},
            {'role':'user','content':search_results}
            ],
        model = 'gpt-4o-mini',
        stream = False
    )
    return response

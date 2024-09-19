import os
from dotenv import load_dotenv
load_dotenv()
import uuid
from datetime import date

from typing import Final
from openai import OpenAI, NotFoundError # future development
from openai.types.beta.thread import Thread # future development

from pinecone import Pinecone, ServerlessSpec

OPENAI_KEY: Final = os.getenv('KNOWLEDGE_BASE_AI_KEY')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

NAMESPACE = 'db'

client = OpenAI(api_key = OPENAI_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('knowledge-db')

def fetch_data_from_db():
    all_keys = [x for x in index.list(namespace = NAMESPACE)][0]
    all_vectors = index.fetch(ids = all_keys, namespace=NAMESPACE)
    data = [all_vectors.vectors[key]['metadata'] for key in all_vectors.vectors]
    return data

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_record(problem, solution):
    text = '\n\n'.join((problem,solution))
    embedding = get_embedding(text)
    vectors = [
        {
            'id':str(uuid.uuid4()),
            'values':embedding,
            'metadata':{
                'problem':problem,
                'solution':solution,
                'date_created':str(date.today()),
                'date_modified':str(date.today())
                }
            }
        ]
    result = index.upsert(vectors = vectors, namespace = NAMESPACE)    
    return result


def delete_record_from_vector(key: str):
    index.delete(ids=[key], namespace=NAMESPACE)
    
def modify_record_vector(key: str, text: tuple):
    current_record = index.fetch(ids = [key], namespace = NAMESPACE)
    problem, solution = text
    embedding = get_embedding('\n\n'.join(text))
    index.update(
        id=key,
        values=embedding,
        set_metadata={
            "problem": problem, "solution": solution,
            'date_created':current_record.vectors[key]['metadata']['date_created'],
            'date_modified':str(date.today())
            },
            namespace=NAMESPACE
        )

def vector_search(query_str: str):
    query_emb = get_embedding(query_str)
    results = index.query(
        namespace=NAMESPACE,
        vector=query_emb,
        top_k=5,
        include_values=False,
        include_metadata=True
    )
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

import os
from dotenv import load_dotenv
load_dotenv()
import uuid
from datetime import date

from typing import Final
from openai import OpenAI, NotFoundError # future development
from openai.types.beta.thread import Thread # future development

from pinecone import Pinecone, ServerlessSpec

OPENAI_KEY: Final = os.getenv('KNOWLEDGE_BASE_AI_KEY') # replace with your OpenAI api key
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY') # replace with your Pinecone api key

NAMESPACE = 'db' # This is the name of your relevant namespace in Pinecone db

client = OpenAI(api_key = OPENAI_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('knowledge-db') # this is the name of your Index in Pinecone (the actual storage)

def fetch_data_from_db():
    """
    Fetches data from the database using a specified namespace.

    This function retrieves all keys from the specified namespace and
    fetches the corresponding vectors from the database. It then extracts
    the metadata associated with each vector and returns it as a list.

    Returns:
        list: A list of metadata dictionaries corresponding to the fetched vectors.

    Raises:
        SomeException: Raises an exception if there is an issue with fetching data
                       from the database (replace with specific exceptions as needed).

    Example:
        >>> metadata = fetch_data_from_db()
        >>> print(metadata)
        [{'key1': 'value1'}, {'key2': 'value2'}, ...]
    """    
    all_keys = [x for x in index.list(namespace = NAMESPACE)][0]
    all_vectors = index.fetch(ids = all_keys, namespace=NAMESPACE)
    data = [all_vectors.vectors[key]['metadata'] for key in all_vectors.vectors]
    return data

def get_embedding(text):
    """
    Generates an embedding for the given text using a specified model.

    This function sends the input text to the API and retrieves its corresponding
    embedding vector. The embedding can be used for various applications such as
    text similarity, classification, or feeding into machine learning models.

    Args:
        text (str): The input text for which the embedding is to be generated.

    Returns:
        list: A list representing the embedding vector of the input text.

    Raises:
        SomeException: Raises an exception if there is an issue with API request
                       or the input text is invalid (replace with specific exceptions as needed).

    Example:
        >>> embedding = get_embedding("Sample text for embedding.")
        >>> print(embedding)
        [0.123, 0.456, 0.789, ...]
    """    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_record(problem, solution):
    """
    Adds a record to the database with the specified problem and solution.

    This function creates an embedding for the combined problem and solution text,
    generates a unique ID for the record, and stores the embedding along with its
    associated metadata in the specified namespace of the index.

    Args:
        problem (str): The description of the problem to be recorded.
        solution (str): The solution corresponding to the problem.

    Returns:
        ResultType: The result of the upsert operation (replace with the specific return type).

    Raises:
        SomeException: Raises an exception if the embedding generation or database
                       operation fails (replace with specific exceptions as needed).

    Example:
        >>> result = add_record("Problem description", "Solution description")
        >>> print(result)
        {'success': True, ...}
    """    
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
    '''simple function to delete the record from Pinecone using record key'''
    index.delete(ids=[key], namespace=NAMESPACE)
    
def modify_record_vector(key: str, text: tuple):
    """
    Modifies the existing record in the database with the specified ID.

    This function fetches the current record associated with the provided key,
    generates a new embedding for the updated problem and solution text,
    and updates the record in the index with the new values and metadata.

    Args:
        key (str): The unique identifier of the record to be modified.
        text (tuple): A tuple containing the updated problem and solution strings.

    Returns:
        None: This function does not return a value.

    Raises:
        KeyError: Raises an exception if the specified record does not exist.
        SomeException: Raises other exceptions if the update operation fails
                       (replace with specific exceptions as needed).

    Example:
        >>> modify_record_vector("unique_record_id", ("Updated problem description", "Updated solution description"))
    """    
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
    """
    Performs a vector search in the database based on the provided query string.

    This function generates an embedding for the input query string and
    retrieves the top matching records from the specified namespace in the index.

    Args:
        query_str (str): The search string for which to find similar embeddings.

    Returns:
        list: A list of results that are the closest matches based on the vector representation
              of the query, including their metadata (and possibly other information depending 
              on the implementation).

    Raises:
        SomeException: Raises an exception if there is an issue with the query operation
                       or embedding generation (replace with specific exceptions as needed).

    Example:
        >>> results = vector_search("Sample query text")
        >>> print(results)
        [{'id': 'record1', 'metadata': {...}}, {'id': 'record2', 'metadata': {...}}, ...]
    """    
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
    """
    Generates a structured response based on a user query and corresponding search results.

    This function creates a prompt that instructs the model to summarize and structure
    the provided search results, focusing on answering the user's question effectively.
    It includes guidelines for maintaining the relevance of results, retaining links and
    references, and conveying any specific answers present in the search results.

    Args:
        query (str): The user's question for which the response is sought.
        search_results (str): The raw search results retrieved from the database, formatted as a string.

    Returns:
        ResponseType: The response generated by the model, encapsulating the summarized information
                      and relevant details (replace with the specific return type).

    Raises:
        SomeException: Raises an exception if there is an issue with the API request or
                       processing the response (replace with specific exceptions as needed).

    Example:
        >>> result = get_response("What is the latest update on the project?", search_results)
        >>> print(result)
        "The latest update is as follows... [Links and references included]"
    """    
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
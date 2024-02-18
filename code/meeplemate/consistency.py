import copy
from enum import Enum
from operator import itemgetter
from functools import partial
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers.regex import RegexParser
from langchain_core.output_parsers import StrOutputParser

def embed_responses(model:SentenceTransformer, question, responses):
    tokenizer = model.tokenizer

    # Combine each response with the question
    responses = [f"{question}{tokenizer.sep_token}{response}" for response in responses]

    return model.encode(responses, normalize_embeddings=True)

def find_centroid(embeddings):
    # Convert the list of embeddings to a 2D numpy array if it's not already
    embeddings_array = np.array(embeddings)
    
    # Calculate the centroid by computing the mean of the embeddings array
    centroid = np.mean(embeddings_array, axis=0)

    return centroid

def centroid_of_largest_cluster(embeddings):
    # Perform K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    
    # Get the cluster labels for each embedding
    labels = kmeans.labels_
    
    # Determine the size of each cluster
    cluster_sizes = np.bincount(labels)
    
    # Identify the largest cluster
    largest_cluster = np.argmax(cluster_sizes)
    
    # Return the centroid of the largest cluster
    largest_cluster_centroid = kmeans.cluster_centers_[largest_cluster]

    # Normalize the centroid
    # normalized_centroid = normalize(largest_cluster_centroid.reshape(1, -1), norm='l2')
    centroid = largest_cluster_centroid.reshape(1, -1)
    
    return centroid

def centroids_sorted_by_cluster_size(embeddings, n_clusters=2):
    # Perform K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    
    # Get the cluster labels for each embedding
    labels = kmeans.labels_
    
    # Determine the size of each cluster
    cluster_sizes = np.bincount(labels)
    
    # Sort cluster labels by size (ascending)
    sorted_cluster_labels = np.argsort(cluster_sizes)

    # Return the centroids of the clusters, sorted by size
    return [kmeans.cluster_centers_[i].reshape(1, -1) for i in sorted_cluster_labels]

def sort_embedding_indices(embeddings, query_vector):
    # Ensure the query_vector is a 2D array for cosine_similarity function
    query_vector_2d = query_vector.reshape(1, -1)
    
    # Calculate cosine similarities between the query_vector and all embeddings
    similarities = cosine_similarity(query_vector_2d, embeddings)
    
    # Sort the indices of the embeddings based on their similarity to the query_vector
    sorted_indices = np.argsort(similarities.flatten())[::-1]

    return sorted_indices

def sort_embedding_indices_by_avg_similarity(embeddings):
    # Calculate pair-wise cosine similarities between all embeddings
    similarities = cosine_similarity(embeddings)

    # Find the average similarity for each
    avg_similarity = np.mean(similarities, axis=1)

    # Sort the indices of the embeddings based on their average similarity
    sorted_avg_similarity = np.argsort(avg_similarity)[::-1]
    
    return sorted_avg_similarity

def rank_responses_by_majority_consensus(embedding_model, question, responses):
    # Sort responses for reproducibility
    responses = responses[:]
    responses.sort()

    # Create the embeddings for the responses
    embeddings = embed_responses(embedding_model, question, responses)
    
    # Find the centroid of the embeddings
    
    # centroid = centroid_of_largest_cluster(embeddings)
    # centroids = centroids_sorted_by_cluster_size(embeddings)
    # centroid = centroids[-1]

    sorted_indices = sort_embedding_indices_by_avg_similarity(embeddings)

    # Pray there are more correct responses than wrong ones so that the centroid
    # is closer to a correct answer than a wrong one
    # centroid = find_centroid(embeddings)
    # sorted_indices = sort_embedding_indices(embeddings, centroid)

    # Return the response closest to the centroid
    return [responses[i] for i in sorted_indices]

consensus_prompt_template = """\
I have generated the following responses to the question: {question}

{responses}

Evaluate these responses.
Select the most consistent response based on majority consensus.
Start your answer with "The most consistent response is Response X" (without \
quotes)
"""

DEFAULT_CONSENSUS_PROMPT = ChatPromptTemplate.from_template(
    consensus_prompt_template
)

consensus_prompt_with_context_template = """\
Evaluate the following responses to a board game question based on the provided rules. Among the responses, select the most consistent response based on majority consensus. Start your answer with "The most consistent response is Response X" (without quotes).

> Rules:
>>>
{context}
>>>
> Question: {question}
> Responses:
>>>
{responses}
>>>
> Most consistent response (Start your answer with "The most consistent response is Response X" (without quotes)):
"""

DEFAULT_CONSENSUS_WITH_CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    consensus_prompt_with_context_template
)

document_prompt_template = """{page_content}"""

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(document_prompt_template)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def build_run_ntimes_chain(chain:Runnable, n:int) -> Runnable:
    def duplicate(x):
        # TODO: Should we do a deepcopy here or is that overkill?
        return [copy.deepcopy(x) for _ in range(n)]

    return RunnableLambda(duplicate) | chain.map()


SelectionStrategy = Literal["average_similarity", "centroid", "prompting", "prompting_with_context"]


def format_responses(responses):
    return "\n\n".join(
        [f"Response {i}: {getattr(response, 'content', response)}" for i, response in enumerate(responses)]
    )


def build_prompting_selection_chain(chat_model, target_chain_response_key="answer", prompt=DEFAULT_CONSENSUS_PROMPT):
    output_parser = RegexParser(regex=r".*The most consistent response is Response (\d+).*", output_keys=["response_index"])

    response_index_chain = (
        # {"responses": itemgetter("responses") | RunnableLambda(format_responses), "question": itemgetter("question"), "context": RunnableLambda(itemgetter("documents")) | partial(_combine_documents, document_prompt=DEFAULT_DOCUMENT_PROMPT)}
        {
            "responses": itemgetter("responses") | RunnableLambda(itemgetter(target_chain_response_key)).map() | RunnableLambda(format_responses),
            "question": itemgetter("question"),
            "context": RunnableLambda(itemgetter("documents")) | partial(_combine_documents, document_prompt=DEFAULT_DOCUMENT_PROMPT)
        }
        | prompt
        | chat_model
        | output_parser
        | itemgetter("response_index") 
        | RunnableLambda(int)
    )
    
    pluck_response_chain = RunnableLambda(lambda x: x["responses"][x["response_index"]])
    first_response_chain = RunnableLambda(lambda x: x["responses"][0])

    chain = (
        RunnablePassthrough.assign(response_index=response_index_chain)
        | pluck_response_chain
    ).with_fallbacks([first_response_chain], exceptions_to_handle=[IndexError, ValueError])

    return chain


def build_universal_consistency_chain(embedding_model, target_chain:Runnable, chat_model=None, target_chain_response_key="answer", samples=9, response_selection_strategy:SelectionStrategy="average_similarity") -> Runnable:
    if ("prompting" in response_selection_strategy) and chat_model is None:
        raise ValueError("chat_model must be provided when using PROMPTING or PROMPTING_WITH_CONTEXT")

    # chain that runs target_chain n times and produces a list of responses
    multiple_responses_chain = build_run_ntimes_chain(target_chain, samples)

    def find_majority_consensus_response(data):
        responses = data["responses"]
        question = data["question"]
        # Create a list of response strings
        response_strings = [response[target_chain_response_key].content for response in responses]
        # Create a list of responses ranked by majority consensus
        ranked_responses = rank_responses_by_majority_consensus(embedding_model, question, response_strings)
        print("*** Ranked responses ***")
        print(ranked_responses)
        # Find index of the most consistent response
        response_index = response_strings.index(ranked_responses[0])
        return responses[response_index]
    
    # chain that takes a list of responses and a question and returns the majority consensus response
    if response_selection_strategy == "average_similarity":
        consensus_response_chain = RunnableLambda(find_majority_consensus_response)
    elif response_selection_strategy == "prompting":
        consensus_response_chain = build_prompting_selection_chain(chat_model, target_chain_response_key=target_chain_response_key)
    elif response_selection_strategy == "prompting_with_context":
        consensus_response_chain = build_prompting_selection_chain(chat_model, target_chain_response_key=target_chain_response_key, prompt=DEFAULT_CONSENSUS_WITH_CONTEXT_PROMPT)

    # consensus_response_chain = RunnableLambda(find_majority_consensus_response)

    # put it all together
    consistency_chain = (
        {"responses": multiple_responses_chain, "question": itemgetter("question"), "documents": itemgetter("documents")}
        | consensus_response_chain
    )

    return consistency_chain.with_config({"run_name": "universal-consistency"})
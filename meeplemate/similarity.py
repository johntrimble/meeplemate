from functools import partial
from typing import TypedDict, Any, List
import numpy as np


class Response(TypedDict):
    text: str
    tokens: list[Any]
    logprobs: list[float]


def embedding_similarity_score(
        embedding_model,
        responses:List[Response],
        weighted:bool=False
) -> np.ndarray:
    response_strings = [response["text"] for response in responses]
    vectors = embedding_model.encode(response_strings, normalize_embeddings=True)
    if weighted:
        logprobs = [response["logprobs"] for response in responses]
        weights = np.exp(np.average(logprobs, axis=1))
        vectors = vectors * weights[:, np.newaxis]
    dot_product = vectors.dot(vectors.T)
    return dot_product


def build_embedding_similarity_scorer(embedding_model, weighted=False):
    return partial(embedding_similarity_score, embedding_model, weighted=weighted)


def extract_ngrams(tokens, logprobs=None, ngrams=1):
    """
    Extracts a mapping of ngrams to their logprobs for each occurence of the
    ngram in the tokens. If no logprobs are provided, the logprobs are assumed
    to be 0 for all tokens.
    """
    ngram_data = {}

    if logprobs is None:
        logprobs = [0 for _ in tokens]

    tokens = tuple(tokens)

    for k in range(1, ngrams+1):
        for i in range(len(tokens)-k+1):
            ngram = tokens[i:i+k]
            logprob = logprobs[i:i+k]
            logprob = sum(logprob)
            ngram_data.setdefault(ngram, []).append(logprob)
    
    return ngram_data


def extract_ngrams(tokens, logprobs=None, ngrams=1):
    """
    Extracts a mapping of ngrams to their logprobs for each occurence of the
    ngram in the tokens. If no logprobs are provided, the logprobs are assumed
    to be 0 for all tokens.
    """
    ngram_data = {}

    if logprobs is None:
        logprobs = [0 for _ in tokens]

    tokens = tuple(tokens)

    for k in range(1, ngrams+1):
        for i in range(len(tokens)-k+1):
            ngram = tokens[i:i+k]
            logprob = logprobs[i:i+k]
            logprob = sum(logprob)
            ngram_data.setdefault(ngram, []).append(logprob)
    
    return ngram_data


def compute_normalized_probs(ngram_data, generation_length, ngrams=1):
    """
    Compute the normalized probability for each ngram in the ngram data. The
    normalized probability is the average probability of the ngram occuring
    in the response, normalized by the length of the response and the size of
    the ngrams.
    """
    normalized_probs = {}
    for ngram, logprobs in ngram_data.items():
        # This comes from the paper, and it looks like there has been 3
        # revisions of it, so I'm going to assume they got this right.
        # However, I don't understand the "- 1" in the denominator,
        # shouldn't it be a "+ 1" instead so that the coefficient has a
        # value of 1 for unigrams?
        normalize_prob_ngram_size_coeff = (
            generation_length 
            / 
            (generation_length - len(ngram) - 1) 
            if ngrams > 1 else 1
        )

        # Probability for each occurrence of the ngram
        ngram_probs = np.exp(logprobs)

        # Average probability for the ngram
        avg_ngram_prob = np.average(ngram_probs)

        # Normalized probability for the ngram
        normalized_ngram_prob = normalize_prob_ngram_size_coeff * avg_ngram_prob

        # Store the normalized probability
        normalized_probs[ngram] = normalized_ngram_prob
    
    return normalized_probs


def build_ngram_vectors(responses, ngrams=1, weighted=True, generation_weighted=True):
    """
    Build n-gram vectors optionally weighting by n-gram probability and the
    length normalized probability of the response.
    """
    sample_ngrams = []
    all_ngrams = set()
    for response in responses:
        if weighted or generation_weighted:
            assert response.get("logprobs") is not None, (
                "Logprobs are required for weighted ngram vectors."
            )
            logprobs = response["logprobs"]
        else:
            logprobs = None
        ngram_data = extract_ngrams(response["tokens"], logprobs if weighted else None, ngrams=ngrams)
        ngram_data = compute_normalized_probs(ngram_data, len(response["tokens"]), ngrams=ngrams)

        if generation_weighted:
            generation_probability = np.exp(np.average(logprobs))
            for ngram in ngram_data:
                ngram_data[ngram] *= generation_probability

        sample_ngrams.append(ngram_data)
        all_ngrams.update(ngram_data.keys())
    
    # As an ordered sequence
    all_ngrams = list(all_ngrams)

    # Build the matrix of ngram vectors as np array
    ngram_vectors = np.zeros((len(responses), len(all_ngrams)))
    for i, _ngrams in enumerate(sample_ngrams):
        for j, ngram in enumerate(all_ngrams):
            ngram_vectors[i, j] = _ngrams.get(ngram, 0)
    
    return ngram_vectors, all_ngrams


def _ngram_consistency_score(responses, ngrams=1, weighted=True, generation_weighted=True, base_vocab_size_per_pair=False):
    # Build the ngram vectors
    ngram_vectors, ngram_order = build_ngram_vectors(responses, ngrams=ngrams, weighted=weighted, generation_weighted=generation_weighted)
    
    # Compute the dot product between every pair of vectors
    dot_products = np.dot(ngram_vectors, ngram_vectors.T)

    # Normalize score by vocab size
    if base_vocab_size_per_pair:
        # Construct vocab_size matrix s.t. vocab_size[i, j] = |ngram_vectors[i] U ngram_vectors[j]|
        # Element-wise add each pair of vectors
        vocab_size = np.add(ngram_vectors[:, np.newaxis], ngram_vectors)
        # Compute the number of non-zero elements in each pair of vectors
        vocab_size = np.sum(vocab_size > 0, axis=2)
        scores = dot_products / vocab_size
    else:
        scores = (1 / len(ngram_order)) * dot_products

    return scores


def ngram_consistency_score(responses, ngrams=1):
    return _ngram_consistency_score(responses, ngrams=ngrams, weighted=False, generation_weighted=False)


def weighted_ngram_consistency_score(responses, ngrams=1):
    return _ngram_consistency_score(responses, ngrams=ngrams, weighted=True, generation_weighted=False)


def consensus_weighted_ngram_consistency_score(responses, ngrams=1):
    return _ngram_consistency_score(responses, ngrams=ngrams, weighted=True, generation_weighted=True)


def generalized_self_consistency_score(similarity_scores):
    # Instatiate the array of self-consistency scores
    self_consistency_scores = np.zeros(len(similarity_scores))

    # Compute the self-consistency score for each response
    # Note: We ignore the similarity of a response with itself
    for i in range(len(similarity_scores)):
        self_consistency_scores[i] = (
            (np.sum(similarity_scores[i]) - similarity_scores[i, i]) 
            / 
            (len(similarity_scores) - 1)
        )
    
    return self_consistency_scores

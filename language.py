# function to estimate the probability of a word
def estimate_probability(word: str, previous_n_gram: tuple, n_gram_counts: dict, n_plus1_gram_counts: dict,vocabulary_size: int,
                         k: float=1.0) -> float:

    previous_n_gram = tuple(previous_n_gram)
    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
    n_plus1_gram = previous_n_gram + (word,)  
    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)       
    return (n_plus1_gram_count + k)/(previous_n_gram_count + k * vocabulary_size)

# function to estimate the probabilities of all the words present in the vocabulary
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):

    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

# function to suggest a word based on the predicted word by the base model, The similar word is predicted by the n_gram model present in the NLP directory.
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):

    n = len(list(n_gram_counts.keys())[0]) 
    previous_n_gram = previous_tokens[-n:]
    # calling the estimate_probabilities
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k)

    suggestion = None
    max_prob = 0

    for word, prob in probabilities.items(): 
        if start_with is not None: 
            if not word.startswith(start_with): 
                continue 
        if prob > max_prob: 
            suggestion = word
            max_prob = prob

    return suggestion, max_prob

# function to take inputs like, n_gram_counts_list and previous tokens along with the vocabulary
def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    n_gram_counts = n_gram_counts_list[0]
    n_plus1_gram_counts = n_gram_counts_list[1]
    # calling suggest_a_word for suggestion of a word
    suggestion = suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=k, start_with=start_with)
    return " " + str(suggestion[0])



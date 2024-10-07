import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')

def select_relevent_descriptions(df, max_descriptions_per_product=1250):
    """
    Selects a diverse set of descriptions for each product code in the DataFrame.

    This function aims to select the most interesting and diverse descriptions
    for each product code, using TF-IDF and cosine similarity to measure uniqueness.

    Args:
    df (pd.DataFrame): Input DataFrame containing 'ProductCode' and 'CleanDescription' columns.
    max_descriptions_per_product (int): Maximum number of descriptions to keep for each product code.

    Returns:
    pd.DataFrame: A new DataFrame with selected diverse descriptions.
    """
    df = df.dropna(subset=["CleanDescription"])
    product_groups = df.groupby('ProductCode')
    selected_rows = []

    for product_code, group in product_groups:
        if len(group) <= max_descriptions_per_product:
            selected_rows.append(group)
            continue
        
        unique_descriptions = group.drop_duplicates(["CleanDescription"])
        descriptions = unique_descriptions['CleanDescription'].tolist()

        # Tokenize descriptions into sentences
        sentences = []
        for desc in descriptions:
            if isinstance(desc, str):
                sentences.extend(nltk.sent_tokenize(desc))
        
        if not sentences:
            continue
        
        # Calculate TF-IDF scores for sentences
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate pairwise cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        mean_similarities = cosine_similarities.mean(axis=1)

        # Select the most diverse sentences based on low mean cosine similarity
        diverse_sentence_indices = np.argsort(mean_similarities)[:max_descriptions_per_product]
        diverse_sentences = [sentences[i] for i in diverse_sentence_indices]

        # Keep rows with diverse sentences
        diverse_rows = unique_descriptions[unique_descriptions['CleanDescription'].apply(
            lambda desc: isinstance(desc, str) and any(sent in desc for sent in diverse_sentences))]

        selected_rows.append(diverse_rows)

        # If not enough diverse rows, fill up with additional rows
        if len(diverse_rows) < max_descriptions_per_product:
            additional_rows_needed = max_descriptions_per_product - len(diverse_rows)
            remaining_rows = unique_descriptions[~unique_descriptions.index.isin(diverse_rows.index)]
            
            if not remaining_rows.empty:
                remaining_descriptions = remaining_rows['CleanDescription'].tolist()
                additional_rows = []

                # Calculate the frequency of each diverse sentence in the remaining descriptions
                sentence_freq = Counter()
                for desc in remaining_descriptions:
                    if isinstance(desc, str):
                        for sent in diverse_sentences:
                            if sent in desc:
                                sentence_freq[sent] += 1

                # Invert frequencies to give higher weight to less frequent sentences
                total_count = sum(sentence_freq.values())
                sentence_weights = {sent: (total_count - freq) for sent, freq in sentence_freq.items()}

                # Select additional rows based on these weights
                for _ in range(additional_rows_needed):
                    selected_sent = np.random.choice(diverse_sentences, p=[sentence_weights[sent] / sum(sentence_weights.values()) for sent in diverse_sentences])
                    matching_rows = remaining_rows[remaining_rows['CleanDescription'].apply(
                        lambda desc: isinstance(desc, str) and selected_sent in desc)]
                    if not matching_rows.empty:
                        selected_row = matching_rows.sample(n=1).iloc[0]
                        additional_rows.append(selected_row)
                        remaining_rows = remaining_rows.drop(selected_row.name)

                # Convert additional rows back to DataFrame and append
                if additional_rows:
                    additional_rows_df = pd.DataFrame(additional_rows)
                    selected_rows.append(additional_rows_df)

    # Concatenate all the selected rows into a single DataFrame
    result_df = pd.concat(selected_rows).reset_index(drop=True)

    return result_df

if __name__ == "__main__":
    # Example usage
    pass
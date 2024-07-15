import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter

# Téléchargez les ressources nécessaires de NLTK
nltk.download('punkt')

def get_top_n_interesting_phrases(df, n=1000):
    df = df.dropna(subset=["CleanDescription"])
    product_groups = df.groupby('ProductCode')
    rows_to_keep = []

    for product_code, group in product_groups:
        if len(group) <= n:
            rows_to_keep.append(group)
            continue
        
        unique_descriptions = group.drop_duplicates(["CleanDescription"])
        
        descriptions = unique_descriptions['CleanDescription'].tolist()

        # Tokenize sentences
        sentences = []
        for desc in descriptions:
            if isinstance(desc, str):  # Ensure the description is a string
                sentences.extend(nltk.sent_tokenize(desc))
        
        if not sentences:
            continue
        
        # Calculate TF-IDF scores
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate pairwise cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        mean_similarities = cosine_similarities.mean(axis=1)

        # Select the top N most interesting sentences based on mean cosine similarity
        top_sentence_indices = np.argsort(mean_similarities)[:n]
        top_sentences = [sentences[i] for i in top_sentence_indices]

        # Keep rows with top sentences
        interesting_rows = unique_descriptions[unique_descriptions['CleanDescription'].apply(
            lambda desc: isinstance(desc, str) and any(sent in desc for sent in top_sentences))]

        rows_to_keep.append(interesting_rows)

        # If not enough interesting rows, fill up with additional rows
        if len(interesting_rows) < n:
            additional_rows_needed = n - len(interesting_rows)
            remaining_rows = unique_descriptions[~unique_descriptions.index.isin(interesting_rows.index)]
            
            if not remaining_rows.empty:
                remaining_descriptions = remaining_rows['CleanDescription'].tolist()
                additional_rows = []

                # Calculate the frequency of each top sentence in the remaining descriptions
                sentence_freq = Counter()
                for desc in remaining_descriptions:
                    if isinstance(desc, str):
                        for sent in top_sentences:
                            if sent in desc:
                                sentence_freq[sent] += 1

                # Invert frequencies to give higher weight to less frequent sentences
                total_count = sum(sentence_freq.values())
                sentence_weights = {sent: (total_count - freq) for sent, freq in sentence_freq.items()}

                # Select additional rows based on these weights
                for _ in range(additional_rows_needed):
                    selected_sent = np.random.choice(top_sentences, p=[sentence_weights[sent] / sum(sentence_weights.values()) for sent in top_sentences])
                    matching_rows = remaining_rows[remaining_rows['CleanDescription'].apply(
                        lambda desc: isinstance(desc, str) and selected_sent in desc)]
                    if not matching_rows.empty:
                        selected_row = matching_rows.sample(n=1).iloc[0]
                        additional_rows.append(selected_row)
                        remaining_rows = remaining_rows.drop(selected_row.name)

                # Convert additional rows back to DataFrame and append
                if additional_rows:
                    additional_rows_df = pd.DataFrame(additional_rows)
                    rows_to_keep.append(additional_rows_df)

    # Concatenate all the interesting rows into a single DataFrame
    result_df = pd.concat(rows_to_keep).reset_index(drop=True)

    return result_df

if __name__ == "__main__":
    
    df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\Merged_df_2024-06-11_0.xlsx")
    filtered_df = get_top_n_interesting_phrases(df, n=1000)
    filtered_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\MAX1000_Merged_and_cleand_df_2024-06-11_0.xlsx", index=False)


    
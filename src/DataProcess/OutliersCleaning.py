import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from annoy import AnnoyIndex

from MatrixTree.MatrixTreesProcess import compute_matrices_distance

# Fonction pour trouver des paires similaires avec Annoy
def find_similar_pairs(annoy_index, vectors, df, threshold=0.8, top_k=10):
    similar_pairs = []
    for i, vector in enumerate(vectors):
        if i%10000==0:
            print(f"Vector {i}/{len(vectors)}")
        indices, distances = annoy_index.get_nns_by_vector(vector, top_k, include_distances=True)
        for j, distance in zip(indices, distances):
            if i != j and distance < (1 - threshold) and df['ProductCode'][i] != df['ProductCode'][j]:
                similar_pairs.append((df['SampleCode'][i], df['Laboratory'][i], df['CleanDescription'][i], df['ProductCode'][i], df['ProductName'][i], df['SampleCode'][j], df['Laboratory'][j], df['CleanDescription'][j], df['ProductCode'][j], df['ProductName'][j], 1 - distance))
    return similar_pairs

def extract_description_productcode_error(df, threshold=0.8):

    def _sort_codes(row):
        codes = sorted([row['ProductCode1'], row['ProductCode2']])
        return pd.Series(codes, index=['SortedCode1', 'SortedCode2'])

    # Prétraitement des descriptions
    df['CleanDescription'] = df['CleanDescription'].str.lower()

    df = df.dropna(subset=["CleanDescription", "ProductCode"])
    df = df.drop_duplicates(subset=["CleanDescription", "ProductCode"])
    
    print(f"Df is {len(df)} row long")

    # df = df.sample(n=1000)

    df.reset_index(inplace=True)

    # Transformation des descriptions en vecteurs TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['CleanDescription'])
    print("TFIDF is done")

    # Réduction de la dimensionnalité avec Truncated SVD
    n_components = 2000 # Ajustez selon vos besoins
    svd = TruncatedSVD(n_components=n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    tfidf_matrix_reduced = lsa.fit_transform(tfidf_matrix)
    print("Reduction of the dimension is done")

    # Vérification de la taille de la matrice réduite
    print(f"Old shape{tfidf_matrix.shape}, new shape {tfidf_matrix_reduced.shape}")

    n_trees = 20
    vector_length = tfidf_matrix_reduced.shape[1]

    # Construction de l'index Annoy
    annoy_index = AnnoyIndex(vector_length, 'angular')

    for i, vector in enumerate(tfidf_matrix_reduced):
        annoy_index.add_item(i, vector)

    annoy_index.build(n_trees)

    print("Annoy is build")

    similar_pairs = find_similar_pairs(annoy_index, tfidf_matrix_reduced, df, threshold=threshold)
    print("Similar_pairs are found")

    similar_df = pd.DataFrame(similar_pairs, columns=['SampleCode1', "Laboratory1", 'Description1', 'ProductCode1', 'ProductName1', 'SampleCode2', "Laboratory2", 'Description2', 'ProductCode2', 'ProductName2', 'Similarity'])


    # Appliquer la fonction de tri à chaque paire
    sorted_codes_df = similar_df.apply(_sort_codes, axis=1)

    # Ajouter les colonnes triées au DataFrame original
    similar_df['SortedCode1'] = sorted_codes_df['SortedCode1']
    similar_df['SortedCode2'] = sorted_codes_df['SortedCode2']

    # Supprimer les doublons basés sur les colonnes triées
    unique_similar_df = similar_df.drop_duplicates(subset=['SortedCode1', 'SortedCode2'])

    # Supprimer les colonnes de tri pour revenir au format original
    unique_similar_df = unique_similar_df.drop(columns=['SortedCode1', 'SortedCode2'])

    # Affichage des résultats
    print(f"{len(similar_df)} Descriptions similaires ")

    return unique_similar_df

def correct_parent_similar_description(df, matrix_tree, similarity_thresh=1, max_dist=1):
    
    def _longest_branch(pc1, pc2, matrix_tree):
        len1 = len(matrix_tree[matrix_tree["ProductCode"] == pc1]["Matrix Ancestries"].values[0].split("."))
        len2 = len(matrix_tree[matrix_tree["ProductCode"] == pc2]["Matrix Ancestries"].values[0].split("."))
        return pc1 if len1 > len2 else pc2
    
    # Create a copy of the filtered DataFrame
    df_filtered = df[(df["Similarity"] >= similarity_thresh) & (df["ProductCodeDistance"] <= max_dist)].copy()

    # Create NewProductCode column
    df_filtered["NewProductCode"] = df_filtered.apply(lambda row: _longest_branch(row["ProductCode1"], row["ProductCode2"], matrix_tree), axis=1)
    
    # Create NewProductName column
    df_filtered["NewProductName"] = df_filtered.apply(lambda row: matrix_tree[matrix_tree["ProductCode"] == row["NewProductCode"]]["Nom"].values[0] if any(matrix_tree["ProductCode"] == row["NewProductCode"]) else None, axis=1)
    
    return df_filtered
    
def clean_outliers(matrix_tree, outliers_df):

    # Compute disances
    outliers_df["ProductCodeDistance"] = outliers_df.apply(lambda row: compute_matrices_distance(row['ProductCode1'], row['ProductCode2'], matrix_tree), axis=1)

    # Extract correction
    # cleand_df = correct_parent_similar_description(outliers_df, matrix_tree, similarity_thresh=0.98, max_dist=2)

    return outliers_df

def update_product_codes_and_names(df, df_filtered):
    # Create sets of sample codes to update
    samples_to_update = set(df_filtered['ProductCode1']).union(set(df_filtered['ProductCode2']))
    
    # Create dictionaries for quick lookup
    code_updates = {}
    name_updates = {}
    
    for _, row in df_filtered.iterrows():
        new_code = row['NewProductCode']
        new_name = row['NewProductName']
        code_updates[row['ProductCode1']] = new_code
        code_updates[row['ProductCode2']] = new_code
        name_updates[row['ProductCode1']] = new_name
        name_updates[row['ProductCode2']] = new_name
    
    # Function to update code and name
    def update_row(row):
        if row['SampleCode'] in samples_to_update:
            new_code = code_updates.get(row['ProductCode'], row['ProductCode'])
            new_name = name_updates.get(row['ProductCode'], row['ProductName'])
            return pd.Series({'UpdatedProductCode': new_code, 'UpdatedProductName': new_name})
        else:
            return pd.Series({'UpdatedProductCode': row['ProductCode'], 'UpdatedProductName': row['ProductName']})
    
    # Apply the update function
    updated = df.apply(update_row, axis=1)
    df[['UpdatedProductCode', 'UpdatedProductName']] = updated
    
    return df

if __name__ == "__main__":
    

    matrix_tree = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\matrices_tree.xlsx")

    # df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\Merged_and_cleand_df_2024-06-11_0.xlsx")

    # outliers_df = extract_description_productcode_error(df, threshold=0.8)
    # outliers_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\OUTLIERS_Merged_and_cleand_df_2024-06-11_0.xlsx", index=False)

    outliers_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\OUTLIERS_Merged_and_cleand_df_2024-06-11_0.xlsx")

    cleand_outliers_df = clean_outliers(matrix_tree, outliers_df)
    cleand_outliers_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\OL_COR_Merged_and_cleand_df_2024-06-11_0.xlsx", index=False)

    # correction_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\OL_COR_Merged_and_cleand_df_2024-06-11_0.xlsx")

    # updated_df = update_product_codes_and_names(df, correction_df)
    # updated_df.to_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\Updated_Merged_and_cleand_df_2024-06-11_0.xlsx", index=False)

    
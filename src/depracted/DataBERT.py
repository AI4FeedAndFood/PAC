import pandas as pd
import os
import random


def generate_pairs_df(df, dist_dict, n_choice_dist=5, i_start=0):
  """_summary_

  Args:
      df (_type_): _description_
      dist_dict (_type_): _description_
      n_df_sample (int, optional): _description_. Defaults to 5.
      min_n_class (int, optional): _description_. Defaults to 5.
      i_start (int, optional): _description_. Defaults to 0.
  """
  distances = []
  sentencesA, codesA = [], []
  sentencesB, codesB = [], []

  for i_iter, (index, row) in enumerate(df.iterrows()):

    if i_iter<i_start:
      continue

    sentenceA, codeA = row["CleanDescription"], row["ProductCode"]

    for dist, code_list in dist_dict[codeA].items():

      if len(code_list)>n_choice_dist:
        code_list = random.sample(code_list, n_choice_dist)
      
      n_by_code = int(len(code_list)/n_choice_dist)

      for codeB in code_list:
        df_sentencesB = df.loc[(df["ProductCode"] == codeB) & (df["CleanDescription"] != sentenceA)]
        min_value = min(len(df_sentencesB), n_by_code)
        df_sentencesB = df_sentencesB.sample(n=min_value)

        sentencesB += df_sentencesB["CleanDescription"].values.tolist()
        for _ in range(min_value):
          sentencesA.append(sentenceA)
          codesA.append(codeA)
          codesB.append(codeB)
          distances.append(dist)

    if i_iter%1000==0:
      print("SHOW index :", i_iter, "len df :", len(sentencesA))

    if len(sentencesA)>500000:
      print("SAVE :", i_iter, codeA)
      df_save = pd.DataFrame({"sentenceA": sentencesA, "sentenceB": sentencesB, "distance" : distances, "codeA": codesA, "codeB": codesB})
      path = os.path.join(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\SBERT_Data", f"train_data_SBERT_{index}.xlsx" )
      df_save.to_excel(path, index=False)
      distances = []
      sentencesA, codesA = [], []
      sentencesB, codesB = [], []

  print("SAVE :", len(sentencesA), index)
  df_save = pd.DataFrame({"sentenceA": sentencesA, "sentenceB": sentencesB, "distance" : distances, "codeA": codesA, "codeB": codesB})
  path = os.path.join(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\SBERT_Data", f"train_data_SBERT_FINAL.xlsx" )
  df_save.to_excel(path, index=False)

if __name__ == "__main__":
  print("\n")

  data_train = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\SBERT_Data\train_SBERT_tfidf_mean.xlsx")

  # generate_pairs_df(data_train, dist_dict, n_choice_dist=11, i_start=0)

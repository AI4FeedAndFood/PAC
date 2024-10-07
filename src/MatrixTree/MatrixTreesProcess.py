import pandas as pd
import json
import os

from DataProcess.RawDataPreprocess import set_config

config_path = r"Config\DataConfig.json"
DATA_CONFIG = set_config(config_path, version="VERSION_NG_FR")

seed = 4
train_size = 0.85
test_size = 1 - train_size

class MatrixTree:
    def __init__(self, matrix_paths_list):
        self.root = {}
        self.max_depth = 0
        self.num_matrices = 0
        self.matrix_to_id = {}
        self.id_to_matrix = {}

        for path in matrix_paths_list:
            self.add_matrix(path)

    def add_matrix(self, matrix_path):
        parts = matrix_path.split('.')
        current = self.root
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {}
            current = current[part]
            if i + 1 > self.max_depth:
                self.max_depth = i + 1

        if matrix_path not in self.matrix_to_id:
            self.matrix_to_id[matrix_path] = self.num_matrices
            self.id_to_matrix[self.num_matrices] = matrix_path
            self.num_matrices += 1

    def get_parent(self, matrix_path):
        parts = matrix_path.split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None

    def get_children(self, matrix_path):
        parts = matrix_path.split('.')
        current = self.root
        for part in parts:
            current = current.get(part, {})
        return list(current.keys())

    def get_siblings(self, matrix_path):
        parent = self.get_parent(matrix_path)
        if parent:
            return [sib for sib in self.get_children(parent) if sib != matrix_path.split('.')[-1]]
        return []

    def get_matrix_id(self, matrix_path):
        return self.matrix_to_id.get(matrix_path)

    def get_matrix_from_id(self, matrix_id):
        return self.id_to_matrix.get(matrix_id)

    def get_path_to_root(self, matrix_path):
        parts = matrix_path.split('.')
        path = []
        for i in range(len(parts)):
            path.append('.'.join(parts[:i+1]))
        return path

    def get_sub_pathes(self, matrix_path):
        parts = matrix_path.split('.')
        path = []
        for i in range(len(parts)):
            path.append('.'.join(parts[:i+1]))
        return path
    
    def get_leaves(self, node):
        if not self.get_children(node):
            return [node]
        leaves = []
        for child in self.get_children(node):
            leaves.extend(self.get_leaves(child))
        return leaves

def compute_matrices_distance(n1, n2, matrices_df):
  """Compute the distance beetween two nodes from the matrix tree using a dataframe that summarized all branches of the tree.
  This distance is named Lowest Common Ancestor

  Args:
      n1 (string): The sample code of the node 1
      n2 (string): The sample code of the node 2
      df (pd.DataFrame): DataFrame that describe the branch path for each node to the root as root.parentNode1.parentNode2.Leaf

  Returns:
      Distance (int): The distance between n1 and n2
  """
  try:
    parents1, parents2 = list(matrices_df[matrices_df["ProductCode"] == n1]["Matrix Ancestries"])[0].split("."), list(matrices_df[matrices_df["ProductCode"] == n2]["Matrix Ancestries"])[0].split(".")
    parents1, parents2 = sorted([parents1,parents2], key=lambda x: len(x))

    for i, ancestre_i in enumerate(parents1[::-1]):
        for j, ancestre_j in enumerate(parents2[::-1]):
            if ancestre_i==ancestre_j:
                return i+j
  except:
     print(f"Error with {n1} or {n2}")
     
def distances_matrix(matrices_df, ref_df = None, i_start=0):
  """Compute all distances between all nodes of a Reference_df if given or for all existing nodes from matrices df.

  Args:
      matrices_df (pd.DataFrame): DataFrame that describe the branch path for each node to the root as root.parentNode1.parentNode2.Leaf_
      ref_df (pd.DataFrame, optional): A DataFrame containg a "ProductCode" column containing all productCodes that . Defaults to None.
      i_start (int, optional): _description_. Defaults to 0.

  Returns:
      dist_dict (dictionary): The distance between all productcodes as follow : P1 : {distance0 : [P1], distance1 : [Pi,..], distance2 : ...}
  """
  # All the codes for which we want to calculate the distance of
  all_codes = list(matrices_df["ProductCode"].unique())
  
  # If a ref_df is given it will take all ProductCode in it 
  if type(ref_df) != type(None):
        existing_codes = ref_df["ProductCode"].unique()
        all_codes = [c for c in all_codes if c in existing_codes]

  # Init the returned dict
  dist_dict = {key: {} for key in all_codes}

  # For each code
  for i in range(i_start, len(all_codes)):

    # Save the dict by slice of 100 codes
    if i%100==0:
        print(i)
        name = f"matrices_dist_REFMERGED_{i}.json"
        path = os.path.join(r"C:\Users\CF6P\Desktop\PAC\Data_PAC", name)
        with open(path, 'w') as fp:
            json.dump(dist_dict, fp)

    code_i = all_codes[i]

    # For each unprocessed codes
    for j in range(i, len(all_codes)):
      code_j = all_codes[j]
      
      # Compute distance
      dist = compute_matrices_distance(code_i, code_j, matrices_df)

      # Add the ProductCode j to the distance of the dict of the ProductCode i
      if not dist in list(dist_dict[code_i].keys()):
        dist_dict[code_i].update({dist : [code_j]})
      else:
          dist_dict[code_i][dist].append(code_j)

      # And vice-versa
      if not dist in list(dist_dict[code_j].keys()):
        dist_dict[code_j].update({dist : [code_i]})
      else:
          dist_dict[code_j][dist].append(code_i)
          
  return dist_dict

if __name__ == "__main__":
  print("\nGO")
  matrices_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\matrices_tree.xlsx")
  ref_df = pd.read_excel(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\NG_Merged_Samples.xlsx")
  dist_dict = distances_matrix(matrices_df, ref_df=ref_df)

  with open(r"C:\Users\CF6P\Desktop\PAC\Data_PAC\matrices_dist_REFMERGED_FINAL.json", 'w') as fp:
    json.dump(dist_dict, fp)


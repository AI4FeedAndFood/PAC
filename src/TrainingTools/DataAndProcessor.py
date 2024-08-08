import os
import pandas as pd

import random
import nltk
from nltk.corpus import wordnet
from functools import lru_cache

import cv2
import numpy as np
import fitz 
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from DataProcess.RawDataPreprocess import load_data

def load_train_test_data(folder_path, used_columns, make_test_if_not_exists=False, split_ratio=0.8, seed=4):

    all_files = load_data(folder_path)
    all_dfs = []
    for file in all_files:
        df = pd.read_csv(file) if file.endswith('.csv') else pd.read_excel(file)
        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, ignore_index=True)

    train_base = merged_df[merged_df["Mode"]=="train"]
    test_base = merged_df[merged_df["Mode"]=="test"]

    train_base = train_base.dropna(axis=0, subset=used_columns)
    test_base = test_base.dropna(axis=0, subset=used_columns)

    for c in used_columns:
        print(c, "null rows : ", len(train_base[train_base[c].isnull()]))

    for c in used_columns:
        print(c, "null rows : ", len(test_base[test_base[c].isnull()]))

    return train_base, test_base

def filter_productcode(train, test, min_row=50, max_row=None, seed=4, split_ratio=0.8):

    def _res_max_df(df, max_selected_productcodes, max_row):
        df1 = df[~df["ProductCode"].isin(max_selected_productcodes.index)]
        added_df = [df1]
        for code in max_selected_productcodes.index.tolist():
            code_df = df[df["ProductCode"]==code]
            if len(code_df)>=max_row:
                added_df.append(code_df.sample(n=max_row, random_state=seed))

        return pd.concat(added_df)

    count = train.ProductCode.value_counts()

    # Process min_row
    min_selected_productcodes = count[count>=min_row]

    train = train[train["ProductCode"].isin(min_selected_productcodes.index)]
    test = test[test["ProductCode"].isin(min_selected_productcodes.index)]

    # Process max_row
    count = train.ProductCode.value_counts()
    if MAX_ROW is not None:
        max_selected_productcodes_train = count[count>=MAX_ROW]
        train = _res_max_df(train, max_selected_productcodes_train, MAX_ROW)

        count = test.ProductCode.value_counts()
        max_selected_productcodes_test = count[count>=int(MAX_ROW*(1-split_ratio))]
        test = _res_max_df(test, max_selected_productcodes_test, int(MAX_ROW*(1-split_ratio)))

    print(f"Train contient {len(train)} lignes")
    print(f"Test contient {len(test)} lignes")

    # Check train pc containing test pc
    train_pc = train.ProductCode.unique()
    test_pc = test.ProductCode.unique()
    if set(test_pc).issubset(set(train_pc)):
        print("Train product codes contiennent test product codes")

    # Trouver les SampleCodes en commun (objectif 0)
    train_samples = set(train['SampleCode'])
    test_samples = set(test['SampleCode'])
    common_samples = train_samples.intersection(test_samples)
    print(f"Nombre de SampleCodes en commun : {len(common_samples)}")

    return train, test

def get_product_code_mapping(matrix_tree):
    matrix_tree = matrix_tree.sort_values("Matrix Ancestries")
    product_codes, product_names = matrix_tree["Matrix Ancestries"].tolist(), matrix_tree["ProductName"].tolist()
    return product_codes, product_names

def get_class_count(df, plot_x="ProductName", show=False):

    if plot_x == "ProductName":
        count = df.ProductName.value_counts()
    else:
        count = df.ProductCode.value_counts()

    print(f"Il y a {len(count)} codes produits traités")

    if show:
        print("La distribution des codes produit est :\n")
        plt.figure(figsize=(70,20))
        count.plot(kind='bar')
        plt.show()

    return count

def get_train_test(folder_path, used_columns, min_row=50, max_row=None):
    # Load DataFrames
    train_base, test_base = load_train_test_data(folder_path, used_columns)

    # Filter by ProductCodes
    train, test = filter_productcode(train_base, test_base, min_row=min_row, max_row=max_row)

    get_class_count(train, plot_x="ProductName", show=False)

    return train, test

class TextProcessor:

    def __init__(self, augment=True, max_length=512):
        self.augment = augment
        self.max_length = max_length

        # Assurez-vous que les ressources NLTK nécessaires sont téléchargées
        nltk.download('punkt')
        nltk.download('wordnet')

        self.augmentations = [
            (self.synonym_replacement, 0.05),
            (self.random_deletion, 0.1),
            (self.random_insertion, 0.1),
            (self.random_swap, 0.2)
        ]

    def process_text(self, text):
        # Tokenisation du texte
        tokens = nltk.word_tokenize(text)

        if self.augment:
            # Application des augmentations
            for aug_func, prob in self.augmentations:
                if random.random() < prob:
                    tokens = aug_func(tokens)

        # Rejoindre les tokens et tronquer si nécessaire
        processed_text = ' '.join(tokens)
        return processed_text[:self.max_length]

    def synonym_replacement(self, tokens, n=1):
        new_tokens = tokens.copy()
        random_word_list = list(set([word for word in tokens if word.isalnum()]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_tokens = [synonym if word == random_word else word for word in new_tokens]
                num_replaced += 1
            if num_replaced >= n:
                break

        return new_tokens

    def random_deletion(self, tokens, p=0.1):
        if len(tokens) == 1:
            return tokens
        return [word for word in tokens if random.random() > p or not word.isalnum()]

    def random_insertion(self, tokens, n=1):
        new_tokens = tokens.copy()
        for _ in range(n):
            self.insert_word(new_tokens)
        return new_tokens

    def random_swap(self, tokens, n=1):
        new_tokens = tokens.copy()
        for _ in range(n):
            self.swap_word(new_tokens)
        return new_tokens

    def insert_word(self, tokens):
        if not tokens:
            return tokens
        random_idx = random.randint(0, len(tokens) - 1)
        random_synonym = self.get_synonyms(tokens[random_idx])
        if not random_synonym:
            return tokens
        random_idx_2 = random.randint(0, len(tokens) - 1)
        tokens.insert(random_idx_2, random.choice(list(random_synonym)))

    def swap_word(self, tokens):
        if len(tokens) < 2:
            return tokens
        idx1, idx2 = random.sample(range(len(tokens)), 2)
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                if len(synonym) > 0 and synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)

    def __call__(self, text):
        return self.process_text(text)
    
class ImageProcessor:
    def __init__(self, image_size=(224, 224), cache_dir='cache_images'):
        self.image_size = image_size
        self.cache_dir = cache_dir

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Transformation pour le chargement
        self.load_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transformations pour l'augmentation
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])

    def load(self, image_path):
        cache_path = os.path.join(self.cache_dir, os.path.basename(image_path) + '.pt')

        if os.path.exists(cache_path):
            image_tensor = torch.load(cache_path)
        else:
            file_extension = os.path.splitext(image_path)[1].lower()

            if file_extension in ['.jpg', '.jpeg', '.png']:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif file_extension == '.pdf':
                image = self.load_pdf(image_path)
            else:
                raise ValueError(f"Format de fichier non pris en charge : {file_extension}")

            # Appliquer les transformations de chargement
            image_tensor = self.load_transform(image)
            torch.save(image_tensor, cache_path)

        return image_tensor

    def load_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        if len(doc) == 1:
            image = self.pdf_page_to_numpy(doc[0])
        else:
            min_black_pixels = float('inf')
            selected_image = None
            for page in doc:
                image = self.pdf_page_to_numpy(page)
                black_pixels = np.sum(image < 50)
                if black_pixels < min_black_pixels:
                    min_black_pixels = black_pixels
                    selected_image = image
            image = selected_image
        doc.close()
        return image

    def pdf_page_to_numpy(self, page):
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def augment(self, image_tensor):
        augmented_image = self.augment_transform(image_tensor)
        # Renormaliser l'image après l'augmentation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        augmented_image = (augmented_image - mean) / std
        return augmented_image

    def visualize(self, image_tensor):
        # Convertir le tensor en numpy array
        image = image_tensor.cpu().numpy().transpose(1, 2, 0)

        # Dénormaliser si nécessaire
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Afficher l'image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def __call__(self, image_path):
        raw_image = self.load(image_path)
        return self.augment(raw_image)

class FlexibleMultiModalDataset(Dataset):
    def __init__(self, dataframe, modalities, device, image_processor=None, tokenizer=None, image_augmentation=None, text_processor=None):
        self.dataframe = dataframe
        self.modalities = modalities
        self.device = device
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.image_augmentation = image_augmentation
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataframe)

    @lru_cache(maxsize=None)
    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        item = {}

        if 'text' in self.modalities:
            if not self.modalities['text'].get('use_encoded', False):
                text = row["CleanDescription"]
                if self.text_augmentations:
                    for aug, proba in self.text_augmentations:
                        if random.random() < proba:
                            text = aug(text)

                text = self.tokenize_text(text)
                text = {
                'input_ids': text["input_ids"].squeeze().to(self.device),
                'attention_mask': text["attention_mask"].squeeze().to(self.device),
                'token_type_ids': text["token_type_ids"].squeeze().to(self.device)
                }

            else:
                text = row["EncodedCleanDescription"].squeeze().to(self.device)

            item['text'] = text

        if 'client' in self.modalities:
            item['client'] = torch.tensor(row["EncodedAccountCode"], dtype=torch.float32).squeeze().to(self.device)

        if 'laboratory' in self.modalities:
            item['laboratory'] = torch.tensor(row["EncodedLaboratory"], dtype=torch.float32).squeeze().to(self.device)

        if 'image' in self.modalities:
            image_path = row["ImagePath"]
            image_tensor = self.image_processor.load(image_path)
            if self.image_augmentation:
                image_tensor = self.image_processor.augment(image_tensor)
            item['image'] = image_tensor.to(self.device)

        item['label'] = torch.tensor(row["EncodedProductCode"], dtype=torch.float32).squeeze().to(self.device)

        return item
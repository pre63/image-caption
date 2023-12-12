import pandas as pd
import random
from nltk.translate.bleu_score import sentence_bleu

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import nltk
from collections import Counter
import os
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')

data = pd.read_csv('final.csv')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in nltk.tokenize.word_tokenize(sentence.lower()):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = nltk.tokenize.word_tokenize(text.lower())

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

def generate_caption(model, features, vocab, max_length=20):
    model.eval()
    with torch.no_grad():
        captions = []

        # Starting token
        inputs = torch.tensor(vocab.stoi["<SOS>"]).unsqueeze(0).to(features.device)
        
        for i in range(max_length):
            outputs = model(features, inputs)
            outputs = outputs.squeeze(1)
            _, predicted = outputs.max(dim=1)
            predicted_word = vocab.itos[predicted.item()]
            captions.append(predicted_word)

            if predicted_word == "<EOS>":
                break

            inputs = torch.cat((inputs, predicted.unsqueeze(0)), dim=1)

        # Convert list of words to a sentence
        sentence = ' '.join(captions)

        return sentence


def evaluate_bleu(model, dataloader, vocab, device):
    bleu_scores = []

    for imgs, captions in dataloader:
        features = torch.stack([encoded_features[image_ids.index(id)] for id in imgs])
        features = features.to(device)

        # Generate captions
        generated_captions = [generate_caption(model, feature.unsqueeze(0), vocab) for feature in features]

        # Compare with ground truth captions
        for i in range(len(imgs)):
            references = [caption.split() for caption in captions[i]]
            candidate = generated_captions[i].split()
            bleu_scores.append(sentence_bleu(references, candidate, weights=(0.5, 0.5)))  # Adjust weights as needed

    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


def split_dataset(dataframe, train_size=0.7, val_size=0.2, test_size=0.1, random_state=None):
    """
    Splits the dataset into training, validation, and test sets without using external libraries.

    :param dataframe: A pandas DataFrame containing the dataset.
    :param train_size: Proportion of the dataset to include in the train split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before splitting.
    :return: Three DataFrames corresponding to the train, validation, and test sets.
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("Train, validation, and test sizes must sum to 1.0")

    # Set random seed for reproducibility
    if random_state is not None:
        random.seed(random_state)

    # Shuffle the dataframe
    shuffled_df = dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split indices
    total_rows = len(shuffled_df)
    train_end = int(train_size * total_rows)
    val_end = train_end + int(val_size * total_rows)

    # Split the dataframe
    train_data = shuffled_df.iloc[:train_end]
    val_data = shuffled_df.iloc[train_end:val_end]
    test_data = shuffled_df.iloc[val_end:]

    return train_data, val_data, test_data


class CustomDataset(Dataset):
    def __init__(self, dataframe, vocab, transform=None):
        self.dataframe = dataframe
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        caption = self.dataframe.iloc[idx, 2]
        img_id = self.dataframe.iloc[idx, 1]
        img = Image.open(os.path.join("images/", img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)


        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    
    def collate_fn(self, batch):
        imgs, captions = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.vocab.stoi["<PAD>"])
        return imgs, captions
    


import torch
from tqdm import tqdm

def encode_images(dataloader, encoder, device):
    # Ensure the encoder is in evaluation mode
    encoder.eval()

    # Store all features and corresponding image IDs
    all_features = []
    image_ids = []

    with torch.no_grad():
        for imgs, ids in tqdm(dataloader):
            # Move images to the appropriate device (CPU/GPU)
            imgs = imgs.to(device)

            # Encode images and store features
            features = encoder(imgs)
            all_features.append(features.cpu())
            image_ids.extend(ids)

    # Stack all features into a single tensor
    all_features = torch.cat(all_features, 0)

    return all_features, image_ids


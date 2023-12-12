
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import nltk
from collections import Counter
import os
from config import configs, report
from data import CustomDataset, Vocabulary, transform, generate_caption, evaluate_bleu, split_dataset
from model import DecoderRNN
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

# Load dataset
data = pd.read_csv('final.csv')

# Build vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(data.caption.tolist())

# Hyperparameters
configs = configs(vocab)
batch_size = 64

for config in configs:
    slug, num_epochs, embed_size, hidden_size, vocab_size, num_layers, learning_rate = config
    print(f'embed_size: {embed_size}, hidden_size: {hidden_size}, vocab_size: {vocab_size}, num_layers: {num_layers}, learning_rate: {learning_rate}')

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the decoder
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    if torch.cuda.device_count() > 1:
        decoder = nn.DataParallel(decoder)
    decoder.to(device)

    # Load precomputed features
    loaded_data = torch.load(f'features/{embed_size}.pt', map_location=device)
    encoded_features = loaded_data['features']
    image_ids = loaded_data['image_ids']

    # Split the dataset
    train_data, val_data, test_data = split_dataset(data)

    # Create datasets
    train_dataset = CustomDataset(train_data, vocab, transform=transform)
    val_dataset = CustomDataset(val_data, vocab, transform=transform)
    test_dataset = CustomDataset(test_data, vocab, transform=transform)  # For test set

    # Create DataLoaders
    batch_size = 64  # Adjust as needed
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Track metrics
    train_losses = []
    val_losses = []
    bleu_scores = []

    # Training loop
    for epoch in range(num_epochs):
        decoder.train()
        total_train_loss = 0

        for idx, (imgs, captions) in enumerate(train_dataloader):
            features = torch.stack([encoded_features[image_ids.index(id)] for id in imgs])
            features, captions = features.to(device), captions.to(device)

            # Forward pass
            outputs = decoder(features, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation loop
        decoder.eval()
        total_val_loss = 0
        all_predicted, all_targets = [], []

        with torch.no_grad():
            for imgs, captions in val_dataloader:
                features = torch.stack([encoded_features[image_ids.index(id)] for id in imgs])
                features, captions = features.to(device), captions.to(device)

                # Forward pass
                outputs = decoder(features, captions[:-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        # Compute BLEU score
        bleu_score = evaluate_bleu(decoder, val_dataloader, vocab, device)
        bleu_scores.append(bleu_score)
        print(f'Epoch [{epoch+1}/{num_epochs}], BLEU Score: {bleu_score}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')  # , BLEU: {bleu_score}')

    # Save model and plot
    torch.save(decoder.state_dict(), f'models/{slug}.ckpt')

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    # plt.plot(bleu_scores, label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/BLEU')
    plt.title(f'Training and Validation Loss for {slug}')
    plt.legend()
    plt.savefig(f'reports/{slug}.png')

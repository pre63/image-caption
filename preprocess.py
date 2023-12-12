import torch
from torch.utils.data import  DataLoader
import nltk
import torch
import nltk
import torch
from model import EncoderCNN

from data import CustomDataset, Vocabulary, transform
from model import EncoderCNN
from data import encode_images, data

nltk.download('punkt')

# Build vocabulary
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(data.caption.tolist())

# Create the dataset
dataset = CustomDataset(data, vocab, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)

vocab = Vocabulary(freq_threshold=5)

embed_sizes = [256, 512, 768, 1024, 2048]

for embed_size in embed_sizes:
    print(f'embed_size: {embed_size}')
    encoder = EncoderCNN(embed_size)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected. Initializing Data Parallel...")
        encoder = torch.nn.DataParallel(encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    encoded_features, image_ids = encode_images(dataloader, encoder, device)

    torch.save({
        'features': encoded_features,
        'image_ids': image_ids
    }, f'features/{embed_size}.pt')


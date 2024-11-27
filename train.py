import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.speech_model import SpeechRecognitionModel
from utils.data_loader import SpeechDataset, collate_fn

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Chemins des données
    train_audio_dir = "data/train/wav"
    train_stm_dir = "data/train/stm"
    val_audio_dir = "data/dev/wav"
    val_stm_dir = "data/dev/stm"

    # Vérification des dossiers
    assert os.path.exists(train_audio_dir), f"Le dossier {train_audio_dir} est introuvable."
    assert os.path.exists(train_stm_dir), f"Le dossier {train_stm_dir} est introuvable."
    assert os.path.exists(val_audio_dir), f"Le dossier {val_audio_dir} est introuvable."
    assert os.path.exists(val_stm_dir), f"Le dossier {val_stm_dir} est introuvable."

    # Chargement des datasets
    train_dataset = SpeechDataset(train_audio_dir, train_stm_dir)
    val_dataset = SpeechDataset(val_audio_dir, val_stm_dir)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Définition du modèle
    model = SpeechRecognitionModel(input_dim=13, num_classes=len(set(train_dataset.labels))).to(device)

    # Configuration de l'optimisation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Boucle d’entraînement
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}")

        # Validation à la fin de chaque époque
        validate(model, val_loader, criterion, device)

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), "wav2vec_small.pt")
    print("Modèle sauvegardé dans 'wav2vec_small.pt'.")

def validate(model, val_loader, criterion, device):
    """
    Fonction de validation du modèle.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    train()

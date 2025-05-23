import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm # For progress bar

# --- VAE Model Definition ---
class VAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=256, z_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE Loss Function
def vae_loss(recon_x, x, mu, logvar):
    # BCE expects input in [0,1] range, our VAE output is Sigmoid, so this is fine.
    # The input `x` from DataLoader is normalized to [-1, 1], so denormalize it to [0,1] for BCE.
    x_denorm = (x.view(-1, 784) * 0.5) + 0.5
    BCE = nn.functional.binary_cross_entropy(recon_x, x_denorm, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- CNN Classifier Model Definition ---
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 14x14 -> 14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 14x14 -> 7x7

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Expects input as [batch_size, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Training Functions ---

def train_vae(model, dataloader, optimizer, epochs, device, model_save_path="vae_mnist.pth"):
    model.train()
    print(f"\n--- Training VAE on {device} for {epochs} epochs ---")
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc=f"VAE Epoch {epoch+1}")):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            overall_loss += loss.item()
            optimizer.step()

        print(f"VAE Epoch {epoch+1}/{epochs}, Average Loss: {overall_loss / len(dataloader.dataset):.4f}")

    print("VAE Training complete.")
    torch.save(model.state_dict(), model_save_path)
    print(f"VAE Model saved to {model_save_path}")

def train_classifier(model, dataloader, optimizer, criterion, epochs, device, model_save_path="cnn_classifier_mnist.pth"):
    model.train()
    print(f"\n--- Training CNN Classifier on {device} for {epochs} epochs ---")
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"CNN Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f"CNN Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("CNN Classifier Training complete.")
    torch.save(model.state_dict(), model_save_path)
    print(f"CNN Classifier Model saved to {model_save_path}")

# --- Main Execution ---
if __name__ == "__main__":
    # Hyperparameters
    VAE_INPUT_DIM = 784
    VAE_H_DIM = 256
    VAE_Z_DIM = 20
    LEARNING_RATE_VAE = 1e-3
    LEARNING_RATE_CNN = 1e-3
    BATCH_SIZE = 128
    NUM_EPOCHS_VAE = 15 # Reduced slightly for faster demo, increase for better quality
    NUM_EPOCHS_CNN = 10 # Good starting point for CNN

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST Dataset and DataLoader
    # VAE needs normalization to [-1, 1], CNN typically uses [0, 1] for ImageNet pre-trained
    # But for MNIST, [-1,1] is often fine for CNN too, let's keep consistent for simplicity
    # and adjust BCE loss for VAE
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0, 1]
        transforms.Normalize((0.5,), (0.5,)) # Normalizes to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Train VAE ---
    vae_model = VAE(VAE_INPUT_DIM, VAE_H_DIM, VAE_Z_DIM).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE_VAE)
    train_vae(vae_model, train_loader, vae_optimizer, NUM_EPOCHS_VAE, device, "vae_mnist.pth")

    # --- Train CNN Classifier ---
    cnn_classifier_model = CNNClassifier(num_classes=10).to(device)
    cnn_optimizer = optim.Adam(cnn_classifier_model.parameters(), lr=LEARNING_RATE_CNN)
    cnn_criterion = nn.CrossEntropyLoss() # For classification
    train_classifier(cnn_classifier_model, train_loader, cnn_optimizer, cnn_criterion, NUM_EPOCHS_CNN, device, "cnn_classifier_mnist.pth")

    # --- Optional: Visualize reconstructions and generated samples (from VAE) ---
    # print("\n--- VAE Evaluation Examples ---")
    vae_model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, mu, logvar = vae_model(data)

        # Denormalize for plotting (from [-1,1] to [0,1])
        data_plot = (data * 0.5 + 0.5).cpu().view(-1, 28, 28).numpy()
        recon_batch_plot = recon_batch.cpu().view(-1, 28, 28).numpy()

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data_plot[i], cmap='gray')
            ax.axis('off')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_batch_plot[i], cmap='gray')
            ax.axis('off')
        plt.suptitle("Original (Top) vs. VAE Reconstructed (Bottom) Digits")
        plt.show()

    # --- Optional: Test CNN Classifier Accuracy ---
    print("\n--- CNN Classifier Test Accuracy ---")
    cnn_classifier_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = cnn_classifier_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the CNN Classifier: {100 * correct / total:.2f}%')

    # Run the script with pasting the command: $env:KMP_DUPLICATE_LIB_OK="TRUE" python vae_model.py
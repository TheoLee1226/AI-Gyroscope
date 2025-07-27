import gyro_simulation_dataset as dataset
import gyro_model as model
import gyro_model_RNN as model_RNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm


LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", DEVICE)

print("Loading dataset...")
try:
    train_dataset = dataset.simulation_dataset(train=True)
    print(f"Training dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("Training dataset is empty! Please check if the data files exist and are accessible.")
        exit()
except FileNotFoundError:
    print("training_gyro_simulation_data.npz not found")
    exit()
except Exception as e:
    print("Error loading training dataset: ", e)
    exit()

try:
    validation_dataset = dataset.simulation_dataset(train=False)
    print(f"Validation dataset size: {len(validation_dataset)}")
    if len(validation_dataset) == 0:
        print("Validation dataset is empty! Please check if the data files exist and are accessible.")
        exit()
except FileNotFoundError:
    print("validation_gyro_simulation_data.npz not found")
    exit()
except Exception as e:
    print("Error loading validation dataset: ", e)
    exit()
print("Dataset loaded successfully")

input_size = 6
sequence_length = 1000
output_size = 3

print("Initializing model...")
try:
    # gyro_model = model.gyro_model(input_size, sequence_length, output_size).to(DEVICE)
    gyro_model = model_RNN.gyro_model_RNN(input_size, sequence_length, hidden_size=128, num_layers=10, output_size=output_size, dropout_prob=0.5).to(DEVICE)
    print(gyro_model) 
except Exception as e:
    print("Error initializing model: ", e)
    exit()
print("Model initialized successfully")

print("Training model...")

criterion = nn.MSELoss()
optimizer = optim.Adam(gyro_model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, threshold=0.01, threshold_mode='rel', cooldown=0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loss = []
validation_loss = []

for epoch in range(EPOCHS):
    gyro_model.train()
    
    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", leave=True)
    running_train_loss = 0.0

    for inputs, I, M, g, H, X_0, D_X_0 in pbar_train:
        
        inputs = inputs.to(DEVICE)
        targets = I.to(DEVICE)

        optimizer.zero_grad()
        outputs = gyro_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        running_train_loss += loss.item()
        train_loss.append(loss.item())
        
    gyro_model.eval()
    running_validation_loss = 0.0

    with torch.no_grad():
        for inputs, I, M, g, H, X_0, D_X_0 in validation_loader:
            inputs = inputs.to(DEVICE)
            targets = I.to(DEVICE)

            outputs = gyro_model(inputs)
            loss = criterion(outputs, targets)

            running_validation_loss += loss.item()

        validation_loss.append(running_validation_loss / len(validation_loader))

    print(f"Epoch {epoch+1}/{EPOCHS} [Validation] Average Loss: {running_train_loss / len(train_loader):.8f} | Validation Loss: {running_validation_loss / len(validation_loader):.8f} | Learning Rate: {optimizer.param_groups[0]['lr']:.8f}") 

print("Training complete")
print("Saving model...")

final_validation_loss = 0
gyro_model.eval()
with torch.no_grad():
    for inputs, I, M, g, H, X_0, D_X_0 in validation_loader:
        inputs = inputs.to(DEVICE)
        targets = I.to(DEVICE)

        outputs = gyro_model(inputs)
        loss = criterion(outputs, targets)

        final_validation_loss += loss.item()
final_validation_loss /= len(validation_loader)
print(f"Final Validation Loss: {final_validation_loss:.4f}")

date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = f"model_fitting/models/gyro_model_{date_time}.pth"

try:
    torch.save(gyro_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
except Exception as e:
    print("Error saving model: ", e)
    exit()
print("Model saved successfully")

plt.figure(figsize=(12, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.set_title('Training Loss')
ax2.set_title('Validation Loss')
ax1.set_xlabel('Batch')
ax2.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Loss')
ax1.plot(train_loss, label='Training Loss')
ax2.plot(validation_loss, label='Validation Loss')
ax1.legend()
ax2.legend()
plt.savefig(f'model_fitting/plots/gyro_training_loss_{date_time}.png')
print(f"Training loss plot saved to model_fitting/plots/gyro_training_loss_{date_time}.png")

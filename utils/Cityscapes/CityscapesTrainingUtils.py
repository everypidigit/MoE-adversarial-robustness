from tqdm import tqdm
import torch

device = "mps"


def train_one_epoch_cityscapes(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, targets in tqdm(loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device).long()
        targets = targets.squeeze(1)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate_cityscapes(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation"):
            images = images.to(device)
            targets = targets.to(device).long()
            targets = targets.squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

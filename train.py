"""Train a binary rest-vs-game classifier and evaluate it window by window."""

import copy

import torch
import torch.nn.functional as F

import config
from models import forward_logits


def get_device(name="auto"):
    """'auto' uses CUDA if available, otherwise CPU. Pass 'mps' for the Apple GPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict_logits(model, X, batch_size=config.BATCH_SIZE):
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits.append(forward_logits(model, X[i:i + batch_size]))
    return torch.cat(logits)


def evaluate(model, X, y):
    """Per-window losses (numpy array) and accuracy."""
    logits = predict_logits(model, X)
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    predictions = (logits >= 0).float()
    accuracy = (predictions == y).float().mean().item()
    return losses.cpu().numpy(), accuracy


def train_model(model, train_data, val_data, epochs, seed):
    """Adam + binary cross-entropy with early stopping on validation loss.

    Restores the weights with the lowest validation loss. Returns the number of epochs run.
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    shuffle_generator = torch.Generator().manual_seed(seed)

    best_val_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=shuffle_generator).to(X_train.device)
        for i in range(0, len(X_train), config.BATCH_SIZE):
            batch = order[i:i + config.BATCH_SIZE]
            optimizer.zero_grad()
            logits = forward_logits(model, X_train[batch])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[batch])
            loss.backward()
            optimizer.step()

        val_losses, _ = evaluate(model, X_val, y_val)
        val_loss = val_losses.mean()
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return epoch

import torch
from torch import nn
from torch.optim import Optimizer
import torch.utils
import torch.utils.data
import torch.utils.tensorboard
from tqdm.auto import tqdm

def train_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
):
    model.train()

    train_loss = 0
    train_acc = 0

    for batch, (X, y) in enumerate(dataloader):
        logits = model(X)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.softmax(logits,dim=1).argmax(dim=1)
        train_acc += (pred == y).sum().item() / len(X)

    num_batch = len(dataloader)
    train_loss /= num_batch
    train_acc /= num_batch

    return train_loss, train_acc

def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
):
    model.eval()

    train_loss = 0
    train_acc = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            logits = model(X)
            loss = loss_fn(logits, y)

            train_loss += loss.item()
            pred = torch.softmax(logits,dim=1).argmax(dim=1)
            train_acc += (pred == y).sum().item() / len(X)

        num_batch = len(dataloader)
        train_loss /= num_batch
        train_acc /= num_batch

    return train_loss, train_acc

def train(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        epochs: int,
        writer: torch.utils.tensorboard.writer.SummaryWriter = None
):
    results = {
        "train_loss": [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            ### New: Experiment tracking ###
            # Add loss results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                                tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                                global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                                tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                                global_step=epoch)
            
            # Track the PyTorch model architecture
            writer.add_graph(model=model, 
                                # Pass in an example input
                                input_to_model=torch.randn(32, 3, 224, 224))
            
            
        else:
            pass

    if writer:
        # Close the writerd, after all epoch
        writer.close()
    else:
        pass
    
    return results
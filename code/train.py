import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import average_precision_score

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=3e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with class weights
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([100.0]).to(device)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            precipitation = batch['precipitation'].to(self.device)
            image = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(precipitation, image)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, all_preds, all_labels

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                precipitation = batch['precipitation'].to(self.device)
                image = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(precipitation, image)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, all_preds, all_labels

    def train(self, num_epochs):
        print(f"Training on device: {self.device}")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_preds, train_labels = self.train_epoch()
            train_ap = average_precision_score(train_labels, train_preds)
            
            # Validation phase
            val_loss, val_preds, val_labels = self.validate()
            val_ap = average_precision_score(val_labels, val_preds)
            
            # Log metrics
            wandb.log({
                "train_loss": train_loss,
                "train_ap": train_ap,
                "val_loss": val_loss,
                "val_ap": val_ap,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
            
            print(f"Training Loss: {train_loss:.4f}, AP: {train_ap:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, AP: {val_ap:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved new best model with validation loss: {val_loss:.4f}")

        wandb.finish()
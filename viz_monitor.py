import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output
import numpy as np
import torch
import build_train_model
import os
import json
import time
import csv
from datetime import datetime

plt.style.use('dark_background')
COLORS = ['#00ffc8', '#ff007f', '#f9f871', '#00d2fc']

class DarkMonitor:
    def __init__(self, config=None):
        self.epoch_data = {'train': [], 'val': []}
        self.step_data = {'loss': [], 'grad': []}
        self.samples = []
        self.fig = None
        
        # Logging Setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("training_logs", f"run_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Log Config
        self.config = config
        self.log_config()
        
        # Initialize CSVs
        self.metrics_file = os.path.join(self.log_dir, "metrics.csv")
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1', 'train_prec', 'val_prec', 'train_rec', 'val_rec', 'duration_sec'])
            
        self.batch_file = os.path.join(self.log_dir, "batch_details.csv")
        with open(self.batch_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'grad_norm'])

    def log_config(self):
        # Handle config if it's an object or dict
        cfg_data = "Not Provided"
        if self.config:
            if hasattr(self.config, '__dict__'):
                cfg_data = self.config.__dict__
            elif isinstance(self.config, dict):
                cfg_data = self.config
            else:
                cfg_data = str(self.config)

        info = {
            "timestamp": self.timestamp,
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
            "model_config": cfg_data
        }
        with open(os.path.join(self.log_dir, "run_meta.json"), 'w') as f:
            json.dump(info, f, indent=4)
            
    def save_metrics(self, epoch, metrics, duration):
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                metrics['train_loss'], metrics['val_loss'],
                metrics['train_acc'], metrics['val_acc'],
                metrics['train_f1'], metrics['val_f1'],
                metrics['train_prec'], metrics['val_prec'],
                metrics['train_rec'], metrics['val_rec'],
                duration
            ])
            
    def log_batch_detail(self, epoch, batch_idx, loss, grad):
        with open(self.batch_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch_idx, loss, grad])

    def refresh(self):
        clear_output(wait=True)
        if self.fig: plt.close(self.fig)
        self.fig = plt.figure(figsize=(16, 9), facecolor='#1e1e1e')
        gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[1, 1.5])
        
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.plot(self.epoch_data['train'], 'o-', color=COLORS[0], label='Train')
        ax1.plot(self.epoch_data['val'], 'o-', color=COLORS[1], label='Val')
        ax1.legend(); ax1.set_title("Binary Cross Entropy Loss")
        
        ax2 = self.fig.add_subplot(gs[0, 1])
        if len(self.step_data['loss']) > 10:
            ax2.plot(np.convolve(self.step_data['loss'], np.ones(50)/50, mode='valid'), color=COLORS[2])
        ax2.set_title("Batch Loss")
        
        ax3 = self.fig.add_subplot(gs[1, :])
        for s in self.samples:
            color = COLORS[1] if s['label'] == 1 else COLORS[0]
            label_text = "FAILURE (1)" if s['label'] == 1 else "SUCCESS (0)"
            ax3.plot(s['risks'], color=color, alpha=0.8, linewidth=2, label=label_text)
            
            if s['label'] == 1:
                # Find the biggest jump in risk
                diffs = np.diff(s['risks'], prepend=0)
                idx = np.argmax(diffs)
                if diffs[idx] > 0.1: # Only annotate significant jumps
                    ax3.annotate('Root Cause', (idx, s['risks'][idx]), xytext=(idx, s['risks'][idx]+0.15), 
                                 arrowprops=dict(facecolor='white', shrink=0.05), ha='center')
                    
        ax3.set_ylim(-0.05, 1.05); ax3.set_title("Risk Velocity (Supervised)")
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        # Save visual
        epoch_idx = len(self.epoch_data['train'])
        self.fig.savefig(os.path.join(self.log_dir, f"epoch_{epoch_idx}.png"), facecolor=self.fig.get_facecolor(), edgecolor='none')
        
        display(self.fig)

    def log_batch(self, l, g): self.step_data['loss'].append(l); self.step_data['grad'].append(g)
    def log_epoch(self, t, v, s): self.epoch_data['train'].append(t); self.epoch_data['val'].append(v); self.samples = s; self.refresh()

def calculate_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    preds_binary = (preds > 0.5).astype(int)
    
    tp = np.sum((preds_binary == 1) & (targets == 1))
    tn = np.sum((preds_binary == 0) & (targets == 0))
    fp = np.sum((preds_binary == 1) & (targets == 0))
    fn = np.sum((preds_binary == 0) & (targets == 1))
    
    total = tp + tn + fp + fn + 1e-8
    
    acc = (tp + tn) / total
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    
    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1
    }

def execute_training(model, optimizer, loaders, device, epochs=5, config=None, scheduler=None):
    monitor = DarkMonitor(config=config)
    print(f"Logging training data to: {monitor.log_dir}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_sum = 0
        train_preds, train_targets = [], []
        
        for i, batch in enumerate(loaders['train']):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            s_risk, g_risk = model(batch)
            loss = build_train_model.risk_velocity_loss(s_risk, g_risk, batch['label'], batch['mask'])
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            monitor.log_batch(loss.item(), grad.item())
            monitor.log_batch_detail(epoch, i, loss.item(), grad.item())
            train_sum += loss.item()
            
            train_preds.extend(g_risk.detach().cpu().numpy())
            train_targets.extend(batch['label'].cpu().numpy())
            
        if scheduler:
            scheduler.step()
            
        train_metrics = calculate_metrics(train_preds, train_targets)
            
        # --- VAL ---
        model.eval()
        val_sum = 0
        samples = []
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for i, batch in enumerate(loaders['val']):
                batch = {k: v.to(device) for k, v in batch.items()}
                s_risk, g_risk = model(batch)
                val_sum += build_train_model.risk_velocity_loss(s_risk, g_risk, batch['label'], batch['mask']).item()
                
                val_preds.extend(g_risk.cpu().numpy())
                val_targets.extend(batch['label'].cpu().numpy())
                
                if i == 0:
                    labels, masks = batch['label'].cpu().numpy(), batch['mask'].cpu().numpy()
                    risks = s_risk.cpu().numpy()
                    # Grab a few failures and successes
                    f_idxs = np.where(labels==1)[0][:2]
                    s_idxs = np.where(labels==0)[0][:2]
                    idxs = np.concatenate([f_idxs, s_idxs])
                    for idx in idxs:
                        if idx < len(labels): # Safety check
                            valid = (~masks[idx]).sum()
                            samples.append({'risks': risks[idx, :valid], 'label': labels[idx]})
        
        val_metrics = calculate_metrics(val_preds, val_targets)
        
        # --- LOGGING ---
        monitor.log_epoch(train_sum/len(loaders['train']), val_sum/len(loaders['val']), samples)
        
        full_metrics = {
            'train_loss': train_sum/len(loaders['train']),
            'val_loss': val_sum/len(loaders['val']),
            'train_acc': train_metrics['acc'], 'val_acc': val_metrics['acc'],
            'train_f1': train_metrics['f1'], 'val_f1': val_metrics['f1'],
            'train_prec': train_metrics['prec'], 'val_prec': val_metrics['prec'],
            'train_rec': train_metrics['rec'], 'val_rec': val_metrics['rec']
        }
        monitor.save_metrics(epoch, full_metrics, time.time() - start_time)
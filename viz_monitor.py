import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, clear_output
import numpy as np
import torch
import build_train_model

plt.style.use('dark_background')
COLORS = ['#00ffc8', '#ff007f', '#f9f871', '#00d2fc']

class DarkMonitor:
    def __init__(self):
        self.epoch_data = {'train': [], 'val': []}
        self.step_data = {'loss': [], 'grad': []}
        self.samples = []
        self.fig = None

    def refresh(self):
        clear_output(wait=True)
        if self.fig: plt.close(self.fig)
        self.fig = plt.figure(figsize=(16, 9), facecolor='#1e1e1e')
        gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[1, 1.5])
        
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.plot(self.epoch_data['train'], 'o-', color=COLORS[0], label='Train')
        ax1.plot(self.epoch_data['val'], 'o-', color=COLORS[1], label='Val')
        ax1.legend(); ax1.set_title("Total Training Loss")
        
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
        
        plt.tight_layout(); display(self.fig)

    def log_batch(self, l, g): self.step_data['loss'].append(l); self.step_data['grad'].append(g)
    def log_epoch(self, t, v, s): self.epoch_data['train'].append(t); self.epoch_data['val'].append(v); self.samples = s; self.refresh()

def execute_training(model, optimizer, loaders, device, epochs=5):
    monitor = DarkMonitor()
    loss_cfg = loaders.get('loss_cfg', {})
    for epoch in range(epochs):
        model.train()
        train_sum = 0
        for batch in loaders['train']:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            s_risk, g_risk = model(batch)
            loss = build_train_model.risk_velocity_loss(
                s_risk, g_risk, batch['label'], batch['mask'], batch['step_targets'], **loss_cfg
            )
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            monitor.log_batch(loss.item(), grad.item())
            train_sum += loss.item()
            
        model.eval()
        val_sum = 0
        samples = []
        with torch.no_grad():
            for i, batch in enumerate(loaders['val']):
                batch = {k: v.to(device) for k, v in batch.items()}
                s_risk, g_risk = model(batch)
                val_sum += build_train_model.risk_velocity_loss(
                    s_risk, g_risk, batch['label'], batch['mask'], batch['step_targets'], **loss_cfg
                ).item()
                if i == 0:
                    labels, masks = batch['label'].cpu().numpy(), batch['mask'].cpu().numpy()
                    risks = s_risk.cpu().numpy()
                    # Grab a few failures and successes
                    f_idxs = np.where(labels==1)[0][:2]
                    s_idxs = np.where(labels==0)[0][:2]
                    idxs = np.concatenate([f_idxs, s_idxs])
                    for idx in idxs:
                        valid = (~masks[idx]).sum()
                        samples.append({'risks': risks[idx, :valid], 'label': labels[idx]})
                        
        monitor.log_epoch(train_sum/len(loaders['train']), val_sum/len(loaders['val']), samples)

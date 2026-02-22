import generate_data
import build_train_model
import viz_monitor
import torch

def main():
    print("Step 1: Generating Data...")
    packages_df, scans_df, calendar_df, _ = generate_data.run_simulation()
    
    print("Step 2: Setting up System...")
    model, optimizer, loaders, device, scheduler = build_train_model.setup_system(packages_df, scans_df, calendar_df)
    
    print("Step 3: Executing Training...")
    # Attempt to reconstruct basic config info if possible, or just pass None
    # We know the hardcoded values in build_train_model.py
    config_info = {
        "description": "Temporal Split Config (d_model=128, layers=3, dropout=0.2)",
        "cont_dim": 32,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3
    }
    
    viz_monitor.execute_training(model, optimizer, loaders, device, epochs=10, config=config_info, scheduler=scheduler)
    
    print("Training Complete. Check training_logs/ directory.")

if __name__ == "__main__":
    main()

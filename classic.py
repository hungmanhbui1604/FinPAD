import torch
import os
import yaml
import argparse
import wandb

from data_utils.data_loaders import get_dataloader
from models.classic_models import get_model
from train_utils.metrics import find_optimal_threshold
from train_utils.one_epochs import train_one_epoch_binary, validate_one_epoch_binary, test_one_epoch_binary
from data_utils.transforms import get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Classification')
    parser.add_argument('-c', '--config', type=str, default='./configs/classic_config.yml', help='Path to the configuration file')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create model save directory if it doesn't exist
    os.makedirs(os.path.dirname(config['MODEL_SAVE_PATH']), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb
    wandb.init(
        project="FinPAD",
        name=f"classic_{config['MODEL_NAME']}_{config['YEAR']}_{config['TRAIN_SENSOR']}_{config['TEST_SENSOR']}",
        config=config,
        tags=["spoof_classification"]
    )

    # Get transforms
    transform = get_transforms(config['TRANSFORM_TYPE'])

    # Get data loaders for binary classification
    train_loader, val_loader, train_label_map = get_dataloader(
        year=config['YEAR'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        dataset_path=config['DATASET_PATH'],
        train=True,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT']
    )
    test_loader, test_label_map = get_dataloader(
        year=config['YEAR'],
        train_sensor=config['TRAIN_SENSOR'],
        test_sensor=config['TEST_SENSOR'],
        dataset_path=config['DATASET_PATH'],
        train=False,
        binary_class=config['BINARY_CLASS'],
        transform=transform,
        batch_size=config['BATCH_SIZE'],
        num_workers=config['NUM_WORKERS'],
        val_split=config['VALID_SPLIT']
    )

    # Initialize model, criterion, optimizer, and scheduler
    model, criterion, optimizer, scheduler = get_model(
        model_name=config['MODEL_NAME'],
        num_classes=1,  # Binary classification: real vs spoof
        criterion_type=config['CRITERION_TYPE'],
        optimizer_type=config['OPTIMIZER_TYPE'],
        learning_rate=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY'],
        scheduler_type=config['SCHEDULER_TYPE'],
        num_epochs=config['NUM_EPOCHS'],
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Initialize variables for best model tracking
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(config['NUM_EPOCHS']):
        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']}")
        print('-' * 36)

        train_metrics = train_one_epoch_binary(model, train_loader, criterion, optimizer, device, config)
        val_metrics = validate_one_epoch_binary(model, val_loader, criterion, device, config)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config['LEARNING_RATE'],
            **train_metrics,
            **val_metrics
        })
        
        # Save the best model based on validation loss
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            best_model_state = model.state_dict().copy()
            print(f"New best model found! Saving to {config['MODEL_SAVE_PATH']}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, config['MODEL_SAVE_PATH'])
            
            # Save model to wandb
            wandb.save(config['MODEL_SAVE_PATH'])
        
        scheduler.step()

    # Load the best model for testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Testing phase
    labels, probabilities = test_one_epoch_binary(model, test_loader, device)
    threshold, apcer, bpcer, ace, accuracy = find_optimal_threshold(labels, probabilities, based_on="ace")
    
    # Log final test results to wandb
    test_results = {
        'test/APCER': apcer * 100,
        'test/BPCER': bpcer * 100,
        'test/ACE': ace * 100,
        'test/Accuracy': accuracy * 100,
        'test/Accuracy_star': (1 - ace) * 100,
        'test/Threshold': threshold
    }
    wandb.log(test_results)
    
    print(f"APCER:      {apcer*100:.2f}%")
    print(f"BPCER:      {bpcer*100:.2f}%")
    print(f"ACE:        {ace*100:.2f}%")
    print(f"Accuracy:   {accuracy*100:.2f}%")
    print(f"Accuracy*:  {(1-ace)*100:.2f}%")
    
    # Finish wandb run
    wandb.finish()
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

class IntraSensorBinaryDataset(Dataset):
    def __init__(self, year, sensor, dataset_path, train=True):
        self.transform = transform
        self.samples = []
        self.label_map = {'Live': 0, 'Spoof': 1}

        phase = 'Train' if train else 'Test'
        
        sensor_path = os.path.join(dataset_path, year, sensor, phase)
        if not os.path.isdir(sensor_path):
            raise RuntimeError(f"Dataset directory not found: {sensor_path}")

        for label_name, label_id in self.label_map.items():
            data_path = os.path.join(sensor_path, label_name)
            if not os.path.isdir(data_path):
                raise RuntimeError(f"Data directory not found: {data_path}")

            if label_name == 'Live':
                for img_file in tqdm(os.listdir(data_path), desc=f"Loading {phase} Live"):
                    if img_file.endswith('.bmp'):
                        image_path = os.path.join(data_path, img_file)
                        self.samples.append((image_path, label_id))
            elif label_name == 'Spoof':
                for spoof_material in os.listdir(data_path):
                    spoof_material_path = os.path.join(data_path, spoof_material)
                    if os.path.isdir(spoof_material_path):
                        for img_file in tqdm(os.listdir(spoof_material_path), desc=f"Loading {phase} {spoof_material}"):
                            if img_file.endswith(('.png', '.bmp')):
                                image_path = os.path.join(spoof_material_path, img_file)
                                self.samples.append((image_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label
    

class IntraSensorMultiClassDataset(Dataset):
    def __init__(self, year, sensor, dataset_path, train=True):
        self.transform = transform
        self.samples = []
        self.label_map = {'Live': 0}

        phase = 'Train' if train else 'Test'
        
        sensor_path = os.path.join(dataset_path, year, sensor, phase)
        if not os.path.isdir(sensor_path):
            raise RuntimeError(f"Dataset directory not found: {sensor_path}")

        # Load Live images
        live_path = os.path.join(sensor_path, 'Live')
        if not os.path.isdir(live_path):
            raise RuntimeError(f"Data directory not found: {live_path}")
        
        for img_file in tqdm(os.listdir(live_path), desc=f"Loading {phase} Live"):
            if img_file.endswith(('.png', '.bmp')):
                image_path = os.path.join(live_path, img_file)
                self.samples.append((image_path, self.label_map['Live']))

        # Load Spoof images and create labels dynamically
        spoof_path = os.path.join(sensor_path, 'Spoof')
        if not os.path.isdir(spoof_path):
            raise RuntimeError(f"Data directory not found: {spoof_path}")

        spoof_materials = sorted([d for d in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, d))])
        
        for i, material in enumerate(spoof_materials, 1):
            self.label_map[material] = i
            material_path = os.path.join(spoof_path, material)
            for img_file in tqdm(os.listdir(material_path), desc=f"Loading {phase} {material}"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(material_path, img_file)
                    self.samples.append((image_path, self.label_map[material]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label


class CrossSensorBinaryDataset(Dataset):
    def __init__(self, year, sensor, dataset_path):
        self.transform = transform
        self.samples = []
        self.label_map = {'Live': 0, 'Spoof': 1}

        for phase in ['Train', 'Test']:
            sensor_path = os.path.join(dataset_path, year, sensor, phase)
            if not os.path.isdir(sensor_path):
                raise RuntimeError(f"Dataset directory not found: {sensor_path}")

            for label_name, label_id in self.label_map.items():
                data_path = os.path.join(sensor_path, label_name)
                if not os.path.isdir(data_path):
                    raise RuntimeError(f"Data directory not found: {data_path}")

                if label_name == 'Live':
                    for img_file in tqdm(os.listdir(data_path), desc=f"Loading {phase} Live"):
                        if img_file.endswith(('.png', '.bmp')):
                            image_path = os.path.join(data_path, img_file)
                            self.samples.append((image_path, label_id))
                elif label_name == 'Spoof':
                    for spoof_material in os.listdir(data_path):
                        spoof_material_path = os.path.join(data_path, spoof_material)
                        if os.path.isdir(spoof_material_path):
                            for img_file in tqdm(os.listdir(spoof_material_path), desc=f"Loading {phase} {spoof_material}"):
                                if img_file.endswith(('.png', '.bmp')):
                                    image_path = os.path.join(spoof_material_path, img_file)
                                    self.samples.append((image_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label


class CrossSensorMultiClassDataset(Dataset):
    def __init__(self, year, sensor, dataset_path):
        self.transform = transform
        self.samples = []
        self.label_map = {'Live': 0}

        # Discover all spoof materials from both Train and Test
        all_spoof_materials = set()
        for phase in ['Train', 'Test']:
            spoof_path = os.path.join(dataset_path, year, sensor, phase, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Spoof directory not found in {phase} set: {spoof_path}")
            materials = [d for d in os.listdir(spoof_path) if os.path.isdir(os.path.join(spoof_path, d))]
            all_spoof_materials.update(materials)
        
        sorted_spoof_materials = sorted(list(all_spoof_materials))
        for i, material in enumerate(sorted_spoof_materials, 1):
            self.label_map[material] = i

        # Load data from both Train and Test
        for phase in ['Train', 'Test']:
            sensor_path = os.path.join(dataset_path, year, sensor, phase)
            if not os.path.isdir(sensor_path):
                raise RuntimeError(f"Dataset directory not found: {sensor_path}")

            # Load Live images
            live_path = os.path.join(sensor_path, 'Live')
            if not os.path.isdir(live_path):
                raise RuntimeError(f"Live directory not found in {phase} set: {live_path}")
            for img_file in tqdm(os.listdir(live_path), desc=f"Loading {phase} Live"):
                if img_file.endswith(('.png', '.bmp')):
                    image_path = os.path.join(live_path, img_file)
                    self.samples.append((image_path, self.label_map['Live']))

            # Load Spoof images
            spoof_path = os.path.join(sensor_path, 'Spoof')
            if not os.path.isdir(spoof_path):
                raise RuntimeError(f"Spoof directory not found in {phase} set: {spoof_path}")
            for material in os.listdir(spoof_path):
                if material in self.label_map:
                    label_id = self.label_map[material]
                    material_path = os.path.join(spoof_path, material)
                    if os.path.isdir(material_path):
                        for img_file in tqdm(os.listdir(material_path), desc=f"Loading {phase} {material}"):
                            if img_file.endswith(('.png', '.bmp')):
                                image_path = os.path.join(material_path, img_file)
                                self.samples.append((image_path, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return image, label


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def split_dataset(dataset: Dataset, val_split: float = 0.2, seed: int = 42) -> tuple[Subset, Subset]:
    if not 0 < val_split < 1:
        raise ValueError("Validation split must be between 0 and 1.")

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def get_dataloader(
    intra: bool,
    year: str,
    sensor: str,
    dataset_path: str,
    phase: str,
    binary_class: bool,
    transform: dict[str, transforms.Compose],
    batch_size: int,
    num_workers: int,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    if intra:
        train = phase == 'Train'
        if binary_class:
            dataset = IntraSensorBinaryDataset(year, sensor, dataset_path, train=train)
        else:
            dataset = IntraSensorMultiClassDataset(year, sensor, dataset_path, train=train)
    else: # Cross-sensor
        if binary_class:
            dataset = CrossSensorBinaryDataset(year, sensor, dataset_path)
        else:
            dataset = CrossSensorMultiClassDataset(year, sensor, dataset_path)

    if phase == 'Train':
        train_subset, val_subset = split_dataset(dataset, val_split=val_split, seed=seed)
        train_dataset = TransformedDataset(train_subset, transform['Train'])
        val_dataset = TransformedDataset(val_subset, transform['Test'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
    else: # Test phase
        test_dataset = TransformedDataset(dataset, transform['Test'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader, None


if __name__ == '__main__':
    # Example usage:
    DATASET_PATH = '/home/hmb1604/projects/FinPAD/data'
    YEAR = '2015'
    SENSOR = 'CrossMatch'
    
    # Define transformations
    transform = {
        'Train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'Test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    try:
        # --- Test get_dataloader function ---
        print("--- Testing get_dataloader function ---")

        # 1. Intra-sensor, Binary, Train -> returns train_loader and val_loader
        print("\n1. Intra-sensor, Binary, Train -> train/val split")
        train_loader_1, val_loader_1 = get_dataloader(
            intra=True, year=YEAR, sensor=SENSOR, dataset_path=DATASET_PATH,
            phase='Train', binary_class=True, transform=transform,
            batch_size=32, num_workers=4
        )
        print(f"Train loader 1 batches: {len(train_loader_1)}")
        print(f"Validation loader 1 batches: {len(val_loader_1)}")
        assert val_loader_1 is not None

        # 2. Intra-sensor, Binary, Test -> returns test_loader and None
        print("\n2. Intra-sensor, Binary, Test")
        test_loader_2, val_loader_2 = get_dataloader(
            intra=True, year=YEAR, sensor=SENSOR, dataset_path=DATASET_PATH,
            phase='Test', binary_class=True, transform=transform,
            batch_size=32, num_workers=4
        )
        print(f"Test loader 2 batches: {len(test_loader_2)}")
        assert val_loader_2 is None
        
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the dataset is located at the correct path and has the expected structure.")
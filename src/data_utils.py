"""
Data Utilities for Syngenta Crop Disease Classification

Handles data loading, preprocessing, augmentation, and pipeline creation

"""

import os
import json
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config

# Set random seeds
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


class DataPipeline:
    """
    Complete data pipeline for crop disease classification
    """
    
    def __init__(self, raw_data_dir=None, processed_dir=None):
        """
        Initialize data pipeline
        
        Args:
            raw_data_dir (Path): Path to raw dataset
            processed_dir (Path): Path to save processed splits
        """
        self.raw_data_dir = raw_data_dir or config.RAW_DATA_DIR
        self.processed_dir = processed_dir or config.PROCESSED_DATA_DIR
        self.class_names = []
        self.class_indices = {}
        self.num_classes = 0
        
    def verify_dataset_structure(self):
        """
        Verify dataset exists and has correct structure
        
        Returns:
            bool: True if valid, raises error otherwise
        """
        print("\n" + "="*70)
        print("VERIFYING DATASET STRUCTURE")
        print("="*70)
        
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.raw_data_dir}\n"
                f"Please download PlantVillage dataset and place in data/raw/"
            )
        
        # Get all class directories
        class_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir()]
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.raw_data_dir}")
        
        self.class_names = sorted([d.name for d in class_dirs])
        
        # If limited classes, select subset
        if config.NUM_CLASSES_TO_USE and config.NUM_CLASSES_TO_USE < len(self.class_names):
            print(f"\n⚠ Using subset: {config.NUM_CLASSES_TO_USE} classes (for faster training)")
            self.class_names = self.class_names[:config.NUM_CLASSES_TO_USE]
        
        self.num_classes = len(self.class_names)
        
        # Count images per class
        class_counts = {}
        for class_name in self.class_names:
            class_path = self.raw_data_dir / class_name
            images = list(class_path.glob('*.[jJ][pP][gG]')) + \
                    list(class_path.glob('*.[pP][nN][gG]'))
            class_counts[class_name] = len(images)
        
        print(f"\n✓ Dataset found: {self.raw_data_dir}")
        print(f"✓ Total classes: {self.num_classes}")
        print(f"✓ Total images: {sum(class_counts.values()):,}")
        print(f"✓ Images per class: {min(class_counts.values())} - {max(class_counts.values())}")
        
        print(f"\nClasses (showing first 5):")
        for i, name in enumerate(self.class_names[:5]):
            print(f"  {i+1}. {name}: {class_counts[name]} images")
        if len(self.class_names) > 5:
            print(f"  ... and {len(self.class_names)-5} more classes")
        
        return True
    
    def create_deterministic_splits(self, force_recreate=False):
        """
        Create deterministic train/val/test splits with stratification
        
        Args:
            force_recreate (bool): If True, recreate splits even if exist
            
        Returns:
            tuple: (train_dir, val_dir, test_dir)
        """
        print("\n" + "="*70)
        print("CREATING DETERMINISTIC SPLITS")
        print("="*70)
        
        train_dir = self.processed_dir / "train"
        val_dir = self.processed_dir / "valid"
        test_dir = self.processed_dir / "test"
        
        # Check if splits already exist
        if train_dir.exists() and not force_recreate:
            print(f"\n✓ Splits already exist at {self.processed_dir}")
            print("  Use force_recreate=True to regenerate")
            return train_dir, val_dir, test_dir
        
        # Clear existing splits
        for split_dir in [train_dir, val_dir, test_dir]:
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create splits for each class
        total_train, total_val, total_test = 0, 0, 0
        
        for class_name in self.class_names:
            # Get all images for this class
            class_path = self.raw_data_dir / class_name
            images = list(class_path.glob('*.[jJ][pP][gG]')) + \
                    list(class_path.glob('*.[pP][nN][gG]'))
            
            if len(images) == 0:
                print(f"⚠ Warning: No images found for {class_name}")
                continue
            
            # Convert to strings for sklearn
            image_paths = [str(img) for img in images]
            
            # First split: train + (val+test)
            train_files, temp_files = train_test_split(
                image_paths,
                train_size=config.TRAIN_SPLIT,
                random_state=config.RANDOM_SEED,
                shuffle=True
            )
            
            # Second split: val + test
            val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
            val_files, test_files = train_test_split(
                temp_files,
                train_size=val_ratio,
                random_state=config.RANDOM_SEED,
                shuffle=True
            )
            
            # Copy files to respective directories
            for split_name, files, split_dir in [
                ("train", train_files, train_dir),
                ("valid", val_files, val_dir),
                ("test", test_files, test_dir)
            ]:
                class_split_dir = split_dir / class_name
                class_split_dir.mkdir(parents=True, exist_ok=True)
                
                for file_path in files:
                    src = Path(file_path)
                    dst = class_split_dir / src.name
                    shutil.copy2(src, dst)
                
                if split_name == "train":
                    total_train += len(files)
                elif split_name == "valid":
                    total_val += len(files)
                else:
                    total_test += len(files)
        
        # Save class indices
        self.class_indices = {name: idx for idx, name in enumerate(self.class_names)}
        self._save_class_indices()
        
        print(f"\n✓ Splits created successfully:")
        print(f"  Train: {total_train} images ({config.TRAIN_SPLIT*100:.0f}%)")
        print(f"  Valid: {total_val} images ({config.VAL_SPLIT*100:.0f}%)")
        print(f"  Test:  {total_test} images ({config.TEST_SPLIT*100:.0f}%)")
        print(f"\n✓ Saved to: {self.processed_dir}")
        
        return train_dir, val_dir, test_dir
    
    def _save_class_indices(self):
        """Save class name to index mapping"""
        with open(config.CLASS_INDICES_PATH, 'w') as f:
            json.dump(self.class_indices, f, indent=4)
        print(f"✓ Class indices saved to: {config.CLASS_INDICES_PATH}")
    
    def create_data_generators(self):
        """
        Create ImageDataGenerators with augmentation
        
        Returns:
            tuple: (train_gen, val_gen, test_gen, class_indices)
        """
        print("\n" + "="*70)
        print("CREATING DATA GENERATORS")
        print("="*70)
        
        # Ensure splits exist
        train_dir, val_dir, test_dir = self.create_deterministic_splits()
        
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            **config.AUGMENTATION_CONFIG
        )
        
        # Validation and test generators (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=config.RANDOM_SEED
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        # Update class indices
        self.class_indices = train_generator.class_indices
        self._save_class_indices()
        
        print(f"\n✓ Generators created:")
        print(f"  Train: {train_generator.samples} images")
        print(f"  Valid: {val_generator.samples} images")
        print(f"  Test:  {test_generator.samples} images")
        print(f"  Classes: {len(self.class_indices)}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        
        print(f"\n✓ Augmentation applied to training data:")
        for key, value in config.AUGMENTATION_CONFIG.items():
            print(f"  - {key}: {value}")
        
        return train_generator, val_generator, test_generator, self.class_indices
    
    def get_dataset_statistics(self):
        """
        Calculate dataset statistics
        
        Returns:
            dict: Statistics about the dataset
        """
        stats = {
            'total_classes': self.num_classes,
            'class_names': self.class_names,
            'class_distribution': {}
        }
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.processed_dir / split
            if split_dir.exists():
                split_stats = {}
                for class_name in self.class_names:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        count = len(list(class_dir.glob('*.[jJ][pP][gG]')) + 
                                   list(class_dir.glob('*.[pP][nN][gG]')))
                        split_stats[class_name] = count
                stats['class_distribution'][split] = split_stats
        
        return stats


def load_class_indices(path=None):
    """
    Load class indices from JSON file
    
    Args:
        path (Path): Path to class_indices.json
        
    Returns:
        dict: Class name to index mapping
    """
    path = path or config.CLASS_INDICES_PATH
    with open(path, 'r') as f:
        return json.load(f)


def preprocess_image(image_path, target_size=None):
    """
    Load and preprocess a single image
    
    Args:
        image_path (str): Path to image
        target_size (tuple): Target size (height, width)
        
    Returns:
        np.array: Preprocessed image array
    """
    target_size = target_size or config.IMG_SIZE
    
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=target_size
    )
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


if __name__ == "__main__":
    """Test data pipeline"""
    print("Testing Data Pipeline...")
    
    pipeline = DataPipeline()
    pipeline.verify_dataset_structure()
    pipeline.create_deterministic_splits(force_recreate=False)
    
    train_gen, val_gen, test_gen, class_indices = pipeline.create_data_generators()
    
    print("\n✓ Data pipeline test completed successfully!")
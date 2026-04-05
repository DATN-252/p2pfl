#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
#

"""AID Dataset loader for P2PFL."""

import os
from datasets import Dataset, DatasetDict
import kagglehub
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

class AIDDataset(P2PFLDataset):
    """AID Scene Classification Dataset."""

    def __init__(self, dataset_id: str = "jiayuanchengala/aid-scene-classification-datasets", batch_size: int = 32):
        # 1. Download dataset
        kaggle_path = kagglehub.dataset_download(dataset_id)
        data_dir = os.path.join(kaggle_path, "AID")
        
        # 2. Gather all image paths and labels
        classes = sorted(os.listdir(data_dir))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        image_paths = []
        labels = []
        
        for cls_name in classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(class_to_idx[cls_name])
        
        # 3. Create HuggingFace Dataset
        full_ds = Dataset.from_dict({
            "image": image_paths,
            "label": labels
        })
        
        # 4. Split into train/test (80/20)
        split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
        
        # 5. Initialize base class
        super().__init__(
            data=split_ds,
            train_split_name="train",
            test_split_name="test",
            batch_size=batch_size,
            dataset_name="AID"
        )

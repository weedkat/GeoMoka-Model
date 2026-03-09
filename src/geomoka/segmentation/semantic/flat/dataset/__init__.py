from geomoka.dataset.seg_dataset import GenericDataset

dataset_map = {
    "GenericDataset": GenericDataset,
    #"custom": CustomDataset,
}

def get_dataset(name, **kwargs):
    """Factory function to get dataset instance by name."""
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(dataset_map.keys())}")
    
    return dataset_map[name](**kwargs)
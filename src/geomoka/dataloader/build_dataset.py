from geomoka.dataloader.dataset import GenericDataset

dataset_map = {
    "generic": GenericDataset,
    #"custom": CustomDataset,
}

def get_dataset(name, split_csv, root_dir, metadata):
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(dataset_map.keys())}")
    
    return dataset_map[name](split_csv, root_dir=root_dir, metadata=metadata)
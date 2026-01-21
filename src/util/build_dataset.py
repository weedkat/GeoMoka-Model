from dataloader.dataset import ISPRSPostdam

dataset_map = {
    'ISPRS-Postdam': ISPRSPostdam,
    # Add other datasets here
}

def get_dataset(name, split_csv, root_dir=None, class_dict=None):
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[name]

    return dataset_class(split_csv, root_dir=root_dir, class_dict=class_dict)
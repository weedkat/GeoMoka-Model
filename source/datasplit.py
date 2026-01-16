import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from pathlib import Path

SEED = 10

def seg_split(img_dir, mask_dir, out_dir, labeled_ratio=(), test_val_split=0.3, img_ext='', mask_ext=''):
    X = sorted(list(Path(img_dir).glob(f'*{img_ext}')))
    y = sorted(list(Path(mask_dir).glob(f'*{mask_ext}')))

    X = [p.name for p in X]
    y = [p.name for p in y]

    assert len(X) == len(y)

    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=test_val_split)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    train_df = pd.DataFrame({'Source': X_train, 'Target': y_train})
    val_df = pd.DataFrame({'Source': X_val, 'Target': y_val})
    test_df = pd.DataFrame({'Source': X_test, 'Target': y_test})

    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    for ratio in labeled_ratio:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train, y_train, train_size=ratio, random_state=SEED)
        out_ratio_dir = os.path.join(out_dir, f'{int(ratio*100)}%')
        os.makedirs(out_ratio_dir, exist_ok=True)

        labeled_df = pd.DataFrame({'Source': X_labeled, 'Target': y_labeled})
        unlabeled_df = pd.DataFrame({'Source': X_unlabeled, 'Target': y_unlabeled})
        
        labeled_df.to_csv(os.path.join(out_ratio_dir, 'labeled.csv'), index=False)
        unlabeled_df.to_csv(os.path.join(out_ratio_dir, 'unlabeled.csv'), index=False)
    
    print(f'Split saved to {out_dir}')


if __name__ == '__main__':
    img_dir = 'dataset/ISPRS-Postdam/patches/Images'
    mask_dir = 'dataset/ISPRS-Postdam/patches/Labels'
    out_dir = 'dataset/ISPRS-Postdam/splits'
    labeled_ratio = (0.4, 0.2, 0.1, 0.05)

    seg_split(img_dir, mask_dir, out_dir, labeled_ratio, img_ext='.tif', mask_ext='.tif')
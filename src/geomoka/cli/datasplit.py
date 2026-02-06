import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

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
    parser = argparse.ArgumentParser(description='Segmented Dataset Splitter')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing masks')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for splits')
    parser.add_argument('--test_val_split', type=float, default=0.3, help='Test/validation split ratio')
    parser.add_argument('--labeled_ratio', type=float, nargs='*', default=[0.4, 0.2, 0.1, 0.05], help='Ratios for labeled data splits')
    parser.add_argument('--img_ext', type=str, default='.tif', help='Image file extension')
    parser.add_argument('--mask_ext', type=str, default='.tif', help='Mask file extension')

    args = parser.parse_args()

    seg_split(args.img_dir, args.mask_dir, args.out_dir, args.labeled_ratio, img_ext=args.img_ext, mask_ext=args.mask_ext)
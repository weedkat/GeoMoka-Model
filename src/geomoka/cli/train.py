import argparse

from geomoka.train.trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(description='GeoMoka training CLI')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--save_dir', type=str, default='outputs', help='path to save checkpoints and logs')
    args = parser.parse_args()

    trainer = SegmentationTrainer.from_config(args.config, save_dir=args.save_dir)
    trainer.run()

if __name__ == '__main__':
    main()
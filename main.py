import argparse
import yaml

from src.trainer import ScatSimCLRTrainer, PretextTaskTrainer


def main(args):

    mode = args.mode

    if mode not in ['unsupervised', 'pretext']:
        raise ValueError('Unsupported mode')

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    if mode == 'unsupervised':
        trainer = ScatSimCLRTrainer(config)
    elif mode == 'pretext':
        trainer = PretextTaskTrainer(config)
    trainer.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m',
                        help='Training mode. `unsupervised` - run training only with contrastive loss, '
                             '`pretext` - run training with contrastive loss and pretext task',
                        choices=['unsupervised', 'pretext'])
    parser.add_argument('--config', '-c',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)

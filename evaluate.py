import yaml
import argparse

from src.trainer import ScatSimCLRTrainer


def run_evaluation(config_path):
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    trainer = ScatSimCLRTrainer(config)
    score = trainer.evaluate()
    print(f'Score: {score}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        help='configuration file')
    args = parser.parse_args()
    run_evaluation(args.config)

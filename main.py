import yaml

from src.trainer import ScatSimCLRTrainer, PretextTaskTrainer


def run_unsupervised():
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

    trainer = ScatSimCLRTrainer(config)
    trainer.train()


def run_pretext():
    config = yaml.load(open('./config_pretext.yaml', 'r'), Loader=yaml.FullLoader)

    trainer = PretextTaskTrainer(config)
    trainer.train()


if __name__ == '__main__':
    run_unsupervised()

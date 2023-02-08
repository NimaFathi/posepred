from .utils.args import *
from .main import main


def run():
    parser = get_parser()
    main_args, dataset_args, model_args, evaluation_args = get_args(parser)
    print(f"Main args: {main_args}")
    print(f"Dataset args: {dataset_args}")
    print(f"Model args: {model_args}")
    print(f"Evaluation args: {evaluation_args}")
    main(main_args, dataset_args, model_args, evaluation_args)


if __name__ == '__main__':
    run()

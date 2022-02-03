import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    # Defining the model
    parser.add_argument(
        "--dataset", default="bci3", type=str, help="Dataset(BCI3)"
    )
    parser.add_argument(
        "--run", default="1", type=int, help="Trial run"
    )

    my_args = parser.parse_args()

    return my_args
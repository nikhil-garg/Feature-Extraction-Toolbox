import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Train a reservoir based SNN on biosignals"
    )

    # Defining the model
    parser.add_argument(
        "--dataset", default="bci3", type=str, help="Dataset(BCI3)"
    )

    my_args = parser.parse_args()

    return my_args
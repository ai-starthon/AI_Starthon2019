
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_file_name",
    type=str, default="train_data",
    help=""" Train dataset file name """,
)

parser.add_argument(
    "--train_label_file_name",
    type=str, default="train_label",
    help=""" Train dataset label file name """,
)

parser.add_argument(
    "--valid_file_name",
    type=str, default="valid_data",
    help=""" Validation dataset file name """,
)

parser.add_argument(
    "--valid_label_file_name",
    type=str, default="valid_label",
    help=""" Validation dataset label file name """,
)

parser.add_argument(
    "--test_file_name",
    type=str, default="test_data",
    help=""" Validation dataset file name """,
)

parser.add_argument(
    "--test_label_file_name",
    type=str, default="test_label",
    help=""" Validation dataset label file name """,
)

parser.add_argument(
    "--batch_size",
    type=int, default=512,
    help=""" Maximum batch size for trainer """,
)

parser.add_argument(
    "--max_sequence_len",
    type=int, default=128,
    help=""" Maximum sequence length """
)

parser.add_argument(
    "--char_embed_size",
    type=int, default=200,
    help=""" Word embed size of char-word CNN """
)

parser.add_argument(
    "--filter_sizes",
    type=list, default=[1, 2, 3, 4, 5],
    help=""" Sizes of filters of char-word CNN """
)

parser.add_argument(
    "--sentence_embed_size",
    type=int, default=1000,
    help=""" Sentence embed size of char-word CNN """
)

parser.add_argument(
    "--dropout",
    type=float, default=0.2,
    help=""" Dropout value of char-word CNN """
)

parser.add_argument(
    "--activation",
    type=str, default="relu",
    help=""" Activation function of char-word CNN """
)

parser.add_argument(
    "--num_epochs",
    type=int, default=50,
    help=""" Number of epochs for training """
)

parser.add_argument(
    "--early_stop_threshold",
    type=int, default=10,
    help=""" Threshold for early stopping """
)

parser.add_argument(
    "--log_steps",
    type=int, default=100,
    help=""" Number of steps to log """
)

parser.add_argument(
    "--learning_rate",
    type=float, default=0.003,
    help=""" Learning rate """,
)

parser.add_argument(
    "--seed",
    type=int, default=42,
    help=""" Random seed """,
)


# reserved for nsml
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--iteration", type=str, default="0")
parser.add_argument("--pause", type=int, default=0)


def get_config():
    return parser.parse_args()

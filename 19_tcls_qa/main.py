
import argparse
import json
import os

from claf.config.factory import (
    DataReaderFactory,
    TokenMakersFactory,
    ModelFactory,
)
from claf.config.namespace import NestedNamespace
from claf.config.utils import set_global_seed
from claf.tokens.text_handler import TextHandler
from claf.learn.optimization.optimizer import get_optimizer_by_name
from claf.learn.trainer import Trainer
from claf.tokens.vocabulary import Vocab

import nsml
from nsml import DATASET_PATH, IS_ON_NSML
import torch

from utils import bind_nsml, create_by_factory, create_model, create_data_loader, set_logging_config

LANG_CODE = "ko"
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_and_evaluate(config):
    token_makers = create_by_factory(TokenMakersFactory, config.token)
    tokenizers = token_makers["tokenizers"]
    del token_makers["tokenizers"]

    config.data_reader.tokenizers = tokenizers
    if nsml.IS_ON_NSML:
        config.data_reader.train_file_path = os.path.join(DATASET_PATH, "train", "train_data", config.data_reader.train_file_path)
        config.data_reader.valid_file_path = os.path.join(DATASET_PATH, "train", "train_data", config.data_reader.valid_file_path)

    data_reader = create_by_factory(DataReaderFactory, config.data_reader)
    datas, helpers = data_reader.read()

    # Vocab & Indexing
    text_handler = TextHandler(token_makers, lazy_indexing=True)
    texts = data_reader.filter_texts(datas)

    token_counters = text_handler.make_token_counters(texts)
    text_handler.build_vocabs(token_counters)
    text_handler.index(datas, data_reader.text_columns)

    # Iterator
    datasets = data_reader.convert_to_dataset(datas, helpers=helpers)
    train_loader = create_data_loader(
        datasets["train"],
        batch_size=config.iterator.batch_size,
        shuffle=True,
        cuda_device_id=device)
    valid_loader = create_data_loader(
        datasets["valid"],
        batch_size=config.iterator.batch_size,
        shuffle=False,
        cuda_device_id=device)

    # Model & Optimizer
    model = create_model(token_makers, ModelFactory, config.model, device, helpers=helpers)
    model_parameters = [param for param in model.parameters() if param.requires_grad]

    optimizer = get_optimizer_by_name("adam")(model_parameters)

    if IS_ON_NSML:
        bind_nsml(model, optimizer=optimizer)

    # Trainer
    trainer_config = vars(config.trainer)
    trainer_config["model"] = model
    trainer = Trainer(**trainer_config)
    trainer.train_and_evaluate(train_loader, valid_loader, optimizer)


def test(config):
    NSML_SESSEION = None  # NOTE: need to hard code
    NSML_CHECKPOINT = None  # NOTE: need to hard code

    assert NSML_CHECKPOINT is not None, "You must insert NSML Session's checkpoint for submit"
    assert NSML_SESSEION is not None, "You must insert NSML Session's name for submit"

    set_global_seed(config.seed_num)

    token_makers = create_by_factory(TokenMakersFactory, config.token)
    tokenizers = token_makers["tokenizers"]
    del token_makers["tokenizers"]

    config.data_reader.tokenizers = tokenizers
    data_reader = create_by_factory(DataReaderFactory, config.data_reader)

    def bind_load_vocabs(config, token_makers):
        CHECKPOINT_FNAME = "checkpoint.bin"

        def load(dir_path):
            checkpoint_path = os.path.join(dir_path, CHECKPOINT_FNAME)
            checkpoint = torch.load(checkpoint_path)

            vocabs = {}
            token_config = config.token
            for token_name in token_config.names:
                token = getattr(token_config, token_name, {})
                vocab_config = getattr(token, "vocab", {})

                texts = checkpoint["vocab_texts"][token_name]
                if type(vocab_config) != dict:
                    vocab_config = vars(vocab_config)
                vocabs[token_name] = Vocab(token_name, **vocab_config).from_texts(texts)

            for token_name, token_maker in token_makers.items():
                token_maker.set_vocab(vocabs[token_name])
            return token_makers

        nsml.bind(load=load)

    bind_load_vocabs(config, token_makers)
    nsml.load(checkpoint=NSML_CHECKPOINT, session=NSML_SESSEION)

    # Raw to Tensor Function
    text_handler = TextHandler(token_makers, lazy_indexing=False)
    raw_to_tensor_fn = text_handler.raw_to_tensor_fn(
        data_reader,
        cuda_device=device,
    )

    # Model & Optimizer
    model = create_model(token_makers, ModelFactory, config.model, device)
    trainer = Trainer(model, metric_key="f1")

    if nsml.IS_ON_NSML:
        bind_nsml(model, trainer=trainer, raw_to_tensor_fn=raw_to_tensor_fn)
        if config.nsml.pause:
            nsml.paused(scope=locals())


if __name__ == "__main__":
    """ Config """
    parser = argparse.ArgumentParser()

    # BaseConfig
    parser.add_argument(
        "--base_config", type=str, default="base_config/baseline.json",
        help="CLaF BaseConfig file"
    )

    # NSML
    parser.add_argument(
        "--mode",
        type=str, default="train_and_evaluate",
        help=""" NSML mode setting """,
    )
    parser.add_argument(
        "--iteration",
        type=int, default=0,
        help=""" NSML default setting """,
    )
    parser.add_argument(
        "--pause",
        type=int, default=0,
        help=""" NSML default setting """,
    )
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        defined_config = json.load(f)
    config = NestedNamespace()
    config.load_from_json(defined_config)
    config.nsml = args

    set_logging_config()

    if args.mode == "train_and_evaluate":
        train_and_evaluate(config)
    elif args.mode == "test" or args.mode == "infer":
        test(config)
    else:
        raise ValueError(f"Unrecognized mode. {config.mode}")

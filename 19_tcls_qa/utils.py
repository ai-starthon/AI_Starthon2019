
import logging
import os
import sys

import nsml

import torch
from torch.utils.data import DataLoader


def set_logging_config():
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging_handlers = [stdout_handler]
    logging_level = logging.INFO

    log_path = os.path.join(
        "logs", "baseline.log"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    logging_handlers.append(file_handler)

    logging.basicConfig(
        format="%(asctime)s (%(filename)s:%(lineno)d): [%(levelname)s] - %(message)s",
        handlers=logging_handlers,
        level=logging_level,
    )


def create_by_factory(factory_cls, item_config, params={}):
    return factory_cls(item_config).create(**params)


def create_model(token_makers, factory_cls, model_config, device, helpers=None):
    if helpers is None:
        model_init_params = {}
        predict_helper = {}
    else:
        first_key = next(iter(helpers))
        helper = helpers[first_key]  # get first helper
        model_init_params = helper.get("model", {})
        predict_helper = helper.get("predict_helper", {})

    model_params = {"token_makers": token_makers}
    model_params.update(model_init_params)

    model = create_by_factory(
        factory_cls, model_config, params=model_params
    )
    # Save params
    model.init_params = model_init_params
    model.predict_helper = predict_helper

    model.to(device)
    return model


def create_data_loader(dataset, batch_size=32, shuffle=True, cuda_device_id=None):
    is_cpu = cuda_device_id is None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn(cuda_device_id=cuda_device_id),
        num_workers=0,
        pin_memory=is_cpu,  # only CPU memory can be pinned
    )


def bind_nsml(model, **kwargs):  # pragma: no cover
    if type(model) == torch.nn.DataParallel:
        model = model.module

    CHECKPOINT_FNAME = "checkpoint.bin"

    def infer(test_dir):
        trainer = kwargs["trainer"]
        raw_to_tensor_fn = kwargs["raw_to_tensor_fn"]

        test_data_path = os.path.join(test_dir, "test_data")
        with open(test_data_path, "r", encoding="utf-8") as in_file:
            test_data = in_file.read()

        DELIMETER = "␝"

        output = []
        test_data = test_data.split("\n")
        for line in test_data:
            context, question = line.split(DELIMETER)

            raw_features = {
                "context": context,
                "question": question
            }
            argument = {}
            argument.update(raw_features)
            prediction = trainer.predict(raw_features, raw_to_tensor_fn, argument)

            output.append((prediction["score"], prediction["text"]))

        return output

    def load(dir_path, *args):
        checkpoint_path = os.path.join(dir_path, CHECKPOINT_FNAME)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["weights"])
        model.config = checkpoint["config"]
        model.metrics = checkpoint["metrics"]
        model.init_params = checkpoint["init_params"],
        model.predict_helper = checkpoint["predict_helper"],
        model.train_counter = checkpoint["train_counter"]
        # model.vocabs = load_vocabs(checkpoint)

        if "optimizer" in kwargs:
            kwargs["optimizer"].load_state_dict(checkpoint["optimizer"])

        print(f"Load checkpoints...! {checkpoint_path}")

    def save(dir_path, *args):
        checkpoint_path = os.path.join(dir_path, CHECKPOINT_FNAME)

        # save the model with 'checkpoint' dictionary.
        checkpoint = {
            "config": model.config,
            "init_params": model.init_params,
            "predict_helper": model.predict_helper,
            "metrics": model.metrics,
            "train_counter": model.train_counter,
            "vocab_texts": {k: v.to_text() for k, v in model.vocabs.items()},
            "weights": model.state_dict(),
        }

        if "optimizer" in kwargs:
            checkpoint["optimizer"] = (kwargs["optimizer"].state_dict(),)

        torch.save(checkpoint, checkpoint_path)

        train_counter = model.train_counter
        print(f"Save {train_counter.global_step} global_step checkpoints...! {checkpoint_path}")

    # function in function is just used to divide the namespace.
    nsml.bind(save, load, infer)

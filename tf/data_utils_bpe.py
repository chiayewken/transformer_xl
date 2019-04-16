from __future__ import absolute_import, division, print_function

import json
import os
import pickle

import tensorflow
from absl import flags

from vocabulary_bpe import get_encoder
from data_utils import Corpus


class BPECorpus(Corpus):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = get_encoder()

        if self.dataset == "bookcorpus":
            # assumes train.txt has been sharded to avoid OOM eg train00, train01 etc
            train_path_pattern = os.path.join(path, "train*")
            self.train = tensorflow.gfile.Glob(train_path_pattern)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"))

        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            self.train = tensorflow.gfile.Glob(train_path_pattern)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "valid.txt"))

        else:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"))
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"))

        # not sure what this is for
        self.cutoffs = []
        # if cutoffs = [], tfrecords will be much smaller
        # training will throw ValueError: Dimension -56279 must be >= 0
        # likely means that a value in cutoff exceeded vocab size
        # self.cutoffs = [0, 20000, 40000, 200000] + [len(self.vocab)]


def get_lm_corpus(data_dir, dataset):
    fn = os.path.join(data_dir, "cache.pkl")

    if tensorflow.gfile.Exists(fn):
        print("Loading cached dataset...")
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    else:
        print("Producing dataset...")
        corpus = BPECorpus(data_dir, dataset)

        print("Saving dataset...")
        with open(fn, "wb") as fp:
            pickle.dump(corpus, fp, protocol=2)

        corpus_info = {
            "vocab_size": len(corpus.vocab),
            "cutoffs": corpus.cutoffs,
            "dataset": corpus.dataset,
        }
        with open(os.path.join(data_dir, "corpus-info.json"), "w") as fp:
            json.dump(corpus_info, fp)

    return corpus


def main(unused_argv):
    del unused_argv  # Unused

    corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

    save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
    if not tensorflow.gfile.Exists(save_dir):
        tensorflow.gfile.MakeDirs(save_dir)

    # test mode
    if FLAGS.per_host_test_bsz > 0:
        corpus.convert_to_tfrecords(
            "test",
            save_dir,
            FLAGS.per_host_test_bsz,
            FLAGS.tgt_len,
            FLAGS.num_core_per_host,
            FLAGS=FLAGS,
        )
        return

    for split, batch_size in zip(
        ["train", "valid"], [FLAGS.per_host_train_bsz, FLAGS.per_host_valid_bsz]
    ):

        if batch_size <= 0:
            continue
        print("Converting {} set...".format(split))
        corpus.convert_to_tfrecords(
            split,
            save_dir,
            batch_size,
            FLAGS.tgt_len,
            FLAGS.num_core_per_host,
            FLAGS=FLAGS,
        )


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("data_dir", None, help="Location of the data corpus")
    flags.DEFINE_enum(
        "dataset",
        "wt103",
        ["ptb", "wt2", "wt103", "lm1b", "enwik8", "text8", "bookcorpus"],
        help="Dataset name.",
    )
    flags.DEFINE_integer("per_host_train_bsz", 60, help="train batch size each host")
    flags.DEFINE_integer("per_host_valid_bsz", 60, help="valid batch size each host")
    flags.DEFINE_integer(
        "per_host_test_bsz",
        0,
        help="If > 0, enter test mode and process test set only."
        "Otherwise, process train and dev sets only.",
    )
    flags.DEFINE_integer("tgt_len", 70, help="number of tokens to predict")
    flags.DEFINE_integer("max_batch", -1, help="run in debug mode")
    flags.DEFINE_integer("num_core_per_host", 8, help="8 for TPU v2.")
    flags.DEFINE_bool(
        "debug",
        default=False,
        help="Process only the first batch without shuffle for lm1b.",
    )
    flags.DEFINE_integer("num_procs", 1, help="number of processes")
    flags.DEFINE_integer("num_passes", 10, help="number of passes when use_tpu=True")
    flags.DEFINE_integer("num_shuffle", 4, help="number of shuffles for lm1b")
    flags.DEFINE_bool("use_tpu", True, help="use tpu")

    tensorflow.app.run(main)

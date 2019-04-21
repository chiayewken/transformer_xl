import json
import os
import pickle
from functools import partial
from typing import List

import tensorflow as tf
from absl import flags
from tqdm import tqdm
from multiprocessing import Pool

from data_utils import get_bin_sizes, create_ordered_tfrecords
from vocabulary_bpe import get_encoder, Encoder


def preprocess(
    idx_shard: int,
    train: List[str],
    vocab: Encoder,
    save_dir: str,
    cutoffs: List[int],
    bin_sizes: List[int],
    bsz: int,
    tgt_len: int,
    num_core_per_host: int,
    use_tpu: bool,
) -> (List[str], int):
    path = train[idx_shard]
    data_shard = vocab.encode_file(path, disable_tqdm=True)

    basename = "train-{:03d}".format(idx_shard)
    file_name, num_batch = create_ordered_tfrecords(
        save_dir=save_dir,
        basename=basename,
        data=data_shard,
        batch_size=bsz,
        tgt_len=tgt_len,
        num_core_per_host=num_core_per_host,
        cutoffs=cutoffs,
        bin_sizes=bin_sizes,
        use_tpu=use_tpu,
        disable_tqdm=True,  # tqdm in inner loop doesn't display well
    )
    return [file_name], num_batch


class BPECorpus:
    def __init__(self, path: str, dataset: str, *args, **kwargs):
        self.dataset = dataset
        self.vocab = get_encoder()

        if self.dataset == "bookcorpus":
            # assumes train.txt has been sharded to avoid OOM eg train00, train01 etc
            train_path_pattern = os.path.join(path, "train*")
            self.train = tf.io.gfile.glob(train_path_pattern)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"))

        elif self.dataset == "stories_corpus":
            # assumes train.txt has been sharded to avoid OOM eg train00, train01 etc
            train_path_pattern = os.path.join(path, "*train.txt")
            self.train = tf.io.gfile.glob(train_path_pattern)
            self.valid = self.vocab.encode_file(
                os.path.join(path, "stories.00002.of.1000.dev.txt")
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "stories.00001.of.1000.test.txt")
            )

        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            self.train = tf.io.gfile.glob(train_path_pattern)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "valid.txt"))

        else:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"))
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"))
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"))

        # not sure what this is for
        # self.cutoffs = []
        # if cutoffs = [], tfrecords will be much smaller
        # training will throw ValueError: Dimension -56279 must be >= 0
        # eg 200k cutoff > 57k bpe vocab size)
        # self.cutoffs = [0, 20000, 40000, 200000] + [len(self.vocab)]
        # if self.dataset in ["wt103", "bookcorpus"]:
        #     self.cutoffs = [0, 20000, 40000] + [len(self.vocab)]
        # else:
        self.cutoffs = []

    def convert_to_tfrecords(
        self,
        split: str,
        save_dir: str,
        bsz: int,
        tgt_len: int,
        num_core_per_host: int,
        **kwargs: dict
    ) -> None:
        FLAGS = kwargs.get("FLAGS")
        file_names = []
        use_tpu = FLAGS.use_tpu and not (split == "test" and num_core_per_host == 1)

        if use_tpu:
            record_name = "record_info-{}.bsz-{}.tlen-{}.core-{}.json".format(
                split, bsz, tgt_len, num_core_per_host
            )
        else:
            record_name = "record_info-{}.bsz-{}.tlen-{}.json".format(
                split, bsz, tgt_len
            )
        record_info_path = os.path.join(save_dir, record_name)

        # if large datasets eg bookcorpus, lm1b exist as separate files to avoid OOM
        # self.train will be list of filepaths instead of nparray of tokens
        # we need to encode each train file separately before conversion
        if split == "train" and type(self.train) == list:
            bin_sizes = get_bin_sizes(
                self.valid, bsz // num_core_per_host, tgt_len, self.cutoffs
            )
            num_batch = 0

            # for idx_shard, path in tqdm(enumerate(self.train), total=len(self.train)):
            #     # nested tqdms misbehave in colab
            #     data_shard = self.vocab.encode_file(path, disable_tqdm=True)
            #     basename = "train-{:03d}".format(idx_shard)
            #     file_name, num_batch_ = create_ordered_tfrecords(
            #         save_dir,
            #         basename,
            #         data_shard,
            #         bsz,
            #         tgt_len,
            #         num_core_per_host,
            #         self.cutoffs,
            #         bin_sizes,
            #         use_tpu=use_tpu,
            #         disable_tqdm=True,
            #     )
            #     file_names.append(file_name)
            #     num_batch += num_batch_

            _preprocess = partial(
                preprocess,
                train=self.train,
                vocab=self.vocab,
                save_dir=save_dir,
                cutoffs=self.cutoffs,
                bin_sizes=bin_sizes,
                bsz=bsz,
                tgt_len=tgt_len,
                num_core_per_host=num_core_per_host,
                use_tpu=use_tpu,
            )
            num_files = len(self.train)
            # If processes is None then the number returned by os.cpu_count() is used
            with Pool(processes=None) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(_preprocess, range(num_files)),
                        total=num_files,
                    )
                )
            for res in results:
                file_names.extend(res[0])
                num_batch += res[1]

        else:
            data = getattr(self, split)
            bin_sizes = get_bin_sizes(
                data, bsz // num_core_per_host, tgt_len, self.cutoffs
            )
            file_name, num_batch = create_ordered_tfrecords(
                save_dir,
                split,
                data,
                bsz,
                tgt_len,
                num_core_per_host,
                self.cutoffs,
                bin_sizes,
                num_passes=FLAGS.num_passes if split == "train" and use_tpu else 1,
                use_tpu=use_tpu,
            )
            file_names.append(file_name)

        with open(record_info_path, "w") as fp:
            record_info = {
                "filenames": file_names,
                "bin_sizes": bin_sizes,
                "num_batch": num_batch,
            }
            json.dump(record_info, fp)


def get_lm_corpus(data_dir: str, dataset: str) -> BPECorpus:
    fn = os.path.join(data_dir, "cache.pkl")

    if tf.io.gfile.exists(fn):
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


def main(unused_argv) -> None:
    del unused_argv  # Unused

    corpus = get_lm_corpus(FLAGS.data_dir, FLAGS.dataset)

    save_dir = os.path.join(FLAGS.data_dir, "tfrecords")
    if not tf.io.gfile.exists(save_dir):
        tf.io.gfile.makedirs(save_dir)

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
        [
            "ptb",
            "wt2",
            "wt103",
            "lm1b",
            "enwik8",
            "text8",
            "bookcorpus",
            "stories_corpus",
        ],
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
    tf.app.run(main)

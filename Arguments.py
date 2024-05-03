
from transformers import TrainingArguments
from typing import Any, Optional
from dataclasses import dataclass, field


#.............................................

#### ARGUMENTS


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    override_speaker_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True` and if `speaker_id_column_name` is specified, it will replace current speaker embeddings with a new set of speaker embeddings."
                "If the model from the checkpoint didn't have speaker embeddings, it will initialize speaker embeddings."
            )
        },
    )

    override_vocabulary_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, it will resize the token embeddings based on the vocabulary size of the tokenizer. In other words, use this when you use a different tokenizer than the one that was used during pretraining."
            )
        },
    )

#.............................................................................................


@dataclass
class VITSTrainingArguments(TrainingArguments):
    do_step_schedule_per_epoch: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to perform scheduler steps per epoch or per steps. If `True`, the scheduler will be `ExponentialLR` parametrized with `lr_decay`."
            )
        },
    )

    lr_decay: float = field(
        default=0.999875,
        metadata={"help": "Learning rate decay, used with `ExponentialLR` when `do_step_schedule_per_epoch`."},
    )

    weight_duration: float = field(default=1.0, metadata={"help": "Duration loss weight."})

    weight_kl: float = field(default=1.5, metadata={"help": "KL loss weight."})

    weight_mel: float = field(default=35.0, metadata={"help": "Mel-spectrogram loss weight"})

    weight_disc: float = field(default=3.0, metadata={"help": "Discriminator loss weight"})

    weight_gen: float = field(default=1.0, metadata={"help": "Generator loss weight"})

    weight_fmaps: float = field(default=1.0, metadata={"help": "Feature map loss weight"})
    d_learning_rate: float = field(default=2e-4, metadata={"help": "Feature map loss weight"})
   
    d_adam_beta1: float = field(default=0.8, metadata={"help": "Feature map loss weight"})
    d_adam_beta2: float = field(default=0.99, metadata={"help": "Feature map loss weight"})


#.............................................................................................

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    project_name: str = field(
        default="vits_finetuning",
        metadata={"help": "The project name associated to this run. Useful to track your experiment."},
    )
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    speaker_id_column_name: str = field(
        default=None,
        metadata={
            "help": """If set, corresponds to the name of the speaker id column containing the speaker ids.
                If `override_speaker_embeddings=False`:
                    it assumes that speakers are indexed from 0 to `num_speakers-1`.
                    `num_speakers` and `speaker_embedding_size` have to be set in the model config.

                If `override_speaker_embeddings=True`:
                        It will use this column to compute how many speakers there are.

                Defaults to None, i.e it is not used by default."""
        },
    )
    filter_on_speaker_id: int = field(
        default=None,
        metadata={
            "help": (
                "If `speaker_id_column_name` and `filter_on_speaker_id` are set, will filter the dataset to keep a single speaker_id (`filter_on_speaker_id`)  "
            )
        },
    )

    max_tokens_length: float = field(
        default=450,
        metadata={
            "help": ("Truncate audio files with a transcription that are longer than `max_tokens_length` tokens")
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the input text should be lower cased."},
    )
    do_normalize: bool = field(
        default=False,
        metadata={"help": "Whether the input waveform should be normalized."},
    )
    full_generation_sample_text: str = field(
        default="This is a test, let's see what comes out of this.",
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    uroman_path: str = field(
        default=None,
        metadata={
            "help": (
                "Absolute path to the uroman package. To use if your model requires `uroman`."
                "An easy way to check it is to go on your model card and manually check `is_uroman` in the `tokenizer_config.json,"
                "e.g the French checkpoint doesn't need it: https://huggingface.co/facebook/mms-tts-fra/blob/main/tokenizer_config.json#L4"
            )
        },
    )

#.............................................................................................

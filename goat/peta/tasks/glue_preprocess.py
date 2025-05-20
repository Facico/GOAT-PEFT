from .datasets_preprocess import DatasetPreprocessor, preprocess


class CoLA_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/cola
    """

    def preprocess(self, sentence: str, label: int):
        assert isinstance(sentence, str)
        assert isinstance(label, int)
        input_text = "cola sentence: {}".format(sentence)
        if label in [0, 1]:
            target_text = "acceptable" if label == 1 else "unacceptable"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["sentence"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence, label in zip(example["sentence"], example["label"]):
                _input_text, _target_text = self.preprocess(sentence, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class RTE_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/rte
    """

    def preprocess(self, sentence1, sentence2, label):
        assert isinstance(sentence1, str)
        assert isinstance(sentence2, str)
        assert isinstance(label, int)

        input_text = "rte sentence1: {sentence1} sentence2: {sentence2}".format(
            sentence1=sentence1, sentence2=sentence2
        )
        if label in [0, 1]:
            target_text = "entailment" if label == 0 else "not_entailment"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the RTE dataset into a text-to-text format.
        """
        if isinstance(example["sentence1"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence1, sentence2, label in zip(
                example["sentence1"], example["sentence2"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(sentence1, sentence2, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class MNLI_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/mnli/
    """

    def preprocess(self, hypothesis, premise, label):
        assert isinstance(hypothesis, str)
        assert isinstance(premise, str)
        assert isinstance(label, int)
        input_text = "mnli hypothesis: {hypothesis} premise: {premise}".format(
            hypothesis=hypothesis, premise=premise
        )
        if label in [0, 1, 2]:
            target_text = {0: "entailment", 1: "neutral", 2: "contradiction"}[label]
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the MNLI dataset into a text-to-text format.
        """
        if isinstance(example["hypothesis"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["hypothesis"], example["premise"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for hypothesis, premise, label in zip(
                example["hypothesis"], example["premise"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(hypothesis, premise, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class MRPC_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/mrpc
    """

    def preprocess(self, sentence1: str, sentence2: str, label: int):
        assert isinstance(sentence1, str)
        assert isinstance(sentence2, str)
        assert isinstance(label, int)
        input_text = "mrpc sentence1: {sentence1} sentence2: {sentence2}".format(
            sentence1=sentence1, sentence2=sentence2
        )
        if label in [0, 1]:
            target_text = "not_equivalent" if label == 0 else "equivalent"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the MRPC dataset into a text-to-text format.
        """
        if isinstance(example["sentence1"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence1, sentence2, label in zip(
                example["sentence1"], example["sentence2"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(sentence1, sentence2, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class QNLI_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/qnli
    """

    def preprocess(self, question: str, sentence: str, label: int):
        assert isinstance(question, str)
        assert isinstance(sentence, str)
        assert isinstance(label, int)
        input_text = "qnli question: {question} sentence: {sentence}".format(
            question=question, sentence=sentence
        )
        if label in [0, 1]:
            target_text = "not_entailment" if label == 1 else "entailment"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the QNLI dataset into a text-to-text format.
        """
        if isinstance(example["question"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["question"], example["sentence"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for question, sentence, label in zip(
                example["question"], example["sentence"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(question, sentence, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class QQP_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/qqp
    """

    def preprocess(self, question1, question2, label):
        assert isinstance(
            question1, str
        ), f"question1 must be a string, got {type(question1)}, question1={question1}"
        assert isinstance(
            question2, str
        ), f"question2 must be a string, got {type(question2)}, question2={question2}"
        assert isinstance(
            label, int
        ), f"label must be an int, got {type(label)}, label={label}"
        input_text = "qqp question1: {question1} question2: {question2}".format(
            question1=question1, question2=question2
        )
        if label in [0, 1]:
            target_text = "not_duplicate" if label == 0 else "duplicate"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the QQP dataset into a text-to-text format.
        """
        if isinstance(example["question1"], str):
            # batched
            input_text, target_text = self.preprocess(
                example["question1"], example["question2"], example["label"]
            )
        else:
            # not batched
            input_text, target_text = [], []
            for question1, question2, label in zip(
                example["question1"], example["question2"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(question1, question2, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class SST2_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/sst2
    """

    def preprocess(self, sentence, label):
        assert isinstance(
            sentence, str
        ), f"sentence must be a string, got {type(sentence)}, sentence={sentence}"
        assert isinstance(
            label, int
        ), f"label must be an integer, got {type(label)}, label={label}"
        input_text = "sst2 sentence: {sentence}".format(sentence=sentence)
        if label in [0, 1]:
            target_text = "negative" if label == 0 else "positive"
        else:
            target_text = ""
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the SST2 dataset into a text-to-text format.
        """
        if isinstance(example["sentence"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence, label in zip(example["sentence"], example["label"]):
                _input_text, _target_text = self.preprocess(sentence, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )


class STSB_Preprocessor(DatasetPreprocessor):
    """
    dataset URL: https://huggingface.co/datasets/glue/viewer/stsb
    """

    def preprocess(self, sentence1, sentence2, label):
        assert isinstance(
            sentence1, str
        ), f"sentence1 must be a string, got {type(sentence1)}, sentence1={sentence1}"
        assert isinstance(
            sentence2, str
        ), f"sentence2 must be a string, got {type(sentence2)}, sentence2={sentence2}"
        assert isinstance(
            label, (float, int)
        ), f"label must be a float or an integer, got {type(label)}, label={label}"
        input_text = "stsb sentence1: {sentence1} sentence2: {sentence2}".format(
            sentence1=sentence1, sentence2=sentence2
        )
        target_text = "{:.1f}".format(label)
        return input_text, target_text

    def __call__(self, example):
        """
        Preprocess the STSB dataset into a text-to-text format.
        """
        if isinstance(example["sentence1"], str):
            # not batched
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            # batched
            input_text, target_text = [], []
            for sentence1, sentence2, label in zip(
                example["sentence1"], example["sentence2"], example["label"]
            ):
                _input_text, _target_text = self.preprocess(sentence1, sentence2, label)
                input_text.append(_input_text)
                target_text.append(_target_text)

        return preprocess(
            tokenizer=self.tokenizer,
            input_text=input_text,
            target_text=target_text,
            tokenizer_kwawgs=self.tokenizer_kwargs,
        )

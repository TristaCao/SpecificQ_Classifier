from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
import string
import nltk
from nltk import word_tokenize

torch.manual_seed(1)

class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: int = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = LabelField(label=tags, skip_indexing = True)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            next(f)
            for line in f:
                pairs = line.strip().split(",")
                sentence = pairs[0]
                s_no_punc = sentence.translate(str.maketrans('','',string.punctuation))
                s_tokens = word_tokenize(s_no_punc)
                tags = 0
                if pairs[1] == "1":
                    tags = 1
                #pairs = line.strip().split()
                #sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in s_tokens], tags)
                #yield self.text_to_instance(s_tokens, tags)

class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=2)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            # TODO find a categorical loss function
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

reader = PosDatasetReader()
train_dataset = reader.read("processed_data_lstm.csv")
# TODO Seperate train dev test in data_process_f_raw.py
validation_dataset = reader.read("processed_data_lstm.csv")
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 200
HIDDEN_DIM = 500
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
#print(iterator)
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

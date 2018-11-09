from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
import re

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

#elmo boilerplate
#from allennlp.modules.elmo import Elmo, batch_to_ids
#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

#torch.manual_seed(1)



class CLAFFDatasetReader(DatasetReader):
    """
    DatasetReader for CL-AFF labelled data

        Structure is Number, sentence, concepts, agency, social, ...
        ex: 27691,"A hot kiss with my girl friend last night made my day","romance","yes","yes",...

        First line is labels and should be rejected
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], agency: str = None, social: str = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if agency:
            agency_field = LabelField(labels=agency)
            fields["labels"] = agency_field

        #if social:
            #social_field = LabelField(labels=social)
            #fields["social"] = social_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            #skip line one, check if labeled set
            firstline = next(f)
            isLabeled = firstline.split(',')[2].strip('"') is 'concepts'
            #now, read in data
            #regex to get rid of non-alphanumeric
            remover = re.compile('[\W_]+')
            for line in f:
                sets = line.split(',')
                sentence = sets[1].strip('"').split()
                if isLabeled:
                    agency = sets[3].strip('"')
                    social = sets[4].strip('"')
                else:
                    agency = None
                    social = None
                yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)



class LstmAgency(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        #the line below is important change. the [-1] takes the final output state from the LSTM,
        #effectively creating a many-to-one LSTM model

        #stopping point at 8:35pm on Wednesday, November 7
        #need to compute the loss for both and combine somehow. Consult web resources for
        #two value prediction models online. Once this is resolved it should train.
        tag_logits = self.hidden2tag(encoder_out[-1])
        output = {"agency_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



class LstmSocial(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('social'))

        self.accuracy = CategoricalAccuracy()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                social: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        #the line below is important change. the [-1] takes the final output state from the LSTM,
        #effectively creating a many-to-one LSTM model

        #stopping point at 8:35pm on Wednesday, November 7
        #need to compute the loss for both and combine somehow. Consult web resources for
        #two value prediction models online. Once this is resolved it should train.
        tag_logits = self.hidden2tag(encoder_out[-1])
        output = {"agency_logits": tag_logits}

        if social is not None:
            self.accuracy(tag_logits, agency, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



reader = CLAFFDatasetReader()

train_dataset = reader.read(cached_path('labeled_10k.csv'))
#validation_dataset = reader.read(cached_path('unlabeled_17k.csv'))

#vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
vocab = Vocabulary.from_instances(train_dataset)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#elmo = Elmo(options_file, weight_file, 1, dropout=0)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = LstmAgency(word_embeddings, lstm, vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  #validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)

trainer.train()

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

tag_logits = predictor.predict("The dog ate the apple")['tag_logits']

tag_ids = np.argmax(tag_logits, axis=-1)

print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])


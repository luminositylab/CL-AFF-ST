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
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError
from allennlp.training.metrics.boolean_accuracy import BooleanAccuracy  

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

from simple_seq2vec import SentenceSeq2VecPredictor

#for debug
import time


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
            agency_field = LabelField(label=agency)
            fields["agency"] = agency_field

        if social:
            social_field = LabelField(label=social)
            fields["social"] = social_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            #skip line one, check if labeled set
            firstline = next(f)
            isLabeled = firstline.split(',')[2].strip('"') == 'concepts'
            #now, read in data
            #regex to get rid of non-alphanumeric
            #remover = re.compile('[\W_]+')
            for line in f:
                sets = line.split(',')
                sentence = sets[1].strip('"').split()
                if isLabeled:
                    agency = sets[3].strip('"')
                    social = sets[4].strip('"')
                    if str(agency) != 'no':
                        agency = 'yes'
                    if str(social) != 'no':
                        social = 'yes'
                else:
                    agency = None
                    social = None
                #yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)
                yield self.text_to_instance([Token(word) for word in sentence], str(agency), str(social))



class LstmAgencySocial(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary,

                 lossmetric = torch.nn.MSELoss()) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=1)
        #Define separate accuracies
        self.accuracy_social = BooleanAccuracy()
        self.accuracy_agency = BooleanAccuracy()
        #Common loss
        self.loss = lossmetric
        #self.loss2 = lossmetric
        self.softmax = torch.nn.Softmax(dim=0)



    def forward(self,
                sentence: Dict[str, torch.Tensor],
                agency: torch.Tensor = None,
                social: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)
        print(embeddings.size())
        print(mask.size())
        #time.sleep(20)
        encoder_out_social = self.encoder(embeddings, mask)
        encoder_out_agency = self.encoder(embeddings, mask)
        #print(encoder_out.size())
        #print("Encode:",encoder_out[-1].size())
    
        loutput_social = self.hidden2tag(encoder_out_social)
        loutput_agency = self.hidden2tag(encoder_out_agency)

        #op = self.softmax(loutput)
        op_social = loutput_social
        #print(op)
        op_agency = loutput_agency
        #print("OP size:", op.size())


        output = {"social_score": op_social, "agency_score":op_agency}
        #print(output)
        #print("Size of agency_logits: {}".format(op.size()))
        #print("Size of agency array: {}".format(agency.unsqueeze(dim=1).size()))
        #print("Size of sentence array: {}".format(embeddings.size()))
        #print(agency)


        if social is not None and agency is not None:
            #Compute different accuracies
            self.accuracy_social(torch.round(op_social), social.type(torch.FloatTensor))
            self.accuracy_agency(torch.round(op_agency), agency.type(torch.FloatTensor))
            #Compute common loss
            output["loss"] = self.loss(torch.cat([op_social,op_agency], dim=1),torch.cat([social.unsqueeze(dim=1).type(torch.FloatTensor),agency.unsqueeze(dim=1).type(torch.FloatTensor)],dim=1))
        

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        #Return both the accuracies
        return {"accuracy_social": self.accuracy_social.get_metric(reset), "accuracy_agency": self.accuracy_agency.get_metric(reset) }



reader = CLAFFDatasetReader()

train_dataset = reader.read(cached_path('csv/labeled_9k5.csv'))
validation_dataset = reader.read(cached_path('csv/labeled_k5.csv'))

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

EMBEDDING_DIM = 20
HIDDEN_DIM = 5

#elmo = Elmo(options_file, weight_file, 1, dropout=0)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = LstmAgencySocial(word_embeddings, lstm, vocab)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

iterator = BucketIterator(batch_size=100, sorting_keys=[("sentence", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=500)

trainer.train()


#need to fix predictor
predictor = SentenceSeq2VecPredictor(model, dataset_reader=reader)

testsentence = "The dog ate the apple"
testsentence2 = "I made dinner for all my friends"

agency_output1 = predictor.predict(testsentence)['agency_score']
social_output1 = predictor.predict(testsentence)['social_score']
#agency_output2 = predictor.predict(testsentence2)['agency_score']
#social_output2 = predictor.predict(testsentence2)['social_score']


print("AGENCY For test sentence \'{}\', the output is {}".format(testsentence, agency_output1))
print("SOCIAL For test sentence \'{}\', the output is {}".format(testsentence, social_output1))
#print("For test sentence \'{}\', the output is {}".format(testsentence2, agency_output2))
#print("For test sentence \'{}\', the output is {}".format(testsentence2, social_output2))
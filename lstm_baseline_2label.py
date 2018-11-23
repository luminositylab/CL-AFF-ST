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

from torch.autograd import Variable

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

    def text_to_instance(self, tokens: List[Token], agency:str = None, social:str = None) -> Instance:
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
                #out = [str(agency), str(social)]
                #yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)
                yield self.text_to_instance([Token(word) for word in sentence],str(agency), str(social))



class LstmAgencySocial(Model):

    def __init__(self,

                 word_embeddings: TextFieldEmbedder,

                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary,

                 lossmetric = torch.nn. MSELoss()) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=2)

        self.accuracy = BooleanAccuracy()
        #self.accuracy_agency = BooleanAccuracy()
        self.loss = lossmetric
        #self.loss2 = lossmetric
        self.softmax = torch.nn.Softmax(dim=0)



    def forward(self,
                sentence: Dict[str, torch.Tensor],
                agency: torch.Tensor = None,
                social: torch.Tensor = None) -> torch.Tensor:

        mask = get_text_field_mask(sentence)
        #print(agency)
        embeddings = self.word_embeddings(sentence)
        #print(embeddings)
        #print(mask.size())
        #time.sleep(20)
        encoder_out_social = self.encoder(embeddings, mask)
        #encoder_out_agency = self.encoder(embeddings, mask)
        #print("THIS IS FIRST",encoder_out_agency.size())
        #print(encoder_out.size())
        #print("Encode:",encoder_out[-1].size())
        #the line below is important change. the [-1] takes the final output state from the LSTM,
        #effectively creating a many-to-one LSTM model

        #stopping point at 8:35pm on Wednesday, November 7
        #need to compute the loss for both and combine somehow. Consult web resources for
        #two value prediction models online. Once this is resolved it should train.
        #final = torch.cat((encoder_out_social,encoder_out_agency), dim=1)
        #print("LOOK HERE",final.size())
        loutput_social = self.hidden2tag(encoder_out_social)
        #print("RESULT",loutput_social.size())
        #loutput_agency = self.hidden2tag(encoder_out_agency)
        #print(loutput_social)
        #final = Variable(torch.cat((loutput_social,loutput_agency), dim=1), requires_grad=True)
        #print(final)

        #op = self.softmax(loutput)
        
        #final = self.softmax(final)
        #print(final)

        #print(op)
        #op_agency = self.softmax(loutput_agency)
        #print(loutput_agency)
        #print("OP size:", op.size())
        final = Variable(loutput_social,requires_grad=True)
        #output ={"social_score": op_social, "agency_score": op_agency}
        output = {"score": final}
        #output_social = {"score_social": op_social}
        #output_agency = {"score_agency": op_agency}
        #print(output)
        #print("Size of agency_logits: {}".format(op.size()))
        #print("Size of agency array: {}".format(agency.unsqueeze(dim=1).size()))
        #print("Size of sentence array: {}".format(embeddings.size()))
        #print(agency)


        if social is not None and agency is not None:
            #labels_acc = torch.cat((social,agency),dim=0)
            social_sq = social.unsqueeze(1)
            agency_sq = agency.unsqueeze(1)
            #print(social[14])
            #print(agency[14])
            labels = Variable(torch.cat((social_sq,agency_sq),dim=1))
            #print(labels[14])
            #print(labels.size())
            #labels =  labels.resize_((100,2)).squeeze(dim=0)
            
            #print(labels)
            #print("THIS:",labels.size())
            self.accuracy(torch.round(final), labels.type(torch.FloatTensor))
            #self.accuracy_agency(torch.round(final), labels.type(torch.FloatTensor))
            #self.accuracy_agency(torch.round(op_agency), agency.type(torch.FloatTensor))
            #output["loss"] = self.loss(torch.cat([op_social,op_agency], dim=1),torch.cat([social.unsqueeze(dim=1).type(torch.FloatTensor),agency.unsqueeze(dim=1).type(torch.FloatTensor)],dim=1))
            #output_agency["loss"] = self.loss(torch.cat([op_social,op_agency], dim=1),torch.cat([social.unsqueeze(dim=1).type(torch.FloatTensor),agency.unsqueeze(dim=1).type(torch.FloatTensor)],dim=1))
            output["loss"] = self.loss(final, labels.type(torch.FloatTensor))
            #print("LOSSSSSS1:", loss_social)
            
            #loss_agency = self.loss2(op_agency, agency.unsqueeze(dim=1).type(torch.FloatTensor))
            #print("LOSSSSSSS2:", loss_agency)

            #loss_final = loss_social + loss_agency
            #loss_final.backward(retain_graph=True)
            #output["loss"] = loss_final
            #print(output["social_score"])
            #print(output["agency_score"])

            #output_social["loss"] = loss_social 
            #output_agency["loss"] = loss_agency 
            #print(output_agency)
            #print(output_social)

        # if agency is not None:
        #     self.accuracy(torch.round(op_agency), social.type(torch.FloatTensor))
        #     output["loss"] = self.loss2(op_agency, social.unsqueeze(dim=1).type(torch.FloatTensor))
            #print(output)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset) }



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

model= LstmAgencySocial(word_embeddings, lstm, vocab)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)

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


#need to write predictor
predictor = SentenceSeq2VecPredictor(model, dataset_reader=reader)

testsentence = "The dog ate the apple"
testsentence2 = "I made dinner for all my friends"

agency_output1 = predictor.predict(testsentence)['score'][1]
print("OUTPUT:",agency_output1)
social_output1 = predictor.predict(testsentence)['score'][0]
print("OUTPUT:",social_output1)
#agency_output2 = predictor.predict(testsentence2)['agency_score']
#social_output2 = predictor.predict(testsentence2)['social_score']


print("AGENCY For test sentence \'{}\', the output is {}".format(testsentence, agency_output1))
print("SOCIAL For test sentence \'{}\', the output is {}".format(testsentence, social_output1))
#print("For test sentence \'{}\', the output is {}".format(testsentence2, agency_output2))
#print("For test sentence \'{}\', the output is {}".format(testsentence2, social_output2))
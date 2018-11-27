from typing import Iterator, List, Dict

#import statements for torch and parts in it we need
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


#AllenNLP imports for dataset stuff
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
#import statement for regular expressions (enables parsing out non alphanumerics)
import re
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path

#AllenNLP tokenizing stuff, more dataset related things
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary

#AllenNLP model stuff
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.mean_absolute_error import MeanAbsoluteError
from allennlp.training.metrics.boolean_accuracy import BooleanAccuracy  
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, move_optimizer_to_cuda
from allennlp.predictors import SentenceTaggerPredictor

#Custom seq2vecpredictor implemented based on the seq2seqpredictor example provided by allenNLP
#(not a big difference)
from simple_seq2vec import SentenceSeq2VecPredictor

#ELMo stuff
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.elmo import batch_to_ids
from elmo_cuda import Elmo
from cl_aff_embedders import ELMoTextFieldEmbedder

#for debug
import time
#torch.manual_seed(1)

cuda = torch.device('cuda')

class CLAFFDatasetReaderELMo(DatasetReader):
    """
    DatasetReader for CL-AFF labelled data
        Structure is Number, sentence, concepts, agency, social, ...
    """

    #SingleIdTokenIndexer is the class that links each word in the vocabulary to its token
    #we will be generating ours and thus using the singleidtoken indexer
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        #changing this line is what's important for ELMo vectors. This basically makes it so that the sentence
        #field will contain a sequence of character ids rather than word id tokens, this becomes important
        #when it's actually fed into the ELMo model for generating vectors.
        self.token_indexers = token_indexers or {"character_ids": ELMoTokenCharactersIndexer()}

    #this function converts a sentence into the appropriate instance type and has to be adapted to 
    #the model
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

    #this is the outermost function and it gets automatically called at the reader.read() step in main
    #it yields the outputs of text_to_instance which produces instance objects containing the keyed values
    #for agency, social, and the sentence itself as an iterator of instances. This fxn depends on the dataset
    #in use.
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



class LstmSocialAgency(Model):
    """
    LSTM model for predicting two labels Social and Agency for the CL-AFF labelled data
    """

    def __init__(self,
                 #Type of word embeddings
                 word_embeddings: TextFieldEmbedder,
                 #Type of encoder
                 encoder: Seq2SeqEncoder,

                 vocab: Vocabulary,
                 #Change loss function here
                 lossmetric = torch.nn.MSELoss()) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=2)
        #Initializing accuracy, loss and softmax variables
        self.accuracy = BooleanAccuracy()
        self.loss = lossmetric
        self.softmax = torch.nn.Softmax(dim=0)


    #I have gathered that the trainer method from allenNLP goes through the forward, loss = backward
    #sequence on its own and it searches for the keys in the instances that get passed as the arguments to
    #forward. It also automatically will convert labelField values from their number to their torch.Tensor
    #value when they get passed in, and will pass sentences as dictionaries of words tied to their torch tensor
    #token values. Inside the trainer function the unwrapping of and iterating over of instances is handled, so
    #we implement our forward pass function on the batched set of sentences level.
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                agency: torch.Tensor = None,
                social: torch.Tensor = None) -> torch.Tensor:
        #Mask to pad shorter sentences
        mask = get_text_field_mask(sentence).cuda()
        #Convert input into word embeddings
        embeddings = self.word_embeddings(sentence)

        #for certain debugging purposes
        #time.sleep(20)

        #the encoder is the name for the sequential model we plug in here. Once we implement the filterbank of
        #dilated convolutions, the encoder will be that rather than an LSTM.
        #Since we use a Seq2VecEncoder as input, the last hidden state of the LSTM is returned as the output
        encoder_out = self.encoder(embeddings, mask)

        #the output from hidden2tag, a fully-connected linear layer converting the LSTM hidden state to 
        #the two labels
        lin_output = self.hidden2tag(encoder_out)

        #output_score is a list of 2 variables which update the scores for social and agency class 
        #output_score = lin_output
        #output_score = self.sigmoid(output_score)
        
        output_score = torch.sigmoid(lin_output)
        output = {"score": output_score}
        #output_score = torch.sigmoid(output_score)

        if social is not None and agency is not None:
            #Unsqueeze(reshape) the tags to convert them to concatenatable format
            social_sq = social.unsqueeze(1)
            agency_sq = agency.unsqueeze(1)

            #Concat the two tags as a single variable to be passed into the loss function
            labels = torch.cat((social_sq,agency_sq),dim=1)
            
            #Accuracy(40%) is shit as of now, should improve with elmo word embeddings
            self.accuracy(torch.round(output_score), labels.type(torch.cuda.FloatTensor))
            
            #output["loss"] = self.loss(torch.cat([op_social,op_agency], dim=1),torch.cat([social.unsqueeze(dim=1).type(torch.FloatTensor),agency.unsqueeze(dim=1).type(torch.FloatTensor)],dim=1))
            
            #Single loss function for two label prediciton
            output["loss"] = self.loss(output_score, labels.type(torch.cuda.FloatTensor))
          
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset) }


torch.set_default_tensor_type(torch.cuda.FloatTensor)

################################EITHER USE THIS OR THE cl_aff_embedders.py ELMo embedder######################
print("Downloading the options file for ELMo...")
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
print("Downloading the weight file for ELMo...")
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
print("Done.")
elmo = Elmo(options_file, weight_file, 1, dropout=0)
##############################################################################################################

elmo.cuda()

#this is all to handle reading in the dataset and prepping the vocab for use. This will probably change slightly
#with the ELMo embeddings.
reader = CLAFFDatasetReaderELMo()

train_dataset = reader.read(cached_path('csv/labeled_9k5.csv'))
validation_dataset = reader.read(cached_path('csv/labeled_k5.csv'))

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

#word_embeddings = BasicTextFieldEmbedder({"character_ids": elmo})
word_embeddings = ELMoTextFieldEmbedder({"character_ids": elmo})

EMBEDDING_DIM = elmo.get_output_dim()
HIDDEN_DIM = 50

#initialize the model layers that we will want to change. 
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
################## for dilated convolutions we will be replacing lstm with our custom layer ##################
model= LstmSocialAgency(word_embeddings, lstm, vocab)

model.cuda()

#Set the optimizaer function here
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)
move_optimizer_to_cuda(optimizer)

# nice iterator functions are pretty much the only reason to stick with AllenNLP rn
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


#Predictor working as expected, returns a dictionary as output which is list with scores of social and agency
predictor = SentenceSeq2VecPredictor(model, dataset_reader=reader)


#battery of testing functions. At this point we can also implement the code to read in the test set for computing our system runs
#If the score value is <0.5 the label is YES, else a NO
#Not sure if this is the right thing to do although
testsentence = "my husband called me just to tell me he loved me"
testsentence2 = "I worked out which always makes me feel good"
testsentence3 = "Finally got to watch the new Resident Evil movie"
testsentence4 = "I got to talk to an old friend and reminisce on the good times"
testsentence5 = "I had a great meeting yesterday at work with my boss and a few colleagues and we went out for lunch afterward everybody was excited by the projects we're working on and how efficient our team is"




social_output1 = predictor.predict(testsentence)['score'][0]
agency_output1 = predictor.predict(testsentence)['score'][1]
social_output2 = predictor.predict(testsentence2)['score'][0]
agency_output2 = predictor.predict(testsentence2)['score'][1]
social_output3 = predictor.predict(testsentence3)['score'][0]
agency_output3 = predictor.predict(testsentence3)['score'][1]
social_output4 = predictor.predict(testsentence4)['score'][0]
agency_output4 = predictor.predict(testsentence4)['score'][1]
social_output5 = predictor.predict(testsentence5)['score'][0]
agency_output5 = predictor.predict(testsentence5)['score'][1]

if social_output1 <= 0.5:
    social_out = "YES"
else:
    social_out = "NO"

if agency_output1 <= 0.5:
    agency_out = "YES"
else:
    agency_out = "NO"

print("Social score for test sentence \'{}\', the output is {}".format(testsentence, social_output1))
print("Agency For test sentence \'{}\', the output is {}".format(testsentence, agency_output1))
print("Social Score for test sentence \'{}\', the output is {}".format(testsentence2, social_output2))
print("Agency for test sentence \'{}\', the output is {}".format(testsentence2, agency_output2))
print("Social Score for test sentence \'{}\', the output is {}".format(testsentence3, social_output3))
print("Agency for test sentence \'{}\', the output is {}".format(testsentence3, agency_output3))
print("Social Score for test sentence \'{}\', the output is {}".format(testsentence4, social_output4))
print("Agency for test sentence \'{}\', the output is {}".format(testsentence4, agency_output4))
print("Social Score for test sentence \'{}\', the output is {}".format(testsentence5, social_output5))
print("Agency for test sentence \'{}\', the output is {}".format(testsentence5, agency_output5))
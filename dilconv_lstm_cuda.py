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
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer, move_optimizer_to_cuda
from allennlp.predictors import SentenceTaggerPredictor

#Custom seq2vecpredictor implemented based on the seq2seqpredictor example provided by allenNLP
#(not a big difference)

#ELMo stuff
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.elmo import batch_to_ids

from cl_aff_utils.predictors import SentenceSeq2VecPredictor
from cl_aff_utils.elmo_cuda import Elmo
from cl_aff_utils.embedders import ELMoTextFieldEmbedder

#for debug
import time
import csv
import pandas as pd
import re
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
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



class BigramDilatedConvModel(Model):
    """
    LSTM model for predicting two labels Social and Agency for the CL-AFF labelled data
    """

    def __init__(self,
                 #Type of word embeddings
                 word_embeddings: TextFieldEmbedder,
                 #Type of encoder
                 encoder:Seq2SeqEncoder,
                 vocab: Vocabulary,
                 
                 #Change loss function here
                 lossmetric = torch.nn.MSELoss()) -> None:

        super().__init__(vocab)

        EMBEDDING_SIZE = 1024
        CONV_OUTPUT_SIZE = 25

        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.encoder_1= encoder
        self.encoder_2= encoder
        self.encoder_3= encoder
        self.encoder_4= encoder
        self.encoder_5= encoder
        
        
        

        self.conv_filterbank1 = torch.nn.Conv1d(EMBEDDING_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        self.conv_filterbank2 = torch.nn.Conv1d(EMBEDDING_SIZE,CONV_OUTPUT_SIZE,2,dilation=2,padding=1)
        self.conv_filterbank3 = torch.nn.Conv1d(EMBEDDING_SIZE,CONV_OUTPUT_SIZE,2,dilation=3,padding=2)
        self.conv_filterbank4 = torch.nn.Conv1d(EMBEDDING_SIZE,CONV_OUTPUT_SIZE,2,dilation=4,padding=2)
        self.conv_filterbank5 = torch.nn.Conv1d(EMBEDDING_SIZE,CONV_OUTPUT_SIZE,2,dilation=5,padding=3)
        #self.conv_filterbank_1 = torch.nn.Conv1d(CONV_OUTPUT_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        #self.conv_filterbank_2 = torch.nn.Conv1d(CONV_OUTPUT_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        #self.conv_filterbank_3 = torch.nn.Conv1d(CONV_OUTPUT_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        #self.conv_filterbank_4 = torch.nn.Conv1d(CONV_OUTPUT_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        #self.conv_filterbank_1 = torch.nn.Conv1d(CONV_OUTPUT_SIZE,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)

        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.AdaptiveMaxPool1d(1) 
        self.pool2 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool3 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool4 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool5 = torch.nn.AdaptiveMaxPool1d(1)
        self.drop = torch.nn.Dropout(p=0.2)
        
        self.hidden2tag = torch.nn.Linear(in_features=CONV_OUTPUT_SIZE,
                                          out_features=2)
        #Initializing accuracy, loss and softmax variables
        self.accuracy = BooleanAccuracy()
        self.f1 = F1Measure(positive_label=0)
        self.loss = lossmetric
        self.auc_s = 0

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
        embeddings = self.word_embeddings(sentence).permute(0,2,1)
        #print(embeddings.size())
        #print(mask.size())

        #for certain debugging purposes
        #time.sleep(20)
        #print(embeddings.shape)

        #we perform the set of convolutions followed by relu on top of them
        cset_1 = self.conv_filterbank1(embeddings).permute(0,2,1)
       
        self.relu = torch.nn.ReLU()
        cset_1 = self.relu(cset_1)
        cset_2 = self.conv_filterbank2(embeddings).permute(0,2,1)
        cset_2 = self.relu(cset_2)
        cset_3 = self.conv_filterbank3(embeddings).permute(0,2,1)
        cset_3 = self.relu(cset_3)
        cset_4 = self.conv_filterbank4(embeddings).permute(0,2,1)
        cset_4 = self.relu(cset_4)
        cset_5 = self.conv_filterbank5(embeddings).permute(0,2,1)
        cset_5 = self.relu(cset_5)
        
        #pass the convoluted layers into the encoder 
        encode_out_1 =  self.encoder_1(cset_1, mask)
        encode_out_2 =  self.encoder_2(cset_2,mask)
        encode_out_3 =  self.encoder_3(cset_3, mask)
        encode_out_4 =  self.encoder_4(cset_4, mask)
        encode_out_5 =  self.encoder_5(cset_5, mask)
        

  
        #concat the encoded results
        hidden_representation = torch.cat((encode_out_1,encode_out_2,encode_out_3,encode_out_4,encode_out_5),dim=-1)
        
       

    
        #the output from hidden2tag, a fully-connected linear layer converting the LSTM hidden state to 
        #the two labels
        #hidden_representation = F.relu(hidden_representation)
        #print(hidden_representation.size())
        #
        lin_output = self.hidden2tag(hidden_representation)
        #print("Linear:",lin_output.size())

        #output_score is a list of 2 variables which update the scores for social and agency class 
        #output_score = lin_output
        #output_score = self.sigmoid(output_score)
        
        output_score = torch.sigmoid(lin_output)

        output = {"score": output_score}
        #print(output_score.shape)
        #output_score = torch.sigmoid(output_score)

        if social is not None and agency is not None:
            #Unsqueeze(reshape) the tags to convert them to concatenatable format
            social_sq = social.unsqueeze(1)
            agency_sq = agency.unsqueeze(1)


            #Concat the two tags as a single variable to be passed into the loss function
            labels = torch.cat((social_sq,agency_sq),dim=1)
            #print(type(labels))
            # pred_labels = torch.empty(labels.size())
            # print(type(pred_labels))
            # for i in range(len(output_score)):
            #     if output_score[i][0] < 0.5:
            #         pred_labels[i][0] = 0
            #     else:
            #         pred_labels[i][0] = 1

            #     if output_score[i][1] < 0.5:
            #         pred_labels[i][1] = 0
            #     else:
            #         pred_labels[i][1] = 1               

            #print(pred_labels[0][0])
            
            #print(output_score.size())
            #print(labels.size())
            #print(output_score)
            #print(torch.round(output_score).size())
            
            #Accuracy(40%) is shit as of now, should improve with elmo word embeddings

            self.accuracy(torch.round(output_score), labels.type(torch.cuda.FloatTensor))
            #f1_result = []
            #f1_result = metrics.f1_score(labels, torch.round(output_score))
            soc_true = torch.empty(100,1)
            agen_true = torch.empty(100,1)
            pred_soc = torch.empty(100,1)
            pred_agen = torch.empty(100,1)
            pred_soc_label = torch.empty(100,1)
            pred_agen_label = torch.empty(100,1)
            for i in range(len(labels)):
                soc_true[i] = labels[i][0]
                agen_true[i] = labels[i][1]
                pred_soc[i] = output_score[i][0]
                pred_agen[i] = output_score[i][1]
                pred_soc_label[i] = (torch.round(output_score))[i][0]
                pred_agen_label[i] = (torch.round(output_score))[i][1]


            #print(soc_true.size())
            #print(pred_soc.size())
            #print(pred_soc_label.size())


            self.auc_s =  metrics.roc_auc_score(soc_true.data.cpu().numpy(), pred_soc.data.cpu().numpy())
            print("Social AUC:",self.auc_s)
            #auc_result_s = metrics.auc(fpr_s, tpr_s)
            #print("Social ROC:", auc_result_s)
            #fpr_a, tpr_a, threshold_a =  metrics.roc_curve(agen_true.data.cpu().numpy(), pred_agen.data.cpu().numpy())
            #auc_result_a = metrics.roc_auc_score(fpr_a, tpr_a)
            auc_a = metrics.roc_auc_score(agen_true.data.cpu().numpy(), pred_agen.data.cpu().numpy())
            print("Agency AUC:", auc_a)
            score = f1_score(soc_true.data.cpu().numpy(), pred_soc_label.data.cpu().numpy(), average='weighted')
            print("F1 Social:",score)
            score_f = f1_score(agen_true.data.cpu().numpy(), pred_agen_label.data.cpu().numpy(), average='weighted')
            print("F1 Agency:",score)
            #self.f1(torch.round(output_score), labels.type(torch.cuda.FloatTensor))

            
            #output["loss"] = self.loss(torch.cat([op_social,op_agency], dim=1),torch.cat([social.unsqueeze(dim=1).type(torch.FloatTensor),agency.unsqueeze(dim=1).type(torch.FloatTensor)],dim=1))
            
            #Single loss function for two label prediciton
            output["loss"] = self.loss(output_score.squeeze(), labels.type(torch.cuda.FloatTensor).squeeze())
          
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset), "auc_s": self.auc_s}


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

#lstm = PytorchSec2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first= True))

#initialize the model layers that we will want to change. 
EMBEDDING_DIM = 25
HIDDEN_DIM = 5
lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
################## for dilated convolutions we will be replacing lstm with our custom layer ##################
model= BigramDilatedConvModel(word_embeddings,lstm, vocab)

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



#Test has list of all test sentences

def clean_str(string):
 
    string = re.sub(r"\. \. \.", "\.", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

test = []
with open('csv/test_17k.csv',encoding="utf8", errors='ignore') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        header = next(readCSV)
        for row in readCSV:
            test.append(row[1])

for i in range(len(test)):
    test[i] =  clean_str(test[i])

social_score = []
agency_score = []
social_tag = []
agency_tag = []


#Assign yes/no label based on the prediction
for i in range(len(test)):
    print("Processing test datapoint {}...".format(test[i]))
    print("Number:", i)
    #social_score.append(predictor.predict(test[i])['score'][0])
    #agency_score.append(predictor.predict(test[i])['score'][1])

    if predictor.predict(test[i])['score'][0] < 0.5:
        social_tag.append("Yes")
    else:
        social_tag.append("No")

    if predictor.predict(test[i])['score'][1] < 0.5:
        agency_tag.append("Yes")
    else:
        agency_tag.append("No")



#Make a dict for output
d = {'Sentence':test,'Social':social_tag, 'Agency':agency_tag}
df = pd.DataFrame(d)
print(df)

#Save the sentence, social prediction, agency prediction on the test set in a csv file 
df.to_csv("test_results.csv",sep=',', index=False)






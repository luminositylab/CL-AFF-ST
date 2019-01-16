from typing import Iterator, List, Dict

#import statements for torch and parts in it we need
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import pandas as pd


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
#torch.manual_seed(1)
import csv

from sklearn import metrics
from sklearn.manifold import TSNE
import pylab
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.switch_backend('agg')
import pickle

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


class CLAFFDatasetReaderELMofromDataFrame(DatasetReader):
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
    def _read(self, data) -> Iterator[Instance]:
        #skip line one, check if labeled set
        #firstline = next(f)
        #isLabeled = firstline.split(',')[2].strip('"') == 'concepts'
        #now, read in data
        #regex to get rid of non-alphanumeric
        #remover = re.compile('[\W_]+')
        #we only use this reader to read in the labeled 10k that gets cross val'd
        if data[0]:
            sentence = data[1].split()
            yield self.text_to_instance([Token(word) for word in sentence])
        else:
            for line in data[1].iterrows():
                #sets = line.split(',')
                sentence = line[1][1].split()
                agency = line[1][3]
                social = line[1][4]
                if str(agency) != 'no':
                    agency = 'yes'
                if str(social) != 'no':
                    social = 'yes'
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
                 vocab: Vocabulary,
                 
                 #Change loss function here
                 lossmetric = torch.nn.MSELoss()) -> None:

        super().__init__(vocab)

        EMBEDDING_SIZE = 1024
        WORD_CLASSES = 100
        CONV_OUTPUT_SIZE = 50

        self.word_embeddings = word_embeddings

        self.word_class_probs1 = torch.nn.Linear(in_features = EMBEDDING_SIZE, out_features = WORD_CLASSES)
        self.word_class_probs2 = torch.nn.Linear(in_features = WORD_CLASSES, out_features = WORD_CLASSES)

        self.conv_filterbank1 = torch.nn.Conv1d(WORD_CLASSES,CONV_OUTPUT_SIZE,2,dilation=1,padding=1)
        self.conv_filterbank2 = torch.nn.Conv1d(WORD_CLASSES,CONV_OUTPUT_SIZE,2,dilation=2,padding=1)
        self.conv_filterbank3 = torch.nn.Conv1d(WORD_CLASSES,CONV_OUTPUT_SIZE,2,dilation=3,padding=2)
        self.conv_filterbank4 = torch.nn.Conv1d(WORD_CLASSES,CONV_OUTPUT_SIZE,2,dilation=4,padding=2)
        self.conv_filterbank5 = torch.nn.Conv1d(WORD_CLASSES,CONV_OUTPUT_SIZE,2,dilation=5,padding=3)
        self.pool1 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool2 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool3 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool4 = torch.nn.AdaptiveMaxPool1d(1)
        self.pool5 = torch.nn.AdaptiveMaxPool1d(1)

        self.recurrent_pool = torch.nn.LSTM(CONV_OUTPUT_SIZE, CONV_OUTPUT_SIZE, batch_first=True, bidirectional=True)

        
        self.hidden2tag = torch.nn.Linear(in_features=CONV_OUTPUT_SIZE*5,
                                          out_features=2)
        #Initializing accuracy, loss and softmax variables
        self.accuracy = BooleanAccuracy()
        self.loss = lossmetric
        self.evalmode = False

    def set_evalmode(self, mode: bool):
        self.evalmode = mode

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

        wordclass1 = torch.nn.functional.relu(self.word_class_probs1(embeddings))
        #wordclass2 = torch.nn.functional.relu(self.word_class_probs2(wordclass1))
        #wordclass3 = torch.nn.functional.relu(self.word_class_probs3(wordclass2))
        #wordclass4 = torch.nn.functional.relu(self.word_class_probs4(wordclass3))
        #wordclass5 = torch.nn.functional.relu(self.word_class_probs4(wordclass4))
        final_word_class = torch.sigmoid(self.word_class_probs2(wordclass1)).permute(0,2,1)
        #for certain debugging purposes
        #time.sleep(20)
        #print(embeddings.shape)

        #rather than the encoder we perform the set of convolutions



        cset_1 = self.conv_filterbank1(final_word_class).permute(0,2,1)
        cset_2 = self.conv_filterbank2(final_word_class).permute(0,2,1)
        cset_3 = self.conv_filterbank3(final_word_class).permute(0,2,1)
        cset_4 = self.conv_filterbank4(final_word_class).permute(0,2,1)
        cset_5 = self.conv_filterbank5(final_word_class).permute(0,2,1)
        pool_1 = torch.sum(self.recurrent_pool(cset_1)[1][0],dim=0).squeeze()
        pool_2 = torch.sum(self.recurrent_pool(cset_2)[1][0],dim=0).squeeze()
        pool_3 = torch.sum(self.recurrent_pool(cset_3)[1][0],dim=0).squeeze()
        pool_4 = torch.sum(self.recurrent_pool(cset_4)[1][0],dim=0).squeeze()
        pool_5 = torch.sum(self.recurrent_pool(cset_5)[1][0],dim=0).squeeze()

        hidden_representation = torch.cat((pool_1,pool_2,pool_3,pool_4,pool_5),dim=-1)
        #print(hidden_representation.shape)
        #hidden_representation = torch.cat((lin_comp_1,lin_comp_2,lin_comp_3,lin_comp_4,lin_comp_5),dim=1)
        #encoder_out = torch.nn.ReLU(hidden_representation)
        #the output from hidden2tag, a fully-connected linear layer converting the LSTM hidden state to 
        #the two labels
        lin_output = self.hidden2tag(hidden_representation)

        #output_score is a list of 2 variables which update the scores for social and agency class 
        #output_score = lin_output
        #output_score = self.sigmoid(output_score)
        
        output_score = torch.sigmoid(lin_output)
        if self.evalmode:
            self.os = hidden_representation
        output = {"score": output_score}
        #print(output_score.shape)
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
            output["loss"] = self.loss(output_score.squeeze(), labels.type(torch.cuda.FloatTensor).squeeze())
          
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset) }


class model_evaluator():
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        cuda = torch.device('cuda')

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
        self.reader = CLAFFDatasetReaderELMofromDataFrame()

        train_dataset = self.reader.read([False,train_df])
        validation_dataset = self.reader.read([False,test_df])
        self.vd = validation_dataset

        vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

        #word_embeddings = BasicTextFieldEmbedder({"character_ids": elmo})
        word_embeddings = ELMoTextFieldEmbedder({"character_ids": elmo})

        #initialize the model layers that we will want to change. 
        #lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
        ################## for dilated convolutions we will be replacing lstm with our custom layer ##################
        self.model= BigramDilatedConvModel(word_embeddings, vocab)

        self.model.cuda()

        #Set the optimizaer function here
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        #optimizer = optim.Adam(model.parameters(), lr=0.0001)
        move_optimizer_to_cuda(optimizer)

        # nice iterator functions are pretty much the only reason to stick with AllenNLP rn
        iterator = BucketIterator(batch_size=50, sorting_keys=[("sentence", "num_tokens")])

        iterator.index_with(vocab)

        self.trainer = Trainer(model=self.model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=validation_dataset,
                          patience=10,
                          num_epochs=500)
        self.iterator = iterator
        self.predictor = SentenceSeq2VecPredictor(self.model, dataset_reader=self.reader)
        self.trained = False



    def train(self):
        self.trainer.train()
        self.trained = True
        outputs = []
        labels = []
        self.model.set_evalmode(True)
        #for instance in self.vd:
        #    print("evaluating")
        #    self.model.forward_on_instance(instance)
        #    outputs.append(self.model.os.cpu().data.numpy())
        #    tensdc = instance.as_tensor_dict()
        #   labels.append([tensdc['agency'].cpu().data.numpy(), tensdc['social'].cpu().data.numpy()])
        #print(outputs)
        #print(labels)
        #outputs =np.vstack(outputs)
        #labels = np.vstack(labels)
        #o_social = np.round(outputs[:,0])
        #o_agency = np.round(outputs[:,1])
        #rs = outputs[:,0]
        #ra = outputs[:,1]
        #l_social = labels[:,1]
        #l_agency = labels[:,0]
        #f1_social=metrics.f1_score(l_social,o_social,pos_label=0)
        #f1_agency=metrics.f1_score(l_agency,o_agency,pos_label=0)  
        #auc_social=metrics.roc_auc_score(l_social,rs)
        #auc_agency=metrics.roc_auc_score(l_agency,ra)
        #acc_s = metrics.accuracy_score(l_social,o_social)  
        #acc_a = metrics.accuracy_score(l_agency,o_agency)  
        return [0,0,0,0,0,0]
        #get F1
        #get AUC


    def save_model(self):
        raise NotImplementedError

    def clean_str(self,string):
 
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

    def predict(self, index):
        print("predicting")
        token_indexers = {"character_ids": ELMoTokenCharactersIndexer()}
        test = []
        sentences = []
        hmid = []
        social = []
        agency = []
        tsne_y = []
        csv_output = open("test_results_"+ str(index) +".csv","w+")
        csv_output.write("hmid,sentence,agency,social\n")
        with open('csv/labeled_10k.csv',encoding="utf8", errors='ignore') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            header = next(readCSV)
            self.model.set_evalmode(True)
            print("DONE!!!")
            for row in readCSV:
                print("Processing hmid {}".format(row[0]))
                hmid.append(row[0])
                sentence = self.clean_str(row[1])
                sentences.append(sentence)
                social.append(row[4])
                agency.append(row[3])
                #rint(social)

                instance_in = self.reader.read([True,sentence])[0]
                #print(instance_in)
                self.model.forward_on_instance(instance_in)
                mop = self.model.os.cpu().data.numpy()
                #print("MOP:",mop)
                #print("MOP SHAPE:", mop.shape)
                test.append(mop)
                #print("TEST:",test)
                #print("TEST shape:")
                #print(print(np.asarray(test).shape))
                #print("tsne-y shape:")
                #print(tsne_y)
                #print(print(np.asarray(tsne_y).shape))
        print("Saving the test list")
        with open("test_list.txt", "wb") as fp:   #Pickling
            pickle.dump(test, fp)
        print("Saving the tsne list")
        with open("tsney_list.txt", "wb") as fp:   #Pickling
            pickle.dump(tsne_y, fp)

        social_only = ['yes','no']
        social_agency = ['yes','yes']
        agency_only = ['no','yes']
        not_any = ['no','no']
    
        for i in range(len(social)):
            if social[i] == social_agency[0] and agency[i] == social_agency[1]:
                tsne_y.append(0)
            elif social[i] == social_only[0] and agency[i] == social_only[1]:
                tsne_y.append(1)
            elif social[i] == agency_only[0] and agency[i] == agency_only[1]:
                tsne_y.append(2)
            elif social[i] == not_any[0] and agency[i] == not_any[1]:
                tsne_y.append(3)

        print("GETTING READY FOR TSNE")
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(test)
        #print("LENGTH:",len(tsne_y.target_names))
        print("TSNE DONE")
        target = np.array([0,1,2,3])
        target_ids = range(4)
        #print(target_ids)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c'
        for i, c, label in zip(target_ids, colors, target):
            plt.scatter(X_2d[tsne_y == i, 0], X_2d[tsne_y == i, 1], c=c, label=label)
        plt.legend()
        #i +=1
        #print(i)
        print("saving...")
        pylab.savefig('updateed_latest.png')
        plt.show()
                #csv_output.write(str(row[0])+","+str(sentence)+","+str(mop[1])+","+str(mop[0])+"\n")


    def batch_predict(self):
        raise NotImplementedError

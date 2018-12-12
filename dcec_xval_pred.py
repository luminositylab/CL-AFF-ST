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
import csv
#torch.manual_seed(1)

from sklearn import metrics

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
    def _read(self, df: pd.DataFrame) -> Iterator[Instance]:
        #skip line one, check if labeled set
        #firstline = next(f)
        #isLabeled = firstline.split(',')[2].strip('"') == 'concepts'
        #now, read in data
        #regex to get rid of non-alphanumeric
        #remover = re.compile('[\W_]+')
        #we only use this reader to read in the labeled 10k that gets cross val'd
        for line in df.rows:
            #sets = line.split(',')
            sentence = line['moment'].split()
            agency = line['agency']
            social = line['social']
            if str(agency) != 'no':
                agency = 'yes'
            if str(social) != 'no':
                social = 'yes'
            #out = [str(agency), str(social)]
            #yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)
            yield self.text_to_instance([Token(word) for word in sentence],str(agency), str(social))


class CLAFFDatasetReaderELMoTest(DatasetReader):
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
    def text_to_instance(self, tokens: List[Token], hmid:int = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}


        hmid_field = LabelField(label=hmid,skip_indexing=True)
        fields["hmid"] = hmid_field

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
                hmid = sets[0].strip('"')
                #out = [str(agency), str(social)]
                #yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)
                yield self.text_to_instance([Token(word) for word in sentence],int(hmid))


class CLAFFDatasetReaderTestELMofromDataFrame(DatasetReader):
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
    def text_to_instance(self, tokens: List[Token], hmid:int) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}


        hmid_field = LabelField(label=hmid,skip_indexing=True)
        fields["hmid"] = hmid_field


        return Instance(fields)

    #this is the outermost function and it gets automatically called at the reader.read() step in main
    #it yields the outputs of text_to_instance which produces instance objects containing the keyed values
    #for agency, social, and the sentence itself as an iterator of instances. This fxn depends on the dataset
    #in use.
    def _read(self, df: pd.DataFrame) -> Iterator[Instance]:
        #skip line one, check if labeled set
        #firstline = next(f)
        #isLabeled = firstline.split(',')[2].strip('"') == 'concepts'
        #now, read in data
        #regex to get rid of non-alphanumeric
        #remover = re.compile('[\W_]+')
        #we only use this reader to read in the labeled 10k that gets cross val'd
        for line in df.rows:
            #sets = line.split(',')
            sentence = line['moment'].split()
            hmid = line['hmid']
            if str(agency) != 'no':
                agency = 'yes'
            if str(social) != 'no':
                social = 'yes'
            #out = [str(agency), str(social)]
            #yield self.text_to_instance([Token(remover.sub('',word)) for word in sentence], agency, social)
            yield self.text_to_instance([Token(word) for word in sentence],int(hmid))



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
            self.os = output_score
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
        self.reader = CLAFFDatasetReaderELMo()

        train_dataset = self.reader.read(cached_path('csv/labeled_9k5.csv'))
        validation_dataset = self.reader.read(cached_path('csv/labeled_k5.csv'))
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
        optimizer = optim.Adam(self.model.parameters(), lr=0.5)
        #optimizer = optim.Adam(model.parameters(), l   r=0.0001)
        move_optimizer_to_cuda(optimizer)

        # nice iterator functions are pretty much the only reason to stick with AllenNLP rn
        iterator = BucketIterator(batch_size=50, sorting_keys=[("sentence", "num_tokens")])

        iterator.index_with(vocab)

        self.trainer = Trainer(model=self.model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=train_dataset,
                          validation_dataset=validation_dataset,
                          patience=1,
                          num_epochs=500)
        self.iterator = iterator
        #self.predictor = SentenceSeq2VecPredictor(self.model, dataset_reader=self.reader)
        self.trained = False



    def train(self):
        self.trainer.train()
        self.predictor = SentenceSeq2VecPredictor(self.model, dataset_reader=self.reader) 
        self.trained = True
        outputs = []
        labels = []
        self.model.set_evalmode(True)

        #now we want to read our entire test set in
        Treader = CLAFFDatasetReaderELMoTest()
        test_set = Treader.read(cached_path('csv/test_17k.csv'))
        for instance in test_set:
            print("evaluating")
            self.model.forward_on_instance(instance)
            outputs.append(self.model.os.cpu().data.numpy())
            tensdc = instance.as_tensor_dict()
            labels.append([tensdc['hmid'].cpu().data.numpy()])
        print(outputs)
        print(labels)
        outputs =np.vstack(outputs)
        labels = np.vstack(labels)
        #self.predictor = SentenceSeq2VecPredictor(self.model, dataset_reader=self.reader)  
        return [outputs, labels]
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
        test = []
        hmid = []
        self.predictor = SentenceSeq2VecPredictor(self.model, dataset_reader=self.reader) 
        with open('csv/test_17k.csv',encoding="utf8", errors='ignore') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            header = next(readCSV)
            for row in readCSV:
                hmid.append(row[0])
                test.append(row[1])

        for i in range(len(test)):
            test[i] =  self.clean_str(test[i])

        social_score = []
        agency_score = []
        social_tag = []
        agency_tag = []
        print(test)
        #print(self.predictor.predict)
        #Assign yes/no label based on the prediction
        for i in range(len(test)):
            print("Processing test datapoint {}...".format(test[i]))
            print("Number:", i)
            print(self.predictor.predict(test[i]))
        #social_score.append(predictor.predict(test[i])['score'][0])
        #agency_score.append(predictor.predict(test[i])['score'][1])

            if self.predictor.predict(test[i])['score'][0] < 0.5:
                social_tag.append("Yes")
            else:
                social_tag.append("No")

            if self.predictor.predict(test[i])['score'][1] < 0.5:
                agency_tag.append("Yes")
            else:
                agency_tag.append("No")

        #Make a dict for output
        d = {'hmid':hmid, 'Sentence':test,'Social':social_tag, 'Agency':agency_tag}
        df = pd.DataFrame(d)
        print(df)
        #Save the sentence, social prediction, agency prediction on the test set in a csv file 
        df.to_csv("test_results_"+index+".csv",sep=',', index=False)

    def batch_predict(self):
        raise NotImplementedError

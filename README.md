# CL-AFF-ST
Our codebase for our submission to the AffCon 2019 CL-AFF Shared Task. Still in progress.

## File Structure
`csv\` contains modified versions of the dataset files provided for the competition, we will not be reuploading the original dataset.

`scripts\` contains utility scripts such as our server quick setup script.

`lstm_baseline_2label.py` is a mutlilabel classification baseline implementation of a naive lstm taking in the default AllenNLP embeddings.

`lstm_elmo.py` is a baseline implementation of a naive lstm taking ELMo vectors as input and returning both labels as in `lstm_baseline_2label.py`.

`cl_aff_embedders.py` contains the custom TextFieldEmbedder that can embed ELMo vectors.

`simple_seq2vec.py` contains a custom Seq2VecPredictor to predict the social or agency value for a single sentence.

## Dependencies
Requires the AllenNLP library and Python 3.6 or above to run.

## Next Steps
Unsupervised tasks, continuing characterization of performance, application on other datasets

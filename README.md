# CL-AFF-ST
Our codebase for our submission to the AffCon 2019 CL-AFF Shared Task. Still in progress.

## File Structure
`csv` contains modified versions of the dataset files provided for the competition, we will not be reuploading the original dataset.
`lstm_baseline.py` is a baseline implementation of a naive lstm taking in the default AllenNLP embeddings.
`lstm_elmo.py` (in progress) is a baseline implementation of a naive lstm taking ELMO vectors as input.

## Dependencies
Requires the AllenNLP library and Python 3.6 or above to run.

## Next Steps
Implementing the dilated convolutions model.

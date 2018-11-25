import io
import tarfile
import zipfile
import bz2
import lzma
import gzip
import re
import logging
import warnings
import itertools
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple

from overrides import overrides
import numpy
import torch
from torch.nn.functional import embedding
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params, Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import get_file_extension, cached_path
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed

#elmo boilerplate
from allennlp.modules.elmo import Elmo, batch_to_ids

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("elmoemb")
class ELMoEmb(TokenEmbedder):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:
        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_params``
    Note that if you are using our data API and are trying to embed a
    :class:`~allennlp.data.fields.TextField`, you should use a
    :class:`~allennlp.modules.TextFieldEmbedder` instead of using this directly.
    Parameters
    ----------
    num_embeddings : int:
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    projection_dim : int, (optional, default=None)
        If given, we add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2):
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False):
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False):
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    Returns
    -------
    An Embedding module.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.output_dim = projection_dim or embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None
        print("Downloading the options file for ELMo...")
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        print("Downloading the weight file for ELMo...")
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        print("Done.")

        #pass in the vocabulary for pre-caching with the ELMo model
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)


    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        original_inputs = inputs
        if original_inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        #we need to process the input words into the vector
        #test 1: it should be the case that the character ids are already properly stored in a way that ELMo
        #can parse. In the tutorial, the instances are keyed as "elmo" rather than sentence.
        #2. convert words to character ids
        character_ids = batch_to_ids(sentences)
        #3. feed character ids into the elmo layer
        embedded = self.elmo(character_ids)
        if original_inputs.dim() > 2:
            view_args = list(original_inputs.size()) + [embedded.size(-1)]
            embedded = embedded.view(*view_args)
        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

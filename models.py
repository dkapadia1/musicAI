import torchaudio
import torch
from torchaudio import load
from audiocraft.models import CompressionModel
import pysdtw
from math import e
import torch.nn.functional as F
import demucs.api
class CUDAModel():
    model : CompressionModel = None
    def __init__(self):
      self.model = CompressionModel.get_pretrained('facebook/encodec_32khz', device = "cuda")
    def get_latent_decoding(self, audio_file_location, seconds = 5, start = 0, end = False, remove_voice = False):
        """
        Takes in an audio file location and returns the latent decoding of the audio file.

        Args:
            audio_file_location: The location of the audio file.

        Returns:
            The latent decoding of the audio file.
        """

        # Load the audio file
        audio, sr = load(audio_file_location)
        # Resample the audio to 32000 Hz
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
        audio = resampler(audio)
        #if remove_voice remove voice
        if remove_voice:
            separator = demucs.api.Separator()
            origin, separated = separator.separate_tensor(audio)
            audio = separated['drums'] + separated['bass'] + separated['other']
            del origin, separated
        # Trim the audio to the desired length

        if end:
            if audio.shape[1] > seconds * 32000:
                #print("we good")
                audio = audio[:, -1 * int(seconds * 32000):]
                #print(audio.shape)
                assert audio.shape[1] <= seconds * 32000, "its too long gangy"
            else:
                audio = audio[:, :int(seconds * 32000)]
        else:
            audio = audio[:, :int(seconds * 32000)]
        audio = audio[0].to('cuda')
        audio = audio[None].expand(1, -1, -1)

        # print(torch.cuda.memory_summary(), audio.shape)
        # Encode the audio
        encoding = self.model.encode(audio)
        del audio
        if encoding[1] is None:
            encoding = encoding[0]
            #print(f'{encoding.shape=}')

        tens = self.model.decode_latent(encoding)
        del encoding
        data = {}
        #print(f'{tens.shape=}')
        return tens
    def get_similarity(self, tens1, tens2, fun = 'lin'):
       if fun == "lin":
        return get_similarity_lin(tens1, tens2)
       elif fun == 'sdtw':
        return get_similarity_sdtw(tens1, tens2)
       else:
        raise NotImplementedError()
    def get_similarity_file(self,file1, file2, seconds = 5, start1 = 0, start2 = 0,tens1 = None, tens2 = None, fun = "lin", remove_voice : bool=  False):
        """
        Takes in two audio file locations and returns the cosine similarity between them.

        Args:
            file1: The first audio file location.
            file2: The second audio file location.
            fun: lin or sdtw, the similarity function
        Returns:
            The cosine similarity between the two audio files using similarity function
            """
            #get the latent decoding of the file
        if tens1 is None:
            tens1 = self.get_latent_decoding(file1, seconds = seconds, start = start1, remove_voice=remove_voice)
        if tens2 is None:
            tens2 = self.get_latent_decoding(file2, seconds = seconds, start = start2, remove_voice=remove_voice)
            #get the similarity
        return self.get_similarity(tens1, tens2, fun=fun)
    def convToPercent(self, tens1, tens2, distance, fun):
        if fun == 'sdtw':
            batches = int(min(tens1.size(2), tens2.size(2)) / 50)
            return sigmoid_transform(distance - (batches * 10))
        else:
            return sigmoid_transform(distance)
T = torch.Tensor
def cosine_distance(x: T, y: T) -> T:
    """Computes the pairwise cosine similarity matrix between x and y."""
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)

    # Normalize the vectors in x and y along the last dimension
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    # Expand x and y for pairwise comparisons
    x_exp = x_norm.unsqueeze(2).expand(-1, n, m, d)
    y_exp = y_norm.unsqueeze(1).expand(-1, n, m, d)

    # Compute pairwise cosine distance(1 - similarity)
    similarity = 1 - (x_exp * y_exp).sum(-1)

    return similarity

def get_similarity_lin(tens1, tens2):
    """
  Takes in two 3d tensors and returns the cosine similarity between them.

  Args:
      tens: The first tensor.
      tens2: The second tensor.

  Returns:
      The cosine similarity between the two tensors.
  """
  # Check if the tensors have the same shape
    if tens1.shape != tens2.shape:
        #assumes shape is [1, 128, x]. should never be different but you never know
        assert len(tens1.shape) == len(tens2.shape) and len(tens1.shape) == 3
        # Get the minimum dimension of the tensors
        min_dim = min(tens1.shape[2], tens2.shape[2])
        tens1 = tens1[:, :, :min_dim]
        tens2 = tens2[:, :, :min_dim]
        #sliceing should work because their is only a little amount of information added and removed through the use of larger files
    # Calculate the cosine similarity between the two tensors.
    similarity = torch.nn.functional.cosine_similarity(tens1, tens2, dim=1)

    return(similarity.sum(), tens1, tens2)
def get_similarity_sdtw(tens1, tens2):
    if tens1.shape[2] % 50 != 0:
        tens1 = tens1[:,:, :-1 * (tens1.shape[2] % 50)]
    if tens2.shape[2] % 50 != 0:
        #print(tens2.shape[2] % 50)
        tens2 = tens2[:,:, :-1 * (tens2.shape[2] % 50)]
    #make sure the lengths are the same, if not change them to the minimum
    if tens1.shape[2] != tens2.shape[2]:
        #print(tens1.shape, tens2.shape)
        if tens1.shape[2] > tens2.shape[2]:
            tens1 = tens1[...,:tens2.shape[2]]
        else:
            tens2 = tens2[..., :tens1.shape[2]]

        #get the similarity
    #print(tens1.shape, tens2.shape)
    sdtw = pysdtw.SoftDTW(gamma=1, use_cuda=True, dist_func = cosine_distance)
    perm_tens1 = tens1.permute(0, 2, 1).squeeze(0) 
    perm_tens2 = tens2.permute(0, 2, 1).squeeze(0) 
    split_tens1 = torch.split(perm_tens1, 50, dim=0)
    stacked_tens1 = torch.stack(split_tens1, dim=0)
    split_tens2 = torch.split(perm_tens2, 50, dim=0)
    stacked_tens2 = torch.stack(split_tens2, dim=0)
    del perm_tens1, perm_tens2, split_tens1, split_tens2
    return ((1 - sdtw(stacked_tens1, stacked_tens2)).sum()), tens1, tens2
def sigmoid_transform(x):
    """
    Sigmoid function
    """
    return 1 / (1 + e ** (-x / 100))
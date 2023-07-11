import numpy as np

from ..rnn_layers import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if self.cell_type == 'rnn':
          nn_forward = rnn_forward
          nn_backword = rnn_backward
        elif self.cell_type == 'lstm':
          nn_forward = lstm_forward
          nn_backword = lstm_backward
          
        h, a_cache = affine_forward(features, W_proj, b_proj)
        word, w_cache = word_embedding_forward(captions_in, W_embed)
        h, r_cache = nn_forward(word, h, Wx, Wh, b)
        h, t_cache = temporal_affine_forward(h, W_vocab, b_vocab)
        
        loss, dx = temporal_softmax_loss(h, captions_out, mask)
        
        dx, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dx, t_cache)
        dx, dh, grads['Wx'], grads['Wh'], grads['b'] = nn_backword(dx, r_cache)
        grads['W_embed'] = word_embedding_backward(dx, w_cache)
        dx, grads['W_proj'], grads['b_proj'] = affine_backward(dh, a_cache)
        pass
        
        # 순방향 패스에서는 다음을 수행해야 합니다:                   #
        # (1) Affine 변환을 사용하여 이미지 특징에서 초기 hidden state를 계산합니다. 이렇게 하면 (N, H) 형태의 배열이 생성됩니다.
        # (2) Word embedding 계층을 사용하여 captions_in의 단어를 인덱스에서 벡터로 변환하고, (N, T, W) 형태의 배열을 생성합니다.
        # (3) Vanilla RNN 또는 LSTM( self.cell_type에 따라 다름)을 사용하여 입력 단어 벡터의 시퀀스를 처리하고 모든 타임스텝에 대한 hidden state 벡터를 생성합니다. 이렇게 하면 (N, T, H) 형태의 배열이 생성됩니다.
        # (4) Hidden state를 사용하여 모든 타임스텝에서 어휘에 대한 점수를 계산하기 위해 (temporal) affine 변환을 사용합니다. 이렇게 하면 (N, T, V) 형태의 배열이 생성됩니다.
        # (5) (temporal) softmax를 사용하여 captions_out을 사용하여 손실을 계산하고, 위의 마스크를 사용하여 출력 단어가 <NULL>인 점을 무시합니다.
        #                                                                          
        # 가중치 또는 그들의 그래디언트를 정규화하는 것에 대해 걱정하지 마세요!          
        #                                                                          
        # 역방향 패스에서는 모든 모델 파라미터에 대한 손실의 그래디언트를 계산해야 합니다. 위에서 정의한 loss와 grads 변수를 사용하여 손실과 그래디언트를 저장하세요. grads[k]는 self.params[k]에 대한 그래디언트를 제공해야 합니다.
        #                                                                          
        # 또한 필요한 경우 layers.py에서 함수를 사용할 수 있음을 참고하세요.

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        if self.cell_type == 'rnn':
          nn_forward = rnn_step_forward
          nn_backword = rnn_step_backward
        elif self.cell_type == 'lstm':
          nn_forward = lstm_step_forward
          nn_backword = lstm_step_backward
        
        h_, a_cache = affine_forward(features, W_proj, b_proj)
        word = self._start * np.ones((N), dtype=np.int32)
        c_ = np.zeros_like(h_)
        for i in range(max_length):
          word, w_cache = word_embedding_forward(word, W_embed)
          if self.cell_type == 'rnn':
            h_, cache_ = rnn_step_forward(word, h_, Wx, Wh, b)
          elif self.cell_type == 'lstm':
            h_, c_, cache_ = lstm_step_forward(word, h_, c_, Wx, Wh, b)
          out, t_cache = affine_forward(h_, W_vocab, b_vocab)
          word = np.argmax(out,axis=1)
          captions[:,i] = word
        pass
        
        # TODO: 모델의 테스트 시간 샘플링을 구현하세요. 학습된 affine 변환을 입력 이미지 특징에 적용하여 RNN의 hidden state를 초기화해야 합니다. 
        # RNN에 공급하는 첫 번째 단어는 <START> 토큰이어야 합니다. 이 값은 self._start 변수에 저장되어 있습니다. 각 타임스텝에서 다음을 수행해야 합니다:
        # (1) 학습된 단어 임베딩을 사용하여 이전 단어를 임베딩합니다.
        # (2) 이전 hidden state와 임베딩된 현재 단어를 사용하여 RNN 단계를 수행하고 다음 hidden state를 얻습니다.
        # (3) 학습된 affine 변환을 다음 hidden state에 적용하여 어휘의 모든 단어에 대한 점수를 얻습니다.
        # (4) 점수가 가장 높은 단어를 다음 단어로 선택하고, 그것(단어 인덱스)을 캡션 변수의 적절한 슬롯에 작성합니다.
        # 
        # 간단하게 하기 위해, <END> 토큰이 샘플링된 후에 생성을 중지할 필요는 없지만 원한다면 그렇게 할 수 있습니다.
        # 
        # 힌트: rnn_forward 또는 lstm_forward 함수를 사용할 수 없습니다. 반복문에서 rnn_step_forward 또는 lstm_step_forward를 호출해야 합니다.
        # 
        # 참고: 이 함수에서도 여전히 미니배치를 처리하고 있습니다. 또한 LSTM을 사용하는 경우 첫 번째 cell state를 0으로 초기화하세요.

        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions

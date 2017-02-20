# nmtrain
by: Philip Arthur (philip.arthur30@gmail.com)

A Neural Machine Translation decoder toolkit using [chainer](http://chainer.org).
This toolkit is based on [Luong et al., 2015](http://www.aclweb.org/anthology/D15-1166) and [Arthur et al., 2016](https://aclweb.org/anthology/D/D16/D16-1162.pdf) model. 

## Installation
The installation is done by a running a simple command:

```python3 setup.py install```

To install nmtrain with GPU, please follow the instruction in chainer [webpage](https://github.com/pfnet/chainer#installation-with-cuda).

## Training translation model
To train a translation model, you need a parallel corpus in separate files with the same document length. We expect you to tokenize and lowercase your files before doing the training. Then, the simplest training procedure is to specify the source and target files + the output of the model.

```python3 nmtrain/train-nmtrain.py --src <source_file> --trg <target_file> --model_out <model_out>```

## Decoding
Given the training is finished you can use the saved model for decoding.
You need to specify the source file (not ~~```stdin```~~) with ```--src```.

```python3 nmtrain/nmtrain-decoder.py --src <input_file> --init_model <your_model_out>```

## Training Options
Bellow is the basic training options:
| Usage                | Option                                | Constraint          |
|---                   |---                                    |---                  |
| With GPU             | ```--gpu <gpu_num>            ```     | 0 <= int <= num_gpu |
| Custom batch size    | ```--batch <batch_size>       ```     | int > 0             |
| Custom embed size    | ```--embed <embed_size>       ```     | int > 0             |
| Custom hidden size   | ```--hidden <hidden_size>     ```     | int > 0             |
| Custom num of epoch  | ```--epoch <num_of_epoch>     ```     | int >= 0            |
| Custom LSTM layers   | ```--depth <lstm_layer_size>  ```     | int > 0             |
| Training seed        | ```--seed <some_positive_int> ```     | int >= 0            |
| Network Dropout      | ```--dropout <dropout_rate>   ```     | 0 <= float <= 1     |
| Truncated BPTT       | ```--bptt_len <bptt_len>      ```     | int > 0             |
| LR decay factor      | ```--sgd_lr_decay_factor <factor> ``` | 0 <= float <= 1     |
| LR decay after       | ```--sgd_lr_decay_after <epoch> ```   | int >= 0            |
| Gradient Clipping    | ```--gradient_clipping <size>```      | float > 0.0         |

Description:
- ```bptt_len``` is the number of decoding timestep before bptt (back propagation through time). At the end of the timestep bptt is performed once again.
- ```sgd_lr_decay_factor``` is a constant float that is multiplied to the learning rate, every time decay is called. This decay is triggered if development ppl declines.
- ```sgd_lr_decay_after``` is a constant integer that specified that LR is always decayed after that specified epoch.
- ```gradient_clippping``` normalize the gradient that is bigger than the specified amount.

## Early Stopping & Evaluation
You can use the development set to specify early stopping.
In order to do so, you need to specify both:
- ```--src_dev``` source development file
- ```--trg_dev``` target development file.

By doing this, the training procedure will automatically calculate perplexity at the end of every training epoch (iteration).
Nmtrain will keep track of the lowest development perplexity. 
If after ```early_stop``` iterations the lowest perplexity is not updated, then the training will conclude.

| Option             | Constraint          |
|---                 |---                  |
| ```--src_dev   ``` | PATH                |
| ```--trg_dev   ``` | PATH                |
| ```--early_stop``` | int > 0             |

## Unknown Word Training
Nmtrain will use a special token (<UNK>) to represent words that are excluded in the vocabulary during testing (both for source and target language). The system need to adapt to train this special tokens from the training corpus. Currently there are two ways to train them:

- Exclude rare words with low frequencies from the training sentence. All of these words will be replaced by <UNK>. This is done by specifying ```--unk_cut <unk_cut>``` during training, where words that have frequncy <= unk_cut will be replaced.
- Exclude rare words with low ranks (according to their frequencies). This is done by specifying the size of the vocabulary we want in the system (excpet all the special tokens used by the system). To specify the source and target vocabularies size, we can use the ```--src_max_vocab <src_max_vocab>``` and ```--trg_max_vocab <trg_max_vocab>``` options.

## Model Saving & Training from Middle
Nmtrain will only save the model with the best development perplexity (If development set is provided) specified by ```model_out```.

For some purposes, you might want to keep all the models, this can be done by passing the ```save_models``` flag. Nmtrain will add the suffix "-$EPOCH" at the ```model_out```, so your model will be saved incrementally.

You can also initialize the model with the nmtrain's trained model using ```--init_model``` to start the training from the middle.

| Option              | Constraint          |
|---                  |---                  |
| ```--save_models``` | flag                |
| ```--init_model```  | PATH                |

## Decoding Options

Bellow is the custom usage of decoding:

| Usage                             | Option                              | Constraint         |
|---                                |---                                  |---                 |
| Custom Beam Size                  | ```--beam <beam_size>```            | int >= 1           |
| Custom Word Penalty               | ```--word_penalty <word_penalty>``` | float >= 0.0       |
| Custom Generation Limit           | ```--gen_limit <gen_limit>```       | int >= 1           |

Description:
- ```beam``` is the width of the beam in the beam search. Beam search will conclude if there is no state that yields better probability compared to the worst state that ends with <EOS>.
- ```word_penalty``` is an exp(word_penalty) which is multiplied each time word is added to the sentence. Bigger for longer sentence.
- ```gen_limit``` controls the maximum length of the generated sentece. If the generation exceed the limit, the process is stoped immediately.

## Automatic Evaluation
We support BLEU evaluation at the moment. This is done by passing ```--ref <reference_file>``` during decoding time. Perplexity will also be calculated when reference is provided.

## Lexicons
We support the method of [Arthur et al., 2016](https://aclweb.org/anthology/D/D16/D16-1162.pdf) that uses lexicon to increase the accuracy of content word translation. This is done by specifying ```--lexicon <lexicon>``` during training time. 
- Format
The lexicon file must provide a lexical word translation probability, p(e|f) in the format of "trg src prob" for each line in the file. 
- Strength of Lexicon
The strength of the lexicon can be set by specifying ```--lexicon_alpha <lexicon_alpha>```, with values that closer to zero neglect the lexicon (Must be greater than zero!).

## Future Works
- Multiple GPU supports
- Memory optimization level
- Results on WMT dataset
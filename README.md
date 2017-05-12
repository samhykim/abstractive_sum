# Improving Abstractive Summarization Using LSTMs

Authors: Sam Kim, Sang Goo Kang

Paper: https://web.stanford.edu/class/cs224n/reports/2733282.pdf

## Abstract
Traditionally, summarization has been
approached through extractive methods.
However, they have produced limited results.
More recently, neural sequence-tosequence
models for abstractive text summarization
have shown more promise, although
the task still proves to be challenging.
In this paper, we explore current
state-of-the-art architectures and reimplement
them from scratch. We begin
with a basic sequence-to-sequence model.
We upgrade the model to use our own attention
mechanism and change the encoder
to a bi-directional LSTM. Finally, we introduce
and implement a hybrid pointergenerator
model that points back to words
in the source text if it encounters an outof-vocabulary
(OOV) word. Our qualitative
results show our models successfully
working by producing very similar results to
the ground truth summaries. The pointergenerator
model, it correctly replaces OOV
words with key words from the source, particularly
for numbers. Our models also do
not experiences short-comings from other
models such as repeating summaries.

### Example Summaries:

y*: president trump 's approval rating hit a new low 37 percent since he took office in January , according to the latest gallup poll .

y': trump 's approval rating hits new low

-----------------------------------------------------------------------------------------------------------------------------------------

y*: North Korea marked the final day of US Secretary of State Rex Tillerson's Asia visit Sunday by claiming to have tested a new type of rocket engine, underlining the country's defiance of recent calls for calm on the Korean Peninsula .

y': urgent north korea marks final day of us visit

-----------------------------------------------------------------------------------------------------------------------------------------

y*: southeast asian countries expressed confidence tuesday that their tsunami-hit tourism industries will return to normal , but pleaded with western governments not to warn citizens against travel to disaster-hit areas . in the aftermath of the dec . 26 indian ocean tsunami , many european governments , such as denmark , norway and sweden , issued advisories to their citizens not to travel to devastated areas in indonesia , india , sri lanka , thailand and the maldives .

y': southeast asian countries express confidence on tourism

### Requirements:
tensorflow 0.12.1

matplotlib

numpy

### Dataset:
Gigaword Data set or similarily structured data set

### Preprocessing Data:

Preprocess dataset to output first three sentences and headline of each article in corresponding files
`python utils/preprocessing.py -in "input path" -out "processed_path" -a`

Split into train/dev/test data
`python utils/split_dataset.py -in "processed_path" -out "data_processed" -dev DEVSIZE -test TESTSIZE`

### Configuration

Word embeddings: 128

Hidden states: 256

Batch size: 64

Dropout: 0.75

Num epochs: 10

Learning rate: 0.01

Encoder length: 200

Decoder length: 20

## Training:
`python seq2seq.py -c "checkpoint_path"`

Other flags:
* -l 0 (when running for the first time)
* -a (attention)
* -bi (bidirectional encoder)
* -p (pointer)
* -v (vocab size)

## Predicting:
`python seq2seq.py predict -c "checkpoint_path"`

Other flags:
* -l 0 (when running for the first time)
* -a (attention)
* -bi (bidirectional encoder)
* -p (pointer)
* -v (vocab size)

## Interactive:
`python seq2seq.py interactive -c "checkpoint_path"`

Other flags:
* -a (attention)
* -bi (bidirectional encoder)
* -p (pointer)
* -b (beam search)
* -v (vocab size)

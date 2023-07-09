#config.py
import transformers

#this is the maximum number of tokens in the sentence
MAX_LEN=512

#batch size is small because model is huge
TRAIN_BATCH_SIZE=8
VALID_BATCH_SIZE=4

#number of epochs
EPOCHS=10


BASE_PATH="C:\\Users\\vivek\\Desktop\\tfc"
#define the model for bert files
BERT_PATH=BASE_PATH+"\\input\\bbc"

#trianing file
TRAINING_FILE=BASE_PATH+"\\input\\imdb.csv"

TOKENIZER=transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=True)

MODEL_PATH=BASE_PATH+"\\mt"


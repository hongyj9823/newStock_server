from sklearn.model_selection import train_test_split
import transformers
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from sklearn import preprocessing
from transformers import AutoTokenizer, TFGPT2Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


wholeData = pd.read_csv("stockName_date_upDown_articleTitle_info.csv", encoding = 'UTF-8')

def drop_na_and_duplciates(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.drop(df[(df['UpDown'] < 1000) & (df['UpDown'] > -1000)].index)
    df = df.reset_index(drop=True)
    return df

wholeData = wholeData[['StockName', 'ArticleTitle', 'UpDown']]
wholeData = drop_na_and_duplciates(wholeData)


data = wholeData[['ArticleTitle', 'StockName']]
result = wholeData[['UpDown']]

result[result['UpDown'] > 0] = 1
result[result['UpDown'] < 0] = 0

result = result['UpDown'].astype('int')


train_data, test_data, train_target, test_target = train_test_split(data, result, test_size = 0.01, random_state = 42)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='<s>', eos_token='</s>', pad_token='<pad>')

#train_target = np.asarray(train_target).astype('int').reshape((-1, 1))
#test_target = np.asarray(test_target).astype('int').reshape((-1, 1))

max_seq_len = 128

def convert_examples_to_features(stockName, articleTitle, max_seq_len, tokenizer):

    input_ids = []

    for sent1, sent2 in tqdm(zip(stockName, articleTitle), total=len(stockName)):
        bos_token = [tokenizer.bos_token]
        eos_token = [tokenizer.eos_token]
        sent1_tokens = bos_token + tokenizer.tokenize(sent1) + eos_token
        sent2_tokens = bos_token + tokenizer.tokenize(sent2) + eos_token + ['<unused0>']
        tokens = sent1_tokens + sent2_tokens
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_id = pad_sequences([input_id], maxlen=max_seq_len, value=tokenizer.pad_token_id, padding='post')[0]

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        input_ids.append(input_id)

    input_ids = np.array(input_ids, dtype=int)

    return input_ids

X_train = convert_examples_to_features(train_data['StockName'], train_data['ArticleTitle'], max_seq_len=max_seq_len, tokenizer=tokenizer)
X_test = convert_examples_to_features(test_data['StockName'], test_data['ArticleTitle'], max_seq_len=max_seq_len, tokenizer=tokenizer)



model = TFGPT2Model.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

max_seq_len = 128

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
outputs = model([input_ids_layer])



class TFGPT2ForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(TFGPT2ForSequenceClassification, self).__init__()
        self.gpt = TFGPT2Model.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='softmax',
                                                name='classifier')

    def call(self, inputs):
        outputs = self.gpt(input_ids=inputs)
        cls_token = outputs[0][:, -1]
        prediction = self.classifier(cls_token)

        return prediction

gpus = tf.config.list_physical_devices('GPU')

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print('multi')
elif len(gpus) == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')
    print('gpu')
else:
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
    print('cpu')

with strategy.scope():
    model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])


model.fit(X_train, train_target, epochs=3, batch_size=32)

results = model.evaluate(X_test, test_target, batch_size=1024)
print("test loss, test acc: ", results)



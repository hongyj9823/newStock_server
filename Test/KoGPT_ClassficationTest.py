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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 훈련 데이터 다운로드
'''
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/multinli.train.ko.tsv", filename="multinli.train.ko.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/snli_1.0_train.ko.tsv", filename="snli_1.0_train.ko.tsv")

# 검증 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.dev.ko.tsv", filename="xnli.dev.ko.tsv")

# 테스트 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.test.ko.tsv", filename="xnli.test.ko.tsv")
'''

train_snli = pd.read_csv("snli_1.0_train.ko.tsv", sep='\t', quoting=3)
train_xnli = pd.read_csv("multinli.train.ko.tsv", sep='\t', quoting=3)
val_data = pd.read_csv("xnli.dev.ko.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("xnli.test.ko.tsv", sep='\t', quoting=3)

# 결합 후 섞기
train_data = train_snli.append(train_xnli)
train_data = train_data.sample(frac=1)

def drop_na_and_duplciates(df):
  df = df.dropna()
  df = df.drop_duplicates()
  df = df.reset_index(drop=True)
  return df

# 결측값 및 중복 샘플 제거
train_data = drop_na_and_duplciates(train_data)
val_data = drop_na_and_duplciates(val_data)
test_data = drop_na_and_duplciates(test_data)

tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='<s>', eos_token='</s>', pad_token='<pad>')

'''
print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

tokenizer.decode(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))

for elem in tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"):
  print(tokenizer.decode(elem))

print(tokenizer.tokenize("전율을 일으키는 영화. 다시 보고싶은 영화"))

print(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))

for elem in tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"):
  print(tokenizer.decode(elem))

for elem in tokenizer.encode("happy birthday~!"):
  print(tokenizer.decode(elem))

print(tokenizer.decode(0))
print(tokenizer.decode(1))
print(tokenizer.decode(2))
print(tokenizer.decode(3))
print(tokenizer.decode(4))
'''

max_seq_len = 128

encoded_result = tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화", max_length=max_seq_len, pad_to_max_length=True)
print(encoded_result)
print('길이 :', len(encoded_result))

def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):

    input_ids = []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
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

X_train = convert_examples_to_features(train_data['sentence1'], train_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# 최대 길이: 128
input_id = X_train[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))

X_val = convert_examples_to_features(val_data['sentence1'], val_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# 최대 길이: 128
input_id = X_val[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))

X_test = convert_examples_to_features(test_data['sentence1'], test_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)

train_label = train_data['gold_label'].tolist()
print(train_data.head)
print(train_data['gold_label'])
val_label = val_data['gold_label'].tolist()
test_label = test_data['gold_label'].tolist()

idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(train_label)

y_train = idx_encode.transform(train_label) # 주어진 고유한 정수로 변환
y_val = idx_encode.transform(val_label) # 고유한 정수로 변환
y_test = idx_encode.transform(test_label) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
idx_label = {value: key for key, value in label_idx.items()}
print(label_idx)
print(idx_label)

model = TFGPT2Model.from_pretrained('skt/kogpt2-base-v2', from_pt=True)

max_seq_len = 128

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
outputs = model([input_ids_layer])

print(outputs)
print(outputs[0])
print(outputs[1])
print(outputs[0][:, -1])

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

"""TPU 사용법 : https://wikidocs.net/119990"""

# TPU 작동을 위한 코드
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
  model = TFGPT2ForSequenceClassification("skt/kogpt2-base-v2", num_labels=3)
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

early_stopping = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.001,
    patience=2)

model.fit(
    X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val),
    callbacks = [early_stopping]
)

results = model.evaluate(X_test, y_test, batch_size=1024)
print("test loss, test acc: ", results)


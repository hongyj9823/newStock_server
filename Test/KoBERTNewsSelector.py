import transformers
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력

print('테스트용 리뷰 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력

train_data[:5] # 상위 5개 출력

test_data[:5] # 상위 5개 출력

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
train_data = train_data.reset_index(drop=True)
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

test_data = test_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
test_data = test_data.reset_index(drop=True)
print(test_data.isnull().values.any()) # Null 값이 존재하는지 확인

print(len(train_data))

print(len(test_data))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

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

print(tokenizer.decode(101))

print(tokenizer.decode(102))

print(tokenizer.cls_token, ':', tokenizer.cls_token_id)
print(tokenizer.sep_token, ':' , tokenizer.sep_token_id)

print(tokenizer.pad_token, ':', tokenizer.pad_token_id)

max_seq_len = 128

encoded_result = tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화", max_length=max_seq_len, pad_to_max_length=True)
print(encoded_result)
print('길이 :', len(encoded_result))

# 세그멘트 인풋
print([0]*max_seq_len)

# 마스크 인풋
valid_num = len(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))
print(valid_num * [1] + (max_seq_len - valid_num) * [0])

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):
    
    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
    
    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        # input_id는 워드 임베딩을 위한 문장의 정수 인코딩
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)
       
        # attention_mask는 실제 단어가 위치하면 1, 패딩의 위치에는 0인 시퀀스.
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        
        # token_type_id는 세그먼트 임베딩을 위한 것으로 이번 예제는 문장이 1개이므로 전부 0으로 통일.
        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels

train_X, train_y = convert_examples_to_features(train_data['document'], train_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)

test_X, test_y = convert_examples_to_features(test_data['document'], test_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# 최대 길이: 128
input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))
print('레이블 :',label)

model = TFBertModel.from_pretrained("bert-base-multilingual-cased")

max_seq_len = 128

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
attention_masks_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
token_type_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)

outputs = model([input_ids_layer, attention_masks_layer, token_type_ids_layer])

print(outputs)

print(outputs[0])

print(outputs[1])

class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
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
    model = TFBertForSequenceClassification("klue/bert-base")
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

model.fit(train_X, train_y, epochs=2, batch_size=64, validation_split=0.2)

results = model.evaluate(test_X, test_y, batch_size=1024)
print("test loss, test acc: ", results)

def sentiment_predict(new_sentence):
  input_id = tokenizer.encode(new_sentence, max_length=max_seq_len, pad_to_max_length=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]
  print(score)

  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))

sentiment_predict("보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ")

sentiment_predict("스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다")

sentiment_predict("별 똥같은 영화를 다 보네. 개별로입니다.")

sentiment_predict("이 영화 존잼입니다 대박.")

sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')

sentiment_predict('이딴게 영화냐 ㅉㅉ')

sentiment_predict('감독 뭐하는 놈이냐?')

sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')
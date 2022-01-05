import numpy as np
import pandas as pd
from numpy import genfromtxt
from underthesea import sent_tokenize
from underthesea import word_tokenize
import matplotlib.pyplot as plt
import re

df = pd.read_excel('/content/gdrive/MyDrive/đồ án/dữ liệu mới_v1.1.xlsx')
reviews = df['Review']
x0 = reviews[0]
labels = df['label']

#tiền xử lý dữ liệu 
def text_preprocess(sentence):
    sentence = word_tokenize(sentence, format="text")
    sentence = re.sub(r"[-()\"#/@;:<>{}`+=~|*'.!?,.]", "", sentence)
    sentence = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]',' ',sentence)
    sentence.lower()
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence
review = []
for sentence in reviews:
    review.append(text_preprocess(sentence))
review = [i.lower() for i in review]

#split data 
from sklearn.model_selection import train_test_split
x, x_test, y, y_test = train_test_split(new_review, labels, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

#vector hóa dữ liệu cho x:
from gensim.models import Word2Vec
x_train1 = []
for review in x_train:
    x_train1.append(review.split())
x_val1 = []
for review in x_val:
    x_val1.append(review.split())
x_test1 = []
for review in x_test:
    x_test1.append(review.split())
model = Word2Vec(input_gensim, size=150, window=5, min_count=0, workers=4, sg=1, negative=5)
word_vectors = model.wv
word2index = {token: token_index for token_index, token in enumerate(word_vectors.index2word)} 
sentence_vec1 = []
x_trainn = []
for sentence in x_train1:    
    sentence_vec1 = [word2index[word] for word in sentence]
    x_trainn.append(sentence_vec1)
sentence_vec2 = []
x_vall = []
for sentence in x_val1:    
    sentence_vec2 = [word2index[word] for word in sentence]
    x_vall.append(sentence_vec2)
sentence_vec3 = []
x_testt = []
for sentence in x_test1:    
    sentence_vec3 = [word2index[word] for word in sentence]
    x_testt.append(sentence_vec3)

from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = 350
x_trained = pad_sequences(x_trainn, maxlen=max_length, padding='post')
x_valed = pad_sequences(x_vall, maxlen=max_length, padding='post')
x_tested = pad_sequences(x_testt, maxlen=max_length, padding='post')
data = pad_sequences(word_vec, maxlen=max_length, padding='post')
words = list(model.wv.vocab)

MAX_NB_WORDS = len(word_vectors.vocab)
word_index = {t[0]: i+1 for i,t in enumerate(words)}
WV_DIM = 150
MAX_NB_WORDS = len(word_vectors.vocab)
nb_words = min(MAX_NB_WORDS, len(word_vectors.vocab))
# we initialize the matrix with random numbers
wv_matrix = (np.random.rand(nb_words, WV_DIM) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass 
#vector hóa dữ liệu cho y:
from sklearn.preprocessing import OneHotEncoder

y_onehot1onehot1 = OneHotEncoder(sparse=False)
y_train1 = np.asarray(y_train)
y_train2 = y_train1.reshape((-1, 1))
y_onehot_encoded = y_onehot1onehot1.fit_transform(y_train2)

y_onehot = OneHotEncoder(sparse=False)
y_val1 = np.asarray(y_val)
y_val2 = y_val1.reshape((-1, 1))
y_val3 = y_onehot.fit_transform(y_val2)

y_onehot = OneHotEncoder(sparse=False)
y_val4 = np.asarray(y_test)
y_val5 = y_val4.reshape((-1, 1))
y_val6 = y_onehot.fit_transform(y_val5)

#xây dựng model;
from keras.layers import Input, Flatten
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Dense
sequence_input = Input(shape=(350,), dtype='int32')
embedded_sequences = Embedding(nb_words, output_dim=150,
                               weights=[wv_matrix], input_length=350,
                               trainable=True)(sequence_input)
l_cov1 = Conv1D(200, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)

l_flat = Flatten()(l_pool1)
l_dense1 = Dense(200, activation='relu')(l_flat)
dropout1 = Dropout(rate=0.5)(l_dense1)
l_dense2 = Dense(128, activation='tanh')(dropout1)
dropout2 = Dropout(rate=0.5)(l_dense2)
preds = Dense(3, activation='softmax')(dropout2)
predict = Model(sequence_input, preds)
predict.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), 
              metrics=['acc'])
print("model fitting - simplified convolutional neural network")
predict.summary()

#train model:
hist1 = predict.fit(x_trained, y_onehot_encoded, validation_data=(x_valed, y_val3), batch_size=256, epochs=18)

#test model;
y_pred = predict.predict(x_tested)
y_pred1 = np.argmax(y_pred, axis=1)#làm tròn
y_test1 = np.argmax(y_val6, axis=1)

#visualization kết quả:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_score, classification_report
cm = confusion_matrix(y_test1, y_pred1)
import seaborn as sns
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['positive','neutral','negative'], 
                     columns = ['positive','neutral','negative'])

plt.figure(figsize=(15,10))
sns.heatmap(cm_df, annot=True)
plt.title('Sentiment Analysis \nAccuracy:{0:.3f}'.format(acc))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

history = pd.DataFrame(hist1.history)
plt.figure(figsize=(12,12))
plt.plot(history["loss"], label='Loss')
plt.plot(history["val_loss"], label = 'Val_loss')
plt.title("Loss with pretrained word vectors")
plt.legend()
plt.show()

history = pd.DataFrame(hist1.history)
plt.figure(figsize=(12,12))
plt.plot(history["acc"], label = 'Acc')
plt.plot(history["val_acc"], label='Val_acc')
plt.title("Acc with pretrained word vectors")
plt.legend()
plt.show()
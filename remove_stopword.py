import numpy as np
import pandas as pd
from numpy import genfromtxt
from underthesea import sent_tokenize
from underthesea import word_tokenize
import matplotlib.pyplot as plt
import re
from openpyxl import load_workbook

df = pd.read_excel('neutral data từ sentiment analysis.xlsx')
reviews = df['Review']
x0 = reviews[0]
labels = df['Label']
review = []

#tiền xử lý dữ liệu 
def text_preprocess(sentence):
    sentence = word_tokenize(sentence, format="text")
    sentence = re.sub(r"[-()\"#/@;:<>{}`+=~|*'.!?,.]", "", sentence)
    sentence = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]',' ',sentence)
    sentence.lower()
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

for sentence in reviews:
    review.append(text_preprocess(sentence))
review = [i.lower() for i in review]
stopword = ["ăn", "nhưng", "mình", "có", "là", "thì", "và", "cũng", "mà", "ở", "nên", "lại", "thấy", "đây", "đi", "như", "ra", "cho", "với", "đến", "gọi", "còn", "cái", "lần", "phải", "vụ", "các", "của", "để", "chỉ", "mới", "luôn", "gì", "vì", "một", "rồi", "sẽ", "nào", "nữa", "đã", "lúc", "làm", "chỗ", "khi", "bị", "về", "đó", "mấy", "hay", "vẫn", "từ", "thôi", "sau", "hôm", "bên", "chung", "chắc", "lượng", "giờ", "kiểu", "xong", "thể", "chứ", "thế", "độ", "bảo", "loại", "ngoài", "vậy", "sự", "sao"]

#remove stopword
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)

new_review = []
for line in review:
    new_review.append(remove_stopwords(line))
df['Review'] = new_review
#lưu lại data vào file excel mới 
writer = pd.ExcelWriter('neutral data từ sentiment analysis.xlsx') 
df.to_excel(writer, "Main_data", columns=['Review', 'Label'])
writer.save()
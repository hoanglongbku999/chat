# -*- coding: utf-8 -*-
from flask import Flask, render_template,request
import pickle
import os, string, re
import json
import numpy as np
import random,re
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import jsonify

from pyvi import ViTokenizer, ViPosTagger

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json',encoding='utf-8').read()
intents = json.loads(data_file)


svm_model = "classifier.pk"
with open(svm_model, 'rb') as f:
    classifier = pickle.load(f)

# Các bước tiền xử lý dữ liệu tiếng Việt
# Xóa các icon trong văn bản
def deleteIcon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s)
    return s.strip()

def clean_doc(doc):
    # xóa tất cả dấu câu (!,?..) trong câu
    for punc in string.punctuation:
        doc = doc.replace(punc,' ')
    doc = deleteIcon(doc)
    # Đưa về chữ thường
    doc = doc.lower()
    # Xóa nhiều khoảng trắng
    doc = re.sub('\\s+',' ',doc)
    return doc




def ngram_featue(text,N):
    sentence = text.split(" ")
    grams = [sentence[i:i+N] for i in range(len(sentence)-N+1)]
    result = []
    for gram in grams:
        result.append(" ".join(gram))
    return result
def get_Word_based_POS(text):
    #text = ViTokenizer.tokenize(text)
    tag_pos = ViPosTagger.postagging(text)
    #print(tag_pos)
    vocab = tag_pos[0]
    list_pos = tag_pos[1]
    result = []
    for index,pos in enumerate(list_pos):
        if "N" in pos or "V" in pos or "A" in pos:
            result.append(vocab[index])
    #print(result)
    return result
def get_POS_feature(text):
    tag_pos = ViPosTagger.postagging(text)
    vocab = tag_pos[0]
    list_pos = tag_pos[1]
    result = []
    for index,pos in enumerate(list_pos):
        result.append(pos)
    return result
def extract_feature(text_preproced):
    feature = ngram_featue(text_preproced,2) + ngram_featue(text_preproced,3) + ngram_featue(text_preproced,4)
    feature += get_Word_based_POS(text_preproced) + get_POS_feature(text_preproced)
    return feature


x_train = []

for intent in intents['intents']:
    for pattern in intent['question']:
        #tiền xử lý các câu và đưa vào danh sách các câu
        pattern = clean_doc(pattern)
        #tách từ trong mỗi câu question và đưa vào danh sách các câu
        w = ViTokenizer.tokenize(pattern)
        w_feature = extract_feature(w)
        x_train.append(w_feature)


# Huấn luyện mô hình Linear SVM để phân loại câu hỏi ra tag
vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
X_train_tfidf = vectorizer.fit_transform(x_train)


# Hàm lấy phản hồi dựa trên tag.
def getRespond(tag):
    data_file = open('data.json', encoding='utf-8').read()
    intents_json = json.loads(data_file)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['answer'])
            break
    return result
import glob
def get_response_image(tag):
    data_file = open('data_image.json', encoding='utf-8').read()
    intents_json = json.loads(data_file)
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['answer'])
            break
    if result == "":
        return "None"
    else:
        return result

@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/analysis/", methods=['POST','GET'])
def classify_text():
    text = request.args.get('query')
    s = clean_doc(text)
    s = ViTokenizer.tokenize(s)
    input_feature = extract_feature(s)
    input_feature_tfidf = vectorizer.transform([input_feature])
    intent_input = classifier.predict(input_feature_tfidf)[0]
    response  = getRespond(intent_input)
    response_image = get_response_image(intent_input)
    r = {'query': text.encode('utf8').decode('utf8'), 'response':response.encode('utf8').decode('utf8'), 'path_image': response_image.encode('utf8').decode('utf8')}
    return jsonify(r)
    #return render_template("Result.html", data = [{"query":text, "response" : response}])

if __name__ == "__main__":
    app.run(debug=True)

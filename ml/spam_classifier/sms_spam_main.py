from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import nltk
import pickle


data_type = {
    1:'Spam',
    0:'Not Spam',
}

SpamModel = pickle.load(open(r'ml\spam_classifier\model.pkl','rb'))
Vectorizer = pickle.load(open(r'ml\spam_classifier\vectorizer.pkl','rb'))

def transform_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)
    ps = PorterStemmer()
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

def sms_spam_predict(text,SpamModel,Vectorizer):

    text = transform_text(text)
    input_text = Vectorizer.transform([text])
    res = SpamModel.predict(input_text)[0]
    results = data_type[res]

    return results

# text = "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
# print(sms_spam_pridector(text,SpamModel,Vectorizer))
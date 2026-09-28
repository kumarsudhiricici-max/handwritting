import os,joblib,numpy as np
from features import extract_features

def load(model_path):
    return joblib.load(model_path)

def predict(image,model):
    f,_=extract_features(image)
    score=float(model.predict_proba(f.reshape(1,-1))[0,1])
    pred=int(score>=0.5)
    return pred,score,f

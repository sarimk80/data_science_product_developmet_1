from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer
import numpy as np
import scipy as sc

app = FastAPI()



@app.get("/")
def read_root():
    return {"Hello": "World"}


class RequestModel(BaseModel):
    text:str

class ResponseModel(BaseModel):
    text:str


@app.post("/detect_emotion_binary",response_model=ResponseModel)
def detect_emotion_binary(request:RequestModel):
    tokenizer = AutoTokenizer.from_pretrained("sarim/myModel")
    model = AutoModel.from_pretrained("sarim/myModel",num_labels=27)
    prepared_input = tokenizer.prepare_seq2seq_batch([request.text], return_tensors='pt')
    model = model.to('cpu')
    model.eval()
    model_output = model(**prepared_input,return_dict=True)
    prediction = np.argmax(model_output.logits[0].detach().numpy())
    #index_to_labels[prediction]
    print(prediction)
    return ResponseModel(
        text="prediction.str()"
    )

@app.post("/detect_emotion_full",response_model=ResponseModel)
def detect_emotion_full(request:RequestModel):
    tokenizer = AutoTokenizer.from_pretrained("sarim/myModel")
    model = AutoModel.from_pretrained("sarim/myModel",num_labels=27)
    prepared_input = tokenizer.prepare_seq2seq_batch([request.text], return_tensors='pt')
    model = model.to('cpu')
    model.eval()
    model_output = model(**prepared_input,return_dict=True)
    prediction = np.argmax(model_output.logits[0].detach().numpy())
    return ResponseModel(
        text="negative"
    )


import flwr as fl
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import tensorflow as tf
import model
import functions 
import constantes
from sklearn.metrics import classification_report
# Define the Flower client class
class AutoencoderClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train, labels):
        self.cid = cid
        self.net = net
        self.train = train
        self.labels = labels

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
      self.net.set_parameters(parameters)
      self.net.train(self.train,self.labels, epochs=constantes.EPOCHS_CLIENT)
      return self.net.get_parameters(), len(self.train), {}


    def evaluate(self, parameters, config):
        # Return the evaluation metrics as a tuple
        return 1, 1, {"mse": 1}


net = model.AutoencoderWithClassifier(18,isServer=False, vae=constantes.VAE)
# Start Flower client setting its associated data partition

data=pd.read_csv("data.csv")  
fake_labels=pd.read_csv("labels.csv")  


fl.client.start_client(
        server_address='192.168.53.247:8080',
        client=AutoencoderClient(
          0,net,data, fake_labels
        ).to_client(),
    )

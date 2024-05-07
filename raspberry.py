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
    def __init__(self, cid, net, train, labels, test):
        self.cid = cid
        self.net = net
        self.train = train
        self.test = test
        self.labels = labels

    def get_parameters(self, config):
        return self.net.get_parameters()

    def fit(self, parameters, config):
      self.net.set_parameters(parameters)
      self.net.train(self.train,self.labels, epochs=constantes.EPOCHS_CLIENT)
      return self.net.get_parameters(), len(self.train), {}


    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        test_features = self.test
        test_features = np.array(test_features)
        _, reconstructed_data = self.net.call(test_features)

        mse = float(
            np.mean(np.square(reconstructed_data - test_features))
        )  # Convert to float

        num_samples = len(self.test)  # Replace with the actual number of test samples

        # Return the evaluation metrics as a tuple
        return mse, num_samples, {"mse": mse}


def get_client_fn(inputdim,dataset, fake_labels,test):

    def create_client(client_id):
        client_id = int(client_id)
       
        tf_client = AutoencoderClient(client_id,net,dataset[client_id], fake_labels[client_id], test).to_client()
        return tf_client

    return create_client

net = model.AutoencoderWithClassifier(18,isServer=False, vae=constantes.VAE)
# Start Flower client setting its associated data partition

data=pd.read_csv("data.csv")  
fake_labels=pd.read_csv("labels.csv")  
test=pd.read_csv("test.csv")  

fl.client.start_client(
        server_address='192.168.53.247:8080',
        client=AutoencoderClient(
          0,net,data, fake_labels, test
        ).to_client(),
    )

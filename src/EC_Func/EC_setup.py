import numpy as np
import torch
from src.EC_Func.EC_Classifier_Ball_Tree import EpistemicClassifier

class EC():
    def __init__(self, cfg, train_utils):
        self.cfg = cfg
        self.net = train_utils.net.to("cpu")
        self.layer_interest = cfg['EC']['layer_interest']
        self.metric = cfg['EC']['metric']
        self.p = cfg['EC']['p']
        self.distance = cfg['EC']['distance']
        self.EC_model = EpistemicClassifier(self.net, self.layer_interest, self.metric, self.p)
        self.train_dataloaders = train_utils.train_dataloaders
        self.test_dataloaders = train_utils.test_dataloaders
        # print(self.net)


    def EC_fit(self):
        self.net.eval()

        # the x, y below are just for test, you can use self.train_dataloaders and self.test_dataloaders to access your dataset set input for EC-classifier,
        # you may want to put data to gpu, I've set self.net to cpu in __init__, you may also need to modify it.
        self.h, self.w = self.cfg['data']['image_size']['h'], self.cfg['data']['image_size']['w']
        x = torch.rand(5,3,self.h, self.w)
        y = [1,1,1,1,1]
        x_text = torch.rand(4,3,self.h, self.w)
        y_test = [1,1,1,1]



        # print("start_fit")
        self.EC_model = self.EC_model
        self.EC_model.fit(x, y) # EC.fit(X_train, y_train_int)
        # print("end_fit")

        # pred = self.EC_model.predict_class(x_text, n_neigh=2)
        pred = self.EC_model.predict_class(x_text, dist=self.distance)
        print("pred:", pred) # IMK: max+2, IDN: max+1,
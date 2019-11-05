from devnet_kdd19 import load_model_weight_predict
from utils import aucPerformance, dataLoading

model_path = 'path_of_your_h5_model'
network_depth = 2  # default is 2, if you changed it while training, you should alter it
input_shape = [29]  # [29] is the input shape of `credit card fraud` dataset, if you use other dataset, it may be diff

x_test, y_test = dataLoading('dataset/creditcard_test.npy')

scores = load_model_weight_predict(model_path,
                                   input_shape=input_shape,
                                   network_depth=network_depth,
                                   x_test=x_test)

AUC_ROC, AUC_PR = aucPerformance(scores, y_test)

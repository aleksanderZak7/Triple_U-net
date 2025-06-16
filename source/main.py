import time
import numpy as np

import Config
from utils import my_print
from cross_val import CrossValidation
from train_test import train_model, test_model


if __name__ == "__main__":
    epoches: int = 100
    train: bool = True
    cross_val: bool = False

    conf = Config.config()
    conf.train = train
    conf.epoches = epoches
    
    start = time.time()
    
    if conf.train:
        my_print('data_path:        {}'.format(conf.train_data_path))
        my_print('label_path:       {}'.format(conf.label_path))
        my_print('valid_data_path:  {}'.format(conf.valid_data_path))
        my_print('epoches num:      {}'.format(conf.epoches))

        if cross_val:
            scores = CrossValidation(cv=4, conf=conf)
            scores.run()
            
            scores.cleanup()
            my_print(scores)
            my_print(f"DICE mean: {np.mean(scores.DICE_):.4f}\nDICE std: {np.std(scores.DICE_):.4f}")
        else:
            my_activation = train_model(conf)
            my_print('Total epoches ({})'.format(conf.epoches))
            my_activation.training()
    else:
        my_print('test_data_path: {}'.format(conf.test_data_path))
        my_activation = test_model(conf)
        my_activation.test()
    
    end = time.time()
    my_print('Running time:{}'.format(str(end-start)))
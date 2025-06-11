import time

import Config
from utils import my_print
from train_test import train_model, test_model


if __name__ == '__main__':
    epoches: int = 2
    train: bool = True

    conf = Config.config()
    conf.train = train
    conf.epoches = epoches
    
    start = time.time()
    if conf.train:
        my_print('data_path:        {}'.format(conf.train_data_path))
        my_print('label_path:       {}'.format(conf.label_path))
        my_print('valid_data_path:  {}'.format(conf.valid_data_path))
        my_print('epoches num:      {}'.format(conf.epoches))

        my_activation = train_model(conf)
        my_print('Total epoches ({})'.format(conf.epoches))
        my_activation.training()
        end = time.time()
    else:
        my_print('test_data_path: {}'.format(conf.test_data_path))
        my_activation = test_model(conf)
        my_activation.test()
        end = time.time()
    my_print('Running time:{}'.format(str(end-start)))

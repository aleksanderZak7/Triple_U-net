import time

import Config
from utils import my_print
from test_model import TestModel


if __name__ == "__main__":
    conf = Config.config()
    
    start = time.time()

    my_print('test_data_path: {}'.format(conf.test_data_path))
    my_activation = TestModel(conf)
    my_activation.test()
    
    end = time.time()
    my_print('Running time:{}'.format(str(end-start)))
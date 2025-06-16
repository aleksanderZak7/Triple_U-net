"""
Directory architecture for the dataset used with CrossValidation:

dataset/
    ├── Original/       # original images
    └── Masks/          # ground truth masks for all images
                        # each mask file must have the same filename as its corresponding original image
"""


class config(object):
    def __init__(self):
        self.input_dir: str       = ''  # base directory for the dataset used with CrossValidation
        
        self.edg_path: str        = '' # edge of all the data
        self.label_path: str      = '' # ground truth of all the data
        self.test_data_path: str  = '' # test data path
        self.train_data_path: str = '' # training data path
        self.valid_data_path: str = '' # validation data path used during training
        
        self.save_path: str       = ''
        self.model_path: str      = ''
        
        self.train: int           = 1 # train:1 test:0
        self.cutoff: float        = 0.65 # the cutoff of the prediction map

        self.epoches: int         = 100
        self.batch_size: int      = 16

        self.optim_conf: dict     = {
        'learning_rate':0.0001,
        'weight_decay':0.0001,
        'betas':(0.9, 0.999)
        }
        
        self.lr_scheduler: dict   = {
        'gamma':0.96
        }

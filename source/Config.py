class config(object):
    def __init__(self):
        self.test_data_path: str  = '' # test data path
        
        self.save_path: str       = ''
        self.model_path: str      = ''
        
        self.cutoff: float        = 0.65 # the cutoff of the prediction map
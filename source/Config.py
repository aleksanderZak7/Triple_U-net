class config(object):
    def __init__(self):
        self.test_data_path: str  = '/home/azak/Data' # test data path

        self.save_path: str       = '/home/azak/Triple_U-net/output'
        self.model_path: str      = '/home/azak/Triple_u-net/model.hdf5'

        self.cutoff: float        = 0.65 # the cutoff of the prediction map
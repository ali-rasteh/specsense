from backend import *
from backend import be_np as np, be_scp as scipy
from filter_utils import Filter_Utils




class Near_Field_Utils(Filter_Utils):
    '''
    Near Field Utils class for handling near field channel models according to this paper:
    Hu, Y., Yin, M., Rangan, S., & Mezzavilla, M. (2023). Parametrization and Estimation 
    of High-Rank Line-of-Sight MIMO Channels with Reflected Paths. IEEE Transactions on 
    Wireless Communications.
    '''
    def __init__(self, params):
        super().__init__(params)

    

if __name__ == '__main__':
    test = Near_Field_Utils(params)
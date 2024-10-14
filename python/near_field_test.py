from backend import *
from backend import be_np as np, be_scp as scipy
from near_field_utils import Near_Field_Utils



class Params_Class(object):
    def __init__(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--n_sim", type=int, default=1000, help="number of simulations")
        # params = parser.parse_args()
        params = SimpleNamespace()
        params.overwrite_configs = True

        if params.overwrite_configs:
            self.n_sim = 1000
            


def run_sim(params):
    nf_test = Near_Field_Utils(params)
    print(nf_test.l2_norm(np.array([1,1])))



if __name__ == '__main__':

    params = Params_Class()
    run_sim(params)


from backend import *
from backend import be_np as np, be_scp as scipy
from SigProc_Comm.general import General
from enum import Enum





class DataTypes(Enum):
    fp16 = 0
    complexfp16 = 1
    int16 = 2
    complexint16 = 3
    



class SALSA_Comp(General):
    """
    Class for the SALSA algorithms computational complexity.
    """

    def __init__(self, params):
        """
        Initialize the SALSA class.

        Args:
            params: Parameters for the SALSA algorithm.
        """
        super().__init__(params)

        self.params = params
        
        self.fs = getattr(self.params, 'fs', 983.04e6)
        self.n_cc = getattr(self.params, 'n_cc', 4)
        self.fcc = getattr(self.params, 'fcc', 122.88e6)
        self.n_rx = getattr(self.params, 'n_rx', 32)
        self.n_tx = getattr(self.params, 'n_tx', 32)
        self.l_filt = getattr(self.params, 'l_filt', 32)
        self.l_match_filt = getattr(self.params, 'l_match_filt', 32)
        self.n_filt_stages = getattr(self.params, 'n_filt_stages', 3)
        self.us_rate = getattr(self.params, 'us_rate', 2)
        self.ds_rate = getattr(self.params, 'ds_rate', 2)
        self.n_fft = getattr(self.params, 'n_fft', 1024)
        self.n_sym_sf = getattr(self.params, 'n_sym_sf', 14)
        self.t_sf = getattr(self.params, 't_sf', 125e-6)
        self.n_sc_rb = getattr(self.params, 'n_sc_rb', 12)
        self.n_rb = getattr(self.params, 'n_rb', 69)
        self.n_sc = getattr(self.params, 'n_sc', 828)
        self.n_rs_rb = getattr(self.params, 'n_rs_rb', 4)

        self.ov_samp_rate = getattr(self.params, 'ov_samp_rate', self.fs / (self.fcc * self.n_cc))
        self.t_cc = getattr(self.params, 't_cc', 1 / self.fcc)
        self.f_sc = getattr(self.params, 'f_sc', self.fcc / self.n_fft)
        self.used_bw = getattr(self.params, 'used_bw', self.n_sc * self.f_sc)
        self.t_sym_total = getattr(self.params, 't_sym_total', self.t_sf / self.n_sym_sf)
        self.t_sym = getattr(self.params, 't_sym', 1/self.f_sc)
        self.t_cp = getattr(self.params, 't_cp', self.t_sym_total - self.t_sym)
        self.n_sym_per_sec = getattr(self.params, 'n_sym_per_sec', 1/self.t_sf * self.n_sym_sf)
        self.n_sym_total = getattr(self.params, 'n_sym_total', self.t_sym_total / self.t_cc)
        self.n_cp = getattr(self.params, 'n_cp', self.t_cp / self.t_sym * self.n_fft)
        self.n_rs = getattr(self.params, 'n_rs', self.n_rb * self.n_rs_rb)


        # Size of each data type in bytes
        self.dtype_size = {
            DataTypes.fp16: 2,
            DataTypes.complexfp16 : 4,
            DataTypes.int16: 2,
            DataTypes.complexint16: 4
            }



    def conv1d(self, n=1, nchan_in=1, nchan_out=1,
            kernel_size=3, 
            dtype_weight=DataTypes.fp16, dtype_data=DataTypes.fp16,
            stride =1,
            add_bias=True,
            params_in_msg=False,
            channel_wise=False):
        """
        Performs conv1d operation with nchan_in input channels and nchan_out output channels

        Parameters
        ----------
        n : int
            Number of input samples
        nchan_in : int
            Number of input channels
        nchan_out : int
            Number of output channels
        kernel_size : int
            Size of the convolution kernel
        dtype_weight : DataTypes
            Data type of the weights
        dtype_data : DataTypes
            Data type of the input data
        stride : int
            Stride of the convolution
        add_bias : bool
            Whether to add a bias term
        params_in_msg : bool
            Whether to include the parameters in the message size
        channel_wise : bool
            Whether to perform channel-wise convolution

        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """
        # Compute the number of operations
        if channel_wise:
            if nchan_in != nchan_out:
                raise ValueError('Channel-wise convolution requires nchan_in = nchan_out')
            nchan = nchan_in
        else:
            nchan = nchan_in*nchan_out 
        n_ops = n * nchan * kernel_size

        if add_bias:
            n_ops += n*nchan_out
        n_ops /= stride
        if dtype_weight == DataTypes.complexfp16:
            n_ops *= 2
        if dtype_data == DataTypes.complexfp16:
            n_ops *= 2

        # Get output data type
        if (dtype_weight == DataTypes.complexfp16) or (dtype_data == DataTypes.complexfp16):
            dtype_out = DataTypes.complexfp16
        else:
            dtype_out = DataTypes.fp16

        # Compute message sizes
        msg_in_size = n*nchan_in*self.dtype_size[dtype_data] 
        msg_out_size = n*nchan_out*self.dtype_size[dtype_out] // stride
        
        # Add the parameters if they are in the message
        if params_in_msg:
            msg_in_size +=nchan*kernel_size*self.dtype_size[dtype_weight]
            if add_bias:
                msg_in_size += nchan_out*self.dtype_size[dtype_weight]        

        return n_ops, msg_in_size, msg_out_size
    


    def mat_mult(self, shape0=(1,), shape1=(1,), dtype_arr0=DataTypes.fp16, dtype_arr1=DataTypes.fp16, reduce_dims=None,
                           array1_in_msg=False):
        """
        Multiplies a tensor of shape shape0 and shape1.  The output is optionally averaged
        over the dimensions in reduce_dims.

        Parameters
        ----------
        shape0 : tuple
            Shape of the first tensor
        shape1 : tuple
            Shape of the second tensor
        dtype_arr0 : DataTypes
            Data type of the first tensor
        dtype_arr1 : DataTypes
            Data type of the second tensor
        reduce_dims : list
            List of dimensions to reduce
        array1_in_msg : bool
            Whether to include the second tensor in the message size

        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """

        # Check that the final dimensions match
        if shape0[-1] != shape1[0]:
            raise ValueError('Mismatched dimensions for matrix multiply')
        k = shape1[0]

        # Compute the output shape
        if len(shape0) == 1:
            shape_out0 = list(shape1)
        else:
            shape_out0 = list(shape0[:-1]) + list(shape1)

        # Remove the output dimensions
        if reduce_dims is None:
            reduce_dims = [] 
        if np.isscalar(reduce_dims):
            reduce_dims = [reduce_dims]
        shape_out = [item for i, item in enumerate(shape_out0) if i not in reduce_dims]
        shape_out = np.array(shape_out)


        # Compute the number of operations
        s0 = np.array(shape0)
        s1 = np.array(shape1)
        n_ops = np.prod(s0) * np.prod(s1) / k  
        if dtype_arr0 == DataTypes.complexfp16:
            n_ops *= 2
        if dtype_arr1 == DataTypes.complexfp16:
            n_ops *= 2


        # Get output data type
        if (dtype_arr0 == DataTypes.complexfp16) or (dtype_arr1 == DataTypes.complexfp16):
            dtype_out = DataTypes.complexfp16
        else:
            dtype_out = DataTypes.fp16
                      
        # Get the message sizes
        msg_in_size = np.prod(s0)*self.dtype_size[dtype_arr0] 
        if array1_in_msg:
            msg_in_size += np.prod(s1)*self.dtype_size[dtype_arr1]
        msg_out_size = np.prod(shape_out)*self.dtype_size[dtype_out]
                
        # Compute processing time
        return n_ops, msg_in_size, msg_out_size



    def fc_layer(self, n=1, nchan_in=1, nchan_out=1, dtype=DataTypes.fp16, reduce_dims=None,
                           weight_in_msg=False, add_bias=True):
        """
        Fully connected layer

        Computes the processing parameters for a fully connected layer.

        Parameters
        ----------
        n : int
            Number of samples
        nchan_in : int
            Number of input channels
        nchan_out : int
            Number of output channels
        dtype : DataTypes
            Data type of the input data
        reduce_dims : list
            List of dimensions to reduce
        weight_in_msg : bool
            Whether to include the weights in the message size
        add_bias : bool
            Whether to add a bias term

        Returns
        -------
        """
        if add_bias:
            shape0 = (n, nchan_in+1)
            shape1 = (nchan_in+1, nchan_out)
        else:
            shape0 = (n, nchan_in)
            shape1 = (nchan_in, nchan_out)
        return self.mat_mult(shape0=shape0, shape1=shape1, dtype_arr0=dtype, dtype_arr1=dtype,
                                       reduce_dims=False,
                                       array1_in_msg=weight_in_msg)



    def elemwise_product(self, shape=(1,), dtype=DataTypes.fp16, reduce_dims=None, array1_in_msg=False):
        """
        Elementwise product.

        Computes the processing parameters for an elementwise product.

        Parameters
        ----------
        shape : tuple
            Shape of the input tensors
        dtype : DataTypes
            Data type of the input tensors
        reduce_dims : list
            List of dimensions to reduce
        array1_in_msg : bool
            Whether to include the second tensor in the message size

        Returns
        -------
        """
        n_ops = np.prod(shape)
        if dtype == DataTypes.complexfp16:
            n_ops *= 4

        msg_in_size = np.prod(shape)*self.dtype_size[dtype]
        if array1_in_msg:
            msg_in_size += np.prod(shape)*self.dtype_size[dtype]
        msg_out_size = np.prod(shape)*self.dtype_size[dtype]

        return n_ops, msg_in_size, msg_out_size



    def elemwise_add(self, shape=(1,), dtype=DataTypes.fp16, reduce_dims=None, array1_in_msg=False):
        """
        Elementwise addition of two tensors

        Parameters
        ----------
        shape : tuple
            Shape of the input tensors
        dtype : DataTypes
            Data type of the input tensors
        reduce_dims : list
            List of dimensions to reduce
        array1_in_msg : bool
            Whether to include the second tensor in the message size

        Returns
        -------
        """
        n_ops = np.prod(shape)
        if dtype == DataTypes.complexfp16:
            n_ops *= 2

        msg_in_size = np.prod(shape)*self.dtype_size[dtype]
        if array1_in_msg:
            msg_in_size += np.prod(shape)*self.dtype_size[dtype]
        msg_out_size = np.prod(shape)*self.dtype_size[dtype]

        return n_ops, msg_in_size, msg_out_size



    def fir_downsamp(self, 
              blksize=1, nstage = 0, nchan=1, 
              ntaps=32, dtype_weight=DataTypes.fp16, dtype_data=DataTypes.fp16):
        """
        Computes the processing parameters for a FIR downsampling operation.

        Parameters
        ----------
        blksize : int
            Size of the input block
        nstage : int
            Number of stages in the FIR filter
        nchan : int
            Number of input channels
        ntaps : int
            Number of taps in the FIR filter
        dtype_weight : DataTypes
            Data type of the weights
        dtype_data : DataTypes
            Data type of the input data

        Returns
        ----------
        """
        n_ops = 0
        if nstage < 0:
            raise ValueError('Invalid nstage = %d' % nstage)

        # Compute cost over multiple stages
        if nstage == 0:
            n_ops, msg_in_size, msg_out_size =\
            self.conv1d(
                n=blksize, nchan_in=nchan, nchan_out=nchan,
                kernel_size=ntaps, dtype_weight=dtype_weight, dtype_data=dtype_data,
                stride=1, channel_wise=True)
        else:
            blksize_ = blksize
            for i in range(nstage):
                blksize_ /= 2
                n_opsi, msg_in_sizei, msg_out_sizei =\
                    self.conv1d(
                        n=blksize_, nchan_in=nchan, nchan_out=nchan,
                        kernel_size=ntaps, dtype_weight=dtype_weight, dtype_data=dtype_data,
                        stride=1, channel_wise=True)
                n_ops += n_opsi

                if (i == 0):
                    msg_in_size = msg_in_sizei * 2
                if (i==nstage-1):
                    msg_out_size = msg_out_sizei

        return n_ops, msg_in_size, msg_out_size



    def fir_upsamp(self, blksize=1,nstage=1,nchan=1, ntaps=32, dtype_weight=DataTypes.fp16, dtype_data=DataTypes.fp16):
        """
        Computes the processing parameters for a FIR upsampling operation.

        Parameters
        ----------
        blksize : int
            Size of the input block
        nstage : int
            Number of stages in the FIR filter
        nchan : int
            Number of input channels
        ntaps : int
            Number of taps in the FIR filter
        dtype_weight : DataTypes
            Data type of the weights
        dtype_data : DataTypes
            Data type of the input data

        Returns
        ----------
        """

        n_ops = 0
        if nstage < 0:
            raise ValueError('Invalid nstage = %d' % nstage)

        # Compute cost over multiple stages
        if nstage == 0:
            n_ops, msg_in_size, msg_out_size =\
            self.conv1d(
                n=blksize, nchan_in=nchan, nchan_out=nchan,
                kernel_size=ntaps, dtype_weight=dtype_weight, dtype_data=dtype_data,
                stride=1, channel_wise=True)
        else:
            blksize_ = blksize
            for i in range(nstage):
                n_opsi, msg_in_sizei, msg_out_sizei =\
                    self.conv1d(
                        n=blksize_, nchan_in=nchan, nchan_out=nchan,
                        kernel_size=ntaps, dtype_weight=dtype_weight, dtype_data=dtype_data,
                        stride=1, channel_wise=True)
                
                n_ops += n_opsi
                if (i == 0):
                    msg_in_size = msg_in_sizei
                if (i==nstage-1):
                    msg_out_size = msg_out_sizei

                blksize_ *= 2

        return n_ops, msg_in_size, msg_out_size



    def batchnorm(self, shape=(1,), dtype=DataTypes.fp16):
        """
        Computes the processing parameters for a batch normalization operation.

        Parameters
        -----------
        shape : tuple
            Shape of the input tensor
        dtype : DataTypes
            Data type of the input data

        Returns
        ----------
        """
        return self.elemwise_product(shape=shape, dtype=dtype) 
    


    def dot_product(self, n=1, dtype=DataTypes.fp16, array1_in_msg=False):
        """
        General dot product operations.  We should eventually replace any of these.
        """
        n_ops = n
        if dtype == DataTypes.complexfp16:
            n_ops *= 4
        
        msg_in_size = n*self.dtype_size[dtype]
        if array1_in_msg:
            msg_in_size += n*self.dtype_size[dtype]
        msg_out_size = self.dtype_size[dtype]

        return n_ops, msg_in_size, msg_out_size



    def activation(self, shape=(1,), act_type='relu'):
        """
        Computes the processing parameters for an activation function.

        Parameters
        ----------
        shape : tuple
            Shape of the input tensor
        act_type : str
            Type of activation function ('relu' or 'sigmoid')

        Returns
        ----------
        """
        
        if act_type == 'relu':
            ops_per_act = 3
        elif act_type == 'sigmoid':
            ops_per_act = 5
        else:
            raise ValueError('Unkown activation %s' % act_type)

        dtype = DataTypes.fp16
        msg_in_size = np.prod(shape)*self.dtype_size[dtype]
        msg_out_size = msg_in_size

        # Assume pipelined implementation of activations
        n_ops = np.prod(shape) * ops_per_act
        
        return n_ops, msg_in_size, msg_out_size



    def fft(self, nfft=1024, nchan=1, dtype=DataTypes.complexfp16):
        """
        Computes the processing parameters for an FFT operation.
        Parameters
        ----------
        nfft : int
            Number of FFT points
        nchan : int
            Number of input channels
        dtype : DataTypes
            Data type of the input data
        
        Returns
        ----------
        """

        # n_ops, msg_in_size, msg_out_size =\
        #    self.mat_mult(shape0=(nfft, nfft), shape1=(nfft, nchan), dtype=dtype, reduce_dims=None,
        #                    array1_in_msg=False)

        n_ops = (3/2) * nfft * np.log2(nfft) * nchan
        n_ops *= 4
        if dtype == DataTypes.fp16:
            n_ops /= 2
        msg_in_size = nfft*nchan*self.dtype_size[dtype]
        dtype = DataTypes.complexfp16
        msg_out_size = nfft*nchan*self.dtype_size[dtype]

        return n_ops, msg_in_size, msg_out_size
    

    def ifft(self, nfft=1024, nchan=1, dtype=DataTypes.complexfp16):
        """
        Computes the processing parameters for an FFT operation.
        Parameters
        ----------
        nfft : int
            Number of FFT points
        nchan : int
            Number of input channels
        dtype : DataTypes
            Data type of the input data
        
        Returns
        ----------
        """

        # n_ops, msg_in_size, msg_out_size =\
        #    self.mat_mult(shape0=(nfft, nfft), shape1=(nfft, nchan), dtype=dtype, reduce_dims=None,
        #                    array1_in_msg=False)

        n_ops = (3/2) * nfft * np.log2(nfft) * nchan
        n_ops *= 4
        dtype_ = DataTypes.complexfp16
        msg_in_size = nfft*nchan*self.dtype_size[dtype_]
        msg_out_size = nfft*nchan*self.dtype_size[dtype]

        return n_ops, msg_in_size, msg_out_size



    def maxpool(self, shape=(1,), kernel_size=3, stride=2):
        """
        Computes the processing parameters for a maxpool operation.
        
        Parameters
        ----------
        shape : tuple
            Shape of the input tensor
        kernel_size : int
            Size of the maxpool kernel
        stride : int
            Stride of the maxpool operation
        
        Returns
        ----------
        """

        if np.isscalar(stride):
            stride = [stride] * len(shape)
        if np.isscalar(kernel_size):
            kernel_size = [kernel_size] * len(shape)
        
        nout_vals = round(np.prod(shape) / np.prod(stride))
        n_ops = nout_vals * np.prod(kernel_size)

        dtype = DataTypes.fp16
        msg_in_size = np.prod(shape)*self.dtype_size[dtype]
        msg_out_size = nout_vals*self.dtype_size[dtype]
        
        return n_ops, msg_in_size, msg_out_size



    def maximum(self, shape=(1,)):
        """
        Finds the maximum of a tensor 

        parameters
        ----------
        shape : tuple
            Shape of the input tensor

        Returns
        ----------
        """

        # The factor of two is added to support the maxima tree
        dtype = DataTypes.fp16
        n_ops = np.prod(shape)

        dtype = DataTypes.fp16
        msg_in_size = np.prod(shape)*self.dtype_size[dtype]
        msg_out_size = self.dtype_size[dtype] + self.dtype_size[DataTypes.int16]
        
        return n_ops, msg_in_size, msg_out_size



    def outerprod(self, n=1, dims=None, dtype=DataTypes.fp16, 
                             auto_corr=False, 
                             reduce_dims=False):
        """
        Computes the processing parameters for an outer product

        The operation takes array x0 of size (n,dims[0]) and array 
        x1 of size (n,dims[1]) and outputs a matrix of tensor :
        
            y = x0[:,:,None] *  conj(x1[:,None,:]) 

        of size (n, dims[0], dims[1]).

        If reduce_dims is True, the output is reduced with an additional step
        equivalent to numpy:
        
            y = np.sum(y, axis=0) of size (dims[0],dims[1])

        If auto_corr is True, the operation assumes x0 = x1. 

        Parameters
        ----------
        n : int
            Number of samples
        dims: Either scalar or list of two integers
            Dimensions of the input vectors
        dtype:  DataTypes
            Data type of the input tensors
        auto_corr : bool
            If True, the operation is an auto-correlation
        reduce_dims : bool
            If True, the output is reduced to the size of the input tensors

        Returns
        ----------
        """
        if auto_corr:
            if dims is None:
                dims = 1
            if not np.isscalar(dims):
                raise ValueError('Expected scalar dims when auto-correlation is used')
            dims = [dims, dims]
        else:
            if dims is None:
                dims = [1,1]
        
        n_ops = n * dims[0]*dims[1]
        if dtype == DataTypes.complexfp16:
            n_ops *= 4       


        # Compute input and output message size
        if auto_corr:
            msg_in_size = n*dims[0]
        else:
            msg_in_size = n*(dims[0]+dims[1])
        if reduce_dims:
            msg_out_size = dims[0]*dims[1]
        else:
            msg_out_size = dims[0]*dims[1]*n

        # Modify with the data type
        msg_in_size *= self.dtype_size[dtype]
        msg_out_size *= self.dtype_size[dtype]
          

        return n_ops, msg_in_size, msg_out_size





class SALSA_Comp_5G(SALSA_Comp):
    """
    Class for the SALSA algorithms computational complexity for 5G.
    """
    def __init__(self, params):
        """
        Initialize the SALSA_Comp_5G class.

        Args:
            params: Parameters for the SALSA algorithm.
        """
        super().__init__(params)



    def nco(self):
        """
        Computes the processing parameters for a NCO operation.

        Parameters
        ----------
        None

        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """
        
        n_ops = 0
        msg_in_size = 0
        msg_out_size = 0

        return n_ops, msg_in_size, msg_out_size





    def filtering_to_cc(self, mode='rx'):
        """
        Computes the processing parameters for the filtering to CC operation.
        Parameters
        ----------
        mode : str
            Mode of the operation ('rx' or 'tx')
        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """
        
        n_ops = 0
        msg_in_size = 0
        msg_out_size = 0

        n_ops_, msg_in_size_, msg_out_size_ = \
            self.nco()
        n_ops += n_ops_
        msg_in_size += msg_in_size_
        msg_out_size += msg_out_size_

        # Padding the input data to take into account the continuous nature of the data
        # and the filter length
        blk_size_ = self.n_fft + self.l_filt - 1
        n_ops_, msg_in_size_, msg_out_size_ = \
            self.fir_downsamp(blksize=blk_size_, nstage=self.n_filt_stages, nchan=self.n_cc,
                              ntaps=self.l_filt, dtype_weight=DataTypes.complexfp16, dtype_data=DataTypes.complexfp16)
        n_ops += n_ops_
        msg_in_size += msg_in_size_
        msg_out_size += msg_out_size_


        return n_ops, msg_in_size, msg_out_size
    
    

    def ofdm_processing(self, mode='tx'):
        """
        Computes the processing parameters for the OFDM processing.

        Parameters
        ----------
        None

        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """
        
        # Compute the processing parameters for the OFDM processing
        if mode == 'rx':
            n_ops, msg_in_size, msg_out_size = \
                self.fft(nfft=self.n_fft, nchan=self.n_cc, dtype=DataTypes.complexfp16)
        elif mode == 'tx':
            n_ops, msg_in_size, msg_out_size = \
                self.ifft(nfft=self.n_fft, nchan=self.n_cc, dtype=DataTypes.complexfp16)
        else:
            raise ValueError('Unknown mode %s' % mode)
        
        
        return n_ops, msg_in_size, msg_out_size
    


    def rs_data_demux(self):
        """
        Computes the processing parameters for the RS data demux operation.

        Parameters
        ----------
        None

        Returns
        -------
        n_ops : int
            Number of operations
        msg_in_size : int
            Size of the input message
        msg_out_size : int
            Size of the output message
        """
        
        n_ops = 5 * self.n_cc * (self.n_rs / self.n_fft)        # Assuming 5 operations to extract each RS
        msg_in_size = self.n_fft * self.n_cc * self.dtype_size[DataTypes.complexfp16]
        msg_out_size = msg_in_size

        return n_ops, msg_in_size, msg_out_size
        
        

    # def match_filter(self):
    #     """
    #     Computes the processing parameters for the match filter operation.

    #     Parameters
    #     ----------
    #     None

    #     Returns
    #     -------
    #     n_ops : int
    #         Number of operations
    #     msg_in_size : int
    #         Size of the input message
    #     msg_out_size : int
    #         Size of the output message
    #     """
        
        
    #     self.conv1d(self, n=self., nchan_in=1, nchan_out=1,
    #         kernel_size=3, 
    #         dtype_weight=DataTypes.fp16, dtype_data=DataTypes.fp16,
    #         stride =1,
    #         add_bias=True,
    #         params_in_msg=False,
    #         channel_wise=False)
        
    #     return n_ops, msg_in_size, msg_out_size


from backend import *
from backend import be_np as np, be_scp as scipy




class OFDMEqualizer(Block):
    # pylint: disable=line-too-long
    r"""
    Block that wraps a MIMO equalizer for use with the OFDM waveform

    The parameter ``equalizer`` is a callable (e.g., a function) that
    implements a MIMO equalization algorithm for arbitrary batch dimensions.

    This class pre-processes the received resource grid ``y`` and channel
    estimate ``h_hat``, and computes for each receiver the
    noise-plus-interference covariance matrix according to the OFDM and stream
    configuration provided by the ``resource_grid`` and
    ``stream_management``, which also accounts for the channel
    estimation error variance ``err_var``. These quantities serve as input
    to the equalization algorithm that is implemented by the callable ``equalizer``.
    This block computes soft-symbol estimates together with effective noise
    variances for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    Note
    -----
    The callable ``equalizer`` must take three inputs:

    * **y** ([...,num_rx_ant], tf.complex) -- 1+D tensor containing the received signals.
    * **h** ([...,num_rx_ant,num_streams_per_rx], tf.complex) -- 2+D tensor containing the channel matrices.
    * **s** ([...,num_rx_ant,num_rx_ant], tf.complex) -- 2+D tensor containing the noise-plus-interference covariance matrices.

    It must generate two outputs:

    * **x_hat** ([...,num_streams_per_rx], tf.complex) -- 1+D tensor representing the estimated symbol vectors.
    * **no_eff** (tf.float) -- Tensor of the same shape as ``x_hat`` containing the effective noise variance estimates.

    Parameters
    ----------
    equalizer : `Callable`
        Callable object (e.g., a function) that implements a MIMO equalization
        algorithm for arbitrary batch dimensions

    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used 

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 equalizer,
                 resource_grid,
                 stream_management,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert callable(equalizer)
        assert isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.phy.mimo.StreamManagement)
        self._equalizer = equalizer
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._removed_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

        # Precompute indices to extract data symbols
        mask = resource_grid.pilot_pattern.mask
        num_data_symbols = resource_grid.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")
        self._data_ind = data_ind[...,:num_data_symbols]

    def call(self, y, h_hat, err_var, no):

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # h_hat has shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]

        # err_var has a shape that is broadcastable to h_hat

        # no has shape [batch_size, num_rx, num_rx_ant]
        # or just the first n dimensions of this

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_rx, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        y_eff = self._removed_nulled_scs(y)

        ####################################################
        ### Prepare the observation y for MIMO detection ###
        ####################################################
        # Transpose y_eff to put num_rx_ant last. New shape:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        y_dt = tf.transpose(y_eff, [0, 1, 3, 4, 2])
        y_dt = tf.cast(y_dt, self.cdtype)

        ##############################################
        ### Prepare the err_var for MIMO detection ###
        ##############################################
        # New shape is:
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant, num_tx*num_streams]
        err_var_dt = tf.broadcast_to(err_var, tf.shape(h_hat))
        err_var_dt = tf.transpose(err_var_dt, [0, 1, 5, 6, 2, 3, 4])
        err_var_dt = flatten_last_dims(err_var_dt, 2)
        err_var_dt = tf.cast(err_var_dt, self.cdtype)

        ###############################
        ### Construct MIMO channels ###
        ###############################

        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = tf.transpose(h_hat, perm)

        # Flatten first tthree dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
        h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(h_dt_desired,
                                 [self._stream_management.num_rx,
                                  self._stream_management.num_streams_per_rx],
                                 0)
        h_dt_undesired = split_dim(h_dt_undesired,
                                   [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self.cdtype)
        h_dt_undesired = tf.transpose(h_dt_undesired, perm)

        ##################################
        ### Prepare the noise variance ###
        ##################################
        # no is first broadcast to [batch_size, num_rx, num_rx_ant]
        # then the rank is expanded to that of y
        # then it is transposed like y to the final shape
        # [batch_size, num_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, num_rx_ant]
        no_dt = expand_to_rank(no, 3, -1)
        no_dt = tf.broadcast_to(no_dt, tf.shape(y)[:3])
        no_dt = expand_to_rank(no_dt, tf.rank(y), -1)
        no_dt = tf.transpose(no_dt, [0,1,3,4,2])
        no_dt = tf.cast(no_dt, self.cdtype)

        ##################################################
        ### Compute the interference covariance matrix ###
        ##################################################
        # Covariance of undesired transmitters
        s_inf = tf.matmul(h_dt_undesired, h_dt_undesired, adjoint_b=True)

        #Thermal noise
        s_no = tf.linalg.diag(no_dt)

        # Channel estimation errors
        # As we have only error variance information for each element,
        # we simply sum them across transmitters and build a
        # diagonal covariance matrix from this
        s_csi = tf.linalg.diag(tf.reduce_sum(err_var_dt, -1))

        # Final covariance matrix
        s = s_inf + s_no + s_csi
        s = tf.cast(s, self.cdtype)

        ############################################################
        ### Compute symbol estimate and effective noise variance ###
        ############################################################
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., num_stream_per_rx]
        x_hat, no_eff = self._equalizer(y_dt, h_dt_desired, s)
        # print("x_hat.shape: ", x_hat.shape)

        ################################################
        ### Extract data symbols for all detected TX ###
        ################################################
        # Transpose tensor to shape
        # [num_rx, num_streams_per_rx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers, batch_size]
        x_hat = tf.transpose(x_hat, [1, 4, 2, 3, 0])
        no_eff = tf.transpose(no_eff, [1, 4, 2, 3, 0])

        # Merge num_rx amd num_streams_per_rx
        # [num_rx * num_streams_per_rx, num_ofdm_symbols,...
        #  ...,num_effective_subcarriers, batch_size]
        x_hat = flatten_dims(x_hat, 2, 0)
        no_eff = flatten_dims(no_eff, 2, 0)

        # Put first dimension into the right ordering
        stream_ind = self._stream_management.stream_ind
        x_hat = tf.gather(x_hat, stream_ind, axis=0)
        no_eff = tf.gather(no_eff, stream_ind, axis=0)

        # Reshape first dimensions to [num_tx, num_streams] so that
        # we can compared to the way the streams were created.
        # [num_tx, num_streams, num_ofdm_symbols, num_effective_subcarriers,...
        #  ..., batch_size]
        num_streams = self._stream_management.num_streams_per_tx
        num_tx = self._stream_management.num_tx
        x_hat = split_dim(x_hat, [num_tx, num_streams], 0)
        no_eff = split_dim(no_eff, [num_tx, num_streams], 0)

        # Flatten resource grid dimensions
        # [num_tx, num_streams, num_ofdm_symbols*num_effective_subcarriers,...
        #  ..., batch_size]
        x_hat = flatten_dims(x_hat, 2, 2)
        no_eff = flatten_dims(no_eff, 2, 2)

        # Broadcast no_eff to the shape of x_hat
        no_eff = tf.broadcast_to(no_eff, tf.shape(x_hat))

        # Gather data symbols
        # [num_tx, num_streams, num_data_symbols, batch_size]
        x_hat = tf.gather(x_hat, self._data_ind, batch_dims=2, axis=2)
        no_eff = tf.gather(no_eff, self._data_ind, batch_dims=2, axis=2)

        # Put batch_dim first
        # [batch_size, num_tx, num_streams, num_data_symbols]
        x_hat = tf.transpose(x_hat, [3, 0, 1, 2])
        no_eff = tf.transpose(no_eff, [3, 0, 1, 2])

        return x_hat, no_eff
    
    
    
class SALSA_Equalizer(OFDMEqualizer):
    # pylint: disable=line-too-long
    """
    LMMSE equalization for OFDM MIMO transmissions

    This block computes linear minimum mean squared error (LMMSE) equalization
    for OFDM MIMO transmissions. The OFDM and stream configuration are provided
    by a :class:`~sionna.phy.ofdm.ResourceGrid` and
    :class:`~sionna.phy.mimo.StreamManagement` instance, respectively. The
    detection algorithm is the :meth:`~sionna.phy.mimo.lmmse_equalizer`. The block
    computes soft-symbol estimates together with effective noise variances
    for all streams which can, e.g., be used by a
    :class:`~sionna.phy.mapping.Demapper` to obtain LLRs.

    Parameters
    ----------
    resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
        ResourceGrid to be used

    stream_management : :class:`~sionna.phy.mimo.StreamManagement`
        StreamManagement to be used

    whiten_interference : `bool`, (default `True`)
        If `True`, the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used which
        can be numerically more stable.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `tf.complex`
        Received OFDM resource grid after cyclic prefix removal and FFT

    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
        Channel estimates for all streams from all transmitters

    err_var : [Broadcastable to shape of ``h_hat``], `tf.float`
        Variance of the channel estimation error

    no : [batch_size, num_rx, num_rx_ant] (or only the first n dims), `tf.float`
        Variance of the AWGN

    Output
    ------
    x_hat : [batch_size, num_tx, num_streams, num_data_symbols], `tf.complex`
        Estimated symbols

    no_eff : [batch_size, num_tx, num_streams, num_data_symbols], `tf.float`
        Effective noise variance for each estimated symbol
    """
    def __init__(self,
                 resource_grid,
                 resource_grid_srs,
                 stream_management,
                 whiten_interference=True,
                 precision=None,
                 **kwargs):

        self._whiten_interference = whiten_interference
        
        super().__init__(equalizer=self.equalizer,
                         resource_grid=resource_grid,
                         stream_management=stream_management,
                         precision=precision, **kwargs)
        
        self.srs_mask = tf.zeros_like(self._resource_grid.pilot_pattern.mask[0,0])
        # self.srs_mask = tf.tensor_scatter_nd_update(self.srs_mask, [[i] for i in range(3)], tf.ones([3], dtype=self.srs_mask.dtype))
        srs_mask_np = self.srs_mask.numpy()
        srs_mask_np[:3] = 1.0
        self.srs_mask = tf.convert_to_tensor(srs_mask_np)
        
        self._resource_grid_srs = resource_grid_srs
        self.mode = "data"
        
        self.ltbf_list = []
        
        
    def get_desired_channels(self, h_hat):
        ###############################
        ### Construct MIMO channels ###
        ###############################

        # Reshape h_hat for the construction of desired/interfering channels:
        # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        perm = [1, 3, 4, 0, 2, 5, 6]
        h_dt = tf.transpose(h_hat, perm)

        # Flatten first tthree dimensions:
        # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt = flatten_dims(h_dt, 3, 0)

        # Gather desired and undesired channels
        ind_desired = self._stream_management.detection_desired_ind
        ind_undesired = self._stream_management.detection_undesired_ind
        h_dt_desired = tf.gather(h_dt, ind_desired, axis=0)
        h_dt_undesired = tf.gather(h_dt, ind_undesired, axis=0)

        # Split first dimension to separate RX and TX:
        # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        h_dt_desired = split_dim(h_dt_desired,
                                 [self._stream_management.num_rx,
                                  self._stream_management.num_streams_per_rx],
                                 0)
        h_dt_undesired = split_dim(h_dt_undesired,
                                   [self._stream_management.num_rx, -1], 0)

        # Permutate dims to
        # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
        #  ..., num_rx_ant, num_streams_per_rx(num_Interfering_streams_per_rx)]
        perm = [2, 0, 4, 5, 3, 1]
        h_dt_desired = tf.transpose(h_dt_desired, perm)
        h_dt_desired = tf.cast(h_dt_desired, self.cdtype)
        h_dt_undesired = tf.transpose(h_dt_undesired, perm)
        
        return h_dt_desired, h_dt_undesired
        

    def process_srs(self, h_hat):
        # TODO: Change the location of each user's pilot in different srs measurements to be more accurate
        
        pilots = self._resource_grid_srs.pilot_pattern.pilots
        pilots = tf.expand_dims(pilots, axis=2)
        # pilots_indices = (pilots != 0)
        pilots_indices = tf.where(pilots != 0)
        # non_zero_pilots = tf.boolean_mask(pilots, pilots_indices)
        non_zero_pilots = tf.gather_nd(pilots, pilots_indices)
        non_zero_pilots = tf.reshape(non_zero_pilots, [pilots.shape[0], pilots.shape[1], pilots.shape[2], -1])
        
        aa, bb, cc = tf.meshgrid(tf.range(h_hat.shape[0]), tf.range(h_hat.shape[1]), tf.range(h_hat.shape[2]), indexing='ij')  # each is (a, b, c)
        indices_1 = tf.stack([aa, bb, cc], axis=-1)
        indices_1 = tf.reshape(indices_1, (-1, 3))  # shape (a*b*c, 3)
        indices_1 = tf.cast(indices_1, pilots_indices.dtype)
        tiled_1 = tf.tile(indices_1[:, tf.newaxis, :], [1, len(pilots_indices), 1])
        tiled_2 = tf.tile(pilots_indices[tf.newaxis, :, :], [tf.shape(indices_1)[0], 1, 1])
        full_indices = tf.concat([tiled_1, tiled_2], axis=-1)
        full_indices = tf.reshape(full_indices, (-1, len(h_hat.shape)))            
        h_hat_ltbf = tf.gather_nd(h_hat, full_indices)
        
        # pilots_mask = (pilots != 0)
        # h_hat_mask = tf.tile(pilots_mask[tf.newaxis, tf.newaxis, tf.newaxis, :, :, :, :], [h_hat.shape[0], h_hat.shape[1], h_hat.shape[2], 1, 1, 1, 1])
        # h_hat_mask = tf.cast(h_hat_mask, pilots_mask.dtype)
        # h_hat_ltbf = tf.boolean_mask(h_hat, h_hat_mask)
        
        h_hat_ltbf = tf.reshape(h_hat_ltbf, tuple(list(h_hat.shape[:-1])+[-1]))
        h_hat_ltbf, _ = self.get_desired_channels(h_hat_ltbf)
        self.ltbf_list.append(h_hat_ltbf)
        
        if len(self.ltbf_list) == 3:
            self.calc_lt_bf()
        
        
    
    def calc_lt_bf(self, alpha=0.5):
        
        # print(self.ltbf_list[0].shape)
        lbtf_data = tf.concat(self.ltbf_list, axis=3)
        # print("lbtf_data: ", lbtf_data.shape)
        
        # pilots_mask = self._rg_srs.pilot_pattern.mask[0,0]
        # pilots = self._rg_srs.pilot_pattern.pilots
        # print("pilots: ", pilots.shape)
        
        # pilots_mask = tf.reshape(pilots_mask, [-1])
        # pilot_indices = tf.where(pilots_mask)[:,0]
        
        lbtf_data = tf.reshape(lbtf_data, lbtf_data.shape[0:2]+(np.prod(lbtf_data.shape[2:4]),)+lbtf_data.shape[4:])
        # s_hat = tf.reshape(s, s.shape[0:2]+(np.prod(s.shape[2:4]),)+s.shape[4:])
        
        # h_hat = tf.gather(h_hat, pilot_indices, axis=2)
        # s_hat = tf.gather(s_hat, pilot_indices, axis=2)
        
        lbtf_data = tf.transpose(lbtf_data, [0,1,4,3,2])
        print("lbtf_data: ", lbtf_data.shape)
        
        n_srs = tf.shape(lbtf_data)[-1]
        Q_hat = tf.matmul(lbtf_data, lbtf_data, adjoint_b=True) / tf.cast(n_srs, lbtf_data.dtype)
        Q_hat = tf.reduce_sum(Q_hat, axis=[2])
        # s_hat = tf.cast(tf.reduce_mean(tf.abs(s_hat), axis=[2]), Q_hat.dtype)
        # Q_hat = Q_hat + s_hat
        print("Q_hat: ", Q_hat.shape)
        Q_hat_I = tf.eye(Q_hat.shape[2])
        Q_hat_I = tf.cast(Q_hat_I, Q_hat.dtype)
        Q_hat = Q_hat_I + alpha * Q_hat
        print("Q_hat: ", Q_hat.shape)
        exit()
        
        Q_hat_inv = tf.linalg.inv(Q_hat)
        eigvals, eigvecs = tf.linalg.eigh(Q_hat)
        Q_inv_sqrt = eigvecs @ tf.linalg.diag(1.0 / tf.sqrt(eigvals)) @ tf.linalg.adjoint(eigvecs)

        QH = Q_inv_sqrt[:,:,None,:,:] @ lbtf_data
        # QH = tf.transpose(QH, [0,1,2,4,3])
        print("QH: ", QH.shape)

        QH_s, QH_u, QH_v = tf.linalg.svd(QH, full_matrices=False)
        # print("u.shape", QH_u.shape)

        N_LT = 4
        F_0 = QH_u[..., :N_LT]
        print("F_0: ", F_0.shape)
        print("y: ", y.shape)
        
        F_0 = tf.transpose(F_0, [0,1,2,4,3])
        
        self.ltbf_list = []
        
        
    def equalizer(self, y, h, s):
        """Salsa equalizer"""
        print("h: ", h.shape)
        if self.mode == "srs":
            # self.process_srs(h)
            return self.last_result
        else:
            self.last_result = lmmse_equalizer(y, h, s, self._whiten_interference)
            return self.last_result 
        
        pilots_mask = self._resource_grid.pilot_pattern.mask[0,0]
        pilots_mask = tf.reshape(pilots_mask, [-1])
        
        pilot_indices = tf.where(pilots_mask)[:,0]
        pilots = self._resource_grid.pilot_pattern.pilots      
      
        y_pilots = tf.reshape(y, y.shape[0:2]+(np.prod(y.shape[2:4]),)+y.shape[4:])
        y_pilots = tf.gather(y_pilots, pilot_indices, axis=2)
        
        # y_ = tf.transpose(y, [0,1,4,2,3])
        y_pilots = tf.transpose(y_pilots, [0,1,3,2])
        print("F_0: ", F_0.shape)
        print("y_pilots: ", y_pilots.shape)
        
        # z = tf.einsum('abcde,abemn->abcdmn', F_0, y_)
        z = tf.einsum('abcde,abem->abcdm', F_0, y_pilots)
        print("z: ", z.shape)
        
        lambda_reg = 1e-6
        Q_2 = tf.matmul(z, z, adjoint_b=True) + lambda_reg * tf.eye(z.shape[-2], dtype=z.dtype)
        Q_2_inv = tf.linalg.inv(Q_2)
        print("Q_2_inv: ", Q_2_inv.shape)
        X = tf.expand_dims(pilots, axis=0)
        X = tf.tile(X, [z.shape[0], 1, 1, 1])
        X = tf.reshape(X, [z.shape[0], z.shape[1], z.shape[2], X.shape[2], X.shape[3]])     # Double check the shape of pilot for different inputs
        print("X: ", X.shape)
        F_1 = tf.matmul(Q_2_inv, z)
        F_1 = tf.matmul(F_1, X, adjoint_b=True)
        print("F_1: ", F_1.shape)
        F_1 = tf.linalg.adjoint(F_1)
        print("F_1: ", F_1.shape)

        # return x_hat, no_eff
        return lmmse_equalizer(y, h, s, self._whiten_interference) 
        
        
        
        

class SALSA_CIRGenerator:
    """Creates a generator from a given dataset of channel impulse responses

    The generator samples ``num_tx`` different transmitters from the given path
    coefficients `a` and path delays `tau` and stacks the CIRs into a single tensor.

    Note that the generator internally samples ``num_tx`` random transmitters
    from the dataset. For this, the inputs ``a`` and ``tau`` must be given for
    a single transmitter (i.e., ``num_tx`` =1) which will then be stacked
    internally.

    Parameters
    ----------
    a : [batch size, num_rx, num_rx_ant, 1, num_tx_ant, num_paths, num_time_steps], complex
        Path coefficients per transmitter

    tau : [batch size, num_rx, 1, num_paths], float
        Path delays [s] per transmitter

    num_tx : int
        Number of transmitters

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self,
                 a,
                 tau,
                 chan_scale=None,
                 batch_size=1):

        # Copy to tensorflow
        self._a = tf.constant(a, tf.complex64)
        self._tau = tf.constant(tau, tf.float32)
        self._dataset_size = self._a.shape[0]
        if chan_scale is not None:
            self._chan_scale = tf.constant(chan_scale, tf.complex64)
        else:
            self._chan_scale = None

        self.batch_size = batch_size
        self.idx = 0

    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):

        # Generator implements an infinite loop that yields new random samples
        while True:

            a = self._a[self.idx:self.idx+self.batch_size]
            tau = self._tau[self.idx:self.idx+self.batch_size]
            if self.batch_size == 1:
                a = tf.squeeze(a, axis=0)
                tau = tf.squeeze(tau, axis=0)
                
            if self._chan_scale is not None:
                a = a * self._chan_scale[None, None, :, None, None, None]
                
            self.idx += self.batch_size
            if self.idx >= self._dataset_size:
                self.idx = 0

            yield a, tau
            


class SALSA_CIRDataset(ChannelModel):
    def __init__(self, cir_generator, batch_size, num_rx, num_rx_ant, num_tx,
        num_tx_ant, num_paths, num_time_steps, precision=None, **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._cir_generator = cir_generator
        self._batch_size = batch_size
        self._num_time_steps = num_time_steps

        # TensorFlow dataset
        output_signature = (tf.TensorSpec(shape=[num_rx,
                                                 num_rx_ant,
                                                 num_tx,
                                                 num_tx_ant,
                                                 num_paths,
                                                 num_time_steps],
                                          dtype=self.cdtype),
                            tf.TensorSpec(shape=[num_rx,
                                                 num_tx,
                                                 num_paths],
                                          dtype=self.rdtype))
        dataset = tf.data.Dataset.from_generator(cir_generator,
                                            output_signature=output_signature)
        # dataset = dataset.shuffle(32, reshuffle_each_iteration=True)
        self._dataset = dataset.repeat(None)
        self._batched_dataset = self._dataset.batch(batch_size)
        # Iterator for sampling the dataset
        self._iter = iter(self._batched_dataset)

    @property
    def batch_size(self):
        """
        int : Get/set batch size
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set the batch size"""
        self._batched_dataset = self._dataset.batch(value)
        self._iter = iter(self._batched_dataset)
        self._batch_size = value

    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):
        return next(self._iter)
    
    


class MIMO_OFDM(Block):
    """This block simulates OFDM MIMO transmissions over the CDL model.

    Simulates point-to-point transmissions between a UT and a BS.
    Uplink and downlink transmissions can be realized with either perfect CSI
    or channel estimation. ZF Precoding for downlink transmissions is assumed.
    The receiver (in both uplink and downlink) applies LS channel estimation
    and LMMSE MIMO equalization. A 5G LDPC code as well as QAM modulation are
    used.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.
        Time-domain simulations are generally slower and consume more memory.
        They allow modeling of inter-symbol interference and channel changes
        during the duration of an OFDM symbol.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use. Note that "D" and "E" are LOS models that are
        not well suited for the transmissions of multiple streams.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed. For downlink
        transmissions, the transmitter is always assumed to have perfect CSI.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.

    Input
    -----
    batch_size : int
        The batch size, i.e., the number of independent Mote Carlo simulations
        to be performed at once. The larger this number, the larger the memory
        requiremens.

    ebno_db : float
        The Eb/No [dB]. This value is converted to an equivalent noise power
        by taking the modulation order, coderate, pilot and OFDM-related
        overheads into account.

    Output
    ------
    b : [batch_size, 1, num_streams, k], tf.float32
        The tensor of transmitted information bits for each stream.

    b_hat : [batch_size, 1, num_streams, k], tf.float32
        The tensor of received information bits for each stream.
    """

    def __init__(self,
                channel_mode = "cdl",
                scenario = "umi",
                domain = "freq",
                direction = "uplink",
                cdl_model = "A",
                channel_model = None,
                normalize_channel = False,
                channel_add_awgn = False,
                delay_spread = 100e-9,
                perfect_csi = True,
                speed = 0.0,
                cyclic_prefix_length = 20,
                pilot_ofdm_symbol_indices = [2, 11],
                subcarrier_spacing = 30e3,
                carrier_frequency = 2.6e9,
                fft_size = 128,
                num_ofdm_symbols = 14,
                num_sectors = 1,
                num_ut = 4,
                bs_ut_association = None,
                num_ut_ant_row = 1,
                num_ut_ant_col = 1,
                num_bs_ant_row = 8,
                num_bs_ant_col = 4,
                dc_null = False,
                # num_guard_carriers = [5, 6],
                num_guard_carriers = [0,0],
                pilot_pattern = "kronecker",
                num_bits_per_symbol = 2,
                coderate = 0.5
                ):
        super().__init__()

        # Provided parameters
        self._channel_mode = channel_mode
        self._scenario = scenario
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._channel_model = channel_model
        self._normalize_channel = normalize_channel
        self._channel_add_awgn = channel_add_awgn
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices

        # System parameters
        self._carrier_frequency = carrier_frequency
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = fft_size
        self._num_ofdm_symbols = num_ofdm_symbols
        self._num_sectors = num_sectors
        self._num_ut = num_ut
        # Must be a multiple of two if dual-polarized antennas are used
        self._num_ut_ant_row = num_ut_ant_row
        self._num_ut_ant_col = num_ut_ant_col
        self._num_ut_ant = self._num_ut_ant_row * self._num_ut_ant_col
        self._num_bs_ant_row = num_bs_ant_row
        self._num_bs_ant_col = num_bs_ant_col
        self._num_bs_ant = self._num_bs_ant_row * self._num_bs_ant_col
        self._dc_null = dc_null
        self._num_guard_carriers = num_guard_carriers
        self._pilot_pattern = pilot_pattern
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._srs_ofdm_symbol_indices = [11, 12]
        self._num_bits_per_symbol = num_bits_per_symbol
        self._coderate = coderate
        self.bs_ut_association = bs_ut_association

        

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        if self.bs_ut_association is None:
            self.bs_ut_association = np.zeros([self._num_sectors, self._num_ut])
            ut_per_sector = int(self._num_ut/self._num_sectors)
            for s in range(self._num_sectors):   
                self.bs_ut_association[s,s*ut_per_sector:(s+1)*ut_per_sector] = 1
        self._rx_tx_association = self.bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant

        # Required system components
        # self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        
        self._rg_srs = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                            fft_size=self._fft_size,
                            subcarrier_spacing = self._subcarrier_spacing,
                            num_tx=self._num_tx,
                            # num_tx=1,
                            num_streams_per_tx=self._num_streams_per_tx,
                            # num_streams_per_tx=1,
                            cyclic_prefix_length=self._cyclic_prefix_length,
                            num_guard_carriers=self._num_guard_carriers,
                            dc_null=self._dc_null,
                            pilot_pattern=self._pilot_pattern,
                            pilot_ofdm_symbol_indices=self._srs_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)
        self._n_srs = int(self._rg_srs.num_data_symbols * self._num_bits_per_symbol)
        self._k_srs = int(self._n_srs * self._coderate)

        # Configure antenna arrays
        self._ut_array = AntennaArray(num_rows=int(self._num_ut_ant_row/2),
                                      num_cols=int(self._num_ut_ant_col),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)
        
        # self._ut_array = AntennaArray(
        #                          num_rows=self._num_ut_ant_row,
        #                          num_cols=self._num_ut_ant_col,
        #                          polarization="single",
        #                          polarization_type="V",
        #                          antenna_pattern="omni",
        #                          carrier_frequency=self._carrier_frequency)

        
        self._bs_array = AntennaArray(num_rows=int(self._num_bs_ant_row/2),
                                      num_cols=int(self._num_bs_ant_col),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)


        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=self._channel_add_awgn)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=self._channel_add_awgn)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._qam_source = QAMSource(self._num_bits_per_symbol)
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._encoder_srs = LDPC5GEncoder(self._k_srs, self._n_srs)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)
        self._srs_rg_mapper = ResourceGridMapper(self._rg_srs)

        if self._direction == "downlink":
            self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._ls_est_srs = LSChannelEstimator(self._rg_srs, interpolation_type=None)
        # self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._lmmse_equ = SALSA_Equalizer(self._rg, self._rg_srs, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._decoder_srs = LDPC5GDecoder(self._encoder_srs, hard_out=True)
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)
        self._remove_nulled_scs_srs = RemoveNulledSubcarriers(self._rg_srs)
        
        if self._channel_model is not None:
            self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=self._channel_add_awgn,
                                            normalize_channel=self._normalize_channel, return_channel=True)
            self._ofdm_channel_srs = OFDMChannel(self._channel_model, self._rg_srs, add_awgn=self._channel_add_awgn,
                                            normalize_channel=self._normalize_channel, return_channel=True)
        else:
            self._ofdm_channel = None
            self._ofdm_channel_srs = None
            
        self.ltbf_list = []
        
        

    # @tf.function # @tf.function(jit_compile=False) # Run in graph mode. See the following guide: https://www.tensorflow.org/guide/function
    def call(self, batch_size=1, ebno_db=3, no=None, h=None, H=None, mode="data"):
        
        # self.new_topology(batch_size)

        if mode == "data":
            _rg = self._rg
            _rg_mapper = self._rg_mapper
            _n = self._n
            _k = self._k
            _num_tx = self._num_tx
            _num_streams_per_tx = self._num_streams_per_tx
            _ofdm_channel = self._ofdm_channel
            _encoder = self._encoder
            _decoder = self._decoder
            _ls_est = self._ls_est
            _remove_nulled_scs = self._remove_nulled_scs
        elif mode == "srs":
            _rg = self._rg_srs
            _rg_mapper = self._srs_rg_mapper
            _n = self._n_srs
            _k = self._k_srs
            # _num_tx = 1
            _num_tx = self._num_tx
            # _num_streams_per_tx = 1
            _num_streams_per_tx = self._num_streams_per_tx
            _ofdm_channel = self._ofdm_channel_srs
            _encoder = self._encoder_srs
            _decoder = self._decoder_srs
            _ls_est = self._ls_est_srs
            _remove_nulled_scs = self._remove_nulled_scs_srs
        else:
            raise ValueError("Invalid mode or tx_id, please use 'data' or 'srs' and tx_id < num_tx")
            
            
            
        if no is None:
            no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, _rg)
        else:
            no = tf.cast(no, tf.float32)
        b = self._binary_source([batch_size, _num_tx, _num_streams_per_tx, _k])
        c = _encoder(b)
        x = self._mapper(c)
        x_rg = _rg_mapper(x)


        if self._domain == "time":
            # Time-domain simulations

            if h is not None:
                a, tau = h
            elif self._channel_mode == "cdl":
                a, tau = self._cdl(batch_size, _rg.num_time_samples+self._l_tot-1, _rg.bandwidth)
            elif self._channel_mode == "dataset":
                a, tau = self._channel_model(batch_size, _rg.num_time_samples+self._l_tot-1, _rg.bandwidth)
                
            else:
                raise ValueError("Invalid channel mode. Use 'cdl' or 'dataset'.")
            h_time = cir_to_time_channel(_rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=self._normalize_channel)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[...,_rg.cyclic_prefix_length:-1:(_rg.fft_size+_rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:_rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=self._normalize_channel)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            if h is not None:
                a, tau = h
            elif self._channel_mode == "cdl":
                a, tau = self._cdl(batch_size, _rg.num_ofdm_symbols, 1/_rg.ofdm_symbol_duration)
            elif self._channel_mode == "dataset":
                a, tau = self._channel_model(batch_size, _rg.num_ofdm_symbols, 1/_rg.ofdm_symbol_duration)
                # if mode == "srs" and tx_id is not None:
                #     a = tf.gather(a, tx_id, axis=3)
                #     a = tf.expand_dims(a, axis=3)
                #     tau = tf.gather(tau, tx_id, axis=2)
                #     tau = tf.expand_dims(tau, axis=2)
            else:
                raise ValueError("Invalid channel mode. Use 'cdl' or 'dataset'.")

            if H is not None:
                h_freq = H
            else:
                h_freq = cir_to_ofdm_channel(self._frequencies, a, tau, normalize=self._normalize_channel)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            elif self._channel_mode == "dataset" and _ofdm_channel is not None:
                if mode == "data":
                    y, h_freq = _ofdm_channel(x_rg, no)
                elif mode == "srs":
                    y = self._channel_freq(x_rg, h_freq, no)
            else:
                y = self._channel_freq(x_rg, h_freq, no)


        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = _remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = _ls_est (y, no)            
            
        if mode == "srs":
            h_hat = tf.expand_dims(h_hat, axis=-2)
            err_var = tf.expand_dims(err_var, axis=-2)
            self._lmmse_equ.process_srs(h_hat)


        # print("h_hat: ", h_hat.shape)
        self._lmmse_equ.mode = mode
        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        if mode == "srs":
            return None, None
        llr = self._demapper(x_hat, no_eff)
        b_hat = _decoder(llr)

        self.b = b
        self.x = x
        self.x_rg = x_rg
        self.y = y
        self.x_hat = x_hat
        self.b_hat = b_hat

        self.h_hat = h_hat
        self.h_freq = h_freq
        self.no_eff = no_eff

        return b, b_hat
    






if __name__ == "__main__":
    
    UL_SIMS = {
        "ebno_db" : list(np.arange(-5, 20, 4.0)),
        "cdl_model" : ["A", "B", "E"],
        "delay_spread" : 100e-9,
        "domain" : "freq",
        "direction" : "uplink",
        "perfect_csi" : False,
        "speed" : 0.0,
        "cyclic_prefix_length" : 6,
        "pilot_ofdm_symbol_indices" : [2, 11],
        "ber" : [],
        "bler" : [],
        "duration" : None
    }

    start = time.time()

    for cdl_model in UL_SIMS["cdl_model"]:

        model = MIMO_OFDM(domain=UL_SIMS["domain"],
                    direction=UL_SIMS["direction"],
                    cdl_model=cdl_model,
                    delay_spread=UL_SIMS["delay_spread"],
                    perfect_csi=UL_SIMS["perfect_csi"],
                    speed=UL_SIMS["speed"],
                    cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                    pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])

        ber, bler = sim_ber(model,
                            UL_SIMS["ebno_db"],
                            batch_size=256,
                            max_mc_iter=100,
                            num_target_block_errors=1000,
                            target_bler=1e-3)

        UL_SIMS["ber"].append(list(ber.numpy()))
        UL_SIMS["bler"].append(list(bler.numpy()))

    UL_SIMS["duration"] = time.time() - start









from backend import *
from backend import be_np as np, be_scp as scipy






from sionna.phy.mimo import lmmse_matrix
from sionna.phy import config, dtypes
from sionna.phy.utils import expand_to_rank, matrix_pinv
from sionna.phy.mimo.utils import whiten_channel


def lmmse_equalizer(y, h, s, whiten_interference=True, precision=None):
    # pylint: disable=line-too-long
    r""" MIMO LMMSE Equalizer

    This function implements LMMSE equalization for a MIMO link, assuming the
    following model:

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector.
    It is assumed that :math:`\mathbb{E}\left[\mathbf{x}\right]=\mathbb{E}\left[\mathbf{n}\right]=\mathbf{0}`,
    :math:`\mathbb{E}\left[\mathbf{x}\mathbf{x}^{\mathsf{H}}\right]=\mathbf{I}_K` and
    :math:`\mathbb{E}\left[\mathbf{n}\mathbf{n}^{\mathsf{H}}\right]=\mathbf{S}`.

    The estimated symbol vector :math:`\hat{\mathbf{x}}\in\mathbb{C}^K` is given as
    (Lemma B.19) [BHS2017]_ :

    .. math::

        \hat{\mathbf{x}} = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}\mathbf{G}\mathbf{y}

    where

    .. math::

        \mathbf{G} = \mathbf{H}^{\mathsf{H}} \left(\mathbf{H}\mathbf{H}^{\mathsf{H}} + \mathbf{S}\right)^{-1}.

    This leads to the post-equalized per-symbol model:

    .. math::

        \hat{x}_k = x_k + e_k,\quad k=0,\dots,K-1

    where the variances :math:`\sigma^2_k` of the effective residual noise
    terms :math:`e_k` are given by the diagonal elements of

    .. math::

        \mathop{\text{diag}}\left(\mathbb{E}\left[\mathbf{e}\mathbf{e}^{\mathsf{H}}\right]\right)
        = \mathop{\text{diag}}\left(\mathbf{G}\mathbf{H} \right)^{-1} - \mathbf{I}.

    Note that the scaling by :math:`\mathop{\text{diag}}\left(\mathbf{G}\mathbf{H}\right)^{-1}`
    is important for the :class:`~sionna.phy.mapping.Demapper` although it does
    not change the signal-to-noise ratio.

    The function returns :math:`\hat{\mathbf{x}}` and
    :math:`\boldsymbol{\sigma}^2=\left[\sigma^2_0,\dots, \sigma^2_{K-1}\right]^{\mathsf{T}}`.

    Input
    -----
    y : [...,M], `tf.complex`
        Received signals

    h : [...,M,K], `tf.complex`
        Channel matrices

    s : [...,M,M], `tf.complex`
        Noise covariance matrices

    whiten_interference : `bool`, (default `True`)
        If `True`, the interference is first whitened before equalization.
        In this case, an alternative expression for the receive filter is used that
        can be numerically more stable.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    x_hat : [...,K], `tf.complex`
        Estimated symbol vectors

    no_eff : `tf.float`
        Effective noise variance estimates
    """
    # Cast inputs
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    y = tf.cast(y, dtype=cdtype)
    h = tf.cast(h, dtype=cdtype)
    s = tf.cast(s, dtype=cdtype)

    # We assume the model:
    # y = Hx + n, where E[nn']=S.
    # E[x]=E[n]=0
    #
    # The LMMSE estimate of x is given as:
    # x_hat = diag(GH)^(-1)Gy
    # with G=H'(HH'+S)^(-1).
    #
    # This leads us to the per-symbol model;
    #
    # x_hat_k = x_k + e_k
    #
    # The elements of the residual noise vector e have variance:
    # diag(E[ee']) = diag(GH)^(-1) - I
    if not whiten_interference:
        # Compute equalizer matrix G
        g = lmmse_matrix(h, s, precision=precision)
    else:
        # Whiten channel
        y, h = whiten_channel(y, h, s, return_s=False) # pylint: disable=unbalanced-tuple-unpacking

        # Compute equalizer matrix G
        g = lmmse_matrix(h, s=None, precision=precision)

    # print("g: ", g.shape)
    # print("h: ", h.shape)
    # print("y: ", y.shape)
    
    # Compute G @ y
    y = tf.expand_dims(y, -1)
    gy = tf.squeeze(tf.matmul(g, y), axis=-1)

    # Compute G @ H
    gh = tf.matmul(g, h)

    # Compute diag(G @ H)
    d = tf.linalg.diag_part(gh)

    # Compute x_hat = diag(G @ H)^-1 @ G @ y
    x_hat = gy / d

    # Compute residual error variance
    one = tf.cast(1, dtype=d.dtype)
    no_eff = tf.math.real(one/d - one)

    return x_hat, no_eff





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
        # print("x_hat: ", x_hat.shape)

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
        
        self._resource_grid_srs = resource_grid_srs
        self.mode = "comm"
        
        self.ltbf_list = []
        self.G = None
        
        self.verbose_level = 5
        
        
    def print(self, *args, thr=5):
        if self.verbose_level >= thr:
            print(*args)
        
        
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
        # # non_zero_pilots = tf.boolean_mask(pilots, pilots_indices)
        # non_zero_pilots = tf.gather_nd(pilots, pilots_indices)
        # non_zero_pilots = tf.reshape(non_zero_pilots, [pilots.shape[0], pilots.shape[1], pilots.shape[2], -1])
        
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
        
    
    
    def calc_lt_bf(self, alpha=None):
        
        ltbf_data = tf.concat(self.ltbf_list, axis=3)
        self.print("ltbf_data: ", ltbf_data.shape)
        ltbf_data = tf.reshape(ltbf_data, ltbf_data.shape[0:2]+(np.prod(ltbf_data.shape[2:4]),)+ltbf_data.shape[4:])
        s_hat = self.s
        s_hat = tf.cast(tf.reduce_mean(tf.abs(s_hat), axis=[2,3]), ltbf_data.dtype)
        
        ltbf_data = tf.transpose(ltbf_data, [0,1,4,3,2])
        self.print("ltbf_data: ", ltbf_data.shape)
        n_srs = tf.shape(ltbf_data)[-1]
        self.print("n_srs: ", n_srs)
        Q_hat_ue = tf.matmul(ltbf_data, ltbf_data, adjoint_b=True) / tf.cast(n_srs, ltbf_data.dtype)
        self.print("Q_hat_ue: ", Q_hat_ue.shape)
        alpha = tf.cast(alpha, Q_hat_ue.dtype)
        Q_hat_ue = Q_hat_ue * alpha[None, :, :, None, None]
        Q_hat = tf.reduce_sum(Q_hat_ue, axis=[2])
        self.print("Q_hat: ", Q_hat.shape)
        Q_hat_I = tf.eye(Q_hat.shape[2])
        Q_hat_I = tf.tile(Q_hat_I[tf.newaxis, tf.newaxis, :, :], [Q_hat.shape[0], Q_hat.shape[1], 1, 1])
        Q_hat_I = tf.cast(Q_hat_I, Q_hat.dtype)
        self.print("Q_hat_I: ", Q_hat_I.shape)
        # Q_hat = Q_hat + s_hat
        Q_hat = Q_hat_I + Q_hat
        # Q_hat = alpha * Q_hat
        
        # TODO: Compute Q_inv with a polynomial
        # Q_hat_inv = tf.linalg.inv(Q_hat)
        eigvals, eigvecs = tf.linalg.eigh(Q_hat)
        P = eigvecs @ tf.linalg.diag(1.0 / tf.sqrt(eigvals)) @ tf.linalg.adjoint(eigvecs)
        eigvals, eigvecs = tf.linalg.eigh(Q_hat_ue)
        Q_hat_ue_sqrt = eigvecs @ tf.linalg.diag(tf.sqrt(eigvals)) @ tf.linalg.adjoint(eigvecs)
        self.print("P: ", P.shape)
        self.print("ltbf_data: ", ltbf_data.shape)
        # P_ = P @ P @ Q_hat
        # self.print("P_.shape: ", P_.shape)
        # self.print(tf.reduce_mean(tf.abs(P_ - Q_hat_I)))

        HP = tf.matmul(ltbf_data, P[:,:,None,:,:], adjoint_a=True)
        QP = tf.matmul(Q_hat_ue_sqrt, P[:,:,None,:,:], adjoint_a=True)
        self.print("HP: ", HP.shape)
        self.print("QP: ", QP.shape)

        HP_s, HP_u, HP_v = tf.linalg.svd(HP, full_matrices=False)
        QP_s, QP_u, QP_v = tf.linalg.svd(QP, full_matrices=False)

        self.print("HP_v: ", HP_v.shape)
        self.print("QP_v: ", QP_v.shape)
        n_lt = 32
        # self.F_0 = QH_u[..., :n_lt]
        # self.F_0 = tf.transpose(self.F_0, [0,1,2,4,3])
        # self.G = HP_v[..., :n_lt, :]
        # self.print("G: ", self.G.shape)
        self.G = QP_v[..., :n_lt, :]
        self.print("G: ", self.G.shape)
        self.G = self.G @ P[:,:,None,:,:]
        self.print("G: ", self.G.shape)
        # exit()
        
        self.ltbf_list = []
        
        
    def equalizer(self, y, h, s):
        """Salsa equalizer"""
        
        if self.mode == "srs":
            return self.last_result
        else:
            self.last_result = lmmse_equalizer(y, h, s, self._whiten_interference)
            if self.G is None:
                self.s = s
                x_hat, self.no_eff = self.last_result
                return self.last_result
            
            n_batch = y.shape[0]
            n_rx = y.shape[1]
            n_ofdm_symbols = y.shape[2]
            n_sc = y.shape[3]
            n_rx_ant = y.shape[4]
            n_tx = self.G.shape[2]
            n_lt = self.G.shape[3]
            # self.print(n_batch, n_rx, n_ofdm_symbols, n_sc, n_rx_ant, n_tx, n_lt)
        
            pilots_mask = self._resource_grid.pilot_pattern.mask[0,0]
            self.print("pilots_mask: ", pilots_mask.shape)
            pilots_mask = tf.reshape(pilots_mask, [-1])
            self.print("pilots_mask: ", pilots_mask.shape)
            
            pilot_indices = tf.where(pilots_mask)[:,0]
            n_pilot_symbols = int(pilot_indices.shape[0] / n_sc)
            self.print("n_pilot_symbols: ", n_pilot_symbols)
            self.print("pilot_indices: ", pilot_indices.shape)
            pilots = self._resource_grid.pilot_pattern.pilots
        
            self.print("y: ", y.shape)
            shape = [n_batch, n_rx, n_ofdm_symbols*n_sc, n_rx_ant]
            y_pilots = tf.reshape(y, shape)
            self.print("y_pilots: ", y_pilots.shape)
            y_pilots = tf.gather(y_pilots, pilot_indices, axis=2)
            self.print("y_pilots: ", y_pilots.shape)
            shape = [n_batch, n_rx, n_pilot_symbols, n_sc, n_rx_ant]
            y_pilots = tf.reshape(y_pilots, shape)
            self.print("y_pilots: ", y_pilots.shape)
            # y_ = tf.transpose(y, [0,1,4,2,3])
            # y_pilots = tf.transpose(y_pilots, [0,1,3,2])
            # self.print("y_pilots: ", y_pilots.shape)
            self.print("self.G: ", self.G.shape)
            
            
            # z = tf.einsum('abcde,abemn->abcdmn', F_0, y_)
            z = tf.einsum('abcde,abfge->abcdfg', self.G, y_pilots)
            self.print("z: ", z.shape)
            n_sc_rb = 1
            n_rb = n_sc // n_sc_rb
            n_rs_rb = n_pilot_symbols * n_sc_rb
            self.print("n_sc_rb: ", n_sc_rb)
            shape = [n_batch, n_rx, n_tx, n_lt, n_pilot_symbols, n_rb, n_sc_rb]
            z = tf.reshape(z, shape)
            self.print("z: ", z.shape)
            z = tf.transpose(z, [0,1,2,5,3,4,6])
            self.print("z: ", z.shape)
            z = tf.reshape(z, [n_batch, n_rx, n_tx, n_rb, n_lt, n_rs_rb])
            self.print("z: ", z.shape)
            # test = tf.random.normal(shape=[n_sc])
            # self.print(test[:10])
            # test = tf.reshape(test, [n_rb, n_sc_rb])
            # self.print(test[:2])
            # test = tf.reshape(test, [n_sc])
            # self.print(test[:10])
            
            lambda_reg = 1e-6
            Q_2 = tf.matmul(z, z, adjoint_b=True) + lambda_reg * tf.eye(n_lt, dtype=z.dtype)
            Q_2_inv = tf.linalg.inv(Q_2)
            self.print("Q_2_inv: ", Q_2_inv.shape)
            self.print("pilots: ", pilots.shape)
            n_s = pilots.shape[1]
            X = pilots
            X = tf.tile(X[tf.newaxis, tf.newaxis, :, :, :], [n_batch, n_rx, 1, 1, 1])
            X = tf.reshape(X, [n_batch, n_rx, n_tx, n_s, n_pilot_symbols, n_sc])     # Double check the shape of pilot for different inputs
            self.print("X: ", X.shape)
            X = tf.reshape(X, [n_batch, n_rx, n_tx, n_s, n_pilot_symbols, n_rb, n_sc_rb])
            self.print("X: ", X.shape)
            X = tf.transpose(X, [0,1,2,5,3,4,6])
            self.print("X: ", X.shape)
            X = tf.reshape(X, [n_batch, n_rx, n_tx, n_rb, n_s, n_rs_rb])
            self.print("X: ", X.shape)
            self.F_1 = tf.matmul(z, Q_2_inv, adjoint_a=True)
            self.print("self.F_1: ", self.F_1.shape)
            self.F_1 = tf.matmul(X, self.F_1)
            self.print("self.F_1: ", self.F_1.shape)
            self.print("self.G: ", self.G.shape)
            G = tf.tile(self.G[:,:,:,tf.newaxis,:,:], [1, 1, 1, n_rb, 1, 1])
            self.F = tf.matmul(self.F_1, G)
            self.print("self.F: ", self.F.shape)
            self.print("y: ", y.shape)
            
            F_ = tf.tile(self.F[:,:,:,:,tf.newaxis,:,:], [1, 1, 1, 1, n_sc_rb, 1, 1])
            self.print("F_: ", F_.shape)
            F_ = tf.reshape(F_, [n_batch, n_rx, n_tx, n_sc, n_s, n_rx_ant])
            self.print("F_: ", F_.shape)
            F_ = tf.tile(F_[:,:,:,tf.newaxis,:,:,:], [1, 1, 1, n_ofdm_symbols, 1, 1, 1])
            self.print("F_: ", F_.shape)
            F_ = tf.squeeze(F_, axis=-2)
            self.print("F_: ", F_.shape)
            F_ = tf.reshape(F_, [n_batch, n_rx, n_ofdm_symbols, n_sc, n_tx, n_rx_ant])
            self.print("F_: ", F_.shape)
            # exit()
            # F_ = tf.random.normal(F_.shape)
            # F_ = tf.complex(F_, F_) * 1e6
            # x_hat = tf.einsum('abcdef,abmdf->abmdc', F_, y)
            # self.print("x_hat: ", x_hat.shape)
            self.print("h: ", h.shape)
        
            
            # Compute G @ y
            g = F_
            y = tf.expand_dims(y, -1)
            gy = tf.squeeze(tf.matmul(g, y), axis=-1)

            # Compute G @ H
            gh = tf.matmul(g, h)

            # Compute diag(G @ H)
            d = tf.linalg.diag_part(gh)

            # Compute x_hat = diag(G @ H)^-1 @ G @ y
            # x_hat = gy
            epsilon = tf.constant(1e-10, dtype=d.dtype)
            x_hat = gy / (d + epsilon)
            # Clip x_hat to avoid extreme values
            # x_hat = tf.clip_by_value(x_hat, -1, 1)

            # Compute residual error variance
            one = tf.cast(1, dtype=d.dtype)
            no_eff = tf.math.real(one/d - one)
            # no_eff = self.no_eff
    
    
            return x_hat, no_eff
            
            
            
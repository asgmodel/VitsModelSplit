from transformers.configuration_utils import PretrainedConfig



#.............................................



class VitsConfig(PretrainedConfig):
    model_type = "vits"

    def __init__(
        self,
        vocab_size=38,
        hidden_size=192,
        num_hidden_layers=6,
        num_attention_heads=2,
        window_size=4,
        use_bias=True,
        ffn_dim=768,
        layerdrop=0.1,
        ffn_kernel_size=3,
        flow_size=192,
        spectrogram_bins=513,
        hidden_act="relu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_stochastic_duration_prediction=True,
        num_speakers=1,
        speaker_embedding_size=0,
        upsample_initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        leaky_relu_slope=0.1,
        depth_separable_channels=2,
        depth_separable_num_layers=3,
        duration_predictor_flow_bins=10,
        duration_predictor_tail_bound=5.0,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.5,
        duration_predictor_num_flows=4,
        duration_predictor_filter_channels=256,
        prior_encoder_num_flows=4,
        prior_encoder_num_wavenet_layers=4,
        posterior_encoder_num_wavenet_layers=16,
        wavenet_kernel_size=5,
        wavenet_dilation_rate=1,
        wavenet_dropout=0.0,
        speaking_rate=1.0,
        noise_scale=0.667,
        noise_scale_duration=0.8,
        sampling_rate=16_000,
        discriminator_kernel_size=5,
        discriminator_stride=3,
        discriminator_periods=[2, 3, 5, 7, 11],
        discriminator_period_channels=[1, 32, 128, 512, 1024],
        discriminator_scale_channels=[1, 16, 64, 256, 1024],
        segment_size=8192,
        hop_length=256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_size = window_size
        self.use_bias = use_bias
        self.ffn_dim = ffn_dim
        self.layerdrop = layerdrop
        self.ffn_kernel_size = ffn_kernel_size
        self.flow_size = flow_size
        self.spectrogram_bins = spectrogram_bins
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_stochastic_duration_prediction = use_stochastic_duration_prediction
        self.num_speakers = num_speakers
        self.speaker_embedding_size = speaker_embedding_size
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope
        self.depth_separable_channels = depth_separable_channels
        self.depth_separable_num_layers = depth_separable_num_layers
        self.duration_predictor_flow_bins = duration_predictor_flow_bins
        self.duration_predictor_tail_bound = duration_predictor_tail_bound
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_dropout = duration_predictor_dropout
        self.duration_predictor_num_flows = duration_predictor_num_flows
        self.duration_predictor_filter_channels = duration_predictor_filter_channels
        self.prior_encoder_num_flows = prior_encoder_num_flows
        self.prior_encoder_num_wavenet_layers = prior_encoder_num_wavenet_layers
        self.posterior_encoder_num_wavenet_layers = posterior_encoder_num_wavenet_layers
        self.wavenet_kernel_size = wavenet_kernel_size
        self.wavenet_dilation_rate = wavenet_dilation_rate
        self.wavenet_dropout = wavenet_dropout
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.sampling_rate = sampling_rate

        # used for training
        self.discriminator_kernel_size = discriminator_kernel_size
        self.discriminator_stride = discriminator_stride
        self.discriminator_periods = discriminator_periods
        self.discriminator_period_channels = discriminator_period_channels
        self.discriminator_scale_channels = discriminator_scale_channels
        self.segment_size = segment_size
        self.hop_length = hop_length

        if len(upsample_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                f"The length of `upsample_kernel_sizes` ({len(upsample_kernel_sizes)}) must match the length of "
                f"`upsample_rates` ({len(upsample_rates)})"
            )

        super().__init__(**kwargs)

#.............................................................................................

from transformers import PretrainedConfig


# Update config based on unit testing/debugging scripts
# Second version of the decoder
class LlamaDecoderConfig(PretrainedConfig):
    model_type = "LlamaDecoder"

    def __init__(
        # Set ALL parameters to standard Llama config settings so I don't need to do the debugging step during test time
        self,
        vocab_size=128256, #32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=128001, #0
        bos_token_id=128000, #1
        eos_token_id=128001, #2
        is_decoder=True,
        add_cross_attention=True,
        use_cache=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.use_cache = use_cache
   
from transformers import PretrainedConfig


# Encoder
class LlamaEncoderConfig(PretrainedConfig):
    model_type = 'LlamaEncoder'
    def __init__(self, pooling_mode="mean", max_length=2098, hidden_size=4096, **kwargs):
        super().__init__(**kwargs)
        self.pooling_mode = pooling_mode
        self.max_length = max_length
        self.hidden_size = hidden_size
  
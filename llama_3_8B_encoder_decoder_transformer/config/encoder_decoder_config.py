import copy

from transformers import PretrainedConfig

class EncoderDecoderConfig(PretrainedConfig):
    model_type = "encoder-decoder"

    def __init__(self, encoder_config=None, decoder_config=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder_config
        self.decoder = decoder_config
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config, decoder_config, **kwargs):
        return cls(encoder_config, decoder_config, **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict() if self.encoder is not None else None
        output["decoder"] = self.decoder.to_dict() if self.decoder is not None else None
        output["model_type"] = self.__class__.model_type
        return output

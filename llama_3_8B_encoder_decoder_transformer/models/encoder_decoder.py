import torch
import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes import optim
from peft import PeftModel, prepare_model_for_kbit_training

from transformers import EncoderDecoderConfig, EncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..config.encoder_config import LlamaEncoderConfig
from .config.decoder_config import LlamaDecoderConfig
from .models.encoder import LlamaEncoder
from .models.decoder import LlamaDecoder

import tqdm
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")

print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Encoder model path
encoder_model_path = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

# Create the encoder
print("Creating encoder...")
encoder_config = LlamaEncoderConfig.from_pretrained(encoder_model_path, hidden_size=4096, force_download=True)
encoder = LlamaEncoder(encoder_config)
encoder = encoder.to(device)  # Move the encoder to the proper device
print("Encoder created.")

print("Creating decoder...")
# Create the decoder config
decoder_config = LlamaDecoderConfig(
    vocab_size=128256,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    intermediate_size=11008,
    max_position_embeddings=2048,
    is_decoder=True,
    add_cross_attention=True,
)

# Create the decoder
decoder = LlamaDecoder(decoder_config)
decoder = decoder.to(device)  # Move the decoder to the proper device
print("Decoder created.")


if hasattr(decoder, 'wte'):
    print(f"Decoder embedding size: {decoder.wte.weight.shape[0]}")
    if decoder.wte.weight.shape[0] != len(tokenizer):
        print("Warning: Decoder embedding size doesn't match tokenizer vocabulary size.")
        print("Reinitializing the decoder's word embeddings.")

        old_embedding = decoder.wte
        new_embedding = nn.Embedding(len(tokenizer), old_embedding.embedding_dim)

        nn.init.normal_(new_embedding.weight, mean=0, std=old_embedding.embedding_dim ** -0.5)
        new_embedding.weight.data[:old_embedding.num_embeddings, :] = old_embedding.weight.data

        decoder.wte = new_embedding

        if hasattr(decoder, 'lm_head'):
            old_lm_head = decoder.lm_head
            new_lm_head = nn.Linear(old_lm_head.in_features, len(tokenizer))
            
            nn.init.normal_(new_lm_head.weight, std=0.02)
            nn.init.zeros_(new_lm_head.bias)
            
            new_lm_head.weight.data[:old_lm_head.out_features, :] = old_lm_head.weight.data
            new_lm_head.bias.data[:old_lm_head.out_features] = old_lm_head.bias.data
            
            decoder.lm_head = new_lm_head

        print(f"Updated decoder embedding size: {decoder.wte.weight.shape[0]}")
        if hasattr(decoder, 'lm_head'):
            print(f"Updated lm_head output size: {decoder.lm_head.out_features}")

# Create the EncoderDecoderConfig
encoder_decoder_config = EncoderDecoderConfig(
    encoder=encoder_config.to_dict(),
    decoder=decoder_config.to_dict(),
)

print("Expected Config Class:", EncoderDecoderModel.config_class)
print("Actual Config Class:", type(encoder_decoder_config))

# Create the EncoderDecoderModel
print("Creating encoder-decoder model...")
encoder_decoder_model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
encoder_decoder_model.config.decoder_start_token_id = tokenizer.bos_token_id
encoder_decoder_model.config.pad_token_id = tokenizer.pad_token_id
encoder_decoder_model.config.vocab_size = len(tokenizer)
encoder_decoder_model.config.eos_token_id = tokenizer.eos_token_id
encoder_decoder_model = encoder_decoder_model.to(device)  # Move the entire model to the proper device
print("Encoder-decoder model created.")

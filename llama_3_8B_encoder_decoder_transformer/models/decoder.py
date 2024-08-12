import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from llm2vec import LLM2Vec

import bitsandbytes as bnb
from peft import PeftModel, prepare_model_for_kbit_training

import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, LlamaForCausalLM, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from ..config.decoder_config import LlamaDecoderConfig

import tqdm
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, output_attentions=False, use_cache=False):
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * (self.head_dim ** -0.5)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # New
        if attention_mask is not None:
            # Debug information
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"attention_mask size: {attention_mask.numel()}")
            print(f"expected shape: [{bsz}, {self.num_heads}, {tgt_len}, {src_len}]")

            # Ensure attention_mask has the correct shape
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Expand attention_mask to [bsz, num_heads, tgt_len, src_len]
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            
            # Reshape to [bsz * num_heads, tgt_len, src_len]
            attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
            
            # Convert attention mask to float and replace 0s with -inf and 1s with 0
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_weights.dtype).min

        attn_weights = attn_weights + attention_mask if attention_mask is not None else attn_weights

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.bmm(attn_weights, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (key_states, value_states)
            
        return outputs


class LlamaDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None, 
        encoder_attention_mask=None, 
        use_cache=False, 
        output_attentions=False,
        past_key_value=None
    ):
        # self.to(hidden_states.device)
        # # Unpack hidden_states if it's a tuple
        # if isinstance(hidden_states, tuple):
        #     hidden_states = hidden_states[0]

        # Ensure all inputs are on the same device as the model's parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = next(self.parameters()).device

        # Check if hidden_states is a tuple, and if so, get the first tensor's device
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]  # Extract the tensor from the tuple
            device = hidden_states[0].to(device)
        else:
            device = hidden_states.to(device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
    
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(device)
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)

        # Move the module to the device of the hidden states
        self.to(device)
            
        # Layer normalization and attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Self Attention layer processing
        self_attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self_attn_outputs[0]

        # Add residual connection
        hidden_states = residual + hidden_states

        # Feed-forward network
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Add residual connection
        hidden_states = residual + hidden_states

        # Move the layer back to CPU to save GPU memory
        # self.to('cpu')
        # self.to(device)

        # Collect outputs
        outputs = (hidden_states,)
        if use_cache:
            outputs += (self_attn_outputs[1],)  # Add next cache
        if output_attentions:
            outputs += (self_attn_outputs[1],)  # Add self attention weights
        
        return outputs


@dataclass
class LlamaDecoderModelOutput:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None


class LlamaDecoder(PreTrainedModel): # LlamaPreTrainedModel
    config_class = LlamaDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = True

        # Load the base Llama model
        self.base_model = LlamaForCausalLM.from_pretrained(
            "onyrotssih/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",
        )
        print(f"Initialized LlamaDecoder with base_model: {type(self.base_model)}")

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.h = nn.ModuleList([LlamaDecoderBlock(config) for _ in tqdm(range(config.num_hidden_layers), desc="Initializing Decoder Blocks")])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:])
        return reordered_past

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, head_mask=None, inputs_embeds=None, position_ids=None, use_cache=None, output_attentions=None, token_type_ids=None, output_hidden_states=None, return_dict=None, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move inputs to the appropriate device
        input_ids = input_ids.to(device) if input_ids is not None else None
        hidden_states = inputs_embeds.to(device) if inputs_embeds is not None else None
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        encoder_hidden_states = encoder_hidden_states.to(device) if encoder_hidden_states is not None else None
        encoder_attention_mask = encoder_attention_mask.to(device) if encoder_attention_mask is not None else None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.float16)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(torch.float16)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(torch.float16)
        
        if head_mask is None:
            head_mask = [None] * self.num_layers
        if isinstance(head_mask, list):
            # Convert list to a tensor and expand it to 5D
            if head_mask[0] is None:
                head_mask = torch.ones((self.num_layers, 1, 1, 1, 1), device=input_ids.device)  # Adjust dimensions as needed
            else:
                head_mask = torch.tensor(head_mask, dtype=torch.float32, device=input_ids.device)
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Expand from 2D to 5D
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) 
    
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
    
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.num_layers)
        else:
            past_length = past_key_values[0][0].size(-2)
    
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # New
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
    
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)

        # Debugging: print out the shape of encoder_hidden_states
        if encoder_hidden_states is not None:
            # print("Shape of encoder_hidden_states:", encoder_hidden_states.size())
            pass

        # Debugging: Ensure encoder_hidden_states has three dimensions [batch_size, sequence_length, features]
        if encoder_hidden_states is not None and encoder_hidden_states.dim() == 2:
            # Assuming the second dimension is features, add a sequence length of 1
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            # print("Adjusted Shape of encoder_hidden_states:", encoder_hidden_states.size())
            pass
    
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Ensure position_ids are provided or create them based on input_ids
        if position_ids is None:
            if input_ids is not None:
                # Create position_ids from input_ids: range from 0 to input length
                position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)  # Expand to match batch size
            else:
                # Handle case where neither input_ids nor position_ids are provided
                raise ValueError("You must provide either input_ids or position_ids")
    
        # Prepare inputs
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
    
        output_shape = input_shape + (hidden_states.size(-1),)
    
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # Wherever next_decoder_cache is, that's where presents used to be
        next_decoder_cache = () if use_cache else None

        # Prepare inputs
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        position_embeds = self.wpe(position_ids) if position_ids is not None else 0
        hidden_states = inputs_embeds + position_embeds
    
        hidden_states = self.drop(hidden_states)
  
        for i, layer_module in enumerate(tqdm(self.h, desc="Processing Layers", leave=False)):
            layer_module.to(device)#(cpu_device)
            if output_hidden_states:
                all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states.to(device),
                    attention_mask.to(device) if attention_mask is not None else None,
                    encoder_hidden_states.to(device) if encoder_hidden_states is not None else None,
                    encoder_attention_mask.to(device) if encoder_attention_mask is not None else None,
                    past_key_values, #[i].to(device) if past_key_values is not None else None,
                    use_cache,
                    output_attentions
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_values[i] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # Unpack the layer outputs
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
                
            if use_cache:
                next_decoder_cache += (layer_outputs[1] if len(layer_outputs) > 1 else None,)
                
            if output_attentions:
                all_self_attentions += (layer_outputs[2] if len(layer_outputs) > 2 else None,)

            # Clear GPU cache periodically
            if i % 8 == 0:
                torch.cuda.empty_cache()
    
            hidden_states = layer_outputs[0]
            layer_module.to(device) #.to('cpu') # Just changed
            
            if use_cache is True:
                next_decoder_cache += (layer_outputs[1] if len(layer_outputs) > 1 else None,)
    
            if output_attentions:
                # all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                all_self_attentions += (layer_outputs[2] if len(layer_outputs) > 2 else None,)
                if self.config.add_cross_attention:
                    # all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
                    all_cross_attentions += (layer_outputs[3] if len(layer_outputs) > 3 else None,)
    
        hidden_states = self.ln_f(hidden_states)
        # Add a linear layer to produce logits
        logits = nn.Linear(self.config.hidden_size, self.config.vocab_size).to(hidden_states.device)(hidden_states)
    
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Initialize outputs
        next_decoder_cache = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        # Applying gradient checkpointing if applicable
        if hasattr(self, 'gradient_checkpointing') and self.gradient_checkpointing and self.training:
            if use_cache:
                logging.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

            for i, layer_module in enumerate(self.h):
                hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states)
        else:
            # Regular processing of each layer
            for i, layer_module in enumerate(self.h):
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_values[i] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = next_decoder_cache + (layer_outputs[-1],)  # Cache keys and values

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
        if not return_dict:
            return tuple(v for v in [hidden_states, logits, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
    
        return LlamaDecoderModelOutput(
            last_hidden_state=hidden_states,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
            # logits=hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        input_shape = input_ids.shape
    
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
    
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": kwargs.get("encoder_attention_mask", None),
            "use_cache": kwargs.get("use_cache", True),
        }

# Register the custom configuration and model type
AutoConfig.register("LlamaDecoder", LlamaDecoderConfig)
AutoModelForCausalLM.register(LlamaDecoderConfig, LlamaDecoder)
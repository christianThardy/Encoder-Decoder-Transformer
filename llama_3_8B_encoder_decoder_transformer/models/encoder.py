import torch

from transformers.modeling_outputs import BaseModelOutput
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer

from ..config.encoder_config import LlamaEncoderConfig

from llm2vec import LLM2Vec

import bitsandbytes as bnb

from peft import PeftModel, prepare_model_for_kbit_training

import tqdm
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class LlamaEncoder(PreTrainedModel):
    config_class = LlamaEncoderConfig
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
        )

        self.base_peft_config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", #trust_remote_code=True, 
        )

        self.base_peft_model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            config=self.base_peft_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Apply quantization using bitsandbytes
        self.base_peft_model = self.base_peft_model.half()  # Convert to half precision
        if hasattr(bnb.optim, 'prepare_model_for_kbit_training'):
            self.base_peft_model = bnb.optim.prepare_model_for_kbit_training(self.base_peft_model)
        else:
            print("Function 'prepare_model_for_kbit_training' not found in bitsandbytes.optim")

        self.base_peft_model = PeftModel.from_pretrained(
            self.base_peft_model,
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        )
        self.base_peft_model = self.base_peft_model.merge_and_unload()  # This can take several minutes on cpu

        # Loading unsupervised SimCSE model
        self.base_peft_model = PeftModel.from_pretrained(
            self.base_peft_model, 
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
        )

        self.config = config

        self.model = LLM2Vec(self.base_peft_model, self.tokenizer, pooling_mode=self.config.pooling_mode, max_length=self.config.max_length)
    
    # New forward
    def forward(self, input_ids=None, position_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, return_dict=None, output_attentions=None, output_hidden_states=None, encoder_attention_mask=None, encoder_hidden_states=None):
        # device = input_ids.device if input_ids is not None else (inputs_embeds.device if inputs_embeds is not None else torch.device('cpu'))
        device = input_ids.device if input_ids is not None else torch.device("cpu")
        
        if input_ids is not None:
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif inputs_embeds is not None:
            inputs = {'inputs_embeds': inputs_embeds, 'attention_mask': attention_mask}
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        inputs['embed_mask'] = inputs['attention_mask'].clone()

        with tqdm(total=1, desc="Encoder Forward Pass") as pbar:
            outputs = self.model(inputs)
            pbar.update(1)

        if return_dict:
            return BaseModelOutput(last_hidden_state=outputs)
        return outputs

    def encode(self, input):
        return self.model.encode(input)

# Register the custom configuration and model type
AutoConfig.register("LlamaEncoder", LlamaEncoderConfig)
AutoModel.register(LlamaEncoderConfig, LlamaEncoder)        
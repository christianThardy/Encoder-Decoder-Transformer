from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from .config.encoder_config import LlamaEncoderConfig
from .config.decoder_config import LlamaDecoderConfig
from .models.encoder import LlamaEncoder
from .models.decoder import LlamaDecoder

# Encoder-Decoder model/Dataset needs refactored to be in class for cross file usage
from .models.encoder_decoder import CustomEncoderDecoderModel
from .config.encoder_decoder_config import EncoderDecoderConfig
from .data.dataset import InstructionDataset

# Register the custom configuration and model type
AutoConfig.register("LlamaDecoder", LlamaDecoderConfig)
AutoModelForCausalLM.register(LlamaDecoderConfig, LlamaDecoder)

AutoConfig.register("LlamaEncoder", LlamaEncoderConfig)
AutoModel.register(LlamaEncoderConfig, LlamaEncoder)   


# Ensure the tokenizer is set correctly
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")

# Make sure special tokens are set
tokenizer.pad_token = tokenizer.eos_token
encoder_decoder_model.config.decoder_start_token_id = tokenizer.bos_token_id
encoder_decoder_model.config.pad_token_id = tokenizer.pad_token_id
encoder_decoder_model.config.eos_token_id = tokenizer.eos_token_id

# Check some special tokens to ensure they match. If they don't match,
# update so they match
encoder_decoder_model.config.pad_token_id = tokenizer.pad_token_id
encoder_decoder_model.config.bos_token_id = tokenizer.bos_token_id
encoder_decoder_model.config.eos_token_id = tokenizer.eos_token_id


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4, #4
    per_device_eval_batch_size=4, #4
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    gradient_accumulation_steps=32,
    fp16=True,  # Enable mixed precision training
    fp16_opt_level="O2",  # Use more aggressive mixed precision
    fp16_full_eval=True,
    dataloader_num_workers=4,  # Adjust based on your CPU
    dataloader_pin_memory=True,  # Set to False to reduce GPU memory usage
    optim="adamw_torch",
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=encoder_decoder_model)

print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Model vocabulary size: {encoder_decoder_model.decoder.config.vocab_size}")


trainer = Seq2SeqTrainer(
    model=encoder_decoder_model,
    args=training_args,
    train_dataset=train_dataset, # Instruction dataset should probably be a class as well
    data_collator=data_collator,
    tokenizer=tokenizer
    # Add WANDB tracking
    # log_to_wandb=True,  # always use wandb unless you are just testing code.
    # wandb_project="encoder_decoder_training",
    # wandb_log_frequency=30,
    # eval_every_n_wandb_logs=20,
)
# Manually set the model to use gradient checkpointing
encoder_decoder_model.decoder.gradient_checkpointing = True

trainer.train()

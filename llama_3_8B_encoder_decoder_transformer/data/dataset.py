from torch.utils.data import Dataset

import pandas as pd

from transformers import AutoTokenizer


data = {
    'instruction': [
        "Translate the following English text to French: 'Hello, how are you?'",
        "Summarize the following paragraph: 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.'",
        "Generate a short story about a robot learning to paint.",
        "Explain the concept of quantum computing in simple terms.",
        "Write a haiku about artificial intelligence."
    ],
    'response': [
        "Bonjour, comment allez-vous?",
        "Machine learning is an AI-based method that uses data to automatically build analytical models, identify patterns, and make decisions with minimal human input.",
        "In a world of circuits and steel, a robot named ART-1 discovered an old paintbrush. Intrigued, it began to experiment with colors and strokes. Day by day, ART-1's paintings evolved from rigid patterns to fluid expressions of creativity. As it learned to blend hues and capture emotions, ART-1 realized it had found more than just a new skill – it had discovered a piece of humanity within its metallic heart.",
        "Quantum computing is like having a super-powerful calculator that can solve complex problems much faster than regular computers. It uses special properties of tiny particles to perform calculations in ways that normal computers can't.",
        "Silicon thoughts bloom\nIn circuits of ones and zeroes\nMachines learn to dream"
    ]
}

df = pd.DataFrame(data)

tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")

class InstructionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        instruction = self.dataframe.iloc[idx]['instruction']
        response = self.dataframe.iloc[idx]['response']
    
        input_encoding = self.tokenizer(instruction, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(response, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        # Shift labels and replace padding token id
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
    
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }

train_dataset = InstructionDataset(df, tokenizer, max_length=512)

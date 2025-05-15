import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from pathlib import Path
import logging
from typing import List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptModelTrainer:
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "models",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = {
            'additional_special_tokens': [
                'TITLE:', 'ACT:', 'CHARACTER:', 'DIALOGUE:', 'SCENE:'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_dataset(self, processed_scripts_dir: str) -> Dataset:
        """Prepare dataset from processed scripts."""
        all_texts = []
        
        # Load and combine all processed scripts
        for script_file in Path(processed_scripts_dir).glob("*_processed.json"):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
                
            # Convert tokenized data back to text
            text = self._convert_to_text(script_data)
            all_texts.append(text)
            
        # Create dataset
        dataset = Dataset.from_dict({
            'text': all_texts
        })
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def _convert_to_text(self, script_data: Dict) -> str:
        """Convert processed script data back to text format."""
        text_parts = []
        
        # Add title
        title = self.tokenizer.decode(script_data['title'])
        text_parts.append(f"TITLE: {title}\n")
        
        # Add acts
        for act in script_data['acts']:
            act_text = self.tokenizer.decode(act)
            text_parts.append(f"ACT: {act_text}\n")
            
        # Add characters
        for char in script_data['characters']:
            char_text = self.tokenizer.decode(char)
            text_parts.append(f"CHARACTER: {char_text}\n")
            
        # Add dialogue
        for dialogue in script_data['dialogue']:
            char = self.tokenizer.decode(dialogue['character'])
            text = self.tokenizer.decode(dialogue['text'])
            text_parts.append(f"DIALOGUE: {char}: {text}\n")
            
        # Add scene descriptions
        for scene in script_data['scene_descriptions']:
            scene_text = self.tokenizer.decode(scene)
            text_parts.append(f"SCENE: {scene_text}\n")
            
        return "\n".join(text_parts)
        
    def train(
        self,
        dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ) -> None:
        """Train the model on the prepared dataset."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_strategy="epoch",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            save_total_limit=2,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        self.model.save_pretrained(self.output_dir / "final_model")
        self.tokenizer.save_pretrained(self.output_dir / "final_model")
        logger.info("Training completed and model saved.")
        
    def generate_script(
        self,
        theme: str,
        max_length: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a script based on the given theme."""
        prompt = f"TITLE: Generate a movie script about {theme}\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    # Example usage
    trainer = ScriptModelTrainer()
    dataset = trainer.prepare_dataset("data/processed_scripts")
    trainer.train(dataset)
    
    # Generate a sample script
    theme = "a romantic drama about two people from different backgrounds"
    generated_script = trainer.generate_script(theme)
    print(generated_script) 
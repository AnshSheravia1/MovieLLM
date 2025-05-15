import re
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptProcessor:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def clean_script(self, script_text: str) -> str:
        """Clean script text by removing unnecessary elements."""
        # Remove scene transitions
        script_text = re.sub(r'FADE (IN|OUT|TO BLACK|TO WHITE).*?\n', '', script_text)
        
        # Remove camera directions
        script_text = re.sub(r'\(.*?\)', '', script_text)
        
        # Remove page numbers
        script_text = re.sub(r'\d+\n', '', script_text)
        
        # Remove multiple newlines
        script_text = re.sub(r'\n{3,}', '\n\n', script_text)
        
        return script_text.strip()
        
    def extract_sections(self, script_text: str) -> Dict[str, str]:
        """Extract different sections from the script."""
        sections = {
            'title': '',
            'acts': [],
            'characters': [],
            'dialogue': [],
            'scene_descriptions': []
        }
        
        # Extract title
        title_match = re.search(r'Title:\s*(.*?)(?:\n|$)', script_text)
        if title_match:
            sections['title'] = title_match.group(1).strip()
            
        # Extract acts
        acts = re.findall(r'ACT [IV]+.*?(?=ACT [IV]+|$)', script_text, re.DOTALL)
        sections['acts'] = [act.strip() for act in acts]
        
        # Extract character names
        character_pattern = r'^([A-Z][A-Z\s]+)(?=\n)'
        characters = re.findall(character_pattern, script_text, re.MULTILINE)
        sections['characters'] = list(set(characters))
        
        # Extract dialogue
        dialogue_pattern = r'([A-Z][A-Z\s]+)\n(.*?)(?=\n[A-Z][A-Z\s]+|\n\n|$)'
        dialogues = re.findall(dialogue_pattern, script_text, re.DOTALL)
        sections['dialogue'] = [{'character': d[0].strip(), 'text': d[1].strip()} for d in dialogues]
        
        # Extract scene descriptions
        scene_pattern = r'INT\.|EXT\.|INT\./EXT\.'
        scenes = re.split(scene_pattern, script_text)
        sections['scene_descriptions'] = [scene.strip() for scene in scenes if scene.strip()]
        
        return sections
        
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=True)
        
    def process_script(self, script_path: Path, output_dir: Path) -> bool:
        """Process a single script file."""
        try:
            # Read script
            with open(script_path, 'r', encoding='utf-8') as f:
                script_text = f.read()
                
            # Clean script
            cleaned_text = self.clean_script(script_text)
            
            # Extract sections
            sections = self.extract_sections(cleaned_text)
            
            # Tokenize sections
            tokenized_sections = {
                'title': self.tokenize_text(sections['title']),
                'acts': [self.tokenize_text(act) for act in sections['acts']],
                'characters': [self.tokenize_text(char) for char in sections['characters']],
                'dialogue': [
                    {
                        'character': self.tokenize_text(d['character']),
                        'text': self.tokenize_text(d['text'])
                    }
                    for d in sections['dialogue']
                ],
                'scene_descriptions': [
                    self.tokenize_text(desc) for desc in sections['scene_descriptions']
                ]
            }
            
            # Save processed data
            output_path = output_dir / f"{script_path.stem}_processed.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tokenized_sections, f, indent=2)
                
            logger.info(f"Processed script: {script_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing script {script_path.name}: {str(e)}")
            return False
            
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """Process all scripts in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for script_file in input_path.glob("*.txt"):
            self.process_script(script_file, output_path)

if __name__ == "__main__":
    # Example usage
    processor = ScriptProcessor()
    processor.process_directory("data/raw_scripts", "data/processed_scripts") 
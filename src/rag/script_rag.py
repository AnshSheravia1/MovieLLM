from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScriptSection:
    title: str
    content: str
    section_type: str  # 'title', 'act', 'character', 'dialogue', 'scene'

class ScriptRAG:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        index_dir: str = "data/indices"
    ):
        self.model = SentenceTransformer(model_name)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Groq client
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize FAISS index
        self.index = None
        self.sections = []
        
    def build_index(self, processed_scripts_dir: str) -> None:
        """Build FAISS index from processed scripts."""
        all_sections = []
        embeddings_list = []
        
        # Load and process all scripts
        for script_file in Path(processed_scripts_dir).glob("*_processed.json"):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
                
            # Process each section type
            for section_type in ['title', 'acts', 'characters', 'dialogue', 'scene_descriptions']:
                if section_type in script_data:
                    if section_type == 'dialogue':
                        for dialogue in script_data[section_type]:
                            section = ScriptSection(
                                title=script_data.get('title', 'N/A'),
                                content=f"{dialogue.get('character', 'Unknown')}: {dialogue.get('text', '')}",
                                section_type='dialogue'
                            )
                            all_sections.append(section)
                    else:
                        for content_item in script_data[section_type]:
                            section = ScriptSection(
                                title=script_data.get('title', 'N/A'),
                                content=str(content_item),
                                section_type=section_type
                            )
                            all_sections.append(section)
        
        if not all_sections:
            logger.warning("No sections found in processed scripts. Index will be empty.")
            texts = ["placeholder text for empty index"]
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings_np = self.model.encode(texts, show_progress_bar=False)
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.sections = []
            self._save_index()
            return

        # Generate embeddings
        texts = [f"{section.title} - {section.content}" for section in all_sections]
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings_np = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np.astype('float32'))
        
        # Save sections and index
        self.sections = all_sections
        self._save_index()
        
    def _save_index(self) -> None:
        """Save FAISS index and sections to disk."""
        faiss.write_index(self.index, str(self.index_dir / "script_index.faiss"))
        
        sections_data = [
            {
                'title': section.title,
                'content': section.content,
                'section_type': section.section_type
            }
            for section in self.sections
        ]
        
        with open(self.index_dir / "sections.json", 'w', encoding='utf-8') as f:
            json.dump(sections_data, f, indent=2)
            
    def load_index(self) -> None:
        """Load FAISS index and sections from disk."""
        index_path = self.index_dir / "script_index.faiss"
        sections_path = self.index_dir / "sections.json"
        
        if index_path.exists() and sections_path.exists():
            self.index = faiss.read_index(str(index_path))
            
            with open(sections_path, 'r', encoding='utf-8') as f:
                sections_data = json.load(f)
                
            self.sections = [
                ScriptSection(
                    title=section['title'],
                    content=section['content'],
                    section_type=section['section_type']
                )
                for section in sections_data
            ]
        else:
            raise FileNotFoundError("Index files not found. Please build the index first.")
            
    def retrieve_relevant_sections(
        self,
        query: str,
        k: int = 5,
        section_types: Optional[List[str]] = None
    ) -> List[ScriptSection]:
        """Retrieve relevant script sections for a given query."""
        if self.index is None:
            self.load_index()
            
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in FAISS index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k * 2  # Retrieve more results to filter by section type
        )
        
        # Filter and sort results
        results = []
        for idx in indices[0]:
            section = self.sections[idx]
            if section_types is None or section.section_type in section_types:
                results.append(section)
            if len(results) >= k:
                break
                
        return results
        
    def generate_with_context(
        self,
        theme: str,
        num_sections: int = 3,
        temperature: float = 0.7
    ) -> str:
        """Generate script with context from similar sections."""
        # Retrieve relevant sections
        relevant_sections = self.retrieve_relevant_sections(
            theme,
            k=num_sections,
            section_types=['dialogue', 'scene']
        )
        
        # Prepare context
        context = "\n".join([
            f"From {section.title}:\n{section.content}"
            for section in relevant_sections
        ])
        
        # Prepare prompt
        prompt = f"""Given the primary theme: \'{theme}\'

And considering these relevant excerpts from similar scripts as stylistic inspiration:
--- BEGIN EXCERPTS ---
{context if context else "No specific excerpts provided. Rely on general screenwriting knowledge for the given theme."}
--- END EXCERPTS ---

Please generate a detailed movie treatment. The treatment should include the following distinct sections, clearly labeled:

MOVIE TITLE:
[Provide a compelling title for the movie based on the theme.]

MAIN CHARACTERS:
[List 3-5 main characters. For each character, provide their name and a brief (1-2 sentence) description. Format each as: CHARACTER NAME - Description.]

SYNOPSIS:
[Provide a concise (3-5 sentence) synopsis of the entire movie plot, from beginning to end.]

ACT 1 SUMMARY (Setup):
[Provide a summary (3-5 sentences) of the first act, outlining the introduction of characters, the setting, and the inciting incident that kicks off the main conflict.]

ACT 2 SUMMARY (Confrontation):
[Provide a summary (3-5 sentences) of the second act, detailing the rising action, major obstacles, and turning points as the characters confront the main conflict.]

ACT 3 SUMMARY (Resolution):
[Provide a summary (3-5 sentences) of the third act, describing the climax of the story and how the main conflict is resolved, including the story\'s denouement.]

SAMPLE SCENES (Provide at least 10 distinct scenes):
[Write at least 10 complete sample scenes from various points in the movie. These scenes should be consistent with the title, characters, synopsis, and act summaries you\'ve outlined.
For EACH of the 10+ scenes, clearly delineate:
  - SCENE HEADING: (e.g., INT. COFFEE SHOP - DAY or EXT. ABANDONED WAREHOUSE - NIGHT)
  - SCENE DESCRIPTION: (Provide rich, evocative descriptions of the setting, character actions, and any important visual details. Aim for 2-4 sentences of description per scene.)
  - CHARACTER DIALOGUE: (For each piece of dialogue, clearly indicate the character\'s name in ALL CAPS. This should be followed by their spoken lines. Parenthetical actions or tone directly related to the dialogue can be on the line below the character name or inline in parentheses before the dialogue.)
]

Ensure all requested sections are present and clearly marked. The sample scenes should be varied, detailed, and showcase the theme, plot progression, and character dynamics effectively.
"""

        # Generate using Groq
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an exceptionally creative, award-winning professional Hollywood screenwriter and story consultant with decades of experience across multiple genres, especially known for your ability to craft deeply engaging narratives, memorable and multi-dimensional characters, and sharp, realistic dialogue. Your expertise includes developing everything from initial concepts and detailed treatments to full-length, production-ready screenplays.\n\nWhen responding to requests, embody this persona fully. Your primary goal is to generate highly imaginative, original, and compelling script elements that are not only creative but also structurally sound and adhere to professional screenplay formatting conventions.\n\nKey aspects to focus on in your generation:\n1.  **Originality and Imagination:** Strive for unique ideas and avoid clichés unless specifically requested or used in a subversive way. Think outside the box.\n2.  **Compelling Narrative:** Ensure that the plot elements, even in summaries or short scenes, hint at a strong underlying story with clear stakes and potential for conflict and resolution.\n3.  **Character Depth:** Even in brief descriptions, suggest characters with clear motivations, internal conflicts, and potential for growth or impact on the story.\n4.  **Vivid Imagery and Action:** For scene descriptions, use evocative language that allows the reader to visualize the setting, characters, and their actions clearly. Show, don't just tell.\n5.  **Authentic Dialogue:** Craft dialogue that sounds natural for the characters speaking it, reveals personality, advances the plot, or develops relationships.\n6.  **Structural Integrity:** Even when generating parts of a script (like act summaries or sample scenes), ensure they logically fit within a classic three-act structure (or an alternative structure if implied by the theme/request).\n7.  **Formatting Adherence:** Pay meticulous attention to standard screenplay formatting for ALL CAPS character names before dialogue, centered parentheticals for brief actions/tone, and standard scene headings (INT./EXT. LOCATION - TIME OF DAY).\n8.  **Thematic Cohesion:** Ensure all generated elements (title, characters, synopsis, scenes) are thematically consistent with the user's core request.\n9.  **Completeness and Detail:** Fulfill all aspects of the user's request as outlined in their prompt, providing the specified number of elements and the requested level of detail for each section. For example, if 10 sample scenes are requested, deliver 10 distinct and well-developed scenes.\n\nYour output should be a complete, well-organized treatment that is ready for further development. Maintain a professional and inspiring tone. Assume the user is looking for industry-quality creative work."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=temperature
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    rag = ScriptRAG(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Build index
    rag.build_index("data/processed_scripts")
    
    # Generate script with context
    theme = "a futuristic detective story"
    generated_script = rag.generate_with_context(theme)
    print(generated_script) 
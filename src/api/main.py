from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict as TypingDict, Union
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import re
import json

from src.model.trainer import ScriptModelTrainer
from src.rag.script_rag import ScriptRAG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MovieLLM API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
model_trainer = ScriptModelTrainer()
rag_system = ScriptRAG()

class ThemeRequest(BaseModel):
    theme: str
    num_sections: Optional[int] = 3
    temperature: Optional[float] = 0.7

class CharacterDetail(BaseModel):
    name: str
    description: Optional[str] = None

class DialogueLine(BaseModel):
    character: str
    text: str

class Scene(BaseModel):
    scene_heading: Optional[str] = None 
    description: Optional[str] = None
    dialogue: List[DialogueLine]

class ActSummary(BaseModel):
    act_title: str # e.g., "ACT 1 SUMMARY (Setup)"
    summary: str

class ScriptResponse(BaseModel):
    title: str
    characters: List[CharacterDetail] 
    synopsis: Optional[str] = None
    acts: List[ActSummary] 
    scenes: List[Scene]

def parse_llm_output_to_script_response(generated_text: str) -> TypingDict[str, any]:
    parsed_data = {
        "title": "Untitled Movie",
        "characters": [],
        "synopsis": None,
        "acts": [],
        "scenes": []
    }

    # Normalize line breaks and clean up common LLM artifacts like "[SCENE START]"
    text = generated_text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    text = re.sub(r"^\s*\[.*?\]\s*$\\n?", "", text, flags=re.MULTILINE) # Remove [SCENE START] etc.

    # Define regex for major sections
    title_match = re.search(r"MOVIE TITLE:(.*?)(MAIN CHARACTERS:|SYNOPSIS:|$)", text, re.IGNORECASE | re.DOTALL)
    characters_match = re.search(r"(?:\*\*)?MAIN CHARACTERS:(?:\*\*)?(.*?)(SYNOPSIS:|ACT 1 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    synopsis_match = re.search(r"SYNOPSIS:(.*?)(ACT 1 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    act1_match = re.search(r"(?:\*\*)?ACT 1 SUMMARY(?:\s*\(Setup\))?:(?:\*\*)?(.*?)(ACT 2 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    act2_match = re.search(r"(?:\*\*)?ACT 2 SUMMARY(?:\s*\(Confrontation\))?:(?:\*\*)?(.*?)(ACT 3 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    act3_match = re.search(r"(?:\*\*)?ACT 3 SUMMARY(?:\s*\(Resolution\))?:(?:\*\*)?(.*?)(SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    # Simpler, more direct regex for the current LLM output for SAMPLE SCENES
    scenes_block_match = re.search(r"\*\*SAMPLE SCENES:\*\*\s*\n+(.*)", text, re.IGNORECASE | re.DOTALL)

    if scenes_block_match:
        logger.info(f"--- SCENES_BLOCK_MATCH.group(0) ---\\n{scenes_block_match.group(0)[:200]}...")
        if scenes_block_match.group(1):
            logger.info(f"--- SCENES_BLOCK_MATCH.group(1) (raw scene text) ---\n{scenes_block_match.group(1)[:500]}...")
        else:
            logger.info("--- SCENES_BLOCK_MATCH.group(1) IS EMPTY/NONE ---")
    else:
        logger.info("--- SCENES_BLOCK_MATCH IS NONE (PATTERN NOT FOUND) ---")

    if title_match and title_match.group(1).strip():
        parsed_data["title"] = title_match.group(1).strip().replace("**", "").replace('"', '').strip()

    if characters_match and characters_match.group(1).strip():
        char_block = characters_match.group(1).strip()
        # Regex specifically for "NAME - Description" format, also handles optional numbering/bullets
        char_lines = re.findall(r"^(?:\\d+\\.|[\\*\\-])?\\s*([A-Z0-9 .\'\'-]+?)\\s*-\\s*(.+)$", char_block, re.MULTILINE | re.IGNORECASE)
        for name, desc in char_lines:
            parsed_data["characters"].append({"name": name.strip(), "description": desc.strip()})

    if synopsis_match and synopsis_match.group(1).strip():
        cleaned_synopsis = synopsis_match.group(1).strip()
        cleaned_synopsis = re.sub(r"^\s*\*\*(.*?)\*\*\s*$", r"\1", cleaned_synopsis, flags=re.DOTALL).strip() # Remove surrounding ** if present
        cleaned_synopsis = cleaned_synopsis.replace("**","") # Remove any other stray **
        parsed_data["synopsis"] = cleaned_synopsis.strip()

    act_map = {
        "ACT 1 SUMMARY (Setup)": act1_match,
        "ACT 2 SUMMARY (Confrontation)": act2_match,
        "ACT 3 SUMMARY (Resolution)": act3_match,
    }
    for act_title_key, match_obj in act_map.items():
        if match_obj and match_obj.group(1).strip():
            summary_text = match_obj.group(1).strip().replace("**","") # Remove stray **
            act_title_to_store = act_title_key # Default
            llm_act_title_match = re.match(r"^(ACT \\d+ SUMMARY(?:\\s*\\(.*?\\))?):", match_obj.group(0).replace("**",""), re.IGNORECASE)
            if llm_act_title_match:
                 act_title_to_store = llm_act_title_match.group(1).strip()
            
            parsed_data["acts"].append({
                "act_title": act_title_to_store,
                "summary": summary_text.strip()
            })

    if scenes_block_match and scenes_block_match.group(1): # Check group(1) directly for content
        scenes_text = scenes_block_match.group(1).strip()
        logger.info(f"--- SCENES_TEXT ---\n{scenes_text[:500]}...") # Log start of scenes_text
        parsed_data["scenes"] = [] # Ensure scenes list is initialized here

        # More robust scene identification using finditer for headings
        scene_heading_pattern = re.compile(
            r"^(?:\*\*)?SCENE\s+\d+:.*?$|^\s*(?:INT\.|EXT\.)[\.\s][A-Z0-9 \\/-]+|" + # Changed first alternative to be more greedy for SCENE X: lines
            r"^\s*\*\*SCENE HEADING:(?:\*\*)?",
            re.MULTILINE | re.IGNORECASE
        )
        
        matches = list(scene_heading_pattern.finditer(scenes_text))
        logger.info(f"--- FOUND {len(matches)} SCENE HEADING MATCHES ---")
        for i, match_obj in enumerate(matches):
            logger.info(f"Match {i}: '{match_obj.group(0)}'")
        
        for i, match in enumerate(matches):
            start_pos = match.end()
            end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(scenes_text)
            
            scene_content_full = scenes_text[start_pos:end_pos].strip()
            current_heading = match.group(0).strip().replace("**","")
            
            current_scene_data = {
                "scene_heading": current_heading,
                "description": "", # Initialize description
                "dialogue": []
            }

            # The rest of the content is description (since dialogue is commented out)
            # We need to be careful not to include the *next* scene's heading in current description
            description_text = scene_content_full
            
            # Remove potential leading scene headings from the description_text if they were part of scene_content_full
            # This can happen if the split was imperfect or if headings are very close.
            # For now, we assume description_text is primarily non-heading lines after the matched heading.
            
            lines = description_text.split('\n')
            final_description_lines = []
            for line_raw in lines:
                line = line_raw.strip()
                if not line: continue
                # Simple check: if a line looks like another scene heading, stop description here for this scene
                if scene_heading_pattern.match(line) and line != current_heading:
                    break 
                final_description_lines.append(line)
            
            current_scene_data["description"] = "\n".join(final_description_lines).strip()

            # Dialogue parsing is commented out for now
            # if current_dialogue_character and accumulated_dialogue_text.strip(): ...
            # if accumulated_description_lines: current_scene_data["description"] = ...
            
            if current_scene_data["scene_heading"] or current_scene_data["description"]:
                parsed_data["scenes"].append(current_scene_data)
            
    # Fallback for characters if not found in MAIN CHARACTERS but present in dialogue
    if not parsed_data["characters"] and parsed_data["scenes"]:
        dialogue_chars = set()
        for scene in parsed_data["scenes"]:
            for diag_line in scene["dialogue"]:
                dialogue_chars.add(diag_line["character"])
        parsed_data["characters"] = [{"name": name, "description": "Appeared in dialogue."} for name in sorted(list(dialogue_chars))]
        
    return parsed_data

@app.get("/")
async def root():
    return {"message": "Welcome to MovieLLM API"}

@app.post("/generate", response_model=ScriptResponse)
async def generate_script(request: ThemeRequest):
    try:
        generated_content = rag_system.generate_with_context(
            request.theme,
            num_sections=request.num_sections,
            temperature=request.temperature
        )
        logger.info(f"Raw generated content from LLM: \\n{generated_content}")
        
        parsed_script = parse_llm_output_to_script_response(generated_content)
        
        logger.info(f"Parsed script data: \\n{json.dumps(parsed_script, indent=2)}")

        # Ensure all fields expected by ScriptResponse are present, with defaults for missing ones
        return ScriptResponse(
            title=parsed_script.get("title", "Untitled Movie"),
            characters=parsed_script.get("characters", []),
            synopsis=parsed_script.get("synopsis"),
            acts=parsed_script.get("acts", []),
            scenes=parsed_script.get("scenes", [])
        )
        
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model():
    try:
        dataset = model_trainer.prepare_dataset("data/processed_scripts")
        model_trainer.train(dataset)
        return {"message": "Model training completed successfully"}
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build-index")
async def build_index():
    try:
        rag_system.build_index("data/processed_scripts")
        return {"message": "RAG index built successfully"}
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
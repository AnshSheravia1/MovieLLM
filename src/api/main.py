from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Any, Optional, Dict as TypingDict, Union
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



def parse_llm_output_to_script_response(generated_text: str) -> TypingDict[str, Any]:
    parsed = {
        "title": "Untitled Movie",
        "characters": [],
        "synopsis": None,
        "acts": [],
        "scenes": []
    }

    # Normalize newlines
    text = generated_text.replace("\\r\\n", "\n").replace("\\r", "\n")

    # TITLE
    m = re.search(r"MOVIE TITLE:\s*""?(.*?)""?\s*(?:MAIN CHARACTERS:|SYNOPSIS:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        parsed["title"] = m.group(1).strip().strip('"')

    # CHARACTERS
    m = re.search(r"MAIN CHARACTERS:(.*?)(?:SYNOPSIS:|ACT 1 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        for line in m.group(1).strip().splitlines():
            line = line.strip().lstrip("*- ")
            if not line: continue
            parts = line.split("-", 1)
            name = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ""
            parsed["characters"].append({"name": name, "description": desc})

    # SYNOPSIS
    m = re.search(r"SYNOPSIS:(.*?)(?:ACT 1 SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        parsed["synopsis"] = m.group(1).strip()

    # ACTS
    for i in range(1, 4):
        m = re.search(rf"ACT {i} SUMMARY.*?:\s*(.*?)(?=ACT {i+1} SUMMARY|SAMPLE SCENES:|$)", text, re.IGNORECASE | re.DOTALL)
        if m:
            parsed["acts"].append({
                "act_title": f"ACT {i} SUMMARY",
                "summary": m.group(1).strip()
            })

    # SCENES — strictly split only on “SCENE <number>:”
    scene_block = re.search(r'SAMPLE SCENES[:\s]*(.*)', text, re.IGNORECASE | re.DOTALL)
    if scene_block:
        scenes_text = scene_block.group(1).strip()
        # 1) split only on "SCENE <digits>:" markers (keeping the marker)
        parts = re.split(r'(?i)(?=(?:SCENE\s+\d+:))', scenes_text)
        for part in parts:
            part = part.strip()
            if not part or not re.match(r'(?i)^SCENE\s+\d+:', part):
                continue   # ignore anything not starting with "SCENE X:"
            lines = part.splitlines()
            heading = lines[0].strip()            # e.g. "SCENE 3:"
            body   = lines[1:]                     # everything after heading
            desc_lines, dialogues = [], []
            current_char, msgs = None, []

            for ln in body:
                ln = ln.strip()
                if not ln:
                    continue
                # if it exactly matches uppercase+digits and ends with "- DAY"/"- NIGHT", treat as description line
                if re.match(r'^(?:INT\.|EXT\.)', ln, re.IGNORECASE):
                    desc_lines.append(ln)
                    continue
                # if it’s an uppercase line with ≤ 3 words, treat as speaker
                if re.fullmatch(r'[A-Z0-9 \'\-\.]+', ln) and len(ln.split()) <= 3:
                    if current_char and msgs:
                        dialogues.append({
                            "character": current_char,
                            "text": ' '.join(msgs).strip()
                        })
                    current_char, msgs = ln, []
                else:
                    if current_char:
                        msgs.append(ln)
                    else:
                        desc_lines.append(ln)

            # flush last dialogue
            if current_char and msgs:
                dialogues.append({
                    "character": current_char,
                    "text": ' '.join(msgs).strip()
                })

            parsed["scenes"].append({
                "scene_heading": heading,
                "description": ' '.join(desc_lines).strip() or None,
                "dialogue": dialogues
            })

    # Fallback characters from dialogues
    if not parsed["characters"]:
        chars = {d["character"] for s in parsed["scenes"] for d in s["dialogue"]}
        parsed["characters"] = [{"name": c, "description": ""} for c in sorted(chars)]

    return parsed


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
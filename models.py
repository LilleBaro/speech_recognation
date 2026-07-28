from pydantic import BaseModel, Field

class Post(BaseModel):
    transcription: str = Field(min_length=3, max_length=256)
    sentiment: str = Field(min_length=3, max_length=256)
    confidence_score: int 

from pydantic import BaseModel, ConfigDict
from typing import Optional

class CategoryRequestBody(BaseModel):
  take: int
  
class ClusterRequestBody(BaseModel):
  take: int

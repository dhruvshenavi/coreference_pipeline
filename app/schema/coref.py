from pydantic import BaseModel, HttpUrl

class Article(BaseModel): # the schema for the input request for coreference api. change this according to orchestration method

    content: str | None = None
    url: HttpUrl
    

class Coref_Article(BaseModel): # the schema for the input request for preprocessing api. change this according to orchestration method

    content: str | None = None
    url: HttpUrl
    chains: list | None = None # this will hold the coreference chains in string format. change this according to orchestration method   
    

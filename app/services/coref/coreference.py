from app.schema.coref import Coref_Article, Article
from app.services.coref.coreferee_code import resolve_text, build_nlp
import spacy
import coreferee
import re

def ready_data(data: Article) -> Coref_Article:
    chains, resolved_text = resolve_text(data.content, build_nlp())

    # Convert spans into JSON-friendly format
    json_chains = []
    for spans in chains:
        chain_list = []
        for sp in spans:
            chain_list.append({
                "text": sp.text,
                "start": sp.start,
                "end": sp.end
            })
        json_chains.append(chain_list)

    # Now safe to create response
    data_out = Coref_Article(
        content=resolved_text,
        url=data.url,
        chains=json_chains
    )
    return data_out




import rdflib
import difflib
import pandas as pd
from src.bot.message_processor import AttributeRecognizer, MessageCleaner, MessageDecomposer





class QueryGenerator:
    def __init__(self):
        pass

    def generate_query(self, message_output):
        # Estrai entità e relazioni dall'output
        entities = message_output.get('entities', {})
        pos_tags = message_output.get('pos_tags', {})

        # Controlla se mancano entità o relazioni
        if not entities:
            return "No entity recognized"
        if not pos_tags:
            return "No relation recognized"

        # Prende la prima entità e la prima relazione
        entity_id_full, entity_label = list(entities.items())[0]  # es. 'http://www.wikidata.org/entity/Q47703', 'The Godfather'
        relation_id_full, relation_label = list(pos_tags.items())[0]   # es. 'http://www.wikidata.org/prop/direct/P57', 'director'

        # Estrarre solo l'identificatore dalla relazione (es. P57) e dall'entità (es. Q47703)
        entity_id = entity_id_full.split('/')[-1]
        relation_id = relation_id_full.split('/')[-1]

        # Converte il label della relazione in camel case
        relation_label = self._to_camel_case(relation_label)

        # Caso 1: Se la relazione è associata a un dato letterale (es. data, numero)
        if relation_id in ['P577', 'P2142', 'P345']:  # Esempi di proprietà letterali come 'publication date' o 'box office'
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?{relation_label} WHERE {{
                ?entity rdfs:label "{entity_label}"@en .
                ?entity wdt:{relation_id} ?{relation_label} .
            }}
            """
            
        # Caso 2: Se la relazione è "node description"
        elif relation_label == "nodeDescription":
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX schema: <http://schema.org/>

            SELECT ?description WHERE {{
                ?entity rdfs:label "{entity_label}"@en .
                ?entity schema:description ?description .
            }}
            """
            
            
        # Caso 3: Relazioni standard con entità collegate
        else:
            query = f"""
            PREFIX ddis: <http://ddis.ch/atai/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>

            SELECT ?{relation_label} WHERE {{
                ?entity rdfs:label "{entity_label}"@en .
                ?entity wdt:{relation_id} ?{relation_label}Item .
                ?{relation_label}Item rdfs:label ?{relation_label} .
            }}
        """  
            
        
        return query

    def _to_camel_case(self, label):
        # Trasforma la stringa in camel case
        words = label.split(' ')
        camel_case_label = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        return camel_case_label
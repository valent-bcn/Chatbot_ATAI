import difflib
import rdflib
from message_processor import AttributeRecognizer
class QueryGenerator:
    def __init__(self, df_relations):
        self.df_relations = df_relations
        self.recognizer = AttributeRecognizer('relations_expanded.csv')

    def generate_query(self, message_output):
        entities = message_output.get('entities', {})
        pos_tags = message_output.get('pos_tags', {})

        if not entities or not pos_tags:
            return None  # Se non ci sono entità o verbi, non possiamo generare una query.

        # Prendiamo la prima entità e il primo verbo trovati
        entity_text, entity_type = list(entities.items())[0]  # es. 'Titanic', 'MOV'
        action_text, action_tag = list(pos_tags.items())[0]   # es. 'directed', 'VBD'

        # Troviamo la relazione più plausibile
        relation, relation_label = self._map_action_to_relation(action_text)
        if not relation:
            return None  # Se non troviamo una relazione, non possiamo generare la query.

        # Trasformiamo il relation_label in camel case
        relation_label = self._to_camel_case(relation_label)

        # Debug: Stampa l'ID della relazione e il nome della relazione
        print(f"Relation ID: {relation}, Relation Label: {relation_label}")
        print(f"Entity Text: {entity_text}")

        query = f"""
        PREFIX ddis: <http://ddis.ch/atai/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX schema: <http://schema.org/>

        SELECT ?{relation_label} WHERE {{
            ?person rdfs:label "{entity_text}"@en .
            ?person wdt:{relation} ?{relation_label}Item .
            ?{relation_label}Item rdfs:label ?{relation_label} .
        }}
        
        Limit 1
        """
        return query

    def _to_camel_case(self, label):
        # Trasforma la stringa in camel case, partendo dalla prima parola in minuscolo e capitalizzando le altre
        words = label.split(' ')
        camel_case_label = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        return camel_case_label


    def _map_action_to_relation(self, action):
        # Cerchiamo le relazioni che sono linguisticamente simili al verbo dell'azione
        possible_relations = difflib.get_close_matches(action, self.df_relations['Label'], n=1, cutoff=0.6)
        
        if possible_relations:
            relation_row = self.df_relations[self.df_relations['Label'] == possible_relations[0]]
            if not relation_row.empty:
                # Otteniamo solo il codice della relazione (es. P57) senza il prefisso completo
                relation_id = relation_row['ID'].values[0].split('/')[-1]  # estrae solo P57 dall'URI
                relation_label = relation_row['Label'].values[0]
                return relation_id, relation_label
        return None, None

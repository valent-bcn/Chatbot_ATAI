from flair.data import Sentence
from flair.models import SequenceTagger
import re
import pandas as pd
from flashtext import KeywordProcessor
from rapidfuzz import process, fuzz
import os
import random

from src.bot.query_generator import QueryGenerator
from src.bot.sparql_queries import SPARQLQuerySolver

FILM_PATH = 'dataset/films_clean.csv'

class MessageCleaner:
    def __init__(self):
        self.wh_pattern = re.compile(r'\b(who|what|when|where)\b', re.IGNORECASE)

    def clean(self, message: str) -> str:
        # Sostituisce ogni occorrenza di '-' con '–'
        message = message.replace('-', '–')
        
        match = self.wh_pattern.search(message)
        if match:
            start_index = match.start()
            return message[start_index:].rstrip('?').strip()
        
        return message
    def cleanResponse(self, response: str) -> str:
        return response.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace("'", '')
    
class AttributeRecognizer:
    def __init__(self):
        data = pd.read_csv('dataset/relations.csv')
        self.relations_dict = {row['Label']: row['ID'] for _, row in data.iterrows()}
        
        # Configura KeywordProcessor per trovare il match perfetto più lungo
        self.keyword_processor = KeywordProcessor()
        for label in self.relations_dict.keys():
            self.keyword_processor.add_keyword(label)

    def recognize(self, recognized_labels: str) -> tuple:
        # Cerca match perfetti usando KeywordProcessor per ottenere il match più lungo
        perfect_matches = self.keyword_processor.extract_keywords(recognized_labels)
        if perfect_matches:
            longest_match = max(perfect_matches, key=len)
            return self.relations_dict[longest_match], longest_match
        
        # Se non c'è un match perfetto, cerca il match più plausibile
        best_match_label, score, _ = process.extractOne(
            recognized_labels, list(self.relations_dict.keys()), scorer=fuzz.WRatio
        )
        
        # Restituisce il match più plausibile se trovato
        return (self.relations_dict[best_match_label], best_match_label) if best_match_label else (None, None)
    

class MessageDecomposer:
    def __init__(self):
        film_path = FILM_PATH
        humans_path = 'dataset/humans_clean.csv'
        entities_path = 'dataset/entities_clean.csv'
        #relations_path = 'dataset/relations.csv'

        self.cleaner = MessageCleaner()
        self.ner_tagger = SequenceTagger.load('ner')
        self.relations_recognizer = AttributeRecognizer()

        # Carica i dataset
        self.film_dataset = pd.read_csv(film_path)
        self.humans_dataset = pd.read_csv(humans_path)
        self.entities_dataset = pd.read_csv(entities_path)

        self.keyword_processor = KeywordProcessor()
        for film in self.film_dataset['Label']:
            self.keyword_processor.add_keyword(film)

        self.keyword_processor2 = KeywordProcessor()
        for human in self.humans_dataset['Label']:
            self.keyword_processor2.add_keyword(human)

        self.keyword_processor3 = KeywordProcessor()
        for entity in self.entities_dataset['Label']:
            self.keyword_processor3.add_keyword(entity)

    def _find_entity(self, message: str) -> tuple:
        # Cerca un film
        film_matches = self.keyword_processor.extract_keywords(message)
        if film_matches:
            longest_match = max(film_matches, key=len)
            entity_row = self.film_dataset.loc[self.film_dataset['Label'] == longest_match].iloc[0] #if we want to take the first one that matches
            modified_message = message.replace(longest_match, 'AAA') #AAA as a safe placeholder, consider putting this as a global class variable
            return modified_message, {entity_row['ID']: longest_match}, "movie"

        # Cerca un umano
        human_matches = self.keyword_processor2.extract_keywords(message)
        if human_matches:
            longest_match = max(human_matches, key=len)
            entity_row = self.humans_dataset.loc[self.humans_dataset['Label'] == longest_match].iloc[0]
            modified_message = message.replace(longest_match, 'AAA')
            return modified_message, {entity_row['ID']: longest_match}, "human"

        # Cerca un'entità generica
        entity_matches = self.keyword_processor3.extract_keywords(message)
        if entity_matches:
            longest_match = max(entity_matches, key=len)
            entity_row = self.entities_dataset.loc[self.entities_dataset['Label'] == longest_match].iloc[0]
            modified_message = message.replace(longest_match, 'AAA')
            return modified_message, {entity_row['ID']: longest_match}, "general"

        # Cerca il miglior match plausibile
        all_entities = self.entities_dataset['Label'].tolist()
        best_match = process.extractOne(message, all_entities, scorer=fuzz.ratio)
        if best_match and best_match[1] >= 90:  # Definisce un criterio di alta somiglianza
            matched_label = best_match[0]  # Etichetta trovata nel dataset, non l'input
            if matched_label in self.film_dataset['Label'].values:
                entity_row = self.film_dataset.loc[self.film_dataset['Label'] == matched_label].iloc[0]
                modified_message = message.replace(matched_label, 'AAA')
                return modified_message, {entity_row['ID']: matched_label}, "movie"
            elif matched_label in self.humans_dataset['Label'].values:
                entity_row = self.humans_dataset.loc[self.humans_dataset['Label'] == matched_label].iloc[0]
                modified_message = message.replace(matched_label, 'AAA')
                return modified_message, {entity_row['ID']: matched_label}, "human"
            else:
                entity_row = self.entities_dataset.loc[self.entities_dataset['Label'] == matched_label].iloc[0]
                modified_message = message.replace(matched_label, 'AAA')
                return modified_message, {entity_row['ID']: matched_label}, "general"

        return message, {}, None

    def decompose(self, message: str) -> dict:
        cleaned_message = self.cleaner.clean(message)
        sentence = Sentence(cleaned_message)

        # Cerca l'entità nei film, umani o entità generiche e sostituisci con "AAA" se match perfetto o plausibile
        modified_message, entities, entity_type = self._find_entity(cleaned_message)

        # Se viene trovata un'entità
        if entities:
            # Controllo per relazioni specifiche
            if "when" in cleaned_message.lower():
                publication_date_relation = process.extractOne("publication date", list(self.relations_recognizer.relations_dict.keys()), scorer=fuzz.WRatio)
                if publication_date_relation:
                    relation_id = self.relations_recognizer.relations_dict[publication_date_relation[0]]
                    return {
                        'pos_tags': {relation_id: publication_date_relation[0]},
                        'entities': entities,
                        'type': entity_type
                    }

            # Controllo per "cast member"
            actor_terms = ["actors", "acted", "act", "actor", "acts", "actors", "casted", "casts", "cast"]
            if any(term in cleaned_message.lower() for term in actor_terms):
                cast_member_relation = process.extractOne("cast member", list(self.relations_recognizer.relations_dict.keys()), scorer=fuzz.WRatio)
                if cast_member_relation:
                    relation_id = self.relations_recognizer.relations_dict[cast_member_relation[0]]
                    return {
                        'pos_tags': {relation_id: cast_member_relation[0]},
                        'entities': entities,
                        'type': entity_type
                    }

            # Controllo per "node description"
            if re.search(r'\b(is|are)\s*AAA\b', modified_message, re.IGNORECASE):
                node_description_relation = process.extractOne("node description", list(self.relations_recognizer.relations_dict.keys()), scorer=fuzz.WRatio)
                if node_description_relation:
                    relation_id = self.relations_recognizer.relations_dict[node_description_relation[0]]
                    return {
                        'pos_tags': {relation_id: node_description_relation[0]},
                        'entities': entities,
                        'type': entity_type
                    }

            # Cerca altre relazioni normalmente
            relation_id, relation_label = self.relations_recognizer.recognize(modified_message)
            pos_tags = {relation_id: relation_label} if relation_id else {}
            return {
                'pos_tags': pos_tags,
                'entities': entities,
                'type': entity_type
            }

        # Fallback NER solo se nessun film, umano o entità generica è stato trovato
        self.ner_tagger.predict(sentence)
        ner_entities = sentence.get_spans('ner')
        ner_dict = {}
        for ent in ner_entities:
            entity_match = process.extractOne(ent.text, self.entities_dataset['Label'].tolist(), scorer=fuzz.ratio)
            if entity_match and entity_match[1] >= 90:
                entity_id = self.entities_dataset.loc[self.entities_dataset['Label'] == entity_match[0], 'ID'].values[0]
                ner_dict[entity_id] = entity_match[0]

        # Cerca la relazione nella frase completa
        relation_id, relation_label = self.relations_recognizer.recognize(cleaned_message)
        pos_tags = {relation_id: relation_label} if relation_id else {}
        print(f"NER: ", ner_dict.items())
        return {
            'pos_tags': pos_tags,
            'entities': ner_dict,
            'type': "general"
        }

class MessageComposer:
    def __init__(self, SPARQLQuerySolver, EmbeddingResolver, QueryGenerator):
        self.sparqlsolver = SPARQLQuerySolver
        self.embbsolver = EmbeddingResolver
        self.query_generator = QueryGenerator
        self.film_dataset = pd.read_csv(FILM_PATH)
        #self.keyword_processor = KeywordProcessor()

    def find_id_labels(self, label):
        entity_row = self.film_dataset.loc[self.film_dataset['Label'] == label]
        print(entity_row)
        return {entity_row['ID']: label for _, entity_row in entity_row.iterrows()}

    def compose(self, messagedecomposed):
        # Check if the label of the entity of the messagedecomposed is multiple
        print(messagedecomposed['entities'].items())
        id_labels = self.find_id_labels(messagedecomposed['entities'][list(messagedecomposed['entities'].keys())[0]])
        if len(id_labels) > 1:
            message_result = "According to our knowledge graph, there are multiple entities that you could be referring to. The answers are:"
            for id, label in id_labels.items():
                messagedecomposed['entities'] = {id: label}
                # Generate the SPARQL query
                sparql_query = self.query_generator.generate_query(messagedecomposed)
                # Get the result from the knowledge graph
                kg_result = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else "No details found"
                # Get the node info, due that there's different IDs with the same label, we need to get the node info for each ID
                # relation_label == "nodeDescription"
                messagedecomposed['pos_tags'] = {id, "nodeDescription"} # I should pass anyway the relation_id, but 'nodeDescription' suffices to call the relation
                sparql_query = self.query_generator.generate_query(messagedecomposed)
                node_info = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else None
                # Append the result
                message_result += f"\nFor {node_info}, the found result is {kg_result}"
            return message_result


        # Generate the SPARQL query, assuming that there's only one entity valid
        sparql_query = self.query_generator.generate_query(messagedecomposed)

        # Ger either the result from the graph and the result from the embeddings
        kg_result = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else None  # "No query generated"
        print(kg_result)
        embedding_result = self.embbsolver.find_most_plausible_responses(messagedecomposed, top_n=3)

        # Now we need to differentiate, if the kg_result is empty then we return the embedding result, and viceversa
        if not kg_result:
            return f"According to our embeddings, the answer to your question is {embedding_result[0]}. However these are also some plausible answers {', '.join(embedding_result[1:])}."
        elif len(kg_result) > 8: # We give a special response in case the kg_result has a high number of responses.
            return f"There're {len(kg_result)} found results to your question, the top 3 are {kg_result[:3]}."
        elif embedding_result: # find a way to combine the results, by sets.
            set_kg = set(kg_result)
            set_embedding = set(embedding_result)
            intersection = set_kg.intersection(set_embedding)
            if intersection:
                return f"The answer to your question is {intersection}."
            else: # Then we give either the kg_result or the embedding_result, with a 50 50 chance
                return f"According to our knowledge graph, the answer to your question is {kg_result}." \
                    if random.choice([True, False]) else f"According to our embeddings, " \
                                                         f"the answer to your question is {embedding_result[0]}. " \
                                                         f"However these are also some plausible answers {', '.join(embedding_result[1:])}."
        else:
            return f"According to our knowledge graph, the answer to your question is {kg_result}."


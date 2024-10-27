from flair.data import Sentence
from flair.models import SequenceTagger
import re
import pandas as pd
from flashtext import KeywordProcessor
from rapidfuzz import process, fuzz
import os

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
    
class AttributeRecognizer:
    def __init__(self, relations_path):
        data = pd.read_csv(relations_path)
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
        film_path = 'dataset/films_clean.csv'
        humans_path = 'dataset/humans_clean.csv'
        entities_path = 'dataset/entities_clean.csv'
        relations_path = 'dataset/relations.csv'

        self.cleaner = MessageCleaner()
        self.ner_tagger = SequenceTagger.load('ner')
        self.relations_recognizer = AttributeRecognizer(relations_path)

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
            entity_row = self.film_dataset.loc[self.film_dataset['Label'] == longest_match].iloc[0]
            modified_message = message.replace(longest_match, 'AAA')
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
            actor_terms = ["actors", "acted", "act", "actor", "acts", "actors"]
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

        return {
            'pos_tags': pos_tags,
            'entities': ner_dict,
            'type': "general"
        }

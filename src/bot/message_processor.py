from flair.data import Sentence
from flair.models import SequenceTagger
import re
import pandas as pd
from flashtext import KeywordProcessor
from rapidfuzz import process, fuzz
import os
import random
import copy

FILM_PATH = 'dataset/films_clean.csv'


class DecomposedData:
    def __init__(self, relations: dict, entities: dict):
        self.data = {
            'relations': relations,
            'entities': entities,
        }

    def set_relations(self, relation_dict):
        """Replace the entire relations dictionary."""
        self.data['relations'] = relation_dict
        return self

    def update_relations(self, relation_dict):
        """Update the relations dictionary with provided key-value pairs."""
        self.data['relations'].update(relation_dict)
        return self

    def set_entities(self, entity_dict):
        """Replace the entire entities dictionary."""
        self.data['entities'] = entity_dict
        return self

    def update_entities(self, entity_dict):
        """Update the entities dictionary with provided key-value pairs."""
        self.data['entities'].update(entity_dict)
        return self

    def display(self):
        print("Relations:", self.data['relations'])
        print("Entities:", self.data['entities'])
        print("Type:", self.data['type'])


class MessageCleaner:
    def __init__(self):
        self.wh_pattern = re.compile(r'\b(who|what|when|where)\b', re.IGNORECASE)

    def clean(self, message: str) -> str:
        # Sostituisce ogni occorrenza di '-' con '–'
        message = message.replace(' - ', '–')

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
        relations_path = 'dataset/relations.csv'

        #Data Class
        self.decomposed_data = DecomposedData({}, {})

        self.cleaner = MessageCleaner()

        self.ner_tagger = SequenceTagger.load('ner')
        self.relations_recognizer = AttributeRecognizer()

        # Carica i dataset
        self.film_dataset = pd.read_csv(film_path)
        self.humans_dataset = pd.read_csv(humans_path)
        self.entities_dataset = pd.read_csv(entities_path)
        self.film_double_dataset = pd.read_csv('dataset/film_double.csv')

        self.keyword_processor = KeywordProcessor()
        for film in self.film_dataset['Label']:
            self.keyword_processor.add_keyword(film)

        self.keyword_processor2 = KeywordProcessor()
        for human in self.humans_dataset['Label']:
            self.keyword_processor2.add_keyword(human)

        self.keyword_processor3 = KeywordProcessor()
        for entity in self.entities_dataset['Label']:
            self.keyword_processor3.add_keyword(entity)

    # Clean the message decomposed
    def _clean_decomposed(self):
        self.decomposed_data = DecomposedData({}, {})
    def _find_entity(self, message: str) -> tuple:
        modified_message = message
        entities = {}

        # Funzione interna per cercare e sostituire entità in modo iterativo
        def search_and_replace(keyword_processor, dataset, message):
            found_any = False
            while True:
                matches = keyword_processor.extract_keywords(message)
                if not matches:
                    break  # Esci dal ciclo se non ci sono più match
                longest_match = max(matches, key=len)  # Estrai il match più lungo
                entity_row = dataset.loc[dataset['Label'] == longest_match].iloc[0]
                message = message.replace(longest_match, 'AAA', 1)  # Sostituisci solo la prima occorrenza
                entities[entity_row['ID']] = longest_match
                found_any = True
            return message, found_any

        # Processa film
        modified_message, found_film = search_and_replace(self.keyword_processor, self.film_dataset, modified_message)

        # Processa umani solo se non sono stati trovati film
        found_human = None
        if not found_film:
            modified_message, found_human = search_and_replace(self.keyword_processor2, self.humans_dataset,
                                                               modified_message)

        # Processa entità generiche solo se non sono state trovate entità nei passaggi precedenti
        if not found_film and not found_human:
            modified_message, _ = search_and_replace(self.keyword_processor3, self.entities_dataset, modified_message)

        film_labels = self.film_dataset['Label'].tolist()
        fuzzy_matches = process.extract(modified_message, film_labels, scorer=fuzz.token_set_ratio, limit=None,
                                        score_cutoff=80)

        for fuzzy_match in fuzzy_matches:
            match_label, score = fuzzy_match[0], fuzzy_match[1]
            if score >= 90:  # Soglia per accettare match plausibili
                entity_row = self.film_dataset.loc[self.film_dataset['Label'] == match_label].iloc[0]
                modified_message = modified_message.replace(match_label, 'AAA', 1)
                entities[entity_row['ID']] = match_label

        return modified_message, entities


    def _find_related_films(self, film_label):
        # Trova tutti i film con lo stesso label in film_double
        related_films = self.film_double_dataset[self.film_double_dataset['Label'] == film_label]
        return {row['ID']: row['Label'] for _, row in related_films.iterrows()}

    def decompose(self, message: str) -> DecomposedData:
        # First we clean the decomposed data, in order to avoid any previous data
        self._clean_decomposed()
        cleaned_message = self.cleaner.clean(message)
        sentence = Sentence(cleaned_message)

        # Cerca l'entità nei film, umani o entità generiche e sostituisci con "AAA" se match perfetto o plausibile
        modified_message, entities = self._find_entity(cleaned_message)
        print(modified_message)

        # Se viene trovata un'entità
        if entities:
            # Controllo per relazioni specifiche
            if "when" in cleaned_message.lower():
                publication_date_relation = process.extractOne("publication date",
                                                               list(self.relations_recognizer.relations_dict.keys()),
                                                               scorer=fuzz.WRatio)
                if publication_date_relation:
                    relation_id = self.relations_recognizer.relations_dict[publication_date_relation[0]]
                    self.decomposed_data.set_relations({relation_id: publication_date_relation[0]}).set_entities(entities)
                    return self.decomposed_data # TODO: Else??

            # Controllo per "cast member"
            actor_terms = ["actors", "acted", "act", "actor", "acts", "actors", "casted", "casts", "cast"]
            if any(term in cleaned_message.lower() for term in actor_terms):
                cast_member_relation = process.extractOne("cast member",
                                                          list(self.relations_recognizer.relations_dict.keys()),
                                                          scorer=fuzz.WRatio)
                if cast_member_relation:
                    relation_id = self.relations_recognizer.relations_dict[cast_member_relation[0]]
                    return self.decomposed_data.set_relations({relation_id: cast_member_relation[0]}).set_entities(entities)
            # Definisci termini relativi a "screenwriter"
            writer_terms = [" write ", " writes ", " written ", " wrote ", " writer ", " screenwriter ",
                            " scriptwriter ", " scripting ", " screenwrited "]
            # Controlla la presenza di questi termini nel messaggio
            if any(term in cleaned_message.lower() for term in writer_terms):
                # Cerca la corrispondenza con la relazione "screenwriter"
                screenwriter_relation = process.extractOne("screenwriter",
                                                           list(self.relations_recognizer.relations_dict.keys()),
                                                           scorer=fuzz.WRatio)
                if screenwriter_relation:
                    relation_id = self.relations_recognizer.relations_dict[screenwriter_relation[0]]
                    return self.decomposed_data.set_relations({relation_id: screenwriter_relation[0]}).set_entities(entities)

            recommendation_terms = [" recommend ", " recommend ", " recommendation ", " recommended ", " recommends ",
                                    " similar ", " like ", " suggest ", " suggestion ", " suggested ", " recommend ",
                                    " recommends ", ]
            if any(term in modified_message.lower() for term in recommendation_terms):
                # Crea una copia temporanea del dizionario per l'iterazione
                temp_entities = entities.copy()
                # Dopo aver identificato le entità, verifica se hanno corrispondenti in film_double
                for entity_id, label in temp_entities.items():
                    if label in self.film_double_dataset['Label'].values:
                        related_entities = self._find_related_films(label)
                        entities.update(related_entities)  # Aggiungi le entità correlate senza duplicare

                return self.decomposed_data.set_entities(entities).set_relations({})

            # Controllo per "node description"
            if re.search(r'\b(is|are)\s*AAA\b', modified_message, re.IGNORECASE):
                node_description_relation = process.extractOne("node description", list(self.relations_recognizer.relations_dict.keys()), scorer=fuzz.WRatio)
                if node_description_relation:
                    relation_id = self.relations_recognizer.relations_dict[node_description_relation[0]]
                    return self.decomposed_data.set_relations({relation_id: node_description_relation[0]}).set_entities(entities)

            # Cerca altre relazioni normalmente
            relation_id, relation_label = self.relations_recognizer.recognize(modified_message)
            relations = {relation_id: relation_label} if relation_id else {}
            return self.decomposed_data.set_relations(relations).set_entities(entities)

        # Fallback NER solo se nessun film, umano o entità generica è stato trovato
        self.ner_tagger.predict(sentence)
        ner_entities = sentence.get_spans('ner')
        ner_dict = {}
        for ent in ner_entities:
            entity_match = process.extractOne(ent.text, self.entities_dataset['Label'].tolist(), scorer=fuzz.ratio)
            if entity_match and entity_match[1] >= 90:
                entity_id = \
                self.entities_dataset.loc[self.entities_dataset['Label'] == entity_match[0], 'ID'].values[0]
                ner_dict[entity_id] = entity_match[0]

        # Cerca la relazione nella frase completa
        relation_id, relation_label = self.relations_recognizer.recognize(cleaned_message)
        relations = {relation_id: relation_label} if relation_id else {}
        print(f"NER: ", ner_dict.items())
        return self.decomposed_data.set_relations(relations).set_entities(ner_dict)

class MessageComposer:
    def __init__(self, SPARQLQuerySolver, EmbeddingResolver, QueryGenerator, RecommendationSolver):
        self.sparqlsolver = SPARQLQuerySolver
        self.embbsolver = EmbeddingResolver
        self.query_generator = QueryGenerator
        self.film_dataset = pd.read_csv(FILM_PATH)
        self.recommsolver = RecommendationSolver

    def find_id_labels(self, label):
        entity_row = self.film_dataset.loc[self.film_dataset['Label'] == label]
        print(entity_row)
        return {entity_row['ID']: label for _, entity_row in entity_row.iterrows()}

    def compose(self, messagedecomposed: DecomposedData):
        decomposed = messagedecomposed.data.copy()
        # Assuming you are retrieving the first entity ID from the 'entities' dictionary
        entity_labels = list(decomposed['entities'].values())
        if not decomposed['relations']: # If there are no relations, we assume is a Recommendation
            recommendation_dict = self.recommsolver.process_recommendation_direct(decomposed['entities'])
            # Give the node description for each entity of the recommendation dict
            message_result = f"Here's a list of recommendations that may interest you: \n"
            for id, label in recommendation_dict.items():
                local_dict = {'entities': {id: label}, 'relations': {'': 'node description'}}
                sparql_query = self.query_generator.generate_query(local_dict)
                node_info = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else None

                message_result += f"\n- {label} ({node_info}) \n"
            #We dont need to provide the id, only the label, and the node info
            return message_result
        elif entity_labels:
            first_entity_label = entity_labels[0]
            id_labels = self.find_id_labels(first_entity_label)

            if len(id_labels) > 1:
                message_result = "According to our knowledge graph, there are multiple entities that you could be referring to, the answers are:"
                for id, label in id_labels.items():
                    local_dict = decomposed.copy()
                    local_dict['entities'] = {id: label}
                    # Generate the SPARQL query
                    sparql_query = self.query_generator.generate_query(local_dict)
                    # Get the result from the knowledge graph
                    kg_result = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else "No details found"
                    # Get the node info
                    local_dict['relations'] = {'': "node description"}
                    sparql_query = self.query_generator.generate_query(local_dict)
                    node_info = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else None
                    # Append the result if found, if not append a message
                    message_result += f"\nFor {label} ({node_info}), the found result is {kg_result}" if kg_result else f"\nFor {label} ({node_info}), no details found."
                return message_result

        # Similar changes should be made to the rest of your method as needed

        # Generate the SPARQL query, assuming that there's only one entity valid
        sparql_query = self.query_generator.generate_query(decomposed)

        # Ger either the result from the graph and the result from the embeddings
        kg_result = self.sparqlsolver.solveQuery(sparql_query) if sparql_query else None  # "No query generated"
        embedding_result = self.embbsolver.find_most_plausible_responses(decomposed, top_n=3)
        print(kg_result, embedding_result)
        if not kg_result and not embedding_result:
            return "Oh :-( No results found. Please try again with a different question. \
            Check maybe the spelling of the Film or Person you are looking for :)"
        # Now we need to differentiate, if the kg_result is empty then we return the embedding result, and viceversa
        if not kg_result:
            return f"According to our embeddings, the answer to your question is {embedding_result[0]}. However these are also some plausible answers {', '.join(embedding_result[1:])}."
        elif len(kg_result) > 8: # We give a special response in case the kg_result has a high number of responses.
            return f"There are {len(kg_result)} found results to your question, the top 3 are {kg_result[:3]}."
        elif embedding_result: # find a way to combine the results, by sets.
            set_kg = set(kg_result)
            set_embedding = set(embedding_result)
            intersection = set_kg.intersection(set_embedding)
            if intersection:
                return f"The answer to your question is {intersection}."
            else:
                return f"Our information systems don't give us a single answer to your question:"\
                       f"\nConsidering the knowedge graph the answer to your question is {kg_result}. "\
                       f"\nHowever, our embeddings suggest that the answer to your question is {embedding_result[0]}. " \
                                                         f"Here are also some plausible answers {', '.join(embedding_result[1:])}."
        else:
            return f"According to our knowledge graph, the answer to your question is {kg_result}."


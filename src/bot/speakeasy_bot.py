from speakeasypy import Speakeasy
import time
import re
from src.bot.sparql_queries import SPARQLQuerySolver  # Importa il solver delle query SPARQL
from src.bot.message_processor import MessageDecomposer, MessageComposer
from src.bot.query_generator import QueryGenerator
from embeddings.embeddings import EmbeddingResolver  # Importa l'EmbeddingResolver

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password):
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.solver = SPARQLQuerySolver()  # Solver per le query SPARQL
        self.message_decomposer = MessageDecomposer()  # Inizializza il decompositore di messaggi
        self.query_generator = QueryGenerator()
        self.embedding_resolver = EmbeddingResolver()  # Inizializza l'EmbeddingResolver
        self.message_composer = MessageComposer(self.solver, self.embedding_resolver, self.query_generator)

        self.speakeasy.login()

    def listen(self):
        while True:
            rooms = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new message #{message.ordinal}: '{message.message}'")
                    response = self.process_message(message.message)
                    room.post_messages(response.encode('utf-8').decode('latin-1'))
                    room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new reaction #{reaction.message_ordinal}: '{reaction.type}'")
                    room.post_messages(f"Received your reaction: '{reaction.type}'")
                    room.mark_as_processed(reaction)

    def process_message(self, message):
        message = message.strip()
        
        if self.is_sparql_query(message):
            # Risolvi la query direttamente se è una query SPARQL già formata
            result = self.solver.solveQuery(message)
            return f"It looks like you're trying to solve a SPARQL query.\nHere's the result: {result}" if result else "No results found."
        
        elif self.is_factual_question(message):
            # 1. Decostruisci il messaggio con MessageDecomposer
            message_decomposed = self.message_decomposer.decompose(message)

            # 2. Construisci il messagio dada la decomposizione
            response = self.message_composer.compose(message_decomposed).replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace("'", '')
            return response #TODO: consider doing all the cleaning also by a method of the class MessageCleaner
        else:
            return "Thanks for your message, we are processing your request."

    def is_sparql_query(self, message: str) -> bool:
        sparql_patterns = [r"\bPREFIX\b", r"\bSELECT\b", r"\bWHERE\b", r"\bORDER BY\b", r"\bLIMIT\b", r"\bASK\b", r"\bCONSTRUCT\b", r"\bDESCRIBE\b"]
        return any(re.search(pattern, message, re.IGNORECASE) for pattern in sparql_patterns)

    def is_factual_question(self, message: str) -> bool:
        factual_keywords = ["who", "what", "where", "when", "how", "why"]
        return any(keyword.lower() in message.lower() for keyword in factual_keywords)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())

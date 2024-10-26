from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
import re
from src.bot.sparql_queries import SPARQLQuerySolver  # Importa il solver delle query SPARQL
from src.bot.message_processor import MessageCleaner, MessageDecomposer
from src.bot.query_generator import QueryGenerator
import pandas as pd
import os


DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2


class Agent:
    def __init__(self, username, password):
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.solver = SPARQLQuerySolver()  # Solver per le query SPARQL
        self.message_decomposer = MessageDecomposer()  # Inizializza il decompositore di messaggi
        self.query_generator = QueryGenerator()

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
            return f"I see it's a SPARQL query. Here is the result: {result}"
        
        elif self.is_factual_question(message):
            # 1. Decostruisci il messaggio con MessageDecomposer
            message_output = self.message_decomposer.decompose(message)

            # 2. Genera la query SPARQL usando il QueryGenerator
            sparql_query = self.query_generator.generate_query(message_output)
            
            if sparql_query:
                # 3. Risolvi la query usando SPARQLQuerySolver
                result = self.solver.solveQuery(sparql_query)
                return f"I think that it is {result}"
            else:
                return "I couldn't generate a query for this question."
        
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

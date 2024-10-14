from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
import re
from src.bot.sparql_queries import SPARQLQuerySolver  # Importa il solver delle query SPARQL

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

class Agent:
    def __init__(self, username, password):
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.solver = SPARQLQuerySolver()  # Inizializza l'oggetto che gestisce le query SPARQL
        self.speakeasy.login()

    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new message #{message.ordinal}: '{message.message}' - {self.get_time()}")
                    response = self.process_message(message.message)
                    room.post_messages(response.encode('utf-8').decode('latin-1'))
                    room.mark_as_processed(message)

                for reaction in room.get_reactions(only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new reaction #{reaction.message_ordinal}: '{reaction.type}' - {self.get_time()}")
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

    def process_message(self, message):
        message = message.strip()
        if self.is_sparql_query(message):
            result = self.solver.solveQuery(message)
            return f"I see it's a SPARQL query. Here is the result: {result}"
        elif self.is_factual_question(message):
            return "I understand it's a factual question. Our team is working on the response."
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

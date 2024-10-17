from flair.data import Sentence
from flair.models import SequenceTagger
import re

class MessageCleaner:
    def __init__(self):
        # Compile a regex that matches the beginning of the interrogative part of the question
        self.wh_pattern = re.compile(r'\b(who|what|when|where)\b', re.IGNORECASE)

    def clean(self, message: str) -> str:
        # Find the position where the first "wh" word occurs
        match = self.wh_pattern.search(message)
        if match:
            # Extract the substring from the "wh" word to the end, excluding the final "?"
            start_index = match.start()
            cleaned_message = message[start_index:].rstrip('?').strip()
            return cleaned_message
        return message  # Return the original message if no "wh" word is found
class MessageDecomposer:
    def __init__(self):
        self.cleaner = MessageCleaner()
        self.ner_tagger = SequenceTagger.load('ner')
        self.pos_tagger = SequenceTagger.load('pos-fast')  # Load POS tagging model

    def decompose(self, message: str) -> dict:
        cleaned_message = self.cleaner.clean(message)
        sentence = Sentence(cleaned_message)

        # Predict NER to identify and segregate entities
        self.ner_tagger.predict(sentence)
        ner_entities = sentence.get_spans('ner')

        # Create a set of character spans to exclude (these cover the entity spans)
        exclude_indices = set()
        for ent in ner_entities:
            for token in ent.tokens:
                exclude_indices.add(token.idx - 1)  # token.idx is 1-based

        # Create a new Sentence object from the tokens not covered by any entity span
        filtered_tokens = [token for i, token in enumerate(sentence.tokens) if i not in exclude_indices]

        # Create a new Sentence object from the filtered tokens for POS tagging
        filtered_sentence = Sentence(' '.join(token.text for token in filtered_tokens))

        # Predict POS tags for the filtered sentence
        self.pos_tagger.predict(filtered_sentence)

        # Collect POS tags and return them in a dictionary
        pos_tags = {token.text: token.labels[0].value for token in filtered_sentence
                    if token.labels[0].value in ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    and token.text not in ['is', 'was', 'were', 'are', 'be', 'been', 'being']}

        return {
            'pos_tags': pos_tags,
            'entities': {ent.text: ent.tag  for ent in ner_entities}
        }

# Example usage
decomposer = MessageDecomposer()
result = decomposer.decompose("Hi, could you tell me Who is the director of Star Wars: Episode VI - Return of the Jedi?")
#result = decomposer.decompose("Hi, could you tell me What is the genre of the Titanic?")
#result = decomposer.decompose("Who is the screenwriter of The Masked Gang: Cyprus?")
#result = decomposer.decompose('When was "The Godfather" released?')
print("POS Tags:", result['pos_tags'])
print("Entities:", result['entities'])

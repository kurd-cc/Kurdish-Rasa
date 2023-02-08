from __future__ import annotations
from typing import Any, Dict, Text, List, Optional
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.tokenizers.tokenizer import Tokenizer, Token
from klpt.tokenize import Tokenize
from klpt.preprocess import Preprocess
from klpt.stem import Stem
from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.graph import ExecutionContext, GraphComponent


@DefaultV1Recipe.register(
    component_types=[DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER],
    is_trainable=False,
)
class KLPTTokenizer(Tokenizer):
    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        return ["ku"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # default don't load custom dictionary
            "dictionary_path": None,
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        super().__init__(config)
        self._tokenizer = Tokenize("Kurmanji", "Latin")
        self._preprocessor = Preprocess("Kurmanji", "Latin")
        self._stemmer = Stem("Kurmanji", "Latin")

    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource,
               execution_context: ExecutionContext) -> GraphComponent:
        return super().create(config, model_storage, resource, execution_context)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        preprocessed = self._preprocessor.preprocess(message.get(attribute))
        all_words = self._tokenizer.word_tokenize(preprocessed)
        tokens = []
        for word in all_words:
            lemmas = self._stemmer.lemmatize(word)
            if len(lemmas) > 0:
                word = lemmas[0]
            if word in self._preprocessor.stopwords:
                continue
            else:
                tokens.append(Token(word, 0))
        return tokens

    @staticmethod
    def required_packages() -> List[Text]:
        return ["klpt"]

import csv
import yake
import nltk
import networkx
import word_forms.word_forms
#import word_forms.lemmatizer
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from itertools import combinations
#from __future__ import annotations

class Article:

    def __init__(self) -> None:
        self.title = None
        self.description = None
        self.content = None
        self.publication_date = None
        self.source_name = None

        self.sentiment_score = None
        self.keywords = None

    def calculate_sentiment_score(self) -> None:
        sentiment_analyzer = ArticleSentimentAnalyzer()

        if self.content is not None:
            self.sentiment_score = sentiment_analyzer.analyze_article_sentiment(self)

    def extract_keywords(self) -> None:
        keywords_extractor = ArticleKeywordsExtractor()

        if self.content is not None:
            self.keywords = keywords_extractor.extract_article_keywords(self)

class Module: # Singleton

    _self = None

    def __new__(_class) -> 'Module':
        if _class._self is None:
            _class._self = super().__new__(_class)

        return _class._self
    
class ConcreteModule(Module):
    
    def __init__(self) -> None:
        # tu sa zavola Module.__new__(self)
        # ...
        pass

class CSVReader(Module):
    
    def get_articles(self, csv_file_name: str) -> list[dict]:
        articles = []
        dataset_articles = self._get_data(csv_file_name)

        for dataset_article in dataset_articles:
            article = Article()

            article.title = dataset_article[0]
            article.description = dataset_article[1]
            article.content = dataset_article[2]
            article.source_name = csv_file_name

            article.sentiment_score = float(dataset_article[3])
            article.keywords = set(keyword.lower() for keyword in dataset_article[4].split(","))

            articles.append(article)

        return articles
    
    def _get_data(self, csv_file_name: str) -> list[list]:
        data = []

        with open(csv_file_name, mode = 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                data.append(row)

        return data[1:]

# abstract class
class BasicKeywordsExtractor(Module):

    MAX_NGRAM_SIZE = 1
    def extract_keywords(self, text: str) -> list[str]:
        raise NotImplementedError("Abstract method must be implemented!")

class YakeKeywordsExtractor(BasicKeywordsExtractor):

    KEYWORD_COUNT = 20

    def extract_keywords(self, text: str) -> list[str]:
        extractor = yake.KeywordExtractor(n = BasicKeywordsExtractor.MAX_NGRAM_SIZE, lan = "en", top = YakeKeywordsExtractor.KEYWORD_COUNT)

        yake_keywords = extractor.extract_keywords(text)
        keywords = [yake_keyword[0] for yake_keyword in yake_keywords]

        return keywords
    
class KeybertKeywordsExtractor(BasicKeywordsExtractor):

    KEYWORD_COUNT = 20

    def extract_keywords(self, text: str) -> list[str]:
        extractor = KeyBERT()

        keybert_keywords = extractor.extract_keywords(text, keyphrase_ngram_range = (1, BasicKeywordsExtractor.MAX_NGRAM_SIZE), stop_words = "english", top_n = KeybertKeywordsExtractor.KEYWORD_COUNT)
        keywords = [keybert_keyword[0] for keybert_keyword in keybert_keywords]

        return keywords
    
class WordLemmatizer(Module):

    def lemmatize_word(self, word: str) -> str:
        #lemmatizer = word_forms.lemmatizer
        lemmatizer = WordNetLemmatizer()

        lower_word = word.lower()
        lemmatized_word = lemmatizer.lemmatize(lower_word, pos = "n")

        return lemmatized_word
    
class NounsFilter(Module):

    MINIMAL_WORD_LENGTH = 3

    def __init__(self) -> None:
        #nltk.download("averaged_perceptron_tagger")
        #nltk.download('universal_tagset')
        pass
    
    def filter_words_dataset(self, words_dataset_file_name: str, nouns_dataset_file_name: str) -> None:
        # words_dataset_file_name = "words_dataset.txt" / "words_small_dataset.txt"
        # nouns_dataset_file_name = "nouns_dataset.txt" / "nouns_small_dataset.txt"
        with open(words_dataset_file_name, "r") as words_dataset_file:
            with open(nouns_dataset_file_name, "w") as nouns_dataset_file:

                for word in words_dataset_file:
                    cleaned_word = word.strip()
                    if (self._is_noun(cleaned_word) and len(cleaned_word) >= NounsFilter.MINIMAL_WORD_LENGTH):
                        nouns_dataset_file.write(cleaned_word + "\n")

    def _is_noun(self, word: str) -> bool:
        part_of_speech = nltk.pos_tag([word], lang = "eng", tagset = "universal")[0][1]

        return part_of_speech == "NOUN"

class WordFormsGenerator(Module):

    def generate_nouns(self, word: str) -> set[str]:
        generator = word_forms.word_forms

        forms = generator.get_word_forms(word)
        nouns = forms["n"]

        return nouns
    
class WordsEvaluator(Module):

    NOUNS_DATASET_FILE_NAME = "nouns_dataset.txt" # "nouns_small_dataset.txt"
    MAXIMAL_VALUE = 99999
    
    def get_most_common_word(self, words: list[str]) -> str | None:
        maximal_value = WordsEvaluator.MAXIMAL_VALUE
        most_common_word = None

        for word in words:
            word_value = self._evaluate_word(word, maximal_value)

            if word_value < maximal_value:
                most_common_word = word
                maximal_value = word_value

        # no most common word
        if maximal_value == WordsEvaluator.MAXIMAL_VALUE:
            return None
        
        return most_common_word
    
    def _evaluate_word(self, word: str, maximal_value: int) -> int:
        value = 0

        with open(WordsEvaluator.NOUNS_DATASET_FILE_NAME, "r") as nouns_dataset_file:

            for noun in nouns_dataset_file:
                value += 1
                cleaned_noun = noun.strip()

                if word == cleaned_noun:
                    return value
                
                # optimalization
                if value > maximal_value:
                    return WordsEvaluator.MAXIMAL_VALUE
                
        # word is not in nouns_dataset_file
        return WordsEvaluator.MAXIMAL_VALUE
    
class AdjustedKeywordsExtractor(Module):

    def extract_adjusted_keywords(self, basic_keywords_extractor: BasicKeywordsExtractor, text: str) -> list[str]:
        lemmatizer = WordLemmatizer()
        nouns_generator = WordFormsGenerator()
        words_evaluator = WordsEvaluator()

        adjusted_keywords = []
        extracted_keywords = basic_keywords_extractor.extract_keywords(text)

        for keyword in extracted_keywords:
            lemmatized_keyword = lemmatizer.lemmatize_word(keyword)
            nouns = nouns_generator.generate_nouns(lemmatized_keyword)
            most_common_noun = words_evaluator.get_most_common_word(list(nouns))
            
            # no most common noun
            if most_common_noun is None:
                most_common_noun = lemmatized_keyword
            
            # klucove slova sa mozu opakovat
            adjusted_keywords.append(most_common_noun)

        return adjusted_keywords

class SentenceSentimentAnalyzer(Module):

    MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def analyze_sentence(self, sentence: str) -> float:
        analyzer = pipeline("sentiment-analysis", model = SentenceSentimentAnalyzer.MODEL)

        result = analyzer(sentence)
        sentiment = result[0]["label"]

        if sentiment == "POSITIVE":
            sentiment_score = result[0]["score"]
        elif sentiment == "NEGATIVE":
            sentiment_score = 1 - result[0]["score"]

        # return number between 0 and 1 (0 = negative, 0.5 = neutral, 1 = positive)
        return sentiment_score
    
class TextSummarizer(Module):

    MODEL = "facebook/bart-large-cnn"

    def summarize_text(self, text: str) -> str:
        summarizer = pipeline("summarization", model = TextSummarizer.MODEL)

        result = summarizer(text)
        summary_text = result[0]["summary_text"]

        return summary_text
    
class SentenceTokenizer(Module):

    def __init__(self) -> None:
        #nltk.download('punkt')
        pass

    def tokenize(self, text: str) -> list[str]:
        sentences = sent_tokenize(text)

        return sentences

class ArticleSentimentAnalyzer(Module):

    def analyze_article_sentiment(self, article: Article) -> float:
        article_summarizer = TextSummarizer()
        sentence_tokenizer = SentenceTokenizer()

        article_summarization = article_summarizer.summarize_text(article.content)
        summarization_sentences = sentence_tokenizer.tokenize(article_summarization)
        summarization_sentiment_score = self._calculate_sentiment_score(summarization_sentences)

        #description_sentences = self.sentence_tokenizer.tokenize(article.description)
        #description_sentiment_score = self._calculate_sentiment_score(description_sentences)

        #title_sentences = self.sentence_tokenizer.tokenize(article.title)
        #title_sentiment_score = self._calculate_sentiment_score(title_sentences)

        total_sentiment_score = summarization_sentiment_score
        #total_sentiment_score = (summarization_sentiment_score + description_sentiment_score) / 2
        #total_sentiment_score = (summarization_sentiment_score + description_sentiment_score + headline_sentiment_score) / 3

        return total_sentiment_score
    
    def _calculate_sentiment_score(self, sentences: list[str]) -> float:
        sentence_sentiment_analyzer = SentenceSentimentAnalyzer()

        total_sentiment_score = 0

        for sentence in sentences:
            sentiment_score = sentence_sentiment_analyzer.analyze_sentence(sentence)
            total_sentiment_score += sentiment_score

        total_sentiment_score /= len(sentences)

        return total_sentiment_score
    
class ArticleKeywordsExtractor(Module):

    TOP_KEYWORDS_COUNT = 5

    def extract_article_keywords(self, article: Article) -> set[str]:
        keywords_extractor = AdjustedKeywordsExtractor()

        # zoradene podla relevantnosti
        keybert_keywords = keywords_extractor.extract_adjusted_keywords(KeybertKeywordsExtractor(), article.content)
        yake_keywords = keywords_extractor.extract_adjusted_keywords(YakeKeywordsExtractor(), article.content)

        top_keybert_keywords = keybert_keywords[:ArticleKeywordsExtractor.TOP_KEYWORDS_COUNT]
        top_yake_keywords = yake_keywords[:ArticleKeywordsExtractor.TOP_KEYWORDS_COUNT]

        top_keywords_unification = set(top_keybert_keywords).union(set(top_yake_keywords))
        all_keywords_intersection = set(keybert_keywords).intersection(set(yake_keywords))
        article_keywords = top_keywords_unification.union(all_keywords_intersection)

        return article_keywords
    
class CombinationsCreator(Module):

    def create_combinations(self, elements: set) -> set:
        elements_combinations = set()

        for combination in combinations(elements, 2):
            elements_combinations.add(frozenset(combination))

        return elements_combinations

class ComplexNetwork:

    def __init__(self) -> None:
        self.graph = networkx.Graph()

    def process_articles(self, articles: list[Article]) -> None:
        combinations_creator = CombinationsCreator()

        for article in articles:
            
            for keyword in article.keywords:
                self._add_node(keyword)
                
            keywords_combinations = tuple(combinations_creator.create_combinations(article.keywords))
            for keywords_combination in keywords_combinations:
                keywords_combination_tuple = tuple(keywords_combination)
                self._add_edge(keywords_combination_tuple[0], keywords_combination_tuple[1])

    def _add_node(self, node: str) -> None:
        if not self.graph.has_node(node):
            self.graph.add_node(node, size = 0)

        self.graph.nodes[node]["size"] += 1

    def _add_edge(self, first_node: str, second_node: str) -> None:
        if not self.graph.has_edge(first_node, second_node):
            self.graph.add_edge(first_node, second_node, width = 0)

        self.graph.edges[first_node, second_node]["width"] += 1

    def get_number_of_nodes(self) -> int:
        return self.graph.number_of_nodes()
    
    def get_number_of_edges(self) -> int:
        return self.graph.number_of_edges()

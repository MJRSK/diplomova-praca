import modules

# abstract class
class Tester:

    ARTICLE_DATASET = "articles_dataset.csv"

    def test(self) -> None:
        raise NotImplementedError("Abstract method must be implemented!")

class SentimentAnalysisTester(Tester):

    def test(self) -> None:
        csv_reader = modules.CSVReader()
        articles = csv_reader.get_articles(Tester.ARTICLE_DATASET)

        self._deviation_test(articles)

    def _deviation_test(self, articles: list[modules.Article]) -> None:
        sentiment_analyzer = modules.ArticleSentimentAnalyzer()

        maximum_deviation = 0
        total_deviation = 0
        total_dataset_sentiment = 0
        total_estimated_sentiment = 0

        for article in articles:
            dataset_sentiment = article.sentiment_score
            estimated_sentiment = sentiment_analyzer.analyze_article_sentiment(article)
            sentiment_deviation = abs(dataset_sentiment - estimated_sentiment)

            if (sentiment_deviation > maximum_deviation):
                maximum_deviation = sentiment_deviation

            total_deviation += sentiment_deviation
            total_dataset_sentiment += dataset_sentiment
            total_estimated_sentiment += estimated_sentiment

            print(f"Article ({article.title}) Processed!")
            print(f"Dataset Sentiment = {dataset_sentiment}, Estimated Sentiment = {estimated_sentiment}, Deviation = {sentiment_deviation}")
            
        average_deviation = total_deviation / len(articles)
        average_dataset_sentiment = total_dataset_sentiment / len(articles)
        average_estimated_sentiment = total_estimated_sentiment / len(articles)

        print(f"Maximum Deviation = {maximum_deviation}")
        print(f"Average Deviation = {average_deviation}")
        print(f"Average Dataset Sentiment = {average_dataset_sentiment}")
        print(f"Average Estimated Sentiment = {average_estimated_sentiment}")
        print(f"|Average Dataset Sentiment - Average Estimated Sentiment| = {abs(average_dataset_sentiment - average_estimated_sentiment)}")            

# pomocna trieda
class AuxiliaryKeywordsExtractor(modules.BasicKeywordsExtractor):

    def extract_keywords(self, article_keywords: set[str]) -> list[str]:
        return list(article_keywords)

class KeywordsExtractionTester(Tester):

    def test(self) -> None:
        csv_reader = modules.CSVReader()
        articles = csv_reader.get_articles(Tester.ARTICLE_DATASET)

        self._successful_keywords_test(articles)

    def _successful_keywords_test(self, articles: list[modules.Article]) -> None:
        keywords_extractor = modules.AdjustedKeywordsExtractor()

        total_final_keyword_count = 0
        total_successful_keyword_count = 0
        total_jaccard_index = 0

        for article in articles:
            # pouzitie pomocnej triedy pre testovanie
            original_keywords = keywords_extractor.extract_adjusted_keywords(AuxiliaryKeywordsExtractor(), article.keywords)

            # prisposobene klucove slova sa v listoch mozu opakovat
            keybert_keywords = keywords_extractor.extract_adjusted_keywords(modules.KeybertKeywordsExtractor(), article.content)
            yake_keywords = keywords_extractor.extract_adjusted_keywords(modules.YakeKeywordsExtractor(), article.content)

            # 1. metoda
            #final_keywords = set(keybert_keywords).intersection(set(yake_keywords))

            # 2. metoda
            #final_keywords = set(keybert_keywords).union(set(yake_keywords))

            # 3. metoda
            TOP_KEYWORDS_COUNT = 7
            top_keywords_unification = set(yake_keywords[:TOP_KEYWORDS_COUNT]).union(keybert_keywords[:TOP_KEYWORDS_COUNT])
            all_keywords_intersection = set(keybert_keywords).intersection(set(yake_keywords))
            final_keywords = top_keywords_unification.union(all_keywords_intersection)

            total_final_keyword_count += len(final_keywords)
            successful_keywords = set(original_keywords).intersection(final_keywords)
            total_successful_keyword_count += len(successful_keywords)
            jaccard_index = self._get_jaccard_index(set(original_keywords), set(final_keywords))
            total_jaccard_index += jaccard_index

            print(f"Original (Adjusted) Keywords: {original_keywords}")
            print(f"Keybert (Adjusted) Keywords: {keybert_keywords}")
            print(f"Yake (Adjusted) Keywords: {yake_keywords}")

            print(f"Final Keywords: {final_keywords}")
            print(f"Successful Keywords: {successful_keywords}")
            print(f"Jaccard Index = {jaccard_index}")

        average_final_keyword_count = total_final_keyword_count / len(articles)
        average_successful_keyword_count = total_successful_keyword_count / len(articles)
        average_jaccard_index = total_jaccard_index / len(articles)

        print(f"Average Final Keywords Count = {average_final_keyword_count}")
        print(f"Average Successful Keywords Count = {average_successful_keyword_count}")
        print(f"Average Jaccard Index = {average_jaccard_index}")

    def _get_jaccard_index(self, first_set: set[str], second_set: set[str]) -> float:
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        return len(intersection) / len(union)

class WordsEvaluatorTester(Tester):

    def test(self) -> None:
        csv_reader = modules.CSVReader()
        articles = csv_reader.get_articles(Tester.ARTICLE_DATASET)

        self._success_rate_test(articles)

    def _success_rate_test(self, articles: list[modules.Article]) -> None:
        keybert_basic_keywords_extractor = modules.KeybertKeywordsExtractor()
        yake_basic_keywords_extractor = modules.YakeKeywordsExtractor()
        words_evaluator = modules.WordsEvaluator()
        lemmatizer = modules.WordLemmatizer()
        nouns_generator = modules.WordFormsGenerator()

        total_keywords_count = 0
        total_adjusted_keywords_count = 0

        for article in articles:
            extracted_keybert_keywords = keybert_basic_keywords_extractor.extract_keywords(article.content)
            extracted_yake_keywords = yake_basic_keywords_extractor.extract_keywords(article.content)

            for keyword in extracted_keybert_keywords:
                total_keywords_count += 1
                lemmatized_keyword = lemmatizer.lemmatize_word(keyword)
                nouns = nouns_generator.generate_nouns(lemmatized_keyword)
                most_common_noun = words_evaluator.get_most_common_word(list(nouns))

                if most_common_noun is not None:
                    total_adjusted_keywords_count += 1

            for keyword in extracted_yake_keywords:
                total_keywords_count += 1
                lemmatized_keyword = lemmatizer.lemmatize_word(keyword)
                nouns = nouns_generator.generate_nouns(lemmatized_keyword)
                most_common_noun = words_evaluator.get_most_common_word(list(nouns))

                if most_common_noun is not None:
                    total_adjusted_keywords_count += 1

        words_evaluator_success_rate = total_adjusted_keywords_count / total_keywords_count
        print(f"Words Evaluator Success Rate = {words_evaluator_success_rate}")

class ComplexNetworkTester(Tester):

    def test(self) -> None:
        csv_reader = modules.CSVReader()
        articles = csv_reader.get_articles(Tester.ARTICLE_DATASET)

        self._nodes_and_edges_test(articles)

    def _nodes_and_edges_test(self, articles: list[modules.Article]) -> None:
        complex_network = modules.ComplexNetwork()
        complex_network.process_articles(articles)

        print(complex_network.graph)
        sorted_nodes = sorted(complex_network.graph.nodes(data = True), key = lambda x: x[1]["size"], reverse = True)
        sorted_edges = sorted(complex_network.graph.edges(data = True), key = lambda x: x[2]["width"], reverse = True)
        print(sorted_nodes)
        print(sorted_edges)
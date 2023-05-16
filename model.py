import pandas as pd
import numpy as np
import bz2
import pickle


class SentimentRecommenderModel:
    ROOT_MODEL_PATH = "models/"
    # SENTIMENT_MODEL = "best_sentiment_model.pkl"
    SENTIMENT_MODEL = "best_sentiment_model.pbz2"
    TFIDF_VECTORIZER = "tfidf.pkl"
    BEST_RECOMMENDER = "best_recommendation_model.pkl"
    CLEAN_DATAFRAME = "clean_data.pkl"

    def __init__(self):
        # self.sentiment_model = pickle.load(open(
        #     SentimentRecommenderModel.ROOT_MODEL_PATH + SentimentRecommenderModel.SENTIMENT_MODEL, 'rb'))
        self.sentiment_model = pickle.load(bz2.BZ2File(
            SentimentRecommenderModel.ROOT_MODEL_PATH + SentimentRecommenderModel.SENTIMENT_MODEL, 'rb'))
        self.tfidf_vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_MODEL_PATH + SentimentRecommenderModel.TFIDF_VECTORIZER)
        self.recommender = pickle.load(open(
            SentimentRecommenderModel.ROOT_MODEL_PATH + SentimentRecommenderModel.BEST_RECOMMENDER, 'rb'))
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_MODEL_PATH + SentimentRecommenderModel.CLEAN_DATAFRAME, 'rb'))

    def get_top5_recommendations(self, user_name):
        if user_name not in self.recommender.index:
            print(f"The User '{user_name}' does not exist. Please try again.")
            return None
        else:
            # Get all the reviews of the top 20 products from the recommender system by this user
            recommendations = list(
                self.recommender.loc[user_name].sort_values(ascending=False)[0:20].index)
            # Check if these records are also part of the dataset used for sentiment analysis
            temp = self.cleaned_data[self.cleaned_data.id.isin(recommendations)]
            # Transform these text into numeric using the TF-IDF created earlier
            X = self.tfidf_vectorizer.transform(
                temp["reviews_lemmatized"].values.astype(str))
            # Feed this input to the sentiment analysis model to get predictions of sentiments
            temp["predicted_sentiment"] = self.sentiment_model.predict(X)
            temp = temp[['name', 'predicted_sentiment']]
            # Calculate the total count of sentiments for each user
            temp_grouped = temp.groupby('name', as_index=False).count()
            # Now calculate the count of 'positive' sentiments for each user
            temp_grouped["pos_review_count"] = temp_grouped.name.apply(lambda x: temp[(
                temp.name == x) & (temp.predicted_sentiment == 1)]["predicted_sentiment"].count())
            # Rename 'predicted_sentiment' to 'total_review_count'
            temp_grouped.rename(
                columns={'predicted_sentiment': 'total_review_count'}, inplace=True)
            # For each product, calculate the % of positive sentiments across all the sentiments of that product
            temp_grouped['pos_sentiment_percent'] = np.round(
                temp_grouped["pos_review_count"]/temp_grouped["total_review_count"]*100, 2)
            pred_df = temp_grouped[["name", "pos_sentiment_percent"]][:5]

            return pred_df.sort_values('pos_sentiment_percent', ascending=False)

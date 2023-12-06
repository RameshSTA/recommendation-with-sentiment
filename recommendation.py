
'''
the main class of the recommendation system is created here, 
*importing all the  dependecies and models for processing the clean dataset for recommending the products. 
*The  product is  recommended based on the sentiment created in the sentiment notebook, and as we  do not have user  full list of reivew, we just recommend the product with high positive reveiws. 
'''
import torch
import tensorflow as tf
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel, TFDistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ProductRecommendation:
    """
    Product recommendation system leveraging BERT-based embeddings and sentiment analysis.
    """

    def __init__(self, model, tokenizer, data):
        """
        Initialize ProductRecommendation class.

        Args:
        - model: BERT-based model for embedding generation
        - tokenizer: Tokenizer for preprocessing text
        - data: DataFrame containing product information and sentiments
        """
        self.model = model
        self.tokenizer = tokenizer
        self.data = data

    def preprocess_review(self, review_text):
        """
        Tokenize and preprocess the review text.

        Args:
        - review_text: Input review text

        Returns:
        - processed_text: Processed tokenized text
        """
        processed_text = self.tokenizer(review_text, return_tensors='pt', padding=True, truncation=True)
        return processed_text

    def generate_review_embedding(self, processed_review):
        """
        Generate embedding for the review text.

        Args:
        - processed_review: Processed tokenized review text

        Returns:
        - review_embedding: Embedding for the review text
        """
        with torch.no_grad():
            outputs = self.model(**processed_review)
            review_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
        return review_embedding

    def recommend_products_by_sentiment(self, review_text, top_n=1):
        """
        Recommend products based on sentiment similarity to the input review.

        Args:
        - review_text: Input review text
        - top_n: Number of products to recommend (default: 5)

        Returns:
        - recommended_products: List of recommended product names
        """
        processed_review = self.preprocess_review(review_text)
        review_embedding = self.generate_review_embedding(processed_review)

        similarities = {}
        for idx, product_name in enumerate(self.data['name']):
            product_processed_text = self.tokenizer(product_name, return_tensors='pt', padding=True, truncation=True)
            product_embedding = self.generate_review_embedding(product_processed_text)
            similarity = cosine_similarity(review_embedding, product_embedding)[0][0]
            sentiment =  self.data.iloc[idx]['reviews_sentiment']
            if sentiment == 1:  # Considering products with positive sentiment
                similarities[idx] = similarity

        sorted_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommended_products = [self.data.iloc[idx]['name'] for idx, _ in sorted_products]
        return recommended_products

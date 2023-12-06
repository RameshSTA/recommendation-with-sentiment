# Importing necessary dependencies
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from recommendation import ProductRecommendation  # this is comming from the recommendation.py  
import torch
import csv

# Initialize Flask app
app = Flask(__name__)

# Load BERT model and tokenizer   which are used for recommending the products 
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# The cleaned data set with sentiment  of the product reviews is saved in the directory
# this will be loaded and in the csv formats
data = pd.read_csv('/Users/ramesh/Desktop/System Recommendation/model/clean_data.csv')

# Route for the homepage
@app.route('/')
def homepage():
    """
    Renders the homepage with a clean dataset that is processed by using BERT models
    """
    #firstly the dataset is converted to html document and  it will be  read in frontend
    html_table = data.to_html()
    return render_template("index.html", table=html_table)

# Route for the product page
@app.route('/product')
def index():
    """
    Renders the product review page.
    """
    return render_template('product_review.html')

# Route for recommending products based on review text
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Recommends products based on review text input.
    Expects a JSON object with 'review_text' key in the request.
    """
    review_text = request.json['review_text']
    product_recommender = ProductRecommendation(model, tokenizer, data)
    recommended_products = product_recommender.recommend_products_by_sentiment(review_text)
    return jsonify({'recommended_products': recommended_products})

# Route for the 'About' page
@app.route('/About')
def about():
    """
    Renders the about page which simply explain about the  projects and give some contact information for further any inquiries.
    """
    return render_template('about.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

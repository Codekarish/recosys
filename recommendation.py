import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from bs4 import BeautifulSoup
import requests
import warnings
from openai import OpenAI
import re
import json
from waitress import serve
from geopy.distance import geodesic
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# Your OpenCage API key
OPENCAGE_API_KEY = 'dd29b25918284b32a3f12bd68ebecae8'
# OpenAI API key
OPENAI_API_KEY = 'sk-proj-pHkywA2mDUiHGcCWRM4TT3BlbkFJz3qlbeGgfG8C70iHX7WK'

def fetch_data(url, params, min_results=100):
    all_properties = []
    current_page = 1
    while len(all_properties) < min_results:
        try:
            params['page'] = current_page
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            properties = data.get('data', [])
            if not properties:
                break  # Exit loop if no more data
            all_properties.extend(properties)
            current_page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    return pd.DataFrame(all_properties)

def extract_numeric_bedrooms(description):
    """ Extracts numeric values from bedroom descriptions. """
    if pd.isnull(description):
        return np.nan
    
    # Handle special cases
    if 'studio' in description.lower() or 'bedsitter' in description.lower():
        return 0  # Assuming studio/bedsitter has 0 bedrooms
    
    # Extract number from the string
    match = re.search(r'\d+', description)
    if match:
        return int(match.group(0))
    elif '+' in description:
        # For '5+ bedroom', assume a threshold or max value; here we use 5
        return 5
    return np.nan

def clean_data(df):
    if 'description' in df.columns:
        df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text() if pd.notnull(x) else "")
    
    if 'bedrooms' in df.columns:
        # Extract numeric values from the 'bedrooms' column
        df['bedrooms'] = df['bedrooms'].apply(extract_numeric_bedrooms)
    
    # Print data types and sample to debug
    print("\nData types after cleaning:")
    print(df.dtypes)
    print("\nSample data from 'bedrooms' column after conversion:")
    print(df['bedrooms'].head())
    
    df = df.fillna('')
    return df

def convert_bedrooms_to_integers(df):
    def parse_bedroom_value(value):
        if pd.isnull(value):
            return None
        if isinstance(value, str):
            # Handle cases like '5+ bedroom'
            if '+' in value:
                return 5
            # Use re.search and check for None before calling .group()
            match = re.search(r'\d+', value)
            if match:
                return int(match.group())
            else:
                return None
        if isinstance(value, (float, int)):
            return int(value)
        return None

    df['bedrooms'] = df['bedrooms'].apply(parse_bedroom_value)
    df['bedrooms'] = df['bedrooms'].astype('Int64')
    return df

def preprocess_text(df, query_keywords):
    def weight_keywords(text, keywords):
        for keyword in keywords:
            text = re.sub(f"(?i){keyword}", f"{keyword} {keyword}", text)
        return text

    df['weighted_text'] = df['name'] + ' ' + df['description']
    df['weighted_text'] = df['weighted_text'].apply(lambda x: weight_keywords(x, query_keywords))

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['weighted_text'])
    return vectorizer, X

def encode_categorical_data(df):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_data = df[['property_type', 'listing_type', 'location']]
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    return encoder, encoded_categorical_data

def combine_features(tfidf_matrix, encoded_categorical_data):
    return np.hstack((tfidf_matrix.toarray(), encoded_categorical_data))

def extract_keywords(query):
    keywords = re.findall(r'\b\w+\b', query.lower())
    return keywords

def format_description(description, property_id, max_length=50):
    if len(description) > max_length:
        return f"{description[:max_length]}... <a href='/property/{property_id}'>Read more</a>"
    return description

def parse_query(query):
    price = None
    bedrooms = None
    location = None
    agent_name = None

    price_match = re.search(r'(\d+)\s*(ksh|kes|shillings|sh)', query, re.IGNORECASE)
    if price_match:
        price = int(price_match.group(1))

    bedrooms_match = re.search(r'(\d+)\s*(bedroom|bedrooms)', query, re.IGNORECASE)
    if bedrooms_match:
        bedrooms = int(bedrooms_match.group(1))

    location_match = re.search(r'at\s*([\w\s]+)', query, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip().lower()

    agent_match = re.search(r'agent\s*([\w\s]+)', query, re.IGNORECASE)
    if agent_match:
        agent_name = agent_match.group(1).strip().lower()

    return price, bedrooms, location, agent_name

def get_lat_lon(location):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['results']:
        return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    return None, None

def get_distance(loc1, loc2):
    return geodesic(loc1, loc2).km

def get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df, k=100):
    price, bedrooms, location, agent_name = parse_query(query)
    query_keywords = extract_keywords(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Debug prints
    print("Price:", price)
    print("Bedrooms:", bedrooms)
    print("Location:", location)
    print("Agent Name:", agent_name)
    print("Query Keywords:", query_keywords)
    print("Query Vector Shape:", query_vector.shape)
    print("Similarities Shape:", similarities.shape)

    # Filter by price range if provided
    if price:
        df['price_amount'] = df['price'].apply(lambda x: x['price'] if isinstance(x, dict) and 'price' in x else 0)
        df = df[(df['price_amount'] <= price * 1.1) & (df['price_amount'] >= price * 0.9)]

    # Calculate distances if location is provided
    if location:
        user_lat, user_lon = get_lat_lon(location)
        if user_lat is not None and user_lon is not None:
            user_location = (user_lat, user_lon)
            df['distance'] = df.apply(lambda row: get_distance(user_location, (row['lat'], row['lon'])), axis=1)
            df = df.sort_values(by='distance')

    # Prioritize by number of bedrooms if provided
    if bedrooms:
        if 'bedrooms' in df.columns:
            df['bedroom_diff'] = df['bedrooms'].apply(lambda x: abs(x - bedrooms) if pd.notnull(x) else np.nan)
            df = df.sort_values(by='bedroom_diff')

    # Prioritize by agent name if provided
    if agent_name:
        if 'agent' in df.columns:
            df = df[df['agent'].str.contains(agent_name, case=False, na=False)]

    df['score'] = similarities
    top_k_indices = df.nlargest(k, 'score').index
    top_k_properties = df.loc[top_k_indices]

    # Debug prints
    print("Top K Properties (indices):", top_k_indices)
    print("Top K Properties (sample):")
    print(top_k_properties[['id', 'score', 'distance', 'bedrooms', 'price_amount']].head())

    results = top_k_properties[['id', 'image', 'submission_type', 'location', 'bedrooms', 'agent', 'description', 'score']].copy()
    results['description'] = results.apply(lambda row: format_description(row['description'], row['id']), axis=1)
    return results

def main():
    url = 'https://api.example.com/properties'
    params = {'status': 'available', 'per_page': 100}
    min_results = 100
    data = fetch_data(url, params, min_results)
    print("\nData fetched successfully.")
    cleaned_data = clean_data(data)
    print("\nData cleaned successfully.")
    converted_data = convert_bedrooms_to_integers(cleaned_data)
    print("\nBedroom data converted successfully.")
    query = "2 bedroom house for sale in Kilimani for 10,000,000 KES"
    vectorizer, tfidf_matrix = preprocess_text(converted_data, extract_keywords(query))
    print("\nText data preprocessed successfully.")
    encoder, encoded_categorical_data = encode_categorical_data(converted_data)
    print("\nCategorical data encoded successfully.")
    combined_data = combine_features(tfidf_matrix, encoded_categorical_data)
    print("\nFeatures combined successfully.")
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, converted_data)
    print("\nRecommendations generated successfully.")
    print(recommendations)

if __name__ == "__main__":
    main()

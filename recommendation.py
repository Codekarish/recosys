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

def fetch_data(url, params, min_results=20):
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
    submission_type = None

    # Extract price
    price_match = re.search(r'(\d+)\s*(ksh|kes|shillings|sh)', query, re.IGNORECASE)
    if price_match:
        price = int(price_match.group(1))

    # Extract number of bedrooms
    bedrooms_match = re.search(r'(\d+)\s*(bedroom|bedrooms)', query, re.IGNORECASE)
    if bedrooms_match:
        bedrooms = int(bedrooms_match.group(1))

    # Extract location
    location_match = re.search(r'in\s*([\w\s]+)', query, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip().lower()

    # Extract agent name
    agent_match = re.search(r'agent\s*([\w\s]+)', query, re.IGNORECASE)
    if agent_match:
        agent_name = agent_match.group(1).strip().lower()

    # Extract submission type
    if 'rent' in query.lower():
        submission_type = 'rent'
    elif 'sale' in query.lower():
        submission_type = 'sale'

    return price, bedrooms, location, agent_name, submission_type

def get_lat_lon(location):
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if data['results']:
        return data['results'][0]['geometry']['lat'], data['results'][0]['geometry']['lng']
    return None, None

def get_distance(loc1, loc2):
    return geodesic(loc1, loc2).km

def get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, df, k=25):
    # Parse the query
    try:
        price, bedrooms, location, agent_name, submission_type = parse_query(query)
    except ValueError:
        # Handle cases where parse_query returns fewer values
        price, bedrooms, location, agent_name, submission_type = (None, None, None, None, None)

    query_keywords = extract_keywords(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Ensure the length of similarities matches the length of df
    if len(similarities) != len(df):
        raise ValueError("Length of similarities does not match length of DataFrame")

    # Handle missing price field and apply submission type filter
    df['price_amount'] = df.apply(lambda row: row['amount'] if 'amount' in row and pd.notnull(row['amount']) 
                                  else (row['sale_price'] if 'sale_price' in row and pd.notnull(row['sale_price'])
                                  else (row['rent'] if 'rent' in row and pd.notnull(row['rent']) else None)), axis=1)

    if submission_type:
        df = df[df['submission_type'].str.lower() == submission_type.lower()]

    # Filter by price range if provided
    if price:
        lower_bound = price * 0.5
        upper_bound = price * 1.5
        df = df[(df['price_amount'] <= upper_bound) & (df['price_amount'] >= lower_bound)]
        # Recompute similarities after filtering
        filtered_indices = df.index
        similarities = similarities[filtered_indices]

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
            df['agent_name'] = df['agent'].apply(lambda x: x['name'].strip().lower() if pd.notnull(x) else 'unknown')
            df = df[df['agent_name'] == agent_name]

    # Apply weights to the similarity scores
    weights = {
        'location': 40,
        'property_type': 20,
        'size': 10,
        'submission_type': 10,
        'price': 25,
        'creation_date': 0.5,
        'property_setting': 0.5,
        'customer_modifiers': 0.5,
        'user_metrics': 0.5,
    }

    weighted_similarities = similarities.copy()

    # Apply weights
    for i, row in df.iterrows():
        if 'location' in query_keywords and 'location' in df.columns:
            weighted_similarities[i] += weights['location'] * 0.1
        if 'property_type' in query_keywords and 'property_type' in df.columns:
            weighted_similarities[i] += weights['property_type'] * 0.1
        if 'size' in query_keywords and 'size' in df.columns:
            weighted_similarities[i] += weights['size'] * 0.1
        if 'submission_type' in query_keywords and 'submission_type' in df.columns:
            weighted_similarities[i] += weights['submission_type'] * 0.1
        if 'price' in query_keywords and 'price_amount' in df.columns:
            weighted_similarities[i] += weights['price'] * 0.1
        if 'creation_date' in query_keywords and 'created_at' in df.columns:
            weighted_similarities[i] += weights['creation_date'] * 0.05
        if 'property_setting' in query_keywords and 'property_setting' in df.columns:
            weighted_similarities[i] += weights['property_setting'] * 0.05
        if 'customer_modifiers' in query_keywords and 'customer_modifiers' in df.columns:
            weighted_similarities[i] += weights['customer_modifiers'] * 0.05
        if 'user_metrics' in query_keywords and 'user_metrics' in df.columns:
            weighted_similarities[i] += weights['user_metrics'] * 0.05

    indices = weighted_similarities.argsort()[-k:][::-1]
    recommendations = df.iloc[indices].copy()
    recommendations['score'] = (weighted_similarities[indices] * 100).round().astype(int)

    # Extract and format data
    if 'images' in recommendations.columns:
        recommendations['image'] = recommendations['images'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'No Image')
    else:
        recommendations['image'] = 'No Image'

    recommendations['description'] = recommendations['description'].apply(lambda x: ' '.join(x.split()[:100]) + ('...' if len(x.split()) > 100 else ''))

    if 'agent' in recommendations.columns:
        recommendations['agent'] = recommendations['agent'].apply(lambda x: x['name'] if pd.notnull(x) else 'Unknown')
    else:
        recommendations['agent'] = 'Unknown'

    recommendations['location'] = recommendations['location'].apply(lambda x: x if pd.notnull(x) else 'Unknown')

    recommendations['ID'] = recommendations['id'].apply(lambda x: f'<a href="/property/{x}">{x}</a>')

    return recommendations[['ID', 'image', 'submission_type', 'bedrooms', 'agent', 'location', 'price_amount', 'description', 'score']].to_dict('records')



def main(query):
    url = 'https://sapi.hauzisha.co.ke/api/properties/search'
    params = {'per page': 300, }
    min_results = 20
    data = fetch_data(url, params, min_results)
    print("\nData fetched successfully.")
    
    cleaned_data = clean_data(data)
    print("\nData cleaned successfully.")
    
    converted_data = convert_bedrooms_to_integers(cleaned_data)
    print("\nBedroom data converted successfully.")
    
    vectorizer, tfidf_matrix = preprocess_text(converted_data, extract_keywords(query))
    print("\nText data preprocessed successfully.")
    
    encoder, encoded_categorical_data = encode_categorical_data(converted_data)
    print("\nCategorical data encoded successfully.")
    
    combined_data = combine_features(tfidf_matrix, encoded_categorical_data)
    print("\nFeatures combined successfully.")
    
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, encoder, encoded_categorical_data, converted_data)
    print("\nRecommendations generated successfully.")
    
    # Return the recommendations instead of printing them
    return recommendations

if __name__ == "__main__":
    query = "2 bedroom house for sale in Kilimani for 10,000,000 KES"  # Example query
    main(query)

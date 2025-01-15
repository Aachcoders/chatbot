import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import logging
from typing import List, Dict, Optional, Any
import uuid
import re
from dataclasses import dataclass, field

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Datasets
try:
    PRODUCTS_DF = pd.read_csv('Updated_Products_Dataset.csv')
    ORDERS_DF = pd.read_csv('Orders_Dataset.csv')
    USERS_DF = pd.read_csv('Users_Dataset.csv')
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    PRODUCTS_DF = pd.DataFrame()
    ORDERS_DF = pd.DataFrame()
    USERS_DF = pd.DataFrame()

# Initialize GPT-2 Models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

class DataProcessor:
    @staticmethod
    def clean_and_validate(text: str, max_length: int = 500) -> str:
        """Sanitize input text"""
        try:
            # Remove potentially harmful characters
            cleaned_text = re.sub(r'[<>]', '', str(text))
            # Truncate if too long
            return cleaned_text[:max_length]
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return ""

class ProductSearchEngine:
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df
    
    def search_products(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Advanced product search"""
        try:
            # Convert query to lowercase
            query = query.lower()
            
            # Create search scoring mechanism
            def calculate_score(row):
                score = 0
                name = str(row['ProductName']).lower()
                description = str(row.get('Description', '')).lower()
                
                # Scoring logic
                if query in name:
                    score += 3
                if query in description:
                    score += 2
                
                return score
            
            # Apply scoring
            search_results = self.products_df.copy()
            search_results['search_score'] = search_results.apply(calculate_score, axis=1)
            
            # Return top k results
            return search_results.nlargest(top_k, 'search_score')[
                ['ProductID', 'ProductName', 'Price', 'Description']
            ]
        
        except Exception as e:
            logger.error(f"Product search error: {e}")
            return pd.DataFrame()

class OrderAnalyzer:
    def __init__(self, orders_df: pd.DataFrame):
        self.orders_df = orders_df
    
    def get_user_order_history(self, user_id: str) -> pd.DataFrame:
        """Retrieve user's order history"""
        try:
            return self.orders_df[self.orders_df['UserID'] == user_id]
        except Exception as e:
            logger.error(f"Order history retrieval error: {e}")
            return pd.DataFrame()
    
    def get_recent_order_status(self, user_id: str) -> Dict[str, Any]:
        """Get the most recent order status for a user"""
        try:
            user_orders = self.get_user_order_history(user_id)
            if user_orders.empty:
                return {"status": "No recent orders"}
            
            # Get most recent order
            recent_order = user_orders.iloc[-1]
            return {
                "order_id": recent_order['OrderID'],
                "product": recent_order['ProductName'],
                "status": recent_order['Status'],
                "order_date": recent_order['OrderDate'],
                "delivery_date": recent_order['DeliveryDate']
            }
        except Exception as e:
            logger.error(f"Recent order status error: {e}")
            return {"status": "Error retrieving order status"}

class UserProfileAnalyzer:
    def __init__(self, users_df: pd.DataFrame):
        self.users_df = users_df
    
    def get_user_details(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user details"""
        try:
            user = self.users_df[self.users_df['UserID'] == user_id]
            if user.empty:
                return {"error": "User not found"}
            
            return user.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"User details retrieval error: {e}")
            return {"error": "Error retrieving user details"}

class RecommendationEngine:
    def __init__(self, products_df: pd.DataFrame, orders_df: pd.DataFrame):
        self.products_df = products_df
        self.orders_df = orders_df
    
    def generate_recommendations(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        """Generate personalized product recommendations"""
        try:
            # Get user's previous purchases
            user_orders = self.orders_df[self.orders_df['UserID'] == user_id]
            purchased_products = user_orders['ProductID'].unique()
            
            # Recommend products not previously purchased
            recommendations = self.products_df[
                ~self.products_df['ProductID'].isin(purchased_products)
            ]
            
            # Add some randomization and relevance
            return recommendations.sample(n=min(top_k, len(recommendations)))
        
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return pd.DataFrame()

class AIAssistant:
    def __init__(
        self, 
        products_df: pd.DataFrame, 
        orders_df: pd.DataFrame, 
        users_df: pd.DataFrame
    ):
        self.products_search = ProductSearchEngine(products_df)
        self.order_analyzer = OrderAnalyzer(orders_df)
        self.user_analyzer = UserProfileAnalyzer(users_df)
        self.recommendation_engine = RecommendationEngine(products_df, orders_df)
    
    def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process user query and generate contextual response"""
        try:
            # Normalize query
            query = query.lower()
            
            # Handle different query types
            if any(keyword in query for keyword in ['search', 'find', 'looking for']):
                # Product search
                search_results = self.products_search.search_products(query)
                return {
                    'type': 'product_search',
                    'results': search_results.to_dict(orient='records'),
                    'response': f"I found {len(search_results)} matching products."
                }
            
            elif user_id and any(keyword in query for keyword in ['order', 'status', 'track']):
                # Order status
                order_status = self.order_analyzer.get_recent_order_status(user_id)
                return {
                    'type': 'order_status',
                    'details': order_status,
                    'response': f"Your latest order status: {order_status.get('status', 'Unknown')}"
                }
            
            elif user_id and any(keyword in query for keyword in ['recommend', 'suggest']):
                # Personalized recommendations
                recommendations = self.recommendation_engine.generate_recommendations(user_id)
                return {
                    'type': 'recommendations',
                    'results': recommendations.to_dict(orient='records'),
                    'response': f"Here are {len(recommendations)} personalized recommendations."
                }
            
            else:
                # General query fallback
                return {
                    'type': 'general',
                    'response': "I'm here to assist you! Please specify if you're looking for products, order status, or recommendations."
                }
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {'error': str(e)}

# Streamlit Application
def main():
    """Streamlit main application"""
    st.title("🤖 E-commerce AI Assistant")
    
    # Load datasets
    try:
        products_df = pd.read_csv('Updated_Products_Dataset.csv')
        orders_df = pd.read_csv('Orders_Dataset.csv')
        users_df = pd.read_csv('Users_Dataset.csv')
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return
    
    # Initialize AI Assistant
    assistant = AIAssistant(products_df, orders_df, users_df)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and generate response
        with st.chat_message("assistant"):
            try:
                # Create a mock user for demonstration
                mock_user_id = "U001"  # Example user ID
                
                # Process query
                result = assistant.process_query(prompt, mock_user_id)
                
                # Display response based on query type
                if result['type'] == 'product_search':
                    st.write("Matching Products:")
                    for product in result['results']:
                        st.write(f"- {product['ProductName']} (${product['Price']:.2f}): {product['Description']}")
                
                elif result['type'] == 'order_status':
                    order_details = result['details']
                    st.write(f"Order ID: {order_details['order_id']}, Status: {order_details['status']}, Delivery Date: {order_details['delivery_date']}")
                
                elif result['type'] == 'recommendations':
                    st.write("Personalized Recommendations:")
                    for product in result['results']:
                        st.write(f"- {product['ProductName']} (${product['Price']:.2f})")
                
                else:
                    st.markdown(result['response'])
            
            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                st.markdown("I'm sorry, there was an error processing your request.")

if __name__ == "__main__":
    main()

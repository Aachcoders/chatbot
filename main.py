
import streamlit as st
import pandas as pd
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
import os
from dotenv import load_dotenv

# Transformers imports
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrderInfo:
    order_id: str
    product_name: str
    status: str
    order_date: str
    delivery_date: str
    tracking_id: str

class ProductSearchEngine:
    def __init__(self, products_df: pd.DataFrame):
        """Initialize search engine with text-based matching"""
        self.products_df = products_df
        
    def search_products(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Search products using keyword matching"""
        query = query.lower()
        # Score products based on keyword matches
        scores = self.products_df.apply(
            lambda row: sum(
                q in str(row['ProductName']).lower() or 
                q in str(row['Description']).lower() or
                q in str(row['Category']).lower()
                for q in query.split()
            ), 
            axis=1
        )
        return self.products_df.iloc[scores.nlargest(top_k).index]

class EcommerceAssistant:
    def __init__(self, products_df: pd.DataFrame, orders_df: pd.DataFrame, users_df: pd.DataFrame):
        self.products_df = products_df
        self.orders_df = orders_df
        self.users_df = users_df
        self.search_engine = ProductSearchEngine(products_df)
        
        # Initialize GPT-2 model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using GPT-2"""
        try:
            # Encode the input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate response
            output = self.model.generate(
                input_ids, 
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True
            )
            
            # Decode the response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm having trouble processing your request. Could you please rephrase?"

    def get_order_status(self, user_id: str) -> Optional[OrderInfo]:
        """Get latest order status for user"""
        user_orders = self.orders_df[
            self.orders_df['UserID'] == user_id
        ].sort_values('OrderDate', ascending=False)
        
        if user_orders.empty:
            return None
            
        recent_order = user_orders.iloc[0]
        product = self.products_df[
            self.products_df['ProductID'] == recent_order['ProductID']
        ].iloc[0]
        
        return OrderInfo(
            order_id=recent_order['OrderID'],
            product_name=product['ProductName'],
            status=recent_order['Status'],
            order_date=recent_order['OrderDate'],
            delivery_date=recent_order.get('DeliveryDate', 'Not available'),
            tracking_id=recent_order.get('TrackID', 'Not available')
        )

    def get_user_recommendations(self, user_id: str) -> pd.DataFrame:
        """Get personalized product recommendations"""
        user_orders = self.orders_df[self.orders_df['UserID'] == user_id]
        if user_orders.empty:
            return self.products_df.nlargest(5, 'Price')
            
        purchased_products = self.products_df[
            self.products_df['ProductID'].isin(user_orders['ProductID'])
        ]
        
        purchased_categories = purchased_products['CategoryID'].unique()
        
        recommended_products = self.products_df[
            (self.products_df['CategoryID'].isin(purchased_categories)) &
            (~self.products_df['ProductID'].isin(purchased_products['ProductID']))
        ].sample(n=min(5, len(self.products_df)))
        
        return recommended_products

    def process_message(self, message: str, user_id: str = None) -> str:
        """Process user message and generate response"""
        try:
            # Process different types of queries
            if any(word in message.lower() for word in ['search', 'find', 'looking', 'show']):
                products = self.search_engine.search_products(message)
                prompt = f"User is looking for products. Here are some relevant products:\n"
                for _, product in products.iterrows():
                    prompt += f"- {product['ProductName']} (${product['Price']}): {product['Description']}\n"
                prompt += "\nHelp the user by explaining these products and their benefits."
                
            elif user_id and any(word in message.lower() for word in ['order', 'status', 'tracking']):
                order_info = self.get_order_status(user_id)
                if not order_info:
                    return "I couldn't find any orders for your account. Would you like to start shopping?"
                
                prompt = f"Order Status:\n"
                prompt += f"Order ID: {order_info.order_id}\n"
                prompt += f"Product: {order_info.product_name}\n"
                prompt += f"Status: {order_info.status}\n"
                prompt += f"Order Date: {order_info.order_date}\n"
                prompt += f"Expected Delivery: {order_info.delivery_date}\n"
                prompt += "Provide a helpful customer service response."
                
            elif user_id and any(word in message.lower() for word in ['recommend', 'suggestion']):
                recommendations = self.get_user_recommendations(user_id)
                prompt = "Personalized Product Recommendations:\n"
                for _, product in recommendations.iterrows():
                    prompt += f"- {product['ProductName']} (${product['Price']}): {product['Description']}\n"
                prompt += "\nExplain why these products might interest the user based on their purchase history."
            
            else:
                # General conversation
                prompt = message
            
            # Generate response using GPT-2
            response = self.generate_response(prompt)
            return response
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error processing your request. Please try again or contact support."

def initialize_app():
    """Initialize Streamlit app and load data"""
    st.set_page_config(
        page_title="E-commerce Assistant",
        page_icon="🛍️",
        layout="wide"
    )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    try:
        products_df = pd.read_csv("Updated_Products_Dataset.csv")
        orders_df = pd.read_csv("Orders_Dataset.csv")
        users_df = pd.read_csv("Users_Dataset.csv")
        
        if 'assistant' not in st.session_state:
            st.session_state.assistant = EcommerceAssistant(
                products_df=products_df,
                orders_df=orders_df,
                users_df=users_df
            )
        return True
        
    except Exception as e:
        st.error(f"Error initializing app: {str(e)}")
        return False

def main():
    """Main application function"""
    if not initialize_app():
        return
    
    st.title("🛍️ Smart E-commerce Assistant")
    
    with st.sidebar:
        st.header("👤 User Authentication")
        if not st.session_state.logged_in:
            user_id = st.text_input("Enter User ID")
            if st.button("Login"):
                if user_id in st.session_state.assistant.users_df['User ID'].values:
                    st.session_state.user_id = user_id
                    st.session_state.logged_in = True
                    st.success(f"Welcome back, User {user_id}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid User ID")
        else:
            st.success(f"Logged in as User {st.session_state.user_id}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.logged_in = False
                st.experimental_rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("How can I assist you with your shopping today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.assistant.process_message(
                prompt,
                st.session_state.user_id if st.session_state.logged_in else None
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

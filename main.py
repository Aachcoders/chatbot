import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import groq
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
from transformers import BertTokenizer, BertModel
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

class EnhancedIntentClassifier(nn.Module):
    def __init__(self, n_intents, n_categories):
        super(EnhancedIntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.intent_classifier = nn.Linear(256, n_intents)
        self.category_classifier = nn.Linear(256, n_categories)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = F.relu(self.fc2(x))
        intent_logits = self.intent_classifier(features)
        category_logits = self.category_classifier(features)
        return intent_logits, category_logits

class EnhancedChatBot:
    def __init__(self, groq_api_key: str, products_df, categories_df, orders_df, users_df, sellers_df):
        self.groq_client = groq.Client(api_key=groq_api_key)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load datasets
        self.products_df = products_df
        self.categories_df = categories_df
        self.orders_df = orders_df
        self.users_df = users_df
        self.sellers_df = sellers_df
        
        # Initialize conversation history and context
        self.messages = []  # Store conversation history
        self.context = {}
        
        # Load intents and initialize classifier
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)
        self.intent_classifier = self.initialize_intent_classifier()
        
        # Create system prompt
        self.system_prompt = self.create_system_prompt()
    
    def create_system_prompt(self) -> str:
        return """You are an AI shopping assistant for an e-commerce platform. You help customers with:
        1. Finding products
        2. Checking order status
        3. Getting personalized recommendations
        4. Answering questions about products, categories, and sellers
        5. Processing returns and complaints
        6. Greets people and user and be friendly with them
        Be concise, helpful, and friendly. Always provide specific product details when available.
        For prices, always format as $XX.XX. For ratings, show them as X.X/5.
        """
    
    def initialize_intent_classifier(self) -> EnhancedIntentClassifier:
        n_intents = len(self.intents['intents'])
        n_categories = len(self.categories_df)
        model = EnhancedIntentClassifier(n_intents, n_categories)
        model.to(self.device)
        return model
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]
    
    def classify_intent(self, user_input: str) -> Tuple[str, float]:
        encoded_input = self.tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_token_type_ids=False
        ).to(self.device)
        
        with torch.no_grad():
            intent_logits, _ = self.intent_classifier(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask']
            )
            intent_probs = F.softmax(intent_logits, dim=1)
            max_prob, intent_idx = torch.max(intent_probs, dim=1)
            
        intent = self.intents['intents'][intent_idx.item()]['tag']
        confidence = max_prob.item()
        
        return intent, confidence
    
    def search_products(self, query: str) -> pd.DataFrame:
        query = query.lower()
        matching_products = self.products_df[
            self.products_df['ProductName'].str.lower().str.contains(query) |
            self.products_df['Description'].str.lower().str.contains(query)
        ]
        
        if matching_products.empty:
            matching_categories = self.categories_df[
                self.categories_df['CategoryName'].str.lower().str.contains(query) |
                self.categories_df['Description'].str.lower().str.contains(query)
            ]
            if not matching_categories.empty:
                category_ids = matching_categories['CategoryID'].tolist()
                matching_products = self.products_df[
                    self.products_df['CategoryID'].isin(category_ids)
                ]
        
        return matching_products
    
    def format_product_info(self, product: Dict) -> str:
        return (
            f"- {product['ProductName']}: ${product['Price']:.2f}\n"
            f"  Rating: {product['Rating']}/5\n"
            f"  Description: {product['Description']}\n"
        )
    
    def get_product_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
        user_orders = self.orders_df[self.orders_df['UserID'] == user_id]
        
        if user_orders.empty:
            return self.get_popular_products(n_recommendations)
            
        ordered_products = user_orders['ProductID'].unique()
        ordered_categories = self.products_df[
            self.products_df['ProductID'].isin(ordered_products)
        ]['CategoryID'].unique()
        
        recommendations = self.products_df[
            self.products_df['CategoryID'].isin(ordered_categories)
        ].sort_values('Rating', ascending=False).head(n_recommendations)
        
        return recommendations.to_dict('records')
    
    def get_popular_products(self, n_products: int = 5) -> List[Dict]:
        return self.products_df.sort_values('Rating', ascending=False).head(n_products).to_dict('records')
    
    def get_order_status(self, user_id: str) -> Dict:
        user_orders = self.orders_df[self.orders_df['UserID'] == user_id].sort_values('OrderDate', ascending=False)
        
        if user_orders.empty:
            return {"error": "No orders found for this user."}
            
        recent_order = user_orders.iloc[0]
        product_name = self.products_df[
            self.products_df['ProductID'] == recent_order['ProductID']
        ]['ProductName'].iloc[0]
        
        return {
            'order_id': recent_order['OrderID'],
            'product_name': product_name,
            'status': recent_order['Status'],
            'delivery_date': recent_order['DeliveryDate'],
            'delivered_by': recent_order['DeliveredBy'],
            'tracking_id': recent_order['TrackID']
        }
    
    def process_message(self, user_input: str, user_id: str = None) -> str:
        # Add user message to conversation history
        self.add_message("user", user_input)
        
        # Get intent classification
        intent, confidence = self.classify_intent(user_input)
        
        # Prepare messages for Groq API
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages
        ]
        
        try:
            if confidence > 0.7:  # High confidence in intent classification
                if "product search" in intent.lower():
                    matching_products = self.search_products(user_input)
                    if matching_products.empty:
                        response = "I couldn't find any products matching your search. Would you like to see our popular products instead?"
                    else:
                        response = "Here are the products I found:\n"
                        for _, product in matching_products.head(5).iterrows():
                            response += self.format_product_info(product.to_dict())
                
                elif "order status" in intent.lower() and user_id:
                    status_info = self.get_order_status(user_id)
                    if "error" in status_info:
                        response = status_info["error"]
                    else:
                        response = (
                            f"Your order for {status_info['product_name']} "
                            f"(Order ID: {status_info['order_id']}) is {status_info['status']}.\n"
                            f"Tracking ID: {status_info['tracking_id']}\n"
                        )
                        if status_info['status'] == 'Delivered':
                            response += f"Delivered on {status_info['delivery_date']} by {status_info['delivered_by']}"
                
                elif "recommendation" in intent.lower() and user_id:
                    recommendations = self.get_product_recommendations(user_id)
                    response = "Based on your history, you might like:\n"
                    for product in recommendations:
                        response += self.format_product_info(product)
                
                else:
                    # Use Groq for other intents
                    completion = self.groq_client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=200
                    )
                    response = completion.choices[0].message.content
            
            else:  # Low confidence, use Groq for general conversation
                completion = self.groq_client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                response = completion.choices[0].message.content
            
            # Add response to conversation history
            self.add_message("assistant", response)
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

def load_data():
    """Load and prepare the datasets"""
    products_df = pd.read_csv("Updated_Products_Dataset.csv")
    categories_df = pd.read_csv("Categories_Dataset.csv")
    orders_df = pd.read_csv("Orders_Dataset.csv")
    users_df = pd.read_csv("Users_Dataset.csv")
    sellers_df = pd.read_csv("Sellers_Dataset_Updated.csv")
    
    return products_df, categories_df, orders_df, users_df, sellers_df

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
async def main():
    st.title("E-commerce AI Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for user authentication
    with st.sidebar:
        st.header("User Authentication")
        user_id = st.text_input("Enter User ID", key="user_id_input")
        if st.button("Login"):
            st.session_state.user_id = user_id
            st.success(f"Logged in as User ID: {user_id}")
    
    # Initialize chatbot if not in session state
    if 'chatbot' not in st.session_state:
        # Load your API key and data
        groq_api_key = "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"
        products_df, categories_df, orders_df, users_df, sellers_df = load_data()
        
        # Initialize chatbot
        st.session_state.chatbot = EnhancedChatBot(
            groq_api_key=groq_api_key,
            products_df=products_df,
            categories_df=categories_df,
            orders_df=orders_df,
            users_df=users_df,
            sellers_df=sellers_df
        )
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.process_message(
                    prompt, 
                    st.session_state.user_id
                )
                st.write(response)
        
        # Update conversation history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

def run_streamlit():
    asyncio.run(main())

if __name__ == "__main__":
    run_streamlit()
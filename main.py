import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional

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
        self.products_df = products_df
        
    def search_products(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Search products using keyword matching"""
        query = query.lower()
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

class MistralAssistant:
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"):
        """Initialize the Mistral model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()  # Set to evaluation mode
            self.max_length = 2048
            logger.info("Mistral model initialized successfully")
        except Exception as e:
            logger.error(f"Error loading Mistral model: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        """Generate response using Mistral model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating the response."

class EcommerceAssistant:
    def __init__(self, products_df: pd.DataFrame, orders_df: pd.DataFrame, users_df: pd.DataFrame):
        self.products_df = products_df
        self.orders_df = orders_df
        self.users_df = users_df
        self.search_engine = ProductSearchEngine(products_df)
        
        try:
            self.mistral = MistralAssistant()
        except Exception as e:
            logger.error(f"Failed to initialize Mistral: {str(e)}")
            self.mistral = None
            
        self.templates = {
            'product_search': """
            As a helpful shopping assistant, provide recommendations for the following products:
            {products}
            Consider the user's query: {query}
            Highlight key features and suggest the best options based on the query.
            """,
            'order_status': """
            Provide a friendly update about the following order:
            Order Details: {order_details}
            Include delivery estimates and any relevant tracking information.
            """,
            'recommendations': """
            Based on the user's purchase history:
            {purchase_history}
            Suggest relevant products they might enjoy:
            {recommendations}
            Explain why each product would be a good fit.
            """
        }

    def format_product_info(self, products_df: pd.DataFrame) -> str:
        product_info = []
        for _, product in products_df.iterrows():
            info = {
                'name': product['ProductName'],
                'price': f"${product['Price']:.2f}",
                'description': product['Description'],
                'category': product.get('Category', 'N/A')
            }
            product_info.append(info)
        return str(product_info)

    def get_order_status(self, user_id: str) -> Optional[OrderInfo]:
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
        try:
            # Process different types of queries
            if any(word in message.lower() for word in ['search', 'find', 'looking', 'show']):
                products = self.search_engine.search_products(message)
                prompt = self.templates['product_search'].format(
                    products=self.format_product_info(products),
                    query=message
                )
                
            elif user_id and any(word in message.lower() for word in ['order', 'status', 'tracking']):
                order_info = self.get_order_status(user_id)
                if not order_info:
                    return "I couldn't find any orders for your account. Would you like to start shopping?"
                prompt = self.templates['order_status'].format(
                    order_details=str(order_info)
                )
                
            elif user_id and any(word in message.lower() for word in ['recommend', 'suggestion']):
                recommendations = self.get_user_recommendations(user_id)
                prompt = self.templates['recommendations'].format(
                    purchase_history=self.format_product_info(
                        self.products_df[self.products_df['ProductID'].isin(
                            self.orders_df[self.orders_df['UserID'] == user_id]['ProductID']
                        )]
                    ),
                    recommendations=self.format_product_info(recommendations)
                )
            else:
                prompt = message

            # Generate response using Mistral if available
            if self.mistral:
                return self.mistral.generate_response(prompt)
            else:
                # Fallback to template-based responses
                if 'products' in locals():
                    return self._format_product_response(products)
                elif 'order_info' in locals():
                    return self._format_order_response(order_info)
                elif 'recommendations' in locals():
                    return self._format_recommendation_response(recommendations)
                else:
                    return "I apologize, but I can only provide basic responses at the moment. Please try specific queries about products, orders, or recommendations."
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"

    def _format_product_response(self, products: pd.DataFrame) -> str:
        """Format product search results without LLM"""
        response = "Here are some products that match your search:\n\n"
        for _, product in products.iterrows():
            response += (f"• {product['ProductName']}\n"
                        f"  Price: ${product['Price']:.2f}\n"
                        f"  {product['Description']}\n\n")
        return response

    def _format_order_response(self, order_info: OrderInfo) -> str:
        """Format order status without LLM"""
        return (f"Here's your order status:\n\n"
                f"Order ID: {order_info.order_id}\n"
                f"Product: {order_info.product_name}\n"
                f"Status: {order_info.status}\n"
                f"Order Date: {order_info.order_date}\n"
                f"Expected Delivery: {order_info.delivery_date}\n"
                f"Tracking ID: {order_info.tracking_id}")

    def _format_recommendation_response(self, recommendations: pd.DataFrame) -> str:
        """Format recommendations without LLM"""
        response = "Based on your purchase history, you might like:\n\n"
        for _, product in recommendations.iterrows():
            response += (f"• {product['ProductName']}\n"
                        f"  Price: ${product['Price']:.2f}\n"
                        f"  {product['Description']}\n\n")
        return response

def initialize_app():
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
    if not initialize_app():
        return
    
    st.title("🛍️ Smart E-commerce Assistant")
    
    with st.sidebar:
        st.header("👤 User Authentication")
        if not st.session_state.logged_in:
            user_id = st.text_input("Enter User ID")
            if st.button("Login"):
                if user_id in st.session_state.assistant.users_df['UserID'].values:
                    st.session_state.user_id = user_id
                    st.session_state.logged_in = True
                    st.success(f"Welcome back, User {user_id}!")
                    st.rerun()
                else:
                    st.error("Invalid User ID")
        else:
            st.success(f"Logged in as User {st.session_state.user_id}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.logged_in = False
                st.rerun()
    
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

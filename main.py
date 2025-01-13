import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import re
from collections import defaultdict
import numpy as np

class ProductSearchEngine:
    def __init__(self, products_df, categories_df):
        self.products_df = products_df
        self.categories_df = categories_df
        self.search_history = defaultdict(int)
        
    def preprocess_query(self, query: str) -> str:
        """Clean and standardize search query"""
        # Convert to lowercase and remove special characters
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        # Remove extra whitespace
        query = ' '.join(query.split())
        return query
    
    def search_products(self, query: str, filters: Dict = None) -> pd.DataFrame:
        """Advanced product search with filtering"""
        query = self.preprocess_query(query)
        self.search_history[query] += 1
        
        # Basic search in product names and descriptions
        matching_products = self.products_df[
            self.products_df['ProductName'].str.lower().str.contains(query) |
            self.products_df['Description'].str.lower().str.contains(query)
        ]
        
        # Apply category-based search if no direct matches
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
        
        # Apply filters if provided
        if filters:
            if 'min_price' in filters:
                matching_products = matching_products[
                    matching_products['Price'] >= filters['min_price']
                ]
            if 'max_price' in filters:
                matching_products = matching_products[
                    matching_products['Price'] <= filters['max_price']
                ]
            if 'min_rating' in filters:
                matching_products = matching_products[
                    matching_products['Rating'] >= filters['min_rating']
                ]
            if 'categories' in filters:
                matching_products = matching_products[
                    matching_products['CategoryID'].isin(filters['categories'])
                ]
        
        return matching_products.sort_values('Rating', ascending=False)

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            'product_search': [
                r'\b(?:search|find|looking for|show|want)\b.*\b(?:product|item)s?\b',
                r'\bwhere\b.*\b(?:buy|get|purchase)\b',
                r'\b(?:recommend|suggest)\b.*\b(?:product|item)s?\b'
            ],
            'order_status': [
                r'\b(?:order|delivery|shipping|track)\b.*\b(?:status|location|update)\b',
                r'\bwhere\b.*\b(?:order|package|delivery)\b',
                r'\bcheck\b.*\b(?:order|delivery)\b'
            ],
            'recommendation': [
                r'\b(?:recommend|suggest|similar)\b',
                r'\bwhat\b.*\b(?:like|recommend|suggest)\b',
                r'\bshow\b.*\b(?:similar|related)\b'
            ],
            'price_query': [
                r'\b(?:price|cost|how much)\b',
                r'\b(?:expensive|cheap)\b',
                r'\b(?:discount|offer|deal)\b'
            ],
            'login_related': [
                r'\b(?:login|signin|sign in|account|profile)\b',
                r'\b(?:register|signup|sign up)\b'
            ]
        }
    
    def classify(self, text: str) -> str:
        text = text.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        return 'general'

class ResponseGenerator:
    def __init__(self, products_df, categories_df):
        self.products_df = products_df
        self.categories_df = categories_df
        
    def format_product_info(self, product: Dict, detailed: bool = False) -> str:
        category_name = self.categories_df[
            self.categories_df['CategoryID'] == product['CategoryID']
        ]['CategoryName'].iloc[0]
        
        basic_info = (
            f"🏷️ {product['ProductName']}\n"
            f"💰 ${product['Price']:.2f}\n"
            f"⭐ {product['Rating']}/5\n"
        )
        
        if detailed:
            return (
                f"{basic_info}"
                f"📝 Description: {product['Description']}\n"
                f"📦 Category: {category_name}\n"
                f"🏢 Seller ID: {product['SellerID']}\n"
                f"-------------------\n"
            )
        return basic_info
    
    def generate_product_list(self, products: pd.DataFrame, detailed: bool = False) -> str:
        if products.empty:
            return "No products found matching your criteria."
        
        response = "Here are the products I found:\n\n"
        for _, product in products.head(5).iterrows():
            response += self.format_product_info(product.to_dict(), detailed)
        
        if len(products) > 5:
            response += f"\n... and {len(products) - 5} more products"
        
        return response
    
    def format_order_status(self, order_info: Dict) -> str:
        status_emoji = {
            'Pending': '⏳',
            'Processing': '🔄',
            'Shipped': '🚚',
            'Delivered': '✅',
            'Cancelled': '❌'
        }
        
        emoji = status_emoji.get(order_info['status'], '📦')
        
        return (
            f"{emoji} Order Status\n"
            f"Product: {order_info['product_name']}\n"
            f"Order ID: {order_info['order_id']}\n"
            f"Status: {order_info['status']}\n"
            f"Order Date: {order_info['order_date']}\n"
            f"Expected Delivery: {order_info['delivery_date']}\n"
            f"Tracking ID: {order_info['tracking_id']}"
        )

class EnhancedChatBot:
    def __init__(self, products_df, categories_df, orders_df, users_df, sellers_df):
        # Initialize components
        self.search_engine = ProductSearchEngine(products_df, categories_df)
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator(products_df, categories_df)
        
        # Store datasets
        self.products_df = products_df
        self.categories_df = categories_df
        self.orders_df = orders_df
        self.users_df = users_df
        self.sellers_df = sellers_df
        
        # Initialize session data
        self.context = {}
        self.last_products_shown = None
    
    def verify_user(self, user_id: str) -> bool:
        return user_id in self.users_df['UserID'].values
    
    def get_order_status(self, user_id: str) -> Dict:
        user_orders = self.orders_df[self.orders_df['UserID'] == user_id].sort_values('OrderDate', ascending=False)
        
        if user_orders.empty:
            return {"error": "No orders found for this user."}
            
        recent_order = user_orders.iloc[0]
        product_info = self.products_df[self.products_df['ProductID'] == recent_order['ProductID']].iloc[0]
        
        return {
            'order_id': recent_order['OrderID'],
            'product_name': product_info['ProductName'],
            'status': recent_order['Status'],
            'order_date': recent_order['OrderDate'],
            'delivery_date': recent_order['DeliveryDate'],
            'tracking_id': recent_order['TrackID']
        }
    
    def get_recommendations(self, user_id: str) -> List[Dict]:
        user_orders = self.orders_df[self.orders_df['UserID'] == user_id]
        
        if user_orders.empty:
            top_products = self.products_df.nlargest(5, 'Rating').to_dict('records')
            return {'products': top_products, 'message': 'Popular products you might like:'}
        
        ordered_products = user_orders['ProductID'].unique()
        ordered_categories = self.products_df[
            self.products_df['ProductID'].isin(ordered_products)
        ]['CategoryID'].unique()
        
        recommendations = self.products_df[
            (self.products_df['CategoryID'].isin(ordered_categories)) &
            (~self.products_df['ProductID'].isin(ordered_products))
        ].nlargest(5, 'Rating').to_dict('records')
        
        return {
            'products': recommendations,
            'message': 'Based on your purchase history, you might like:'
        }
    
    def process_message(self, user_input: str, user_id: str = None) -> str:
        try:
            intent = self.intent_classifier.classify(user_input)
            
            # Handle login-related queries
            if intent == 'login_related':
                return (
                    "To access personalized features like order status and recommendations, "
                    "please log in using your User ID. You can find the login option in the sidebar."
                )
            
            # Handle product search (no login required)
            if intent == 'product_search':
                products = self.search_engine.search_products(user_input)
                self.last_products_shown = products
                return self.response_generator.generate_product_list(products)
            
            # Handle price queries (no login required)
            if intent == 'price_query':
                if self.last_products_shown is not None and not self.last_products_shown.empty:
                    return self.response_generator.generate_product_list(
                        self.last_products_shown, 
                        detailed=True
                    )
                products = self.search_engine.search_products(user_input)
                return self.response_generator.generate_product_list(products, detailed=True)
            
            # Features requiring login
            if intent in ['order_status', 'recommendation']:
                if not user_id:
                    return (
                        "To check your order status or get personalized recommendations, "
                        "please log in first using your User ID from the sidebar. "
                        "This helps me provide you with accurate information about your orders and preferences."
                    )
                
                if not self.verify_user(user_id):
                    return "The provided User ID is not valid. Please check and try again."
                
                if intent == 'order_status':
                    status_info = self.get_order_status(user_id)
                    if "error" in status_info:
                        return status_info["error"]
                    return self.response_generator.format_order_status(status_info)
                
                if intent == 'recommendation':
                    rec_info = self.get_recommendations(user_id)
                    response = rec_info['message'] + "\n\n"
                    for product in rec_info['products']:
                        response += self.response_generator.format_product_info(product)
                    return response
            
            # General queries and fallback
            return (
                "I can help you with:\n"
                "1. Searching for products and checking prices\n"
                "2. Checking your order status (requires login)\n"
                "3. Getting personalized recommendations (requires login)\n"
                "What would you like to know?"
            )
            
        except Exception as e:
            return (
                "I apologize, but I encountered an error while processing your request. "
                "Please try again or rephrase your question."
            )

def load_data():
    """Load and prepare the datasets"""
    products_df = pd.read_csv("Updated_Products_Dataset.csv")
    categories_df = pd.read_csv("Categories_Dataset.csv")
    orders_df = pd.read_csv("Orders_Dataset.csv")
    users_df = pd.read_csv("Users_Dataset.csv")
    sellers_df = pd.read_csv("Sellers_Dataset_Updated.csv")
    
    return products_df, categories_df, orders_df, users_df, sellers_df

def initialize_session_state():
    """Initialize or reset Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

def main():
    st.title("🛍️ E-commerce AI Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Load data
    products_df, categories_df, orders_df, users_df, sellers_df = load_data()
    
    # Initialize chatbot if not in session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EnhancedChatBot(
            products_df=products_df,
            categories_df=categories_df,
            orders_df=orders_df,
            users_df=users_df,
            sellers_df=sellers_df
        )
    
    # Sidebar for user authentication
    with st.sidebar:
        st.header("👤 User Authentication")
        if not st.session_state.logged_in:
            user_id = st.text_input("Enter User ID (optional)", key="user_id_input")
            if st.button("Login"):
                if st.session_state.chatbot.verify_user(user_id):
                    st.session_state.user_id = user_id
                    st.session_state.logged_in = True
                    st.success(f"Successfully logged in as User ID: {user_id}")
                    st.rerun()
                else:
                    st.error("Invalid User ID. Please try again.")
        else:
            st.success(f"Logged in as User ID: {st.session_state.user_id}")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.logged_in = False
                st.rerun()
    
    # Display welcome message for new sessions
    if not st.session_state.messages:
        st.chat_message("assistant").write(
            "👋 Hello! I'm your E-commerce Assistant. I can help you:\n"
            "• Search for products\n"
            "• Check prices\n"
            "• View order status (requires login)\n"
            "• Get personalized recommendations (requires login)\n\n"
            "What would you like to know?"
        )
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input and processing
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.chatbot.process_message(
                    prompt, 
                    st.session_state.user_id if st.session_state.logged_in else None
                )
                st.write(response)
        
        # Update conversation history
        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

def add_custom_styling():
    """Add custom CSS styling to the Streamlit interface"""
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            background-color: #f0f2f6;
        }
        
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #45a049;
        }
        
        .success-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #d4edda;
            color: #155724;
            margin: 10px 0;
        }
        
        .error-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #f8d7da;
            color: #721c24;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def display_session_info():
    """Display session information in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Session Info")
        
        # Display login status
        status = "🟢 Logged In" if st.session_state.logged_in else "⚪ Not Logged In"
        st.markdown(f"**Status:** {status}")
        
        if st.session_state.logged_in:
            st.markdown(f"**User ID:** {st.session_state.user_id}")
            
        # Display message count
        message_count = len(st.session_state.messages)
        st.markdown(f"**Messages:** {message_count}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

@handle_errors
def load_and_verify_data():
    """Load and verify all required datasets"""
    try:
        products_df = pd.read_csv("Updated_Products_Dataset.csv")
        categories_df = pd.read_csv("Categories_Dataset.csv")
        orders_df = pd.read_csv("Orders_Dataset.csv")
        users_df = pd.read_csv("Users_Dataset.csv")
        sellers_df = pd.read_csv("Sellers_Dataset_Updated.csv")
        
        # Verify required columns
        required_columns = {
            'products_df': ['ProductID', 'ProductName', 'Price', 'Description', 'CategoryID', 'SellerID', 'Rating'],
            'categories_df': ['CategoryID', 'CategoryName', 'Description'],
            'orders_df': ['OrderID', 'UserID', 'ProductID', 'OrderDate', 'Status', 'DeliveryDate', 'TrackID'],
            'users_df': ['UserID'],
            'sellers_df': ['SellerID']
        }
        
        for df_name, columns in required_columns.items():
            df = locals()[df_name]
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in {df_name}: {missing_columns}")
        
        return products_df, categories_df, orders_df, users_df, sellers_df
    
    except FileNotFoundError as e:
        st.error(f"Required dataset file not found: {str(e)}")
        st.stop()
    except ValueError as e:
        st.error(f"Dataset validation error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading datasets: {str(e)}")
        st.stop()

def initialize_app():
    """Initialize the application with all required components"""
    # Set page config
    st.set_page_config(
        page_title="E-commerce AI Assistant",
        page_icon="🛍️",
        layout="wide"
    )
    
    # Add custom styling
    add_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Load and verify data
    return load_and_verify_data()

if __name__ == "__main__":
    # Initialize the application
    data = initialize_app()
    
    if data is not None:
        products_df, categories_df, orders_df, users_df, sellers_df = data
        
        # Create main application layout
        main_container = st.container()
        
        with main_container:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                main()  # Run the main chat interface
            
            with col2:
                display_session_info()  # Display session information
                
                # Additional features for logged-in users
                if st.session_state.logged_in:
                    st.markdown("---")
                    st.markdown("### Quick Actions")
                    
                    if st.button("📦 View Recent Orders"):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": "Show my recent orders"
                        })
                    
                    if st.button("💝 Get Recommendations"):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": "Show me product recommendations"
                        })
    else:
        st.error("Failed to initialize the application. Please check the error messages above.")
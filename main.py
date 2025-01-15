import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

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
        """Initialize search engine with product embeddings"""
        self.products_df = products_df
        self.embeddings = OpenAIEmbeddings()
        self.setup_vector_store()
        
    def setup_vector_store(self):
        """Create FAISS vector store for semantic search"""
        try:
            # Combine product information for better semantic search
            texts = [
                f"{row['ProductName']} {row['Description']} {row['Category']}"
                for _, row in self.products_df.iterrows()
            ]
            
            # Create FAISS index
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"id": i} for i in range(len(texts))]
            )
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def search_products(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Search products using semantic search"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            product_indices = [doc.metadata["id"] for doc, _ in results]
            return self.products_df.iloc[product_indices]
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Fallback keyword-based search"""
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

class EcommerceAssistant:
    def __init__(self, products_df: pd.DataFrame, orders_df: pd.DataFrame, users_df: pd.DataFrame):
        self.products_df = products_df
        self.orders_df = orders_df
        self.users_df = users_df
        self.search_engine = ProductSearchEngine(products_df)
        
        # Initialize ChatGPT model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        
        # Set up conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize prompt templates
        self.templates = {
            'product_search': ChatPromptTemplate.from_template("""
                You are a knowledgeable e-commerce assistant. Based on the following products:
                {products}
                
                And the user's search query: {query}
                
                Provide a helpful response that:
                1. Highlights the most relevant products
                2. Explains why each product matches their needs
                3. Mentions key features and pricing
                4. Suggests alternatives if relevant
                
                Keep the tone friendly and conversational.
            """),
            
            'order_status': ChatPromptTemplate.from_template("""
                You are a helpful customer service agent. For the following order:
                {order_details}
                
                Provide a friendly status update that includes:
                1. Current order status
                2. Expected delivery date
                3. Tracking information if available
                4. Next steps or recommendations
                
                Keep the tone positive and helpful.
            """),
            
            'recommendations': ChatPromptTemplate.from_template("""
                You are a personalized shopping assistant. Based on:
                
                Purchase History:
                {purchase_history}
                
                Recommended Products:
                {recommendations}
                
                Provide personalized recommendations that:
                1. Explain why each product was chosen
                2. Connect it to their past purchases
                3. Highlight key features and benefits
                4. Include pricing information
                
                Keep the tone engaging and personalized.
            """)
        }
        
        # Initialize chains
        self.chains = {
            'product_search': LLMChain(
                llm=self.llm,
                prompt=self.templates['product_search'],
                memory=self.memory
            ),
            'order_status': LLMChain(
                llm=self.llm,
                prompt=self.templates['order_status'],
                memory=self.memory
            ),
            'recommendations': LLMChain(
                llm=self.llm,
                prompt=self.templates['recommendations'],
                memory=self.memory
            )
        }

    def format_product_info(self, products_df: pd.DataFrame) -> str:
        """Format product information for LLM prompts"""
        product_info = []
        for _, product in products_df.iterrows():
            info = (f"Product: {product['ProductName']}\n"
                   f"Price: ${product['Price']:.2f}\n"
                   f"Category: {product.get('Category', 'N/A')}\n"
                   f"Description: {product['Description']}\n")
            product_info.append(info)
        return "\n".join(product_info)

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

    async def process_message(self, message: str, user_id: str = None) -> str:
        """Process user message and generate response"""
        try:
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Process different types of queries
            if any(word in message.lower() for word in ['search', 'find', 'looking', 'show']):
                products = self.search_engine.search_products(message)
                response = await self.chains['product_search'].arun(
                    products=self.format_product_info(products),
                    query=message,
                    callbacks=[st_callback]
                )
                
            elif user_id and any(word in message.lower() for word in ['order', 'status', 'tracking']):
                order_info = self.get_order_status(user_id)
                if not order_info:
                    return "I couldn't find any orders for your account. Would you like to start shopping?"
                response = await self.chains['order_status'].arun(
                    order_details=str(order_info.__dict__),
                    callbacks=[st_callback]
                )
                
            elif user_id and any(word in message.lower() for word in ['recommend', 'suggestion']):
                recommendations = self.get_user_recommendations(user_id)
                response = await self.chains['recommendations'].arun(
                    purchase_history=self.format_product_info(
                        self.products_df[self.products_df['ProductID'].isin(
                            self.orders_df[self.orders_df['UserID'] == user_id]['ProductID']
                        )]
                    ),
                    recommendations=self.format_product_info(recommendations),
                    callbacks=[st_callback]
                )
            else:
                # General conversation
                response = await self.llm.apredict(
                    message,
                    callbacks=[st_callback]
                )
            
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

async def main():
    """Main application function"""
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
            response = await st.session_state.assistant.process_message(
                prompt,
                st.session_state.user_id if st.session_state.logged_in else None
            )
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

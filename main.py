import streamlit as st
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
import logging
from typing import List, Dict, Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GPT-2 Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')

# Ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class TextEmbeddingGenerator:
    @staticmethod
    def generate_embedding(text: str) -> torch.Tensor:
        """
        Generate embedding for given text using GPT-2
        
        Args:
            text (str): Input text to embed
        
        Returns:
            torch.Tensor: Text embedding
        """
        try:
            # Tokenize input
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Use the last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.squeeze()
        
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return torch.zeros(model.config.hidden_size)

class Product:
    def __init__(
        self, 
        product_id: str, 
        name: str, 
        price: float, 
        description: str
    ):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.description = description
        self.embedding = self._generate_embedding()
    
    def _generate_embedding(self) -> torch.Tensor:
        """Generate embedding for product"""
        return TextEmbeddingGenerator.generate_embedding(
            f"{self.name} {self.description}"
        )

class ProductSearchEngine:
    def __init__(self, products: List[Product]):
        self.products = products
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Product]:
        """
        Perform semantic search using embedding similarity
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            List[Product]: Top matching products
        """
        try:
            # Generate query embedding
            query_embedding = TextEmbeddingGenerator.generate_embedding(query)
            
            # Calculate similarities
            similarities = []
            for product in self.products:
                similarity = torch.nn.functional.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    product.embedding.unsqueeze(0)
                ).item()
                
                similarities.append((product, similarity))
            
            # Sort and return top k products
            return [
                product for product, _ in 
                sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            ]
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

class EcommerceAssistant:
    def __init__(self, products_df: pd.DataFrame):
        """
        Initialize assistant with product data
        
        Args:
            products_df (pd.DataFrame): DataFrame of products
        """
        # Convert DataFrame to Product objects
        self.products = [
            Product(
                product_id=str(row['ProductID']),
                name=row['ProductName'],
                price=float(row['Price']),
                description=row.get('Description', '')
            ) for _, row in products_df.iterrows()
        ]
        
        # Initialize search engine
        self.search_engine = ProductSearchEngine(self.products)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query and generate response
        
        Args:
            query (str): User input query
        
        Returns:
            Dict: Response with search results
        """
        try:
            # Perform semantic search
            search_results = self.search_engine.semantic_search(query)
            
            # Generate response using GPT-2
            response = self._generate_response(query, search_results)
            
            return {
                'search_results': [
                    {
                        'product_id': p.product_id,
                        'name': p.name,
                        'price': p.price,
                        'description': p.description
                    } for p in search_results
                ],
                'response': response
            }
        
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return {'error': str(e)}
    
    def _generate_response(
        self, 
        query: str, 
        products: List[Product]
    ) -> str:
        """
        Generate a contextual response based on query and products
        
        Args:
            query (str): User's original query
            products (List[Product]): Matched products
        
        Returns:
            str: Generated response
        """
        try:
            # Prepare context
            product_context = "\n".join([
                f"{p.name} - ${p.price:.2f}: {p.description}" 
                for p in products
            ])
            
            # Construct prompt
            prompt = (
                f"User Query: {query}\n"
                f"Matching Products:\n{product_context}\n"
                "Generate a helpful and engaging response:"
            )
            
            # Tokenize input
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate response
            output = model.generate(
                input_ids, 
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            # Decode response
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm sorry, I couldn't generate a response."

def main():
    """Streamlit main application"""
    st.title("🤖 AI Product Assistant")
    
    # Load products data
    try:
        products_df = pd.read_csv('Updated_Products_Dataset.csv')
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return
    
    # Initialize assistant
    assistant = EcommerceAssistant(products_df)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What product are you looking for?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query and generate response
        with st.chat_message("assistant"):
            try:
                # Process query
                result = assistant.process_query(prompt)
                
                # Display search results
                if 'search_results' in result:
                    st.write("Recommended Products:")
                    for product in result['search_results'][:3]:
                        st.write(f"- {product['name']} (${product['price']:.2f})")
                
                # Display AI-generated response
                if 'response' in result: st.markdown(result['response'])

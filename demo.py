import streamlit as st
import pandas as pd
import os
import random
import datetime
import time # Added for latency measurement
from groq import Groq
from typing import List, Dict
from dotenv import load_dotenv 

# ==============================
# 1. CONFIGURATION
# ==============================
# Load environment variables from the .env file.
load_dotenv() 

# --- Configuration ---
LOG_FILE = 'chat_log.csv' # File to save detailed interactions
FILE_PATHS = {
    'sentiment': 'all-data.csv',
    'structured_data': 'data.csv',
    'transactions': 'Personal_Finance_Dataset.csv'
}
GROQ_MODEL = "llama-3.1-8b-instant" # Fast Groq model suitable for RAG and chat

# ==============================
# 2. DATA LOGGING FUNCTIONALITY
# ==============================

def log_interaction(data: Dict):
    """
    Persistently logs interaction data to a CSV file for research analysis.
    This saves the user query, response, mode, and context.
    """
    try:
        # Check if the log file exists to determine if a header is needed
        file_exists = os.path.exists(LOG_FILE)
        
        # Convert the single interaction to a DataFrame
        log_df = pd.DataFrame([data])
        
        # Append to the CSV file
        log_df.to_csv(
            LOG_FILE, 
            mode='a', 
            header=not file_exists, 
            index=False, 
            encoding='utf-8'
        )
    except Exception as e:
        st.error(f"Error logging data: {e}")


# --- Mock RAG / Vector Store Setup ---
class MockEmbeddings:
    """Simulates a dummy embedding model."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[random.random() for _ in range(384)] for _ in range(len(texts))]

class MockVectorStore:
    """Simulates a vector database for storage and retrieval."""
    def __init__(self):
        self.vectors: Dict[str, str] = {} 

    def add_texts(self, texts: List[str]):
        """Adds text chunks to the mock store."""
        for i, text in enumerate(texts):
            self.vectors[f"doc_{i}"] = text
        if 'rag_ready' in st.session_state:
             st.session_state['rag_ready'] = True

    def similarity_search(self, query: str, k: int = 2) -> List[str]:
        """Simulates retrieval: returns the first k chunks for any query."""
        context_list = list(self.vectors.values())
        return random.sample(context_list, min(k, len(context_list)))


@st.cache_resource
def load_and_process_data():
    """Loads all data and sets up the RAG knowledge base."""
    st.info("Loading and processing financial datasets... This runs once.")
    
    # --- 1. Load Data ---
    try:
        df_sentiment = pd.read_csv(FILE_PATHS['sentiment'], encoding='latin1', header=None, names=['Sentiment', 'Text'])
        df_structured = pd.read_csv(FILE_PATHS['structured_data'])
        df_transactions = pd.read_csv(FILE_PATHS['transactions'])
        df_transactions['Date'] = pd.to_datetime(df_transactions['Date'])
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found. Ensure {FILE_PATHS.values()} are in the same directory. {e}")
        return None, None, None, None

    # --- 2. Setup RAG Knowledge Base (from all-data.csv) ---
    knowledge_base = MockVectorStore()
    documents = df_sentiment[df_sentiment['Sentiment'].isin(['neutral', 'positive'])]['Text'].tolist()
    knowledge_base.add_texts(documents)
    
    st.success("Data loaded and RAG knowledge base initialized.")
    
    # --- 3. Mock User Profile for Session ---
    mock_user_profile = df_structured.sample(1).iloc[0].to_dict()
    
    return knowledge_base, df_transactions, mock_user_profile, df_structured


# --- Groq API Client and Core Logic ---

def get_groq_client():
    """Initializes and returns the Groq client or None if key is missing."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.warning("GROQ_API_KEY environment variable not found. Chatbot will use a mock response.")
        return None
    try:
        return Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

def mock_groq_response(system_prompt: str, user_prompt: str) -> str:
    """Provides a mock response when the Groq API key is missing."""
    if "Decision Support Agent" in system_prompt:
        return "MOCK RESPONSE (API Key Missing): Based on the profile data you provided, I recommend prioritizing debt repayment over new investments for the next three months."
    elif "Financial Education Expert" in system_prompt:
        return "MOCK RESPONSE (API Key Missing): A 401(k) is a retirement savings plan sponsored by an employer. This information was retrieved from the RAG knowledge base (simulated)."
    return "MOCK RESPONSE (API Key Missing): Please set your GROQ_API_KEY environment variable to enable the Groq-accelerated LLM."


def groq_chat_completion(client: Groq, system_prompt: str, user_prompt: str) -> str:
    """Sends prompt to the Groq API."""
    if not client:
        return mock_groq_response(system_prompt, user_prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq API Error: {e}"


def handle_user_query(query: str, knowledge_base: MockVectorStore, df_transactions: pd.DataFrame, mock_user_profile: Dict, client: Groq):
    """
    Routes the user query to the appropriate LLM mode (RAG or Decision Support)
    and logs the interaction details.
    """
    start_time = time.time() # Start timer for latency

    query_lower = query.lower()
    
    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'user_query': query,
        'llm_mode': 'N/A',
        'rag_context': None,
        'profile_used': str(mock_user_profile), 
        'assistant_response': None,
        'latency_sec': 0.0 # Initialize latency
    }
    
    response = ""

    # --- MODE 1: Decision Support / Personalized Advice ---
    if any(keyword in query_lower for keyword in ["budget", "save", "spending", "advice", "analyze", "rent", "groceries", "income"]):
        
        log_data['llm_mode'] = 'Decision Support'
        
        # 1. Gather Context: Create a detailed string from the mock user profile
        profile_str = (
            f"Current User Profile:\n"
            f"Income: ${mock_user_profile.get('Income'):,.0f}, "
            f"Age: {mock_user_profile.get('Age')}, "
            f"Occupation: {mock_user_profile.get('Occupation')}, "
            f"Rent: ${mock_user_profile.get('Rent'):,.0f}, "
            f"Groceries: ${mock_user_profile.get('Groceries'):,.0f}, "
            f"Desired Savings: ${mock_user_profile.get('Desired_Savings'):,.0f}."
        )

        # 2. Gather Transactional Detail
        if any(keyword in query_lower for keyword in ["transactions", "recent spending"]):
            transaction_sample = df_transactions.sort_values('Date', ascending=False).head(5).to_markdown(index=False)
            profile_str += f"\n\nRecent Transactions:\n{transaction_sample}"
        
        system_prompt = (
            "You are a Personal Finance Decision Support Agent. Use the provided User Profile and transaction data "
            "to give a specific, actionable, and data-driven financial recommendation. Be direct and concise. "
            "Address the user's request based on the data."
        )

        user_prompt = f"{profile_str}\n\nUser's Request: {query}"
        
        response = groq_chat_completion(client, system_prompt, user_prompt)

    # --- MODE 2: Financial Education / RAG ---
    elif any(keyword in query_lower for keyword in ["what is", "explain", "define", "tell me about", "history"]):
        
        log_data['llm_mode'] = 'RAG (Education)'
        
        # 1. Retrieve Context from RAG
        rag_context = knowledge_base.similarity_search(query, k=3)
        log_data['rag_context'] = ' | '.join(rag_context) 
        
        system_prompt = (
            "You are a Financial Education Expert. Use the following RAG Context to answer the user's question. "
            "If the context is insufficient, use general financial knowledge. Be informative and educational. "
            f"RAG Context: {' '.join(rag_context)}"
        )

        response = groq_chat_completion(client, system_prompt, query)
        
    else:
        response = "I am the Groq-Accelerated Financial Literacy Chatbot. Please ask me to **'explain'** a concept or give you **'advice'** on your budget."
        log_data['llm_mode'] = 'No Match'
        
    # --- FINAL STEP: LOG THE INTERACTION ---
    end_time = time.time()
    log_data['latency_sec'] = round(end_time - start_time, 4) # Calculate and log latency

    log_data['assistant_response'] = response
    log_interaction(log_data)
    
    return response, log_data['llm_mode']

# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="Groq-Accelerated Financial Chatbot", layout="wide")
    st.title("💰 Financial Literacy Chatbot (Batch Input Mode)")
    st.caption(f"Powered by Groq and the **{GROQ_MODEL}** model.")

    # 1. Load Data and Setup RAG (Cached)
    knowledge_base, df_transactions, mock_user_profile, df_structured = load_and_process_data()

    if knowledge_base is None:
        return # Stop if file loading failed

    # 2. Initialize Groq Client and Session State
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = get_groq_client()
        st.session_state.results = [] # Store batch results here
        st.session_state.mock_user_profile = mock_user_profile

    # Display Mock Profile for transparency
    with st.sidebar:
        st.header("Simulated User Profile (from `data.csv`)")
        st.markdown(f"**Income:** ${st.session_state.mock_user_profile.get('Income', 0):,.0f}")
        st.markdown(f"**Age:** {st.session_state.mock_user_profile.get('Age', 'N/A')}")
        st.markdown(f"**Occupation:** {st.session_state.mock_user_profile.get('Occupation', 'N/A')}")
        st.markdown(f"**Monthly Rent:** ${st.session_state.mock_user_profile.get('Rent', 0):,.0f}")
        st.markdown(f"**Desired Savings:** ${st.session_state.mock_user_profile.get('Desired_Savings', 0):,.0f}")
        st.divider()
        st.success(f"**Research Log:** All interactions saved to `{LOG_FILE}`.")

    # --- Batch Input Section ---
    st.subheader("📝 Enter Queries in Batch (One Query Per Line)")
    
    # Text Area for batch input
    batch_input = st.text_area(
        "Enter your queries here:",
        height=300,
        placeholder="Example Queries:\nWhat is inflation?\nHow can I save on my budget?\nDefine a 401(k)."
    )
    
    process_button = st.button("🚀 Run Batch Queries and Log Results")
    
    st.divider()

    # --- Results Section ---
    st.subheader("📊 Batch Results")
    
    if process_button:
        if not batch_input.strip():
            st.warning("Please enter at least one query in the text area.")
            return

        # 1. Clean and split queries
        raw_queries = batch_input.strip().split('\n')
        queries = [q.strip() for q in raw_queries if q.strip()]
        
        st.session_state.results = []
        
        # 2. Process queries
        with st.spinner(f"Processing {len(queries)} queries... (Groq is fast!)"):
            
            for i, query in enumerate(queries):
                # Run the core logic for each query
                response, mode = handle_user_query(
                    query, 
                    knowledge_base, 
                    df_transactions, 
                    st.session_state.mock_user_profile, 
                    st.session_state.groq_client
                )
                
                st.session_state.results.append({
                    "Query": query,
                    "Mode": mode,
                    "Response (Snippet)": response[:100] + "..." if len(response) > 100 else response,
                    "Full Response": response # For detailed display
                })

        st.success(f"✅ Successfully processed and logged {len(queries)} queries to `{LOG_FILE}`.")

    # 3. Display Results in a Table
    if st.session_state.results:
        # Create a display DataFrame showing only the snippet and mode
        display_df = pd.DataFrame(st.session_state.results)
        st.dataframe(display_df.drop(columns=['Full Response']), use_container_width=True)
        
        # Allow user to expand for full response details
        st.markdown("### Full Response Details")
        for result in st.session_state.results:
            with st.expander(f"Query: **{result['Query']}** (Mode: {result['Mode']})"):
                st.code(result['Full Response'], language='markdown')


if __name__ == "__main__":
    main()
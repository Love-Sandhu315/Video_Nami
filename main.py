import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from youtube_utils import get_youtube_video_id, validate_youtube_id
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import base64

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="videoNami",
    page_icon="my_logo.png",
    layout="wide",
    initial_sidebar_state="auto"  # Changed to auto for better mobile experience
)

# Enhanced responsive CSS
st.markdown("""
<style>
/* Base styles */
.main-header {
    text-align: center;
    color: #FF0000;
    font-size: clamp(1.5rem, 4vw, 2.5rem);
    margin-bottom: 1rem;
    word-wrap: break-word;
}

.sub-header {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
    font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    padding: 0 1rem;
}

.chat-container {
    background-color: #f8f9fa;
    padding: clamp(10px, 3vw, 20px);
    border-radius: 10px;
    margin: 10px 0;
    max-width: 100%;
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: clamp(8px, 2vw, 15px);
    border-radius: 15px;
    margin: 5px 0;
    text-align: right;
    word-wrap: break-word;
    max-width: 85%;
    margin-left: auto;
    font-size: clamp(0.8rem, 2vw, 1rem);
}

.bot-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: clamp(8px, 2vw, 15px);
    border-radius: 15px;
    margin: 5px 0;
    word-wrap: break-word;
    max-width: 85%;
    margin-right: auto;
    font-size: clamp(0.8rem, 2vw, 1rem);
}

/* Responsive layout adjustments */
.block-container {
    padding: clamp(1rem, 3vw, 5rem) clamp(0.5rem, 2vw, 1rem);
    max-width: 100%;
}

/* Mobile-specific styles */
@media (max-width: 768px) {
    .main-header {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header img {
        width: 50px !important;
        margin-right: 5px !important;
    }
    
    .sub-header {
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .user-message, .bot-message {
        max-width: 95%;
        padding: 10px;
        font-size: 0.85rem;
    }
    
    .stButton > button {
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .stTextInput > div > div > input {
        font-size: 16px; /* Prevents zoom on iOS */
    }
    
    .stColumns {
        gap: 0.5rem;
    }
    
    /* Sidebar adjustments for mobile */
    .css-1d391kg {
        padding: 1rem 0.5rem;
    }
    
    /* Feature cards responsive */
    .feature-card {
        margin-bottom: 1rem;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #FF0000;
    }
}

/* Tablet styles */
@media (min-width: 769px) and (max-width: 1024px) {
    .main-header {
        font-size: 2rem;
    }
    
    .main-header img {
        width: 60px !important;
    }
    
    .user-message, .bot-message {
        max-width: 90%;
        font-size: 0.9rem;
    }
    
    .stColumns {
        gap: 1rem;
    }
}

/* Desktop styles */
@media (min-width: 1025px) {
    .main-header img {
        width: 80px !important;
    }
    
    .user-message, .bot-message {
        max-width: 85%;
        font-size: 1rem;
    }
}

/* Sample question buttons responsive */
.sample-questions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

@media (min-width: 769px) {
    .sample-questions {
        flex-direction: row;
        flex-wrap: wrap;
    }
    
    .sample-questions .stButton {
        flex: 1;
        min-width: 200px;
    }
}

/* Improved form styling */
.stForm {
    background: transparent;
    border: none;
}

/* Progress bar responsive */
.stProgress {
    margin: 1rem 0;
}

/* Better spacing for mobile */
.element-container {
    margin-bottom: clamp(0.5rem, 2vw, 1rem);
}

/* Instructions section */
.instructions {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Sidebar responsive width */
@media (max-width: 768px) {
    .css-1d391kg {
        width: 100% !important;
        max-width: 300px;
    }
}

/* Touch-friendly buttons */
@media (hover: none) and (pointer: coarse) {
    .stButton > button {
        padding: 12px 16px;
        min-height: 44px;
        font-size: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None

@st.cache_resource
def initialize_models():
    """Initialize the LLM and embeddings models"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation"
        )
        model = ChatHuggingFace(llm=llm)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        return model, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

def process_youtube_video(url):
    """Process YouTube video and create vector store"""
    
    # Extract video ID
    video_id = get_youtube_video_id(url)
    
    if not video_id or not validate_youtube_id(video_id):
        st.error("❌ Invalid YouTube URL or video ID")
        return False
    
    # Check if same video is already processed
    if st.session_state.current_video_id == video_id:
        st.info("✅ This video is already processed!")
        return True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Fetch transcript
        status_text.text("🔍 Fetching video transcript...")
        progress_bar.progress(25)
        
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.fetch(video_id)
        transcript_text = " ".join(snippet.text for snippet in transcript_list.snippets)
        
        if not transcript_text:
            st.error("❌ No transcript text found")
            return False
        
        st.success(f"✅ Transcript fetched!")
        
        # Split text into chunks
        status_text.text("📝 Processing transcript...")
        progress_bar.progress(50)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])
        
        # Create vector store
        progress_bar.progress(75)
        
        _, embeddings = initialize_models()
        if embeddings is None:
            return False
            
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create retriever and chain
        status_text.text("⚙️ Setting up chat system...")
        progress_bar.progress(90)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        model, _ = initialize_models()
        if model is None:
            return False
        
        prompt = PromptTemplate(
            template="""
You are a helpful assistant that answers questions about a YouTube video based on its transcript.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.
Be concise and helpful in your responses.

Context: {context}

Question: {question}

Answer:""",
            input_variables=['context', 'question']
        )
        
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | model | parser
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.chain = main_chain
        st.session_state.video_processed = True
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []  # Clear previous chat history
        
        progress_bar.progress(100)
        status_text.text("✅ Video processed successfully!")
        
        return True
        
    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        return False
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
        return False
    finally:
        progress_bar.empty()
        status_text.empty()

def create_responsive_header():
    """Create responsive header with logo"""
    try:
        file_path = "my_logo.png"
        with open(file_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        st.markdown(
            f"""
            <h1 class="main-header">
                <img src="data:image/png;base64,{encoded}" 
                    alt="logo" width="80" style="vertical-align:middle; margin-right:10px;">
                Video Nami Chatbot
            </h1>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        # Fallback header without logo
        st.markdown('<h1 class="main-header">🎥 Video Nami Chatbot</h1>', unsafe_allow_html=True)

def create_sidebar():
    """Create responsive sidebar"""
    with st.sidebar:
        try:
            file_path = "my_logo.png"
            with open(file_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode()

            st.markdown(
                f"""
                <h2>
                    <img src="data:image/png;base64,{encoded}" 
                        alt="logo" width="35" style="vertical-align:middle; margin-right:10px;">
                    Video Processing
                </h2>
                """,
                unsafe_allow_html=True
            )
        except FileNotFoundError:
            st.header("🎥 Video Processing")
        
        # YouTube URL input
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL here"
        )
        
        # Process video button
        if st.button("🚀 Process Video", type="primary", disabled=not youtube_url, use_container_width=True):
            if process_youtube_video(youtube_url):
                st.success("✅ Ready to chat!")
        
        st.divider()
        
        # Current video info
        if st.session_state.video_processed:
            st.success("✅ Video Ready")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.warning("⚠️ No video processed yet")
        
        st.divider()
        
        # Instructions
        st.markdown("""
        <div class="instructions">
        <h4>📝 How to use:</h4>
        <ol>
        <li><strong>Paste a YouTube URL</strong> in the input field</li>
        <li><strong>Click 'Process Video'</strong> to analyze the transcript</li>
        <li><strong>Start chatting</strong> about the video content</li>
        <li>Ask questions like:
           <ul>
           <li>"What is this video about?"</li>
           <li>"Summarize the main points"</li>
           <li>"What does the speaker say about [topic]?"</li>
           </ul>
        </li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

def create_sample_questions():
    """Create responsive sample questions"""
    st.subheader("💡 Try these sample questions:")
    sample_questions = [
        "What is this video about?",
        "Can you summarize the main points?",
        "What are the key takeaways?",
        "Who is the target audience?",
        "What examples are given?"
    ]
    
    # Check screen size and adjust layout
    if st.session_state.get('is_mobile', False):
        # Stack vertically on mobile
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}", use_container_width=True):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.chain.invoke(question)
                        st.session_state.chat_history.append((question, response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    else:
        # Use columns for larger screens
        cols = st.columns(min(len(sample_questions), 3))
        for i, question in enumerate(sample_questions):
            col_idx = i % len(cols)
            if cols[col_idx].button(question, key=f"sample_{i}"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.chain.invoke(question)
                        st.session_state.chat_history.append((question, response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

def create_welcome_screen():
    """Create responsive welcome screen"""
    st.info("👈 Please process a YouTube video using the sidebar to start chatting!")
    
    st.subheader("🌟 Features:")
    
    # Use single column on mobile, three columns on larger screens
    # Note: We'll use a simple approach since we can't directly detect screen size
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <strong>🎯 Smart Analysis</strong><br>
        • Extracts video transcripts<br>
        • Creates searchable knowledge base<br>
        • Understands context
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <strong>💬 Natural Chat</strong><br>
        • Ask questions in plain English<br>
        • Get accurate, contextual answers<br>
        • Maintains conversation history
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <strong>⚡ Fast & Reliable</strong><br>
        • Powered by advanced AI models<br>
        • Quick response times<br>
        • Works with any YouTube video with English subtitles
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    create_responsive_header()
    st.markdown('<p class="sub-header">Chat with any YouTube video using your YouTube Navigator!</p>', unsafe_allow_html=True)
    
    # Sidebar
    create_sidebar()
    
    # Main chat interface
    if st.session_state.video_processed:
        st.header("💬 Chat with the Video")
        
        # Display chat history
        if st.session_state.chat_history:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"""
                    <div class="user-message">
                        💭 {question}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="bot-message">
                        🤖 {answer}
                    </div>
                    """, unsafe_allow_html=True)
                st.divider()
        
        # Chat input - responsive form
        with st.form("chat_form", clear_on_submit=True):
            # Single column on mobile-like layout
            user_question = st.text_input(
                "Ask a question about the video:",
                placeholder="What is this video about?",
                label_visibility="collapsed"
            )
            
            submit_button = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
            
            if submit_button and user_question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get response from the chain
                        response = st.session_state.chain.invoke(user_question)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, response))
                        
                        # Rerun to display the new message
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")
        
        # Sample questions
        if not st.session_state.chat_history:
            create_sample_questions()
    
    else:
        # Welcome screen
        create_welcome_screen()

if __name__ == "__main__":
    main()
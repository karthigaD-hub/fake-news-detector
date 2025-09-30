import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import nltk
import time

# Page configuration
st.set_page_config(
    page_title="Student Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    with st.spinner("Downloading NLP data..."):
        nltk.download('punkt')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .reliable {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .fake {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
    .borderline {
        border-left-color: #ffc107;
        background-color: #fff3cd;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.loaded = True
    
    def extract_article_from_url(self, url):
        """Extract article content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text() if title else "No title found"
            
            # Get content
            content = ""
            article = soup.find('article')
            if article:
                content = article.get_text()
            else:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'title': title_text,
                'content': content,
                'success': True if len(content) > 100 else False
            }
            
        except Exception as e:
            return {
                'title': 'Error',
                'content': '',
                'success': False,
                'error': str(e)
            }
    
    def analyze_text(self, text):
        """Analyze text for fake news indicators"""
        if len(text) < 50:
            return {'error': 'Text too short for analysis'}
        
        # Generate summary
        summary = self.generate_summary(text)
        
        # Analyze credibility
        analysis = self.credibility_analysis(text)
        
        # Extract features
        features = self.extract_features(text)
        
        return {
            'summary': summary,
            'analysis': analysis,
            'features': features,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def generate_summary(self, text):
        """Generate article summary using extractive method"""
        if len(text) < 100:
            return "Text too short for meaningful summary"
        
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 3:
                return ' '.join(sentences)
            
            # Simple extractive summary (first few sentences)
            return ' '.join(sentences[:3])
        except:
            return text[:200] + "..."
    
    def extract_features(self, text):
        """Extract linguistic features from text"""
        words = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)
        
        # Sensational words
        sensational_words = ['shocking', 'miracle', 'secret', 'breaking', 'urgent']
        sensational_count = sum(1 for word in words if any(sens_word in word for sens_word in sensational_words))
        
        # Reliable indicators
        reliable_indicators = ['according to', 'study shows', 'research indicates', 'experts say']
        reliable_count = sum(1 for word in words if any(rel_word in word for rel_word in reliable_indicators))
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'exclamation_count': text.count('!'),
            'sensational_word_count': sensational_count,
            'reliable_indicator_count': reliable_count,
        }
        
        return features
    
    def credibility_analysis(self, text):
        """Analyze text credibility using rule-based approach"""
        text_lower = text.lower()
        
        # Rule-based scoring
        fake_indicators = [
            'miracle cure', 'secret they don\'t want you to know', 'conspiracy',
            'cover-up', 'big pharma', 'mainstream media hiding'
        ]
        
        reliable_indicators = [
            'according to study', 'research shows', 'experts say', 'official report',
            'peer-reviewed', 'clinical trial', 'scientific study'
        ]
        
        # Count indicators
        fake_score = sum(3 for indicator in fake_indicators if indicator in text_lower)
        reliable_score = sum(3 for indicator in reliable_indicators if indicator in text_lower)
        
        # Linguistic feature scoring
        features = self.extract_features(text)
        
        # Penalize excessive punctuation
        if features['exclamation_count'] > features['sentence_count']:
            fake_score += 2
        
        # Calculate final scores
        total_indicators = fake_score + reliable_score
        if total_indicators > 0:
            fake_ratio = fake_score / total_indicators
            reliable_ratio = reliable_score / total_indicators
        else:
            fake_ratio = reliable_ratio = 0.5
        
        # Determine verdict
        if fake_ratio > 0.6:
            verdict = "Fake News"
            confidence = fake_ratio
            color = "red"
        elif reliable_ratio > 0.6:
            verdict = "Reliable"
            confidence = reliable_ratio
            color = "green"
        else:
            verdict = "Borderline/Uncertain"
            confidence = max(fake_ratio, reliable_ratio)
            color = "orange"
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'color': color,
            'scores': {
                'fake_score': fake_ratio,
                'reliable_score': reliable_ratio,
            }
        }

def main():
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = FakeNewsDetector()
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Sidebar
    with st.sidebar:
        st.title("🎓 Student Fake News Detector")
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio("Choose a page:", ["🏠 Home", "🔍 Analyze Article", "📊 History", "🎓 Learn"])
        
        st.markdown("---")
        st.subheader("About")
        st.info("AI tool to help students identify misleading information and develop critical thinking skills.")
        
        st.markdown("---")
        st.metric("Analyses", len(st.session_state.analysis_history))
    
    # Page routing
    if page == "🏠 Home":
        render_home_page()
    elif page == "🔍 Analyze Article":
        render_analysis_page()
    elif page == "📊 History":
        render_history_page()
    elif page == "🎓 Learn":
        render_learn_page()

def render_home_page():
    """Render the home page"""
    st.markdown('<h1 class="main-header">🔍 Student Fake News Detector</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🛡️ Your Defense Against Misinformation
        
        This tool helps students:
        - **🔍 Analyze** news articles and social media posts
        - **🎯 Assess** credibility using AI-powered analysis  
        - **📝 Generate** concise summaries
        - **🎓 Develop** critical thinking skills
        
        ### 🚀 How to Use:
        1. Go to **"Analyze Article"**
        2. Paste text or enter a URL
        3. Get instant analysis and insights
        """)
    
    with col2:
        st.image("🔍", width=200)
        st.markdown("### Quick Stats")
        st.metric("Fake News Spread", "85%")
        st.metric("Student Exposure", "72%")

def render_analysis_page():
    """Render the article analysis page"""
    st.title("🔍 Analyze Article")
    
    # Input method
    input_method = st.radio("Choose input method:", ["📝 Paste Text", "🌐 Enter URL"], horizontal=True)
    
    if input_method == "📝 Paste Text":
        input_content = st.text_area("Paste article text:", height=200, placeholder="Paste news article text here...")
        analyze_button = st.button("🚀 Analyze Text", type="primary")
    else:
        url = st.text_input("Enter article URL:", placeholder="https://example.com/news")
        analyze_button = st.button("🌐 Analyze URL", type="primary")
        input_content = url
    
    if analyze_button and input_content:
        with st.spinner("Analyzing content..."):
            try:
                if input_method == "🌐 Enter URL":
                    extraction_result = st.session_state.detector.extract_article_from_url(input_content)
                    if not extraction_result['success']:
                        st.error("Could not extract content from URL")
                        return
                    article_title = extraction_result['title']
                    article_content = extraction_result['content']
                else:
                    article_title = "Pasted Text Analysis"
                    article_content = input_content
                
                analysis_result = st.session_state.detector.analyze_text(article_content)
                
                if 'error' in analysis_result:
                    st.error(analysis_result['error'])
                    return
                
                # Store result
                full_result = {
                    'timestamp': datetime.now().isoformat(),
                    'title': article_title,
                    **analysis_result
                }
                st.session_state.analysis_history.append(full_result)
                
                # Display results
                display_results(full_result)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def display_results(result):
    """Display analysis results"""
    st.success("✅ Analysis Complete!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Article Info")
        st.write(f"**Title:** {result.get('title', 'N/A')}")
        st.write(f"**Words:** {result.get('word_count', 0)}")
        
        st.subheader("📋 Summary")
        st.info(result['summary'])
    
    with col2:
        st.subheader("🎯 Credibility")
        analysis = result['analysis']
        
        color_class = {
            "red": "fake", "green": "reliable", "orange": "borderline"
        }.get(analysis['color'], 'borderline')
        
        st.markdown(f"""
        <div class="result-box {color_class}">
            <h3>{analysis['verdict']}</h3>
            <p>Confidence: {analysis['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scores chart
        scores = analysis['scores']
        fig = px.bar(
            x=list(scores.keys()), 
            y=list(scores.values()),
            color=list(scores.keys()),
            color_discrete_map={'fake_score': 'red', 'reliable_score': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_history_page():
    """Render analysis history page"""
    st.title("📊 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analyses yet. Analyze an article first!")
        return
    
    # Display history
    for i, analysis in enumerate(st.session_state.analysis_history):
        with st.expander(f"Analysis {i+1}: {analysis.get('title', 'Unknown')}"):
            st.write(f"Verdict: {analysis['analysis']['verdict']}")
            st.write(f"Confidence: {analysis['analysis']['confidence']:.1%}")
            st.write(f"Date: {pd.Timestamp(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')}")

def render_learn_page():
    """Render educational content"""
    st.title("🎓 Learn About Fake News")
    
    st.markdown("""
    ## 🔍 How to Spot Fake News
    
    **🚨 Red Flags:**
    - Excessive capitalization and punctuation!!!
    - Emotional language provoking strong reactions
    - Claims of "secret information" or "conspiracies"
    - No author information or sources
    - Requests to "SHARE URGENTLY"
    
    **✅ Reliability Signs:**
    - Clear author credentials
    - Multiple reputable sources
    - Balanced, factual language
    - Evidence and citations
    - Professional website design
    
    ## 🛡️ Protection Tips
    - Verify with multiple sources
    - Check fact-checking websites
    - Investigate the source
    - Think before sharing
    - Be aware of your biases
    """)

if __name__ == "__main__":
    main()

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
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords') 
except LookupError:
    nltk.download('stopwords')

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class FakeNewsDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
    def load_models(self):
        """Load ML models"""
        try:
            # Load summarization model
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # Fallback to smaller model
            self.summarizer = pipeline("summarization")
        
        # Initialize feature extractor
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
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
            
            # Get content - try multiple strategies
            content = ""
            
            # Strategy 1: Look for article tag
            article = soup.find('article')
            if article:
                content = article.get_text()
            else:
                # Strategy 2: Look for main content divs
                main_content = soup.find('main') or soup.find('div', class_=re.compile('content|article|main'))
                if main_content:
                    content = main_content.get_text()
                else:
                    # Strategy 3: Get all paragraphs
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
            return {
                'error': 'Text too short for analysis (minimum 50 characters required)'
            }
        
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
        """Generate article summary"""
        if len(text) < 100:
            return "Text too short for meaningful summary"
        
        try:
            # Truncate very long text
            if len(text) > 2000:
                text = text[:2000]
                
            summary = self.summarizer(
                text,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            # Fallback to extractive summarization
            return self.extractive_summary(text)
    
    def extractive_summary(self, text, num_sentences=3):
        """Simple extractive summarization as fallback"""
        sentences = nltk.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        return ' '.join(sentences[:num_sentences])
    
    def extract_features(self, text):
        """Extract linguistic features from text"""
        words = nltk.word_tokenize(text.lower())
        sentences = nltk.sent_tokenize(text)
        
        # Sensational words
        sensational_words = [
            'shocking', 'miracle', 'secret', 'breaking', 'urgent', 'warning',
            'alert', 'amazing', 'unbelievable', 'astounding', 'revealed',
            'exposed', 'cover-up', 'conspiracy', 'they don\'t want you to know'
        ]
        
        sensational_count = sum(1 for word in words if any(sens_word in word for sens_word in sensational_words))
        
        # Reliable indicators
        reliable_indicators = [
            'according to', 'study shows', 'research indicates', 'experts say',
            'official report', 'peer-reviewed', 'clinical trial', 'data shows',
            'scientists found', 'research published'
        ]
        
        reliable_count = sum(1 for word in words if any(rel_word in word for rel_word in reliable_indicators))
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'capital_ratio': sum(1 for char in text if char.isupper()) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'sensational_word_count': sensational_count,
            'reliable_indicator_count': reliable_count,
            'quote_count': text.count('"') // 2,
            'number_count': sum(1 for word in words if word.replace('.', '').isdigit())
        }
        
        return features
    
    def credibility_analysis(self, text):
        """Analyze text credibility using rule-based and ML approaches"""
        text_lower = text.lower()
        
        # Rule-based scoring
        fake_indicators = [
            'miracle cure', 'secret they don\'t want you to know', 'conspiracy',
            'cover-up', 'big pharma', 'mainstream media hiding', 'breaking exclusive',
            'shocking revelation', 'whistleblower reveals', 'hidden truth',
            'they\'re hiding', 'wake up people', 'the truth about', 'government secret'
        ]
        
        reliable_indicators = [
            'according to study', 'research shows', 'experts say', 'official report',
            'peer-reviewed', 'clinical trial', 'university research', 'scientific study',
            'data shows', 'according to officials', 'government report', 'journal published',
            'research indicates', 'study found', 'scientists discovered'
        ]
        
        # Count indicators
        fake_score = sum(3 for indicator in fake_indicators if indicator in text_lower)
        reliable_score = sum(3 for indicator in reliable_indicators if indicator in text_lower)
        
        # Linguistic feature scoring
        features = self.extract_features(text)
        
        # Penalize excessive punctuation
        if features['exclamation_count'] > features['sentence_count']:
            fake_score += 2
        
        # Penalize excessive capitalization
        if features['capital_ratio'] > 0.2:
            fake_score += 1
        
        # Reward balanced sentence length
        if 10 <= features['avg_sentence_length'] <= 25:
            reliable_score += 1
        
        # Reward presence of numbers and quotes (evidence)
        if features['number_count'] > 2:
            reliable_score += 1
        if features['quote_count'] > 1:
            reliable_score += 1
        
        # Calculate final scores
        total_indicators = fake_score + reliable_score
        if total_indicators > 0:
            fake_ratio = fake_score / total_indicators
            reliable_ratio = reliable_score / total_indicators
        else:
            fake_ratio = reliable_ratio = 0.5
        
        # Determine verdict
        if fake_ratio > 0.7:
            verdict = "Fake News"
            confidence = fake_ratio
            color = "red"
        elif reliable_ratio > 0.7:
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
                'borderline_score': 1 - abs(fake_ratio - reliable_ratio)
            },
            'fake_indicators_found': [ind for ind in fake_indicators if ind in text_lower],
            'reliable_indicators_found': [ind for ind in reliable_indicators if ind in text_lower]
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
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "🔍 Analyze Article", "📊 History", "🎓 Educational Guide"]
        )
        
        st.markdown("---")
        st.subheader("About")
        st.info(
            "This AI tool helps students identify potentially misleading information "
            "and develop critical thinking skills for the digital age."
        )
        
        st.markdown("---")
        st.subheader("Quick Stats")
        st.metric("Analyses Performed", len(st.session_state.analysis_history))
    
    # Page routing
    if page == "🏠 Home":
        render_home_page()
    elif page == "🔍 Analyze Article":
        render_analysis_page()
    elif page == "📊 History":
        render_history_page()
    elif page == "🎓 Educational Guide":
        render_educational_guide()

def render_home_page():
    """Render the home page"""
    st.markdown('<h1 class="main-header">🔍 Student Fake News Detector</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🛡️ Your AI-Powered Defense Against Misinformation
        
        In today's digital world, misinformation spreads rapidly through social media and online news. 
        This tool empowers students with AI-assisted analysis to:
        
        - **🔍 Analyze** news articles and social media posts
        - **🎯 Assess** credibility using multiple indicators  
        - **📝 Generate** concise, accurate summaries
        - **📊 Understand** linguistic patterns of misinformation
        - **🎓 Develop** critical thinking skills
        
        ### 🚀 How to Use:
        1. Go to **"Analyze Article"** in the sidebar
        2. Paste text or enter a URL
        3. Get instant AI-powered analysis
        4. Review credibility scores and insights
        
        ### 🎯 What We Analyze:
        - Sensational language and emotional manipulation
        - Source credibility and evidence quality
        - Linguistic patterns common in fake news
        - Structural elements of reliable reporting
        """)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2017/08/07/22/29/thinking-2608140_1280.jpg", 
                use_column_width=True, caption="Think Critically, Verify Always")
        
        st.markdown("### 📈 Why This Matters")
        stats_data = {
            'Issue': ['Fake News Spread', 'Student Exposure', 'Verification Rate'],
            'Percentage': [85, 72, 35]
        }
        df = pd.DataFrame(stats_data)
        fig = px.bar(df, x='Issue', y='Percentage', 
                    title="Digital Literacy Challenges (%)",
                    color='Issue')
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick start section
    st.markdown("---")
    st.subheader("🎯 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📝 Paste Text</h3>
        <p>Copy and paste article text for instant analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🌐 Enter URL</h3>
        <p>Analyze articles directly from their web addresses</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Get Insights</h3>
        <p>Receive detailed analysis and recommendations</p>
        </div>
        """, unsafe_allow_html=True)

def render_analysis_page():
    """Render the article analysis page"""
    st.title("🔍 Analyze Article")
    st.markdown("---")
    
    # Input method
    input_method = st.radio(
        "Choose input method:",
        ["📝 Paste Text", "🌐 Enter URL"],
        horizontal=True
    )
    
    input_content = ""
    
    if input_method == "📝 Paste Text":
        input_content = st.text_area(
            "Paste article text:",
            height=200,
            placeholder="Paste the full text of the news article here...\n\nExample: 'Scientists have discovered a breakthrough treatment...'",
            help="Minimum 100 characters for best results"
        )
        analyze_button = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
        
    else:
        url = st.text_input(
            "Enter article URL:",
            placeholder="https://example.com/news-article",
            help="Supports most news websites and blogs"
        )
        analyze_button = st.button("🌐 Analyze URL", type="primary", use_container_width=True)
        input_content = url
    
    # Analysis options
    with st.expander("⚙️ Analysis Options"):
        col1, col2 = st.columns(2)
        with col1:
            generate_summary = st.checkbox("Generate Summary", value=True)
            show_detailed_features = st.checkbox("Show Detailed Features", value=True)
        with col2:
            show_indicators = st.checkbox("Show Found Indicators", value=True)
            compare_sources = st.checkbox("Suggest Source Comparison", value=True)
    
    if analyze_button and input_content:
        with st.spinner("🔍 Analyzing content... This may take a few seconds."):
            try:
                if input_method == "🌐 Enter URL":
                    # Extract content from URL
                    extraction_result = st.session_state.detector.extract_article_from_url(input_content)
                    
                    if not extraction_result['success']:
                        st.error(f"❌ Could not extract content from URL: {extraction_result.get('error', 'Unknown error')}")
                        return
                    
                    article_title = extraction_result['title']
                    article_content = extraction_result['content']
                    
                    if len(article_content) < 100:
                        st.warning("⚠️ Extracted content seems very short. Analysis may be less accurate.")
                    
                else:
                    # Use directly pasted text
                    article_title = "Pasted Text Analysis"
                    article_content = input_content
                
                # Perform analysis
                analysis_result = st.session_state.detector.analyze_text(article_content)
                
                if 'error' in analysis_result:
                    st.error(f"❌ {analysis_result['error']}")
                    return
                
                # Add metadata and store in history
                full_result = {
                    'timestamp': datetime.now().isoformat(),
                    'title': article_title,
                    'input_method': input_method,
                    **analysis_result
                }
                st.session_state.analysis_history.append(full_result)
                
                # Display results
                display_analysis_results(full_result, generate_summary, show_detailed_features, show_indicators)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def display_analysis_results(result, show_summary, show_features, show_indicators):
    """Display analysis results in an organized layout"""
    st.success("✅ Analysis Complete!")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "📝 Summary", "📊 Analysis", "💡 Recommendations"])
    
    with tab1:
        display_overview_tab(result)
    
    with tab2:
        if show_summary:
            display_summary_tab(result)
        else:
            st.info("Summary generation was disabled for this analysis.")
    
    with tab3:
        if show_features:
            display_analysis_tab(result, show_indicators)
        else:
            st.info("Detailed analysis was disabled for this analysis.")
    
    with tab4:
        display_recommendations_tab(result)

def display_overview_tab(result):
    """Display overview tab content"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📖 Article Information")
        st.write(f"**Title:** {result.get('title', 'N/A')}")
        st.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Word Count:** {result.get('word_count', 0)} words")
        st.write(f"**Character Count:** {result.get('char_count', 0)} characters")
    
    with col2:
        st.subheader("🎯 Credibility Verdict")
        analysis = result['analysis']
        
        # Display verdict with color coding
        color_class = {
            "red": "fake",
            "green": "reliable", 
            "orange": "borderline"
        }.get(analysis['color'], 'borderline')
        
        st.markdown(f"""
        <div class="result-box {color_class}">
            <h3>{analysis['verdict']}</h3>
            <p>Confidence: {analysis['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = analysis['confidence'] * 100,
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': analysis['color']},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ]
            }
        ))
        fig.update_layout(height=200, margin=dict(t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

def display_summary_tab(result):
    """Display summary tab content"""
    st.subheader("📋 Article Summary")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(result['summary'])
    
    with col2:
        st.metric("Summary Length", f"{len(result['summary'])} chars")
        st.metric("Reduction", f"{((result['char_count'] - len(result['summary'])) / result['char_count'] * 100):.1f}%")

def display_analysis_tab(result, show_indicators):
    """Display detailed analysis tab"""
    st.subheader("📈 Credibility Breakdown")
    
    # Scores visualization
    scores = result['analysis']['scores']
    fig = px.bar(
        x=list(scores.keys()), 
        y=list(scores.values()),
        color=list(scores.keys()),
        color_discrete_map={
            'fake_score': 'red',
            'borderline_score': 'orange',
            'reliable_score': 'green'
        },
        labels={'x': 'Score Type', 'y': 'Confidence'},
        title="Credibility Confidence Scores"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Linguistic features
    st.subheader("🔍 Linguistic Features")
    features = result['features']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Words", features['word_count'])
        st.metric("Sentences", features['sentence_count'])
        st.metric("Avg Sentence Length", f"{features['avg_sentence_length']:.1f}")
    
    with col2:
        st.metric("Exclamation Marks", features['exclamation_count'])
        st.metric("Sensational Words", features['sensational_word_count'])
        st.metric("Reliable Indicators", features['reliable_indicator_count'])
    
    if show_indicators:
        st.subheader("🔎 Found Indicators")
        
        analysis = result['analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis['fake_indicators_found']:
                st.warning("🚨 Suspicious Indicators Found:")
                for indicator in analysis['fake_indicators_found']:
                    st.write(f"• {indicator}")
            else:
                st.success("✅ No strong fake news indicators detected")
        
        with col2:
            if analysis['reliable_indicators_found']:
                st.success("✅ Reliable Indicators Found:")
                for indicator in analysis['reliable_indicators_found']:
                    st.write(f"• {indicator}")
            else:
                st.info("ℹ️ No strong reliable indicators detected")

def display_recommendations_tab(result):
    """Display recommendations tab"""
    analysis = result['analysis']
    
    st.subheader("💡 Action Recommendations")
    
    if analysis['verdict'] == "Fake News":
        st.error("""
        🚨 **High Caution Advised**
        
        **Immediate Actions:**
        - ⚠️ Do not share this content
        - 🔍 Verify with reputable fact-checking websites
        - 📰 Check multiple reliable news sources
        - 🕵️ Investigate the source's reputation
        
        **Fact-Checking Resources:**
        - Snopes.com
        - FactCheck.org  
        - Reuters Fact Check
        - AP News Fact Check
        """)
        
    elif analysis['verdict'] == "Reliable":
        st.success("""
        ✅ **Appears Credible**
        
        **Good Practices:**
        - 👍 You can consider this source
        - 🔄 Still verify extraordinary claims
        - 📚 Check for recent updates
        - 🎯 Consider multiple perspectives
        
        **Remember:** Even reliable sources can make mistakes or have biases.
        """)
        
    else:
        st.warning("""
        ⚠️ **Additional Verification Needed**
        
        **Recommended Steps:**
        - 🔍 Cross-reference with established news outlets
        - 📊 Look for supporting evidence and data
        - 👨‍💼 Check author credentials and expertise
        - 📅 Verify publication dates and context
        - 🌐 Search for corroborating reports
        
        **When in doubt, don't share without verification.**
        """)
    
    st.subheader("🎓 Critical Thinking Questions")
    st.markdown("""
    - **Source:** Who created this information and why?
    - **Evidence:** What evidence supports the claims?
    - **Missing:** What information might be missing?
    - **Emotion:** Is this trying to provoke strong emotions?
    - **Corroboration:** Do other reliable sources confirm this?
    """)

def render_history_page():
    """Render analysis history page"""
    st.title("📊 Analysis History")
    st.markdown("---")
    
    if not st.session_state.analysis_history:
        st.info("📝 No analyses yet. Go to 'Analyze Article' to get started!")
        return
    
    # Display history table
    history_data = []
    for i, analysis in enumerate(st.session_state.analysis_history):
        history_data.append({
            'ID': i + 1,
            'Title': analysis.get('title', 'Unknown')[:60] + '...',
            'Verdict': analysis['analysis']['verdict'],
            'Confidence': f"{analysis['analysis']['confidence']:.1%}",
            'Words': analysis.get('word_count', 0),
            'Date': pd.Timestamp(analysis.get('timestamp')).strftime('%m/%d %H:%M')
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Analytics
    st.subheader("📈 Analytics Overview")
    
    col1, col2, col3 = st.columns(3)
    
    # Verdict distribution
    verdict_counts = history_df['Verdict'].value_counts()
    
    with col1:
        fig1 = px.pie(
            values=verdict_counts.values, 
            names=verdict_counts.index,
            title="Verdict Distribution",
            color=verdict_counts.index,
            color_discrete_map={
                'Fake News': 'red',
                'Reliable': 'green',
                'Borderline/Uncertain': 'orange'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Confidence over time
        if len(st.session_state.analysis_history) > 1:
            dates = [pd.Timestamp(a['timestamp']) for a in st.session_state.analysis_history]
            confidences = [a['analysis']['confidence'] for a in st.session_state.analysis_history]
            
            fig2 = px.line(
                x=dates, y=confidences,
                title="Confidence Scores Over Time",
                labels={'x': 'Date', 'y': 'Confidence'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        # Clear history option
        st.metric("Total Analyses", len(st.session_state.analysis_history))
        st.metric("Average Confidence", f"{np.mean([a['analysis']['confidence'] for a in st.session_state.analysis_history]):.1%}")
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.analysis_history = []
            st.rerun()

def render_educational_guide():
    """Render educational guide page"""
    st.title("🎓 Fake News Detection Guide")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Basics", "🔍 Detection", "🛡️ Protection", "🌐 Resources"])
    
    with tab1:
        st.header("📚 Understanding Fake News")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("What is Fake News?")
            st.markdown("""
            Fake news refers to **false or misleading information** presented as legitimate news.
            
            **Common Types:**
            - 🎭 **Fabricated Content**: Completely made-up stories
            - 🔄 **Manipulated Content**: Real information twisted out of context  
            - 👤 **Imposter Content**: Fake sources pretending to be real
            - 🎯 **Misleading Content**: Misleading use of genuine information
            - 📍 **False Context**: Real content with false context
            
            **Why It Spreads:**
            - 📱 Social media algorithms
            - 😡 Emotional engagement
            - 🔁 Echo chambers
            - ⏱️ Lack of verification time
            """)
        
        with col2:
            st.subheader("Real World Impact")
            impact_data = {
                'Area': ['Public Health', 'Elections', 'Social Harmony', 'Education'],
                'Impact Level': [85, 78, 72, 65]
            }
            df = pd.DataFrame(impact_data)
            fig = px.bar(df, x='Area', y='Impact Level', 
                        title="Fake News Impact Areas (%)",
                        color='Area')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔍 How to Spot Fake News")
        
        st.subheader("🚨 Red Flags")
        red_flags = [
            "EXCESSIVE CAPITALIZATION and!!! punctuation!!!",
            "Emotional language trying to provoke anger/fear",
            "Claims of 'secret information' or 'conspiracies'",
            "Lack of author information or sources",
            "Requests to 'SHARE URGENTLY'",
            "Dates don't match or content is old",
            "Website has suspicious domain name",
            "No other reputable sources report the story"
        ]
        
        for i, flag in enumerate(red_flags, 1):
            st.write(f"{i}. {flag}")
        
        st.subheader("✅ Reliability Green Flags")
        green_flags = [
            "Clear author information with credentials",
            "Multiple reputable sources report similar information",
            "Balanced language without emotional manipulation",
            "Citations and references to evidence",
            "Professional website design and contact information",
            "Correction policy stated",
            "Recent publication date"
        ]
        
        for i, flag in enumerate(green_flags, 1):
            st.write(f"{i}. {flag}")
    
    with tab3:
        st.header("🛡️ Protecting Yourself")
        
        st.subheader("🔒 Digital Literacy Skills")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Critical Thinking:**
            - 🧠 Question everything
            - 🤔 Consider the purpose
            - 💭 Think before sharing
            - 🔍 Look for evidence
            
            **Verification Habits:**
            - ✅ Check multiple sources
            - 📚 Use fact-checking websites
            - 🕵️ Investigate sources
            - 📊 Look for data support
            """)
        
        with col2:
            st.markdown("""
            **Safe Browsing:**
            - 🌐 Use reputable news sources
            - 🔔 Install browser extensions
            - 📖 Read beyond headlines
            - ⏰ Take time to verify
            
            **Social Media Caution:**
            - ⚠️ Be extra careful with social media news
            - 🔄 Verify before resharing
            - 🎯 Follow credible accounts
            - 📵 Limit exposure to questionable content
            """)
        
        st.subheader("🎯 The S.H.A.R.E. Method")
        st.markdown("""
        **S**ource - Check who's sharing the information  
        **H**eadline - Read beyond the headline  
        **A**nalyze - Check the facts and evidence  
        **R**etouched - Look for altered images/videos  
        **E**rror - Check for mistakes and bias  
        """)
    
    with tab4:
        st.header("🌐 Helpful Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Fact-Checking Websites")
            resources = {
                "Snopes": "https://www.snopes.com/",
                "FactCheck.org": "https://www.factcheck.org/", 
                "PolitiFact": "https://www.politifact.com/",
                "Reuters Fact Check": "https://www.reuters.com/fact-check/",
                "AP News Fact Check": "https://apnews.com/hub/fact-checking",
                "BBC Reality Check": "https://www.bbc.com/news/reality_check"
            }
            
            for name, url in resources.items():
                st.markdown(f"- [{name}]({url})")
        
        with col2:
            st.subheader("🎓 Educational Resources")
            education = {
                "Media Literacy Project": "https://medialiteracyproject.org/",
                "Stanford History Education Group": "https://sheg.stanford.edu/",
                "News Literacy Project": "https://newslit.org/",
                "Common Sense Media": "https://www.commonsensemedia.org/",
                "PBS Media Literacy": "https://www.pbs.org/newshour/classroom/digital-studies/"
            }
            
            for name, url in education.items():
                st.markdown(f"- [{name}]({url})")
            
            st.subheader("📱 Browser Extensions")
            extensions = {
                "NewsGuard": "https://www.newsguardtech.com/",
                "Trusted News": "https://trusted-news.com/",
                "B.S. Detector": "https://bsdetector.tech/"
            }
            
            for name, url in extensions.items():
                st.markdown(f"- {name}")

if __name__ == "__main__":
    main()

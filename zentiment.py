import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
import base64

# Configure page
st.set_page_config(
    page_title="🧠 Zentiment - AI Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #DC143C;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #333;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(45deg, #DC143C 0%, #8B0000 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stAlert {
        border-radius: 10px;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #DC143C;
        text-align: center;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #333;
        font-size: 0.9rem;
        border-top: 1px solid #DC143C;
        margin-top: 3rem;
        background-color: #f8f8f8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None

@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

def analyze_sentiment(sentences):
    """Analyze sentiment of sentences using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    
    positive_reviews = []
    negative_reviews = []
    neutral_reviews = []
    detailed_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, review in enumerate(sentences):
        status_text.text(f'Analyzing review {i+1} of {len(sentences)}...')
        sentiment = analyzer.polarity_scores(review)
        compound = sentiment['compound']
        
        result = {
            'review': review,
            'compound': compound,
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu']
        }
        
        if compound >= 0.05:
            positive_reviews.append(review)
            result['category'] = 'Positive'
            result['emoji'] = '😊'
        elif compound <= -0.05:
            negative_reviews.append(review)
            result['category'] = 'Negative'
            result['emoji'] = '😞'
        else:
            neutral_reviews.append(review)
            result['category'] = 'Neutral'
            result['emoji'] = '😐'
            
        detailed_results.append(result)
        progress_bar.progress((i + 1) / len(sentences))
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'positive': positive_reviews,
        'negative': negative_reviews,
        'neutral': neutral_reviews,
        'detailed': detailed_results
    }

def create_pie_chart(data):
    """Create interactive pie chart with Zentiment styling"""
    labels = ['Positive 😊', 'Negative 😞', 'Neutral 😐']
    values = [len(data['positive']), len(data['negative']), len(data['neutral'])]
    colors = ['#DC143C', '#8B0000', '#333333']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        textfont_size=14,
        hovertemplate='<b>%{label}</b><br>Reviews: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': '🎯 Zentiment Analysis Overview',
            'x': 0.5,
            'font': {'size': 22, 'family': 'Arial Black'}
        },
        font=dict(size=14),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_bar_chart(data):
    """Create interactive bar chart with animations"""
    labels = ['Positive 😊', 'Negative 😞', 'Neutral 😐']
    values = [len(data['positive']), len(data['negative']), len(data['neutral'])]
    colors = ['#DC143C', '#8B0000', '#333333']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont_size=16,
        hovertemplate='<b>%{x}</b><br>Count: %{y} reviews<extra></extra>',
        marker_line_color='white',
        marker_line_width=2
    )])
    
    fig.update_layout(
        title={
            'text': '📊 Sentiment Distribution by Count',
            'x': 0.5,
            'font': {'size': 22, 'family': 'Arial Black'}
        },
        xaxis_title='Sentiment Category',
        yaxis_title='Number of Reviews',
        font=dict(size=14),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig

def create_wordcloud(text, title, colormap):
    """Create word cloud with better styling"""
    if not text.strip():
        return None
        
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=80,
        prefer_horizontal=0.7
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def create_sentiment_timeline(data):
    """Create sentiment score timeline"""
    df = pd.DataFrame(data['detailed'])
    df['index'] = range(len(df))
    
    fig = go.Figure()
    
    colors = {'Positive': '#DC143C', 'Negative': '#8B0000', 'Neutral': '#333333'}
    
    for category in ['Positive', 'Negative', 'Neutral']:
        category_data = df[df['category'] == category]
        emoji = category_data['emoji'].iloc[0] if len(category_data) > 0 else ''
        
        fig.add_trace(go.Scatter(
            x=category_data['index'],
            y=category_data['compound'],
            mode='markers+lines',
            name=f'{category} {emoji}',
            marker=dict(color=colors[category], size=10, line=dict(width=2, color='white')),
            line=dict(width=2, color=colors[category]),
            hovertemplate=f'<b>{category}</b><br>Review #%{{x+1}}<br>Score: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '📈 Zentiment Score Timeline',
            'x': 0.5,
            'font': {'size': 22, 'family': 'Arial Black'}
        },
        xaxis_title='Review Number',
        yaxis_title='Sentiment Score (-1 to +1)',
        height=500,
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='red')
    )
    
    return fig

# Main app
def main():
    # Download NLTK data
    download_nltk_data()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Zentiment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### Welcome to Zentiment! 🎉
    Transform your text data into actionable insights with our powerful sentiment analysis engine. 
    Upload reviews, feedback, or any text to discover the emotional tone and sentiment patterns.
    """)
    
    # Sidebar
    st.sidebar.markdown('<p class="sidebar-header">🧠 Zentiment Controls</p>', unsafe_allow_html=True)
    
    input_method = st.sidebar.radio(
        "Choose your input method:",
        ["📄 Upload CSV File", "✍️ Manual Text Entry"],
        help="Select how you want to provide your text data for analysis"
    )
    
    sentences = []
    
    if input_method == "📄 Upload CSV File":
        st.sidebar.markdown("##### 📁 File Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Your CSV should contain a 'review' column with the text to analyze"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'review' not in df.columns:
                    st.sidebar.error("❌ CSV must have a 'review' column!")
                    st.sidebar.info("💡 Make sure your CSV has a column named 'review' containing the text data")
                else:
                    sentences = df['review'].dropna().tolist()
                    st.sidebar.success(f"✅ Successfully loaded {len(sentences)} reviews!")
                    
                    # Show preview
                    with st.sidebar.expander("👀 Preview Data"):
                        st.write(df.head())
                        
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
    
    else:
        st.sidebar.markdown("##### ✍️ Text Input")
        text_input = st.sidebar.text_area(
            "Enter your text for analysis:",
            height=150,
            placeholder="Paste your reviews, feedback, or any text here...\n\nExample:\nI love this product! It's amazing.\nNot happy with the service.\nIt's okay, nothing special.",
            help="Enter text and it will be automatically split into sentences for individual analysis"
        )
        
        if text_input:
            sentences = sent_tokenize(text_input)
            st.sidebar.success(f"✅ Found {len(sentences)} sentences to analyze!")
    
    # Analysis section
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button(
        "🚀 Run Zentiment Analysis", 
        type="primary", 
        use_container_width=True,
        help="Click to start the sentiment analysis process"
    )
    
    if analyze_button:
        if sentences:
            with st.spinner("🧠 Zentiment is analyzing your data..."):
                st.session_state.analyzed_data = analyze_sentiment(sentences)
            st.balloons()
            st.success("🎉 Analysis complete! Check out your results below.")
        else:
            st.error("❌ Please provide some text to analyze first!")
    
    # Display results if available
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        
        # Metrics section
        st.header("📊 Zentiment Overview")
        
        total_reviews = len(data['detailed'])
        positive_count = len(data['positive'])
        negative_count = len(data['negative'])
        neutral_count = len(data['neutral'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📝 Total Reviews", 
                total_reviews, 
                help="Total number of text entries analyzed"
            )
        with col2:
            positive_pct = positive_count/total_reviews*100 if total_reviews > 0 else 0
            st.metric(
                "😊 Positive", 
                positive_count, 
                f"{positive_pct:.1f}%",
                delta_color="normal"
            )
        with col3:
            negative_pct = negative_count/total_reviews*100 if total_reviews > 0 else 0
            st.metric(
                "😞 Negative", 
                negative_count, 
                f"{negative_pct:.1f}%",
                delta_color="inverse"
            )
        with col4:
            neutral_pct = neutral_count/total_reviews*100 if total_reviews > 0 else 0
            st.metric(
                "😐 Neutral", 
                neutral_count, 
                f"{neutral_pct:.1f}%"
            )
        
        # Charts section
        st.header("📈 Interactive Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_pie_chart(data), use_container_width=True)
        with col2:
            st.plotly_chart(create_bar_chart(data), use_container_width=True)
        
        # Timeline
        st.plotly_chart(create_sentiment_timeline(data), use_container_width=True)
        
        # Word Clouds section
        st.header("☁️ Zentiment Word Clouds")
        st.markdown("Discover the most common words in positive and negative feedback:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if data['positive']:
                positive_text = " ".join(data['positive'])
                fig = create_wordcloud(positive_text, "😊 Positive Sentiment Words", "Greens")
                if fig:
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No positive reviews found to generate word cloud")
        
        with col2:
            if data['negative']:
                negative_text = " ".join(data['negative'])
                fig = create_wordcloud(negative_text, "😞 Negative Sentiment Words", "Reds")
                if fig:
                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No negative reviews found to generate word cloud")
        
        # Sample Reviews section
        st.header("📝 Sample Reviews by Sentiment")
        
        tab1, tab2, tab3 = st.tabs(["😊 Positive Samples", "😞 Negative Samples", "😐 Neutral Samples"])
        
        with tab1:
            if data['positive']:
                st.markdown("**Most positive feedback from your data:**")
                sample_positive = data['positive'][:5]
                for i, review in enumerate(sample_positive, 1):
                    st.success(f"**Sample {i}:** {review}")
            else:
                st.info("No positive reviews found in your data")
        
        with tab2:
            if data['negative']:
                st.markdown("**Areas for improvement based on negative feedback:**")
                sample_negative = data['negative'][:5]
                for i, review in enumerate(sample_negative, 1):
                    st.error(f"**Sample {i}:** {review}")
            else:
                st.info("No negative reviews found in your data")
        
        with tab3:
            if data['neutral']:
                st.markdown("**Neutral feedback from your data:**")
                sample_neutral = data['neutral'][:5]
                for i, review in enumerate(sample_neutral, 1):
                    st.info(f"**Sample {i}:** {review}")
            else:
                st.info("No neutral reviews found in your data")
        
        # Detailed Results
        with st.expander("📋 Detailed Analysis Results"):
            st.markdown("**Complete breakdown of every analyzed text with sentiment scores:**")
            df_results = pd.DataFrame(data['detailed'])
            df_results = df_results[['emoji', 'category', 'review', 'compound', 'positive', 'negative', 'neutral']]
            df_results.columns = ['😊😞😐', 'Category', 'Review Text', 'Overall Score', 'Positive', 'Negative', 'Neutral']
            
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button for results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="zentiment_analysis_results.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome section when no analysis has been run
        st.header("🚀 Getting Started with Zentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📁 Upload CSV Method
            - Prepare a CSV file with a **'review'** column
            - Upload it using the sidebar
            - Get instant analysis of all your data
            """)
            
        with col2:
            st.markdown("""
            ### ✍️ Manual Entry Method  
            - Paste text directly into the text area
            - Each sentence will be analyzed separately
            - Perfect for quick tests and samples
            """)
        
        st.markdown("""
        ---
        ### 🧠 How Zentiment Works
        
        Zentiment uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, a lexicon and rule-based 
        sentiment analysis tool specifically designed for social media text. It provides:
        
        - **Compound Score**: Overall sentiment (-1 to +1)
        - **Individual Scores**: Positive, Negative, and Neutral components
        - **Real-time Analysis**: Process thousands of reviews in seconds
        - **Visual Insights**: Interactive charts and word clouds
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🧠 <strong>Zentiment</strong> - AI-Powered Sentiment Analysis Dashboard</p>
        <p>Built with ❤️ using Streamlit, Plotly, and VADER Sentiment Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ai_features.py - Funciones de IA para análisis de noticias
from transformers import pipeline
import streamlit as st

@st.cache_resource(show_spinner=False)
def get_ai_pipelines():
    """Carga los modelos de IA para resumen y análisis de sentimiento."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", 
                             device=-1, max_length=150, min_length=40, do_sample=False)
        sentiment_analyzer = pipeline("sentiment-analysis", 
                                     model="distilbert-base-uncased-finetuned-sst-2-english",
                                     device=-1)
        return {'summarizer': summarizer, 'sentiment': sentiment_analyzer}
    except Exception as e:
        st.error(f"Error cargando modelos IA: {e}")
        return None

def ai_summarize_texts(texts, max_length=150):
    """Genera un resumen automático de múltiples textos usando IA."""
    if not texts:
        return None
    
    pipelines = get_ai_pipelines()
    if not pipelines:
        return None
    
    combined_text = " ".join([t[:500] for t in texts if t])[:2000]
    
    if len(combined_text.split()) < 50:
        return "Texto insuficiente para generar resumen."
    
    try:
        result = pipelines['summarizer'](combined_text, max_length=max_length, min_length=40, do_sample=False)
        if result and len(result) > 0:
            return result[0]['summary_text']
    except Exception as e:
        return f"Error generando resumen: {str(e)}"
    
    return None

def ai_analyze_sentiment(texts):
    """Analiza el sentimiento de múltiples textos y devuelve un veredicto general."""
    if not texts:
        return {'label': 'NEUTRO', 'score': 0.0, 'details': []}
    
    pipelines = get_ai_pipelines()
    if not pipelines:
        return {'label': 'NEUTRO', 'score': 0.0, 'details': []}
    
    results = []
    positive_count = 0
    negative_count = 0
    
    for text in texts[:5]:
        if not text or len(text.split()) < 10:
            continue
        
        try:
            result = pipelines['sentiment'](text[:512])
            if result and len(result) > 0:
                sentiment = result[0]
                results.append(sentiment)
                if sentiment['label'] == 'POSITIVE':
                    positive_count += 1
                else:
                    negative_count += 1
        except Exception:
            continue
    
    if not results:
        return {'label': 'NEUTRO', 'score': 0.0, 'details': []}
    
    total = positive_count + negative_count
    if total == 0:
        return {'label': 'NEUTRO', 'score': 0.0, 'details': results}
    
    positive_ratio = positive_count / total
    avg_score = sum(r['score'] for r in results) / len(results)
    
    if positive_ratio > 0.6:
        label = 'POSITIVO'
    elif positive_ratio < 0.4:
        label = 'NEGATIVO'
    else:
        label = 'NEUTRO'
    
    return {'label': label, 'score': avg_score, 'positive': positive_count, 
            'negative': negative_count, 'total': total, 'details': results}

def ai_generate_prediction(stats_data, sentiment_data, team_name):
    """Genera una predicción basada en estadísticas y sentimiento de noticias."""
    if not stats_data or not sentiment_data:
        return None
    
    home_form = stats_data.get('home_form', 0.5)
    away_form = stats_data.get('away_form', 0.5)
    sentiment_score = sentiment_data.get('score', 0.5)
    sentiment_label = sentiment_data.get('label', 'NEUTRO')
    
    sentiment_boost = 0
    if sentiment_label == 'POSITIVO':
        sentiment_boost = 0.1 * sentiment_score
    elif sentiment_label == 'NEGATIVO':
        sentiment_boost = -0.1 * sentiment_score
    
    adjusted_home_prob = min(0.95, max(0.05, home_form + sentiment_boost))
    adjusted_away_prob = min(0.95, max(0.05, away_form - sentiment_boost))
    
    total = adjusted_home_prob + adjusted_away_prob + 0.3
    home_final = adjusted_home_prob / total
    away_final = adjusted_away_prob / total
    draw_final = 0.3 / total
    
    return {
        'home_win': home_final,
        'draw': draw_final,
        'away_win': away_final,
        'sentiment_impact': sentiment_boost,
        'confidence': 'Alta' if abs(sentiment_boost) > 0.05 else 'Media'
    }

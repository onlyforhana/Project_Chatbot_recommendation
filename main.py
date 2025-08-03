from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, 
    QuickReply, QuickReplyButton, MessageAction, PostbackEvent, PostbackAction
)
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os
import re
from difflib import get_close_matches
import unicodedata

# Neo4j connection
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "theoneandonlyhana"))

# Text Normalization and Spell Correction
class ThaiEngTextNormalizer:
    def __init__(self):
        # Common Thai-English mixed words mapping
        self.mixed_word_mapping = {
            'perfume': 'น้ำหอม',
            'review': 'รีวิว',
            'recommend': 'แนะนำ',
            'new': 'ใหม่',
            'bestseller': 'ขายดี',
            'fresh': 'สดชื่น',
            'sweet': 'หวาน',
            'sexy': 'เซ็กซี่',
            'light': 'เบา',
            'summer': 'หน้าร้อน',
            'winter': 'หน้าหนาว',
            'work': 'ทำงาน',
            'office': 'ออฟฟิศ',
            'date': 'เดท',
            'party': 'ปาร์ตี้',
            'limited': 'ลิมิเต็ด',
            'edition': 'อิดิชั่น',
            'price': 'ราคา',
            'size': 'ขนาด',
            'stock': 'สต็อก',
            'cool': 'เย็น',
            'warm': 'อุ่น',
            'romantic': 'โรแมนติก',
            'intense': 'เข้ม',
            'black': 'ดำ',
            'blue': 'น้ำเงิน',
            'red': 'แดง',
            'pink': 'ชมพู',
            'love': 'รัก',
            'heart': 'หัวใจ',
            'flower': 'ดอกไม้',
            'rose': 'กุหลาব',
            'vanilla': 'วานิลลา',
            'citrus': 'ส้ม',
            'ocean': 'ทะเล',
            'night': 'กลางคืน',
            'morning': 'เช้า',
            'evening': 'เย็น'
        }
        
        # Common Thai spelling variations and corrections
        self.thai_spell_mapping = {
            'สวัสดีครับ': 'สวัสดี',
            'สวัสดีค่ะ': 'สวัสดี',
            'ขอโทษครับ': 'ขอโทษ',
            'ขอโทษค่ะ': 'ขอโทษ',
            'แนะนำหน่อย': 'แนะนำ',
            'ช่วยแนะนำหน่อย': 'แนะนำ',
            'มีอะไรน่าสนใจบ้าง': 'น่าสนใจ',
            'มีอะไรน่าสนใจช่วงนี้': 'น่าสนใจ',
            'กลิ่นไหนเหมาะกับหน้าร้อน': 'หน้าร้อน',
            'น้ำหอมไหนดี': 'แนะนำ',
            'perfumeไหนดี': 'แนะนำ',
            'reviewดี': 'รีวิวดี',
            'รีวิวดีๆ': 'รีวิวดี',
            'ขายดีๆ': 'ขายดี',
            'ใหม่ๆ': 'ใหม่',
            'สดชื่นๆ': 'สดชื่น',
            'หวานๆ': 'หวาน',
            'เซ็กซี่ๆ': 'เซ็กซี่'
        }
        
        # Enhanced intent keywords (Thai-English mixed)
        self.intent_keywords = {
            'greeting': ['สวัสดี', 'หวัดดี', 'ไง', 'hello', 'hi', 'hey'],
            'general_inquiry': ['มีอะไรคุย', 'รู้อะไรบ้าง', 'ช่วยแนะนำ', 'แนะนำ', 'recommend', 'suggestion', 'น่าสนใจ', 'interesting'],
            'product_bestseller': ['ขายดี', 'bestseller', 'ยอดนิยม', 'popular', 'best seller', 'best-seller'],
            'product_new': ['ใหม่', 'new', 'มาใหม่', 'latest', 'newest'],
            'product_reviewed': ['รีวิว', 'review', 'รีวิวดี', 'good review', 'มีรีวิว', 'reviewed'],
            'product_limited': ['limited', 'ลิมิเต็ด', 'limited edition', 'พิเศษ', 'exclusive'],
            'scent_fresh': ['สดชื่น', 'fresh', 'เซฟ', 'safe', 'เบา', 'light', 'น่าสนใจ', 'ใส', 'clean'],
            'scent_sweet': ['หวาน', 'sweet', 'หอมหวาน', 'floral', 'ดอกไม้', 'flower', 'กุหลาบ', 'rose'],
            'scent_sexy': ['เซ็กซี่', 'sexy', 'ดึงดูด', 'attractive', 'เข้ม', 'intense', 'แรง', 'strong'],
            'season_summer': ['หน้าร้อน', 'summer', 'ร้อน', 'hot', 'เย็น', 'cool', 'ฤดูร้อน'],
            'season_winter': ['หน้าหนาว', 'winter', 'หนาว', 'cold', 'อุ่น', 'warm', 'ฤดูหนาว'],
            'occasion_work': ['ทำงาน', 'work', 'ออฟฟิศ', 'office', 'การทำงาน', 'working'],
            'occasion_date': ['เดท', 'date', 'โรแมนติก', 'romantic', 'รัก', 'love'],
            'occasion_party': ['ปาร์ตี้', 'party', 'งานเลี้ยง', 'celebration', 'กลางคืน', 'night']
        }
    
    def normalize_text(self, text):
        """Normalize Thai-English mixed text"""
        # Convert to lowercase for English parts
        normalized = text.lower()
        
        # Remove extra spaces and normalize Unicode
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        normalized = unicodedata.normalize('NFKC', normalized)
        
        # Replace common mixed words
        for eng_word, thai_word in self.mixed_word_mapping.items():
            # Case insensitive replacement
            pattern = re.compile(re.escape(eng_word), re.IGNORECASE)
            normalized = pattern.sub(thai_word, normalized)
        
        # Apply Thai spelling corrections
        for wrong, correct in self.thai_spell_mapping.items():
            normalized = normalized.replace(wrong, correct)
        
        # Remove common Thai particles that don't affect meaning
        particles_to_remove = ['ครับ', 'ค่ะ', 'คะ', 'นะ', 'หน่อย', 'บ้าง', 'เอ่อ', 'อืม']
        for particle in particles_to_remove:
            normalized = normalized.replace(particle, '')
        
        # Clean up extra spaces again
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return normalized
    
    def extract_intent_from_text(self, text):
        """Extract intent from normalized text using keyword matching"""
        normalized_text = self.normalize_text(text)
        
        # Score each intent based on keyword presence
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in normalized_text:
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])
        
        return None, 0

# Enhanced Intent Classification Data with Thai-English mixed examples
intent_data = [
    # Basic greetings
    ['สวัสดี', 'greeting'],
    ['หวัดดี', 'greeting'],
    ['ไง', 'greeting'],
    ['hello', 'greeting'],
    ['hi', 'greeting'],
    ['สวัสดีครับ', 'greeting'],
    ['สวัสดีค่ะ', 'greeting'],
    
    # General inquiry (Thai-English mixed)
    ['มีอะไรคุย', 'general_inquiry'],
    ['รู้อะไรบ้าง', 'general_inquiry'],
    ['ช่วยแนะนำหน่อย', 'general_inquiry'],
    ['มีอะไรน่าสนใจบ้าง', 'general_inquiry'],
    ['มีอะไรน่าสนใจช่วงนี้', 'general_inquiry'],
    ['recommend หน่อย', 'general_inquiry'],
    ['แนะนำ perfume หน่อย', 'general_inquiry'],
    ['ช่วย recommend น้ำหอม', 'general_inquiry'],
    
    # Product category requests (mixed language)
    ['สินค้าขายดี', 'product_bestseller'],
    ['ของขายดี', 'product_bestseller'],
    ['ยอดนิยม', 'product_bestseller'],
    ['bestseller', 'product_bestseller'],
    ['perfume ขายดี', 'product_bestseller'],
    ['น้ำหอม bestseller', 'product_bestseller'],
    
    ['สินค้าใหม่', 'product_new'],
    ['ของใหม่', 'product_new'],
    ['มาใหม่', 'product_new'],
    ['new perfume', 'product_new'],
    ['น้ำหอมใหม่', 'product_new'],
    ['perfume ใหม่', 'product_new'],
    
    ['สินค้าที่มีรีวิว', 'product_reviewed'],
    ['มีรีวิว', 'product_reviewed'],
    ['รีวิวดี', 'product_reviewed'],
    ['good review', 'product_reviewed'],
    ['ขอ perfume ที่ review ดี', 'product_reviewed'],
    ['น้ำหอมที่มี review', 'product_reviewed'],
    ['perfume รีวิวดี', 'product_reviewed'],
    
    ['สินค้า Limited Edition', 'product_limited'],
    ['Limited Edition', 'product_limited'],
    ['ลิมิเต็ด', 'product_limited'],
    ['limited perfume', 'product_limited'],
    ['น้ำหอม limited', 'product_limited'],
    
    # Scent-based recommendations (mixed)
    ['แนะนำกลิ่นสดชื่น', 'scent_fresh'],
    ['กลิ่นสดชื่น', 'scent_fresh'],
    ['กลิ่นเซฟ', 'scent_fresh'],
    ['กลิ่นเบา', 'scent_fresh'],
    ['fresh scent', 'scent_fresh'],
    ['light perfume', 'scent_fresh'],
    ['ขอ perfume fresh', 'scent_fresh'],
    ['น้ำหอม light', 'scent_fresh'],
    
    ['กลิ่นหวาน', 'scent_sweet'],
    ['กลิ่นหอมหวาน', 'scent_sweet'],
    ['หวานๆ', 'scent_sweet'],
    ['sweet perfume', 'scent_sweet'],
    ['ขอ perfume sweet', 'scent_sweet'],
    ['น้ำหอมหวาน', 'scent_sweet'],
    
    ['กลิ่นเซ็กซี่', 'scent_sexy'],
    ['กลิ่นเซ็กซี่ผู้ชาย', 'scent_sexy'],
    ['กลิ่นดึงดูด', 'scent_sexy'],
    ['sexy perfume', 'scent_sexy'],
    ['intense perfume', 'scent_sexy'],
    ['ขอ perfume sexy', 'scent_sexy'],
    
    # Seasonal recommendations (mixed)
    ['กลิ่นไหนเหมาะกับหน้าร้อน', 'season_summer'],
    ['หน้าร้อน', 'season_summer'],
    ['ฤดูร้อน', 'season_summer'],
    ['อากาศร้อน', 'season_summer'],
    ['summer perfume', 'season_summer'],
    ['perfume สำหรับ summer', 'season_summer'],
    ['น้ำหอมหน้าร้อน', 'season_summer'],
    
    ['หน้าหนาว', 'season_winter'],
    ['ฤดูหนาว', 'season_winter'],
    ['อากาศเย็น', 'season_winter'],
    ['winter perfume', 'season_winter'],
    ['perfume หน้าหนาว', 'season_winter'],
    
    # Occasion-based (mixed)
    ['ไปทำงาน', 'occasion_work'],
    ['ออฟฟิศ', 'occasion_work'],
    ['ใส่ทำงาน', 'occasion_work'],
    ['office perfume', 'occasion_work'],
    ['perfume for work', 'occasion_work'],
    ['น้ำหอม office', 'occasion_work'],
    
    ['ไปเดท', 'occasion_date'],
    ['เดท', 'occasion_date'],
    ['โรแมนติก', 'occasion_date'],
    ['date perfume', 'occasion_date'],
    ['romantic perfume', 'occasion_date'],
    ['perfume สำหรับเดท', 'occasion_date'],
    
    ['ไปงานปาร์ตี้', 'occasion_party'],
    ['ปาร์ตี้', 'occasion_party'],
    ['งานเลี้ยง', 'occasion_party'],
    ['party perfume', 'occasion_party'],
    ['perfume ปาร์ตี้', 'occasion_party'],
]

# Initialize text normalizer
text_normalizer = ThaiEngTextNormalizer()

# Normalize intent data
normalized_intent_data = []
for text, intent in intent_data:
    normalized_text = text_normalizer.normalize_text(text)
    normalized_intent_data.append([normalized_text, intent])

# Create DataFrame for intent classification
intent_df = pd.DataFrame(normalized_intent_data, columns=['text', 'intent'])

# Enhanced multilingual sentence transformer
def get_multilingual_encoder():
    """Get best multilingual model for Thai-English mixed text"""
    try:
        # Try to use a more advanced multilingual model
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    except:
        # Fallback to the original model
        return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

encoder = get_multilingual_encoder()

# Train Intent Classifier with normalized data
def train_intent_classifier():
    model_path = 'intent_classifier_normalized.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    # Create and train the model with better parameters for mixed language
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),  # Include trigrams for better context
            max_features=2000,   # Increase features for mixed language
            analyzer='char_wb',  # Character n-grams work better for mixed text
            min_df=1,           # Include rare terms
            token_pattern=r'(?u)\b\w+\b|[^\w\s]'  # Include Thai characters
        )),
        ('nb', MultinomialNB(alpha=0.1))  # Lower smoothing for better precision
    ])
    
    X = intent_df['text']
    y = intent_df['intent']
    
    classifier.fit(X, y)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    return classifier

# Initialize intent classifier
intent_classifier = train_intent_classifier()

# Enhanced product search with better Thai-English support
def search_products_by_intent(intent, query=""):
    query_string = ""
    
    if intent == "product_bestseller":
        query_string = """
        MATCH (p:Product)-[:HAS_STATUS]->(s:Status {name: 'BESTSELLER'}) 
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "product_new":
        query_string = """
        MATCH (p:Product)-[:HAS_STATUS]->(s:Status {name: 'NEW'}) 
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "product_reviewed":
        query_string = """
        MATCH (p:Product)
        WHERE p.review IS NOT NULL 
        AND p.review <> '' 
        AND p.review <> 'No Review'
        AND NOT p.review =~ '.*No Review.*'
        AND p.review =~ '.*[0-9]+.*'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        ORDER BY 
          CASE 
            WHEN p.review =~ '5(\\.0)?/5' THEN 5.0
            WHEN p.review =~ '4\\.[5-9]/5' THEN 4.7
            WHEN p.review =~ '4\\.[0-4]/5' THEN 4.2
            WHEN p.review =~ '3\\.[5-9]/5' THEN 3.7
            ELSE 0.0
          END DESC
        LIMIT 5
        """
    elif intent == "product_limited":
        query_string = """
        MATCH (p:Product)-[:HAS_STATUS]->(s:Status {name: 'Limited Edition'}) 
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "scent_fresh":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'fresh' OR toLower(p.title) CONTAINS 'light' OR 
              toLower(p.title) CONTAINS 'citrus' OR toLower(p.title) CONTAINS 'ocean' OR
              toLower(p.title) CONTAINS 'clean' OR toLower(p.title) CONTAINS 'cool' OR
              toLower(p.title) CONTAINS 'aqua' OR toLower(p.title) CONTAINS 'blue' OR
              toLower(p.title) CONTAINS 'mint' OR toLower(p.title) CONTAINS 'green' OR
              p.title CONTAINS 'สดชื่น' OR p.title CONTAINS 'เซฟ' OR 
              p.title CONTAINS 'เบา' OR p.title CONTAINS 'ใส' OR
              p.title CONTAINS 'น้ำใส' OR p.title CONTAINS 'ทะเล' OR
              p.title CONTAINS 'เย็น' OR p.title CONTAINS 'สด'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "scent_sweet":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'sweet' OR toLower(p.title) CONTAINS 'vanilla' OR 
              toLower(p.title) CONTAINS 'floral' OR toLower(p.title) CONTAINS 'rose' OR
              toLower(p.title) CONTAINS 'flower' OR toLower(p.title) CONTAINS 'pink' OR
              toLower(p.title) CONTAINS 'cherry' OR toLower(p.title) CONTAINS 'peach' OR
              p.title CONTAINS 'หวาน' OR p.title CONTAINS 'ดอกไม้' OR 
              p.title CONTAINS 'กุหลาบ' OR p.title CONTAINS 'หอม' OR
              p.title CONTAINS 'วานิลลา' OR p.title CONTAINS 'ชมพู'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "scent_sexy":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'intense' OR toLower(p.title) CONTAINS 'black' OR 
              toLower(p.title) CONTAINS 'noir' OR toLower(p.title) CONTAINS 'dark' OR
              toLower(p.title) CONTAINS 'deep' OR toLower(p.title) CONTAINS 'red' OR
              toLower(p.title) CONTAINS 'sexy' OR toLower(p.title) CONTAINS 'seductive' OR
              p.title CONTAINS 'เซ็กซี่' OR p.title CONTAINS 'ดำ' OR 
              p.title CONTAINS 'เข้ม' OR p.title CONTAINS 'แรง' OR
              p.title CONTAINS 'ดึงดูด' OR p.title CONTAINS 'แดง'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "season_summer":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'fresh' OR toLower(p.title) CONTAINS 'light' OR 
              toLower(p.title) CONTAINS 'citrus' OR toLower(p.title) CONTAINS 'ocean' OR
              toLower(p.title) CONTAINS 'cool' OR toLower(p.title) CONTAINS 'aqua' OR
              toLower(p.title) CONTAINS 'blue' OR toLower(p.title) CONTAINS 'summer' OR
              toLower(p.title) CONTAINS 'mint' OR toLower(p.title) CONTAINS 'ice' OR
              p.title CONTAINS 'สดชื่น' OR p.title CONTAINS 'เซฟ' OR 
              p.title CONTAINS 'เบา' OR p.title CONTAINS 'ร้อน' OR
              p.title CONTAINS 'เย็น' OR p.title CONTAINS 'สด'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "season_winter":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'warm' OR toLower(p.title) CONTAINS 'intense' OR 
              toLower(p.title) CONTAINS 'rich' OR toLower(p.title) CONTAINS 'deep' OR
              toLower(p.title) CONTAINS 'dark' OR toLower(p.title) CONTAINS 'winter' OR
              toLower(p.title) CONTAINS 'spice' OR toLower(p.title) CONTAINS 'wood' OR
              p.title CONTAINS 'อุ่น' OR p.title CONTAINS 'เข้ม' OR 
              p.title CONTAINS 'หนาว' OR p.title CONTAINS 'แรง' OR
              p.title CONTAINS 'เครื่องเทศ' OR p.title CONTAINS 'ไม้'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "occasion_work":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'light' OR toLower(p.title) CONTAINS 'fresh' OR 
              toLower(p.title) CONTAINS 'clean' OR toLower(p.title) CONTAINS 'subtle' OR
              toLower(p.title) CONTAINS 'office' OR toLower(p.title) CONTAINS 'work' OR
              toLower(p.title) CONTAINS 'professional' OR toLower(p.title) CONTAINS 'classic' OR
              p.title CONTAINS 'เบา' OR p.title CONTAINS 'เซฟ' OR 
              p.title CONTAINS 'ทำงาน' OR p.title CONTAINS 'ออฟฟิศ' OR
              p.title CONTAINS 'เรียบร้อย' OR p.title CONTAINS 'สุภาพ'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "occasion_date":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'romance' OR toLower(p.title) CONTAINS 'love' OR 
              toLower(p.title) CONTAINS 'sexy' OR toLower(p.title) CONTAINS 'seductive' OR
              toLower(p.title) CONTAINS 'date' OR toLower(p.title) CONTAINS 'heart' OR
              toLower(p.title) CONTAINS 'passion' OR toLower(p.title) CONTAINS 'charm' OR
              p.title CONTAINS 'โรแมนติก' OR p.title CONTAINS 'รัก' OR 
              p.title CONTAINS 'เดท' OR p.title CONTAINS 'หัวใจ' OR
              p.title CONTAINS 'ดึงดูด' OR p.title CONTAINS 'เสน่ห์'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif intent == "occasion_party":
        query_string = """
        MATCH (p:Product)
        WHERE toLower(p.title) CONTAINS 'intense' OR toLower(p.title) CONTAINS 'bold' OR 
              toLower(p.title) CONTAINS 'strong' OR toLower(p.title) CONTAINS 'party' OR
              toLower(p.title) CONTAINS 'night' OR toLower(p.title) CONTAINS 'club' OR
              toLower(p.title) CONTAINS 'celebration' OR toLower(p.title) CONTAINS 'festive' OR
              p.title CONTAINS 'ปาร์ตี้' OR p.title CONTAINS 'แรง' OR 
              p.title CONTAINS 'เลี้ยง' OR p.title CONTAINS 'กลางคืน' OR
              p.title CONTAINS 'สนุก' OR p.title CONTAINS 'เฟส'
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    else:
        # ถ้าไม่เจอ intent ที่ตรงกัน ให้แสดงสินค้าทั้งหมด
        query_string = """
        MATCH (p:Product)
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    
    result = graph.run(query_string).data()
    
    # ถ้าไม่มีผลลัพธ์ ให้ fallback เป็นการแสดงสินค้าทั้งหมด
    if not result:
        fallback_query = """
        MATCH (p:Product)
        RETURN p.title AS title, p.price AS price, p.size AS size, 
               p.image_url AS image_url, p.review AS review, p.stock AS stock
        ORDER BY p.review DESC
        LIMIT 5
        """
        result = graph.run(fallback_query).data()
    
    return result

# Enhanced intent response messages (Thai-English friendly)
def get_intent_response_message(intent):
    responses = {
        'greeting': 'สวัสดีค่ะ! ยินดีให้คำแนะนำเรื่องน้ำหอม (perfume) ค่ะ 🌸',
        'general_inquiry': 'ฉันสามารถแนะนำน้ำหอมตามความต้องการของคุณได้ค่ะ เช่น กลิ่นสดชื่น (fresh) กลิ่นหวาน (sweet) หรือตามโอกาสใช้งานค่ะ',
        'scent_fresh': 'แนะนำน้ำหอมกลิ่นสดชื่น (Fresh Scent) สำหรับคุณค่ะ: 🌿',
        'scent_sweet': 'แนะนำน้ำหอมกลิ่นหวาน (Sweet Scent) สำหรับคุณค่ะ: 🌸',
        'scent_sexy': 'แนะนำน้ำหอมกลิ่นเซ็กซี่ (Sexy Scent) สำหรับคุณค่ะ: 🔥',
        'season_summer': 'แนะนำน้ำหอมสำหรับหน้าร้อน (Summer Perfume) ค่ะ: ☀️',
        'season_winter': 'แนะนำน้ำหอมสำหรับหน้าหนาว (Winter Perfume) ค่ะ: ❄️',
        'occasion_work': 'แนะนำน้ำหอมสำหรับใส่ทำงาน (Office Perfume) ค่ะ: 💼',
        'occasion_date': 'แนะนำน้ำหอมสำหรับไปเดท (Date Perfume) ค่ะ: 💕',
        'occasion_party': 'แนะนำน้ำหอมสำหรับงานปาร์ตี้ (Party Perfume) ค่ะ: 🎉',
        'product_bestseller': 'แนะนำน้ำหอม Bestseller ขายดีสำหรับคุณค่ะ: ⭐',
        'product_new': 'แนะนำน้ำหอมใหม่ (New Arrivals) สำหรับคุณค่ะ: ✨',
        'product_reviewed': 'แนะนำน้ำหอมที่มี Review ดีสำหรับคุณค่ะ: 👍',
        'product_limited': 'แนะนำน้ำหอม Limited Edition สำหรับคุณค่ะ: 💎',
    }
    return responses.get(intent, 'ขอโทษค่ะ ฉันไม่เข้าใจคำถามของคุณ กรุณาลองถามใหม่หรือเลือกจากตัวเลือกที่มีค่ะ')

# Basic FAISS for fallback
basic_data = [['สวัสดี', 'สวัสดีค่ะ'], ['ayyo', 'ayyo']]
basic_df = pd.DataFrame(basic_data, columns=['text', 'category'])
basic_text = basic_df['text']
basic_vectors = encoder.encode(basic_text)
basic_vector_dimension = basic_vectors.shape[1]
basic_index = faiss.IndexFlatL2(basic_vector_dimension)
faiss.normalize_L2(basic_vectors)
basic_index.add(basic_vectors)

# Function to get all product titles as a corpus
def get_product_titles_as_corpus():
    query_string = """
    MATCH (p:Product)
    RETURN p.title AS title
    """
    result = graph.run(query_string).data()
    corpus = [record['title'] for record in result if 'title' in record]
    return corpus

corpus = get_product_titles_as_corpus()
print("Product Titles Corpus:", corpus)

# Function to get product details by title
def get_product_details_by_title(title):
    print(f"=== PRODUCT QUERY DEBUG ===")
    print(f"Searching for product title: '{title}'")
    
    # First try exact match
    query_string = f"""
    MATCH (p:Product {{title: $title}})
    RETURN p.title AS title, p.price AS price, p.size AS size, 
           p.image_url AS image_url, p.review AS review, p.stock AS stock
    LIMIT 1
    """
    
    print(f"Query: {query_string}")
    print(f"Parameter: title = '{title}'")
    
    try:
        result = graph.run(query_string, title=title).data()
        print(f"Query result: {result}")
        
        if not result:
            print(f"No exact match found for '{title}', trying case-insensitive search...")
            
            # Try case-insensitive search
            case_insensitive_query = """
            MATCH (p:Product)
            WHERE toLower(p.title) = toLower($title)
            RETURN p.title AS title, p.price AS price, p.size AS size, 
                   p.image_url AS image_url, p.review AS review, p.stock AS stock
            LIMIT 1
            """
            
            result = graph.run(case_insensitive_query, title=title).data()
            print(f"Case-insensitive result: {result}")
            
            if not result:
                print(f"Still no match, trying partial match...")
                
                # Try partial match
                partial_query = """
                MATCH (p:Product)
                WHERE p.title CONTAINS $title OR toLower(p.title) CONTAINS toLower($title)
                RETURN p.title AS title, p.price AS price, p.size AS size, 
                       p.image_url AS image_url, p.review AS review, p.stock AS stock
                LIMIT 1
                """
                
                result = graph.run(partial_query, title=title).data()
                print(f"Partial match result: {result}")
        
        if not result:
            print(f"No product found with title containing '{title}'")
            return f"ไม่พบสินค้าที่ชื่อ '{title}'"
        
        product = result[0]
        print(f"Found product: {product}")
        
        product_details = {
            "title": product['title'],
            "price": product['price'],
            "size": product['size'],
            "image_url": product['image_url'],
            "review": product['review'],
            "stock": product['stock']
        }
        
        print(f"Returning product details: {product_details}")
        print("=== PRODUCT QUERY COMPLETED ===")
        return product_details
        
    except Exception as e:
        print(f"Database query error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"เกิดข้อผิดพลาดในการค้นหาสินค้า '{title}'"

# Function to create Flex Carousel using product data
# Function to create Flex Carousel with interactive buttons
def create_flex_carousel(products):
    import urllib.parse
    bubbles = []
    for i, product in enumerate(products):
        # URL encode the product title to handle spaces and special characters
        encoded_title = urllib.parse.quote(product['title'])
        
        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover",
                "url": product['image_url'],
                "action": {
                    "type": "postback",
                    "data": f"action=view_detail&product_id={i}&title={encoded_title}"
                }
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "text",
                        "text": product['title'],
                        "wrap": True,
                        "weight": "bold",
                        "size": "xl"
                    },
                    {
                        "type": "box",
                        "layout": "baseline",
                        "contents": [
                            {
                                "type": "text",
                                "text": f"Size: {product['size']}",
                                "wrap": True,
                                "weight": "bold",
                                "size": "xs",
                                "color": "#8c8c8c",
                                "flex": 0
                            }
                        ]
                    },
                    {
                        "type": "box",
                        "layout": "baseline",
                        "contents": [
                            {
                                "type": "text",
                                "text": f"⭐ {product['review'] if product['review'] else 'No Review'}",
                                "wrap": True,
                                "weight": "bold",
                                "size": "sm",
                                "color": "#ff5551" if not product['review'] else "#00AA00",
                                "flex": 0
                            }
                        ]
                    },
                    {
                        "type": "box",
                        "layout": "baseline",
                        "contents": [
                            {
                                "type": "text",
                                "text": f"💰 {product['price']}",
                                "wrap": True,
                                "weight": "bold",
                                "size": "lg",
                                "color": "#ff5551",
                                "flex": 0
                            }
                        ]
                    },
                    {
                        "type": "text",
                        "text": f"📦 {product['stock']}",
                        "wrap": True,
                        "size": "xxs",
                        "margin": "md",
                        "color": "#ff5551" if product['stock'] == "Temporarily out of stock" else "#00AA00",
                        "flex": 0
                    }
                ]
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "button",
                        "style": "primary",
                        "action": {
                            "type": "postback",
                            "label": "📋 ดูรายละเอียด",
                            "data": f"action=view_detail&product_id={i}&title={encoded_title}"
                        }
                    },
                    {
                        "type": "button",
                        "style": "secondary",
                        "action": {
                            "type": "postback",
                            "label": "🛒 Add to Cart",
                            "data": f"action=add_cart&product_id={i}&title={encoded_title}"
                        }
                    }
                ]
            }
        }
        bubbles.append(bubble)

    carousel = {
        "type": "carousel",
        "contents": bubbles
    }

    return FlexSendMessage(alt_text="Product Catalog", contents=carousel)
# Function to create detailed product card
def create_detailed_product_card(product):
    import urllib.parse
    encoded_title = urllib.parse.quote(product['title'])
    
    detailed_card = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover",
            "url": product['image_url']
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": product['title'],
                    "weight": "bold",
                    "size": "xl",
                    "wrap": True
                },
                {
                    "type": "separator",
                    "margin": "md"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "spacing": "sm",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "💰 ราคา",
                                    "color": "#aaaaaa",
                                    "size": "sm",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": product['price'],
                                    "wrap": True,
                                    "color": "#ff5551",
                                    "size": "lg",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "📏 ขนาด",
                                    "color": "#aaaaaa",
                                    "size": "sm",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": product['size'],
                                    "wrap": True,
                                    "color": "#666666",
                                    "size": "md",
                                    "flex": 2
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "⭐ รีวิว",
                                    "color": "#aaaaaa",
                                    "size": "sm",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": product['review'] if product['review'] else "ยังไม่มีรีวิว",
                                    "wrap": True,
                                    "color": "#00AA00" if product['review'] else "#ff5551",
                                    "size": "md",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "baseline",
                            "spacing": "sm",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "📦 สต็อก",
                                    "color": "#aaaaaa",
                                    "size": "sm",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": product['stock'],
                                    "wrap": True,
                                    "color": "#00AA00" if product['stock'] != "Temporarily out of stock" else "#ff5551",
                                    "size": "md",
                                    "weight": "bold",
                                    "flex": 2
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "action": {
                        "type": "postback",
                        "label": "🛒 เพิ่มในตะกร้า",
                        "data": f"action=add_cart&title={encoded_title}"
                    }
                },
                {
                    "type": "button",
                    "style": "secondary",
                    "action": {
                        "type": "message",
                        "label": "🔍 ดูสินค้าอื่น",
                        "text": "แนะนำสินค้าอื่น"
                    }
                }
            ]
        }
    }
    
    return FlexSendMessage(alt_text=f"รายละเอียด {product['title']}", contents=detailed_card)

# Function to handle postback events
def handle_postback_event(line_bot_api, event):
    try:
        data = event.postback.data
        user_id = event.source.user_id
        reply_token = event.reply_token
        
        print(f"=== POSTBACK DEBUG ===")
        print(f"Raw postback data: {data}")
        print(f"User ID: {user_id}")
        print(f"Reply Token: {reply_token}")
        
        # Parse postback data - improved parsing
        params = {}
        for param in data.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)  # Split only on first '=' 
                params[key] = value
        
        print(f"Parsed params: {params}")
        
        action = params.get('action')
        product_title = params.get('title', '')
        
        # URL decode the product title
        import urllib.parse
        decoded_title = urllib.parse.unquote(product_title)
        
        print(f"Action: {action}")
        print(f"Encoded Title: {product_title}")
        print(f"Decoded Title: {decoded_title}")
        
        if action == 'view_detail':
            print(f"Processing view_detail for: {decoded_title}")
            
            # Get detailed product information
            product_details = get_product_details_by_title(decoded_title)
            print(f"Product details query result: {product_details}")
            
            if isinstance(product_details, dict):
                print("Creating detailed card...")
                # Create detailed card
                detailed_card = create_detailed_product_card(product_details)
                line_bot_api.reply_message(reply_token, detailed_card)
                print("Detailed card sent successfully")
                
                # Save interaction to chat history
                bot_response = f"แสดงรายละเอียดสินค้า: {decoded_title}"
                save_chat_history_with_relationship(user_id, f"ดูรายละเอียด {decoded_title}", bot_response)
            else:
                print(f"Product not found: {decoded_title}")
                error_message = f"ขอโทษค่ะ ไม่พบข้อมูลรายละเอียดของสินค้า '{decoded_title}'"
                line_bot_api.reply_message(
                    reply_token, 
                    TextSendMessage(text=error_message)
                )
        
        elif action == 'add_cart':
            print(f"Processing add_cart for: {decoded_title}")
            
            # Handle add to cart action
            cart_message = f"✅ เพิ่ม '{decoded_title}' ลงในตะกร้าแล้วค่ะ!\n\n" \
                          f"🛒 ดูตะกร้าสินค้า: /cart\n" \
                          f"💳 สั่งซื้อ: /checkout\n" \
                          f"🔍 ดูสินค้าอื่น: พิมพ์ 'แนะนำ'"
            
            line_bot_api.reply_message(reply_token, TextSendMessage(text=cart_message))
            print("Add to cart message sent")
            
            # Save to cart in Neo4j
            save_to_cart(user_id, decoded_title)
            
            # Save interaction to chat history
            bot_response = f"เพิ่ม {decoded_title} ลงในตะกร้า"
            save_chat_history_with_relationship(user_id, f"เพิ่มในตะกร้า {decoded_title}", bot_response)
        
        else:
            print(f"Unknown action: {action}")
            line_bot_api.reply_message(reply_token, TextSendMessage(text="ขอโทษค่ะ ไม่เข้าใจคำสั่งที่เลือก"))
            
        print("=== POSTBACK PROCESSING COMPLETED ===")
            
    except Exception as e:
        print(f"=== POSTBACK ERROR ===")
        print(f"Error details: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("=== END ERROR ===")
        
        try:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="ขอโทษค่ะ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"))
        except Exception as reply_error:
            print(f"Reply error: {reply_error}")

# Function to save item to cart in Neo4j
def save_to_cart(user_id, product_title):
    query = '''
    MERGE (u:User {user_id: $user_id})
    MATCH (p:Product {title: $product_title})
    MERGE (u)-[r:ADDED_TO_CART]->(p)
    ON CREATE SET r.timestamp = datetime(), r.quantity = 1
    ON MATCH SET r.timestamp = datetime(), r.quantity = r.quantity + 1
    '''
    with graph.session() as session:
        session.run(query, user_id=user_id, product_title=product_title)

# Function to get user's cart
def get_user_cart(user_id):
    query = '''
    MATCH (u:User {user_id: $user_id})-[r:ADDED_TO_CART]->(p:Product)
    RETURN p.title AS title, p.price AS price, p.image_url AS image_url, 
           r.quantity AS quantity, r.timestamp AS added_time
    ORDER BY r.timestamp DESC
    '''
    with graph.session() as session:
        result = session.run(query, user_id=user_id).data()
    return result

# Enhanced message handler with cart commands
def return_message(line_bot_api, tk, user_id, msg):
    # Handle special commands
    if msg.lower() == '/cart' or msg == 'ตะกร้า':
        cart_items = get_user_cart(user_id)
        if cart_items:
            cart_text = "🛒 ตะกร้าสินค้าของคุณ:\n\n"
            total_items = 0
            for item in cart_items:
                cart_text += f"• {item['title']}\n"
                cart_text += f"  จำนวน: {item['quantity']} ชิ้น\n"
                cart_text += f"  ราคา: {item['price']}\n\n"
                total_items += item['quantity']
            
            cart_text += f"รวม {total_items} รายการ\n\n"
            cart_text += "💳 สั่งซื้อ: พิมพ์ '/checkout'\n"
            cart_text += "🗑️ ล้างตะกร้า: พิมพ์ '/clear_cart'"
            
            line_bot_api.reply_message(tk, TextSendMessage(text=cart_text))
        else:
            line_bot_api.reply_message(tk, TextSendMessage(text="🛒 ตะกร้าสินค้าของคุณว่างเปล่าค่ะ\nลองเลือกสินค้าจากรายการแนะนำดูค่ะ"))
        
        save_chat_history_with_relationship(user_id, msg, "แสดงตะกร้าสินค้า")
        return
    
    elif msg.lower() == '/checkout' or msg == 'สั่งซื้อ':
        cart_items = get_user_cart(user_id)
        if cart_items:
            checkout_text = "💳 ขั้นตอนการสั่งซื้อ:\n\n"
            checkout_text += "1. ตรวจสอบรายการสินค้า ✅\n"
            checkout_text += "2. กรอกข้อมูลจัดส่ง 📋\n"
            checkout_text += "3. เลือกวิธีการชำระเงิน 💰\n"
            checkout_text += "4. ยืนยันการสั่งซื้อ ✅\n\n"
            checkout_text += "📞 ติดต่อทีมขาย: 02-xxx-xxxx\n"
            checkout_text += "💬 LINE: @perfumeshop"
            
            line_bot_api.reply_message(tk, TextSendMessage(text=checkout_text))
        else:
            line_bot_api.reply_message(tk, TextSendMessage(text="ตะกร้าสินค้าของคุณว่างเปล่าค่ะ กรุณาเลือกสินค้าก่อน"))
        
        save_chat_history_with_relationship(user_id, msg, "เริ่มกระบวนการสั่งซื้อ")
        return
    
    elif msg.lower() == '/clear_cart' or msg == 'ล้างตะกร้า':
        clear_cart_query = '''
        MATCH (u:User {user_id: $user_id})-[r:ADDED_TO_CART]->()
        DELETE r
        '''
        with graph.session() as session:
            session.run(clear_cart_query, user_id=user_id)
        
        line_bot_api.reply_message(tk, TextSendMessage(text="🗑️ ล้างตะกร้าสินค้าเรียบร้อยแล้วค่ะ"))
        save_chat_history_with_relationship(user_id, msg, "ล้างตะกร้าสินค้า")
        return

    # Apply text normalization first
    normalized_msg = text_normalizer.normalize_text(msg)
    print(f"Original message: {msg}")
    print(f"Normalized message: {normalized_msg}")
    
    # Try keyword-based intent extraction first
    keyword_intent, keyword_score = text_normalizer.extract_intent_from_text(msg)
    print(f"Keyword intent: {keyword_intent}, Score: {keyword_score}")
    
    # Predict intent using ML classifier with normalized text
    try:
        predicted_intent = intent_classifier.predict([normalized_msg])[0]
        confidence = max(intent_classifier.predict_proba([normalized_msg])[0])
        
        print(f"ML predicted intent: {predicted_intent}, Confidence: {confidence}")
        
        # Use keyword intent if it has high score, otherwise use ML prediction
        if keyword_score >= 1 and confidence < 0.8:
            final_intent = keyword_intent
            final_confidence = keyword_score / len(text_normalizer.intent_keywords.get(keyword_intent, []))
            print(f"Using keyword intent: {final_intent}")
        else:
            final_intent = predicted_intent
            final_confidence = confidence
            print(f"Using ML intent: {final_intent}")
        
        # If both confidence is low, fall back to basic FAISS
        if final_confidence < 0.5:
            chk_msg = check_sentence_basic(msg)
            if chk_msg[0] == "สวัสดี":
                bot_response = chk_msg[1]
                line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
                save_chat_history_with_relationship(user_id, msg, bot_response)
                return
                
    except Exception as e:
        print(f"Intent classification error: {e}")
        final_intent = "unknown"
        final_confidence = 0.0

    bot_response = ""

    # Handle different intents
    if final_intent == "greeting":
        bot_response = get_intent_response_message(final_intent)
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
        
    elif final_intent == "general_inquiry":
        quick_reply_items = [
            QuickReplyButton(action=MessageAction(label="🌟 BEST SELLER", text="สินค้าขายดี")),
            QuickReplyButton(action=MessageAction(label="🆕 NEW", text="สินค้าใหม่")),
            QuickReplyButton(action=MessageAction(label="💬 มีรีวิว", text="สินค้าที่มีรีวิว")),
            QuickReplyButton(action=MessageAction(label="🌿 กลิ่นสดชื่น", text="แนะนำกลิ่นสดชื่น")),
            QuickReplyButton(action=MessageAction(label="🛒 ตะกร้า", text="/cart"))
        ]
        quick_reply_buttons = QuickReply(items=quick_reply_items)
        bot_response = get_intent_response_message(final_intent)
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response, quick_reply=quick_reply_buttons))
        
    elif final_intent in ['product_bestseller', 'product_new', 'product_reviewed', 'product_limited',
                         'scent_fresh', 'scent_sweet', 'scent_sexy', 
                         'season_summer', 'season_winter',
                         'occasion_work', 'occasion_date', 'occasion_party']:
        
        products = search_products_by_intent(final_intent, normalized_msg)
        
        if products:
            response_message = get_intent_response_message(final_intent)
            flex_message = create_flex_carousel(products)
            
            # Send text message first, then flex message
            line_bot_api.reply_message(tk, [
                TextSendMessage(text=response_message),
                flex_message
            ])
            bot_response = f"{response_message} (ส่ง Flex Message แสดง {len(products)} รายการ)"
        else:
            bot_response = f"ขอโทษค่ะ ไม่พบสินค้าที่ตรงกับ '{msg}' ลองใช้คำค้นหาอื่นดูค่ะ"
            line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
    
    # Check if message is a specific product title
    elif msg in corpus:
        product_details = get_product_details_by_title(msg)
        if isinstance(product_details, dict):
            detailed_card = create_detailed_product_card(product_details)
            line_bot_api.reply_message(tk, detailed_card)
            bot_response = f"แสดงรายละเอียดสินค้า: {msg}"
        else:
            bot_response = product_details
            line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
    
    else:
        # Suggest similar terms based on normalized text
        suggestions = []
        for intent_name, keywords in text_normalizer.intent_keywords.items():
            for keyword in keywords:
                if keyword in normalized_msg:
                    suggestions.append(get_intent_response_message(intent_name))
                    break
        
        if suggestions:
            bot_response = f"คุณหมายถึง: {', '.join(suggestions[:2])} ใช่ไหมคะ?"
        else:
            bot_response = "ขอโทษค่ะ ฉันไม่เข้าใจคำถามของคุณ\nลองถาม 'แนะนำ perfume' หรือ 'ขอ review ดีๆ' ดูค่ะ 😊\n\nหรือดูตะกร้าสินค้า: /cart"
        
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))

    save_chat_history_with_relationship(user_id, msg, bot_response)

# Basic FAISS check function for fallback
def check_sentence_basic(msg):
    search_vector = encoder.encode(msg)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    distances, ann = basic_index.search(_vector, k=1)
    if distances[0][0] <= 0.4:
        category = basic_df['category'][ann[0][0]]
        sentence = basic_df['text'][ann[0][0]]
    else:
        category = msg
        sentence = msg
    return [sentence, category]

def save_chat_history_with_relationship(user_id, user_message, bot_message):
    query = '''
    MERGE (u:User {user_id: $user_id})
    CREATE (um:UserMessage {message: $user_message, timestamp: datetime()})
    CREATE (bm:BotMessage {message: $bot_message, timestamp: datetime()})
    MERGE (u)-[:SENT]->(um)
    MERGE (bm)-[:REPLIED_WITH]->(u)
    '''
    with graph.session() as session:
        session.run(query, user_id=user_id, user_message=user_message, bot_message=bot_message)

# Initialize Flask app
app = Flask(__name__)

# Enhanced Flask webhook handler
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'kcKoDOsKhahBSiDnrbZTdizLgnwEXsOy2vmdJ/Lpti6eg+2RDPKORbdEGR9zizzLs7kcO1UZa36xWYlQcrIt6WW5sbjWMmAh0SSsW0RK8jF6se6/cLQ3a70+c5pIpjjYe82X3O7VcmH7grDA87DgBAdB04t89/1O/w1cDnyilFU='
        secret = 'e44afc0c53a055472793d01e2a0552a6'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers.get('X-Line-Signature', '')
        
        # Handle different event types
        if json_data.get('events'):
            event = json_data['events'][0]
            
            if event.get('type') == 'message' and event.get('message', {}).get('type') == 'text':
                msg = event['message']['text']
                user_id = event['source']['userId']
                tk = event['replyToken']
                return_message(line_bot_api, tk, user_id, msg)
                
            elif event.get('type') == 'postback':
                # Create event object for postback
                class PostbackEventObj:
                    def __init__(self, data):
                        self.postback = type('obj', (object,), {'data': data['postback']['data']})
                        self.source = type('obj', (object,), {'user_id': data['source']['userId']})
                        self.reply_token = data['replyToken']
                
                postback_event = PostbackEventObj(event)
                handle_postback_event(line_bot_api, postback_event)
                
    except Exception as e:
        print(f"Webhook error: {e}")
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)
from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, QuickReply, QuickReplyButton, MessageAction
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import json

# Neo4j connection
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "theoneandonlyhana"))

# Step 1: Define FAISS index and Sentence Encoding for basic commands
data = [['สวัสดี', 'สวัสดีค่ะ'],
        ['ayyo', 'ayyo'],
        ['มีอะไรคุย', 'สอบถาม'],
        ['รู้อะไรบ้าง', 'สอบถาม'],
        ['ช่วยแนะนำหน่อย', 'สอบถาม'],]
df = pd.DataFrame(data, columns=['text', 'category'])
text = df['text']
encoder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
vectors = encoder.encode(text)
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

# Function to get all product titles as a corpus
def get_product_titles_as_corpus():
    query_string = """
    MATCH (p:Product)
    RETURN p.title AS title
    """
    result = graph.run(query_string).data()
    
    # Extract titles into a list (corpus)
    corpus = [record['title'] for record in result if 'title' in record]
    return corpus

# Usage example
corpus = get_product_titles_as_corpus()
print("Product Titles Corpus:", corpus)

# Function to get product details by title
def get_product_details_by_title(title):
    query_string = f"""
    MATCH (p:Product {{title: '{title}'}})
    RETURN p.title AS title, p.price AS price, p.size AS size, 
           p.image_url AS image_url, p.review AS review, p.stock AS stock
    LIMIT 1
    """
    result = graph.run(query_string).data()
    print("result", result)
    
    if not result:
        return f"ไม่พบสินค้าที่ชื่อ '{title}'"
    
    # Return product details as a dictionary
    product = result[0]
    product_details = {
        "title": product['title'],
        "price": product['price'],
        "size": product['size'],
        "image_url": product['image_url'],
        "review": product['review'],
        "stock": product['stock']
    }
    return product_details

# Function to search products by status or general keywords
def search_products(query, field=None):
    query_string = ""
    if field == "status":
        # ค้นหาสินค้าที่มีสถานะตามที่ระบุ
        query_string = f"""
        MATCH (p:Product)-[:HAS_STATUS]->(s:Status {{name: '{query}'}}) 
        RETURN p.title AS title, p.price AS price, p.size AS size, p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    elif field == "product":
        # ค้นหาสินค้าที่มีรีวิวในรูปแบบตัวเลข/ตัวเลข
        query_string = """
        MATCH (p:Product)
        WHERE p.review =~ '\\d+/\\d+'
        RETURN p.title AS title, p.price AS price, p.size AS size, p.image_url AS image_url, p.review AS review, p.stock AS stock
        LIMIT 5
        """
    
    result = graph.run(query_string).data()
    return result

# Function to format product response
def format_product_response(products):
    if not products:
        return "ไม่พบสินค้าที่คุณค้นหา"
    
    response = "รายการสินค้าที่เกี่ยวข้อง:\n"
    for product in products:
        response += f"🔹 {product['title']} - {product['price']} THB ({product['size']})\n"
        response += f"📊 รีวิว: {product['review'] if product['review'] else 'No review available'}\n"
        response += f"📦 สถานะสินค้า: {product['stock']}\n\n"
    
    return response

# Function to create Flex Carousel using product data
def create_flex_carousel(products):
    bubbles = []
    for product in products:
        bubble = {
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
                                "text": product['size'],
                                "wrap": True,
                                "weight": "bold",
                                "size": "xs",
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
                                "text": product['review'] if product['review'] else "No Review",
                                "wrap": True,
                                "weight": "bold",
                                "size": "sm",
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
                                "text": product['price'],
                                "wrap": True,
                                "weight": "bold",
                                "size": "lg",
                                "flex": 0
                            }
                        ]
                    },
                    {
                        "type": "text",
                        "text": f"Stock: {product['stock']}",
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
                            "type": "uri",
                            "label": "Add to Cart",
                            "uri": "https://line.me/"
                        }
                    },
                    {
                        "type": "button",
                        "action": {
                            "type": "uri",
                            "label": "ดูรายละเอียดเพิ่มเติม",
                            "uri": "https://line.me/"
                        }
                    }
                ]
            }
        }
        bubbles.append(bubble)

    # Create the carousel structure
    carousel = {
        "type": "carousel",
        "contents": bubbles
    }

    # Return the Flex Message
    return FlexSendMessage(alt_text="Product Catalog", contents=carousel)

# Function to return Quick Reply options based on user input
def return_message(line_bot_api, tk, user_id, msg):
    chk_msg = check_sentent(msg)
    bot_response = ""  # เพิ่มตัวแปรเพื่อเก็บข้อความตอบกลับจากบอท

    # ถ้าข้อความตรงกับคำทักทาย
    if chk_msg[0] == "สวัสดี":
        bot_response = chk_msg[1]
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))

    # ถ้าข้อความตรงกับ "สอบถาม" ให้แสดง Quick Reply
    elif chk_msg[0] == "สอบถาม":
        quick_reply_items = [
            QuickReplyButton(action=MessageAction(label="BEST SELLER", text="สินค้าขายดี")),
            QuickReplyButton(action=MessageAction(label="NEW", text="สินค้าใหม่")),
            QuickReplyButton(action=MessageAction(label="มีรีวิว", text="สินค้าที่มีรีวิว")),
            QuickReplyButton(action=MessageAction(label="Limited Edition", text="สินค้า Limited Edition"))
        ]
        quick_reply_buttons = QuickReply(items=quick_reply_items)
        bot_response = "กรุณาเลือกประเภทสินค้า"
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response, quick_reply=quick_reply_buttons))

    # ถ้าผู้ใช้เลือกปุ่ม Quick Reply หรือถามเกี่ยวกับสินค้า
    elif msg in ["สินค้าขายดี", "สินค้าใหม่", "สินค้าที่มีรีวิว", "สินค้า Limited Edition"] or msg in corpus:
        if msg == "สินค้าขายดี":
            products = search_products("BESTSELLER", field="status")
        elif msg == "สินค้าใหม่":
            products = search_products("NEW", field="status")
        elif msg == "สินค้าที่มีรีวิว":
            products = search_products("review", field="product")
        elif msg == "สินค้า Limited Edition":
            products = search_products("Limited Edition", field="status")
        else:
            product_details = get_product_details_by_title(msg)
            if isinstance(product_details, dict):
                bot_response = (
                    f"รายละเอียดสินค้า:\n"
                    f"🔹 ชื่อ: {product_details['title']}\n"
                    f"💵 ราคา: {product_details['price']} THB\n"
                    f"📏 ขนาด: {product_details['size']}\n"
                    f"🖼️ รูปภาพ: {product_details['image_url']}\n"
                    f"📊 รีวิว: {product_details['review'] if product_details['review'] else 'No review available'}\n"
                    f"📦 สถานะสินค้า: {product_details['stock']}\n"
                )
                line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
            else:
                bot_response = product_details
                line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))
            save_chat_history_with_relationship(user_id, msg, bot_response)
            return

        if products:
            flex_message = create_flex_carousel(products)
            line_bot_api.reply_message(tk, flex_message)
            bot_response = "ส่ง Flex Message แสดงรายการสินค้า"
        else:
            bot_response = f"ไม่พบสินค้า{msg}"
            line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))

    # กรณีที่ผู้ใช้ถามเกี่ยวกับสินค้าโดยไม่ใช้ Quick Reply หรือ title
    else:
        products = search_products(msg)
        bot_response = format_product_response(products)
        line_bot_api.reply_message(tk, TextSendMessage(text=bot_response))

    save_chat_history_with_relationship(user_id, msg, bot_response)

# Step 2: Check user's input using FAISS
def check_sentent(msg):
    search_vector = encoder.encode(msg)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    distances, ann = index.search(_vector, k=1)
    if distances[0][0] <= 0.4:
        category = df['category'][ann[0][0]]
        Sentence = df['text'][ann[0][0]]
    else:
        category = msg
        Sentence = msg
    return [ Sentence, category]

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

# Step 3: Flask app and LINE webhook handler
app = Flask(__name__)

@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = 'kcKoDOsKhahBSiDnrbZTdizLgnwEXsOy2vmdJ/Lpti6eg+2RDPKORbdEGR9zizzLs7kcO1UZa36xWYlQcrIt6WW5sbjWMmAh0SSsW0RK8jF6se6/cLQ3a70+c5pIpjjYe82X3O7VcmH7grDA87DgBAdB04t89/1O/w1cDnyilFU='
        secret = 'e44afc0c53a055472793d01e2a0552a6'
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        msg = json_data['events'][0]['message']['text']
        user_id = json_data['events'][0]['source']['userId']
        tk = json_data['events'][0]['replyToken']
        return_message(line_bot_api, tk, user_id, msg)
    except Exception as e:
        print(f"Error: {e}")
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)

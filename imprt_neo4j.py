from py2neo import Graph, Node, Relationship
import json

# เชื่อมต่อกับ Neo4j
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "theoneandonlyhana"))

# โหลดข้อมูล JSON
with open('C:\\Users\\firhana\\socialai\\jomalone\\product_json\\jomalone_products.json', encoding='utf-8') as f:
    data = json.load(f)

# ฟังก์ชันเพื่อสร้างโหนดและความสัมพันธ์
def create_product_node(product):
    # สร้างโหนด Product
    product_node = Node("Product",
                        title=product['title'],
                        size=product['size'],
                        price=product['price'],
                        image_url=product['image_url'],
                        review=product['review'],
                        stock=product['Stock'])
    graph.create(product_node)

    # สร้างหรือเชื่อมโยงโหนด Status
    status_node = graph.nodes.match("Status", name=product['status']).first()
    if status_node is None:
        status_node = Node("Status", name=product['status'])
        graph.create(status_node)
    
    # สร้างความสัมพันธ์ระหว่าง Product และ Status
    graph.create(Relationship(product_node, "HAS_STATUS", status_node))

    # สร้างโหนด Note (Top, Heart, Base) และสร้างความสัมพันธ์
    top_note = Node("Note", type="Top Note", description=product['top_note'])
    heart_note = Node("Note", type="Heart Note", description=product['Heart_Note'])
    base_note = Node("Note", type="Base Note", description=product['Base Note'])

    graph.create(top_note)
    graph.create(heart_note)
    graph.create(base_note)

    # เชื่อมโยง Notes กับ Product
    graph.create(Relationship(product_node, "HAS_TOP_NOTE", top_note))
    graph.create(Relationship(product_node, "HAS_HEART_NOTE", heart_note))
    graph.create(Relationship(product_node, "HAS_BASE_NOTE", base_note))

# ลูปผ่านข้อมูล JSON ทั้งหมดแล้วสร้างโหนดและความสัมพันธ์
for product in data:
    create_product_node(product)

print("Data imported successfully into Neo4j!")

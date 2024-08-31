import pytesseract
from PIL import Image
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text= pytesseract.image_to_string(image)
    return text


def preprocess_text(text):
    text= re.sub(r'\W+', ' ', text)
    text=text.lower()
    return text


def extract_features(text):
    invoice_number = re.search(r'invoice\s+number\s*[:\-]?\s*(\w+)', text)
    total_amount = re.search(r'amount\s*[:\-]?\s*\$?(\d+\.?\d*)', text)
    invoice_date = re.search(r'date\s*[:\-]?\s*(\d{2}-\d{2}-\d{4})', text)
    vendor_name= re.search(r'vendor\s*[:\-]?\s*(\w+)',text)

    return{
        "number": invoice_number.group(1) if invoice_number else '',
        "amount": float(total_amount.group(1)) if total_amount else 0.0,
        "date": invoice_date.group(1) if invoice_date else '',
        "vendor": vendor_name.group(1) if vendor_name else ''

    }


def fuzzy_match(str1, str2):
    return fuzz.ratio(str1, str2)

def compare_invoices(inv1, inv2, cosine_sim):
    number_sim= fuzzy_match(inv1["number"], inv2["number"])
    amount_sim= 1- abs(inv1["amount"] - inv2["amount"])/ max(inv1["amount"], inv2["amount"])
    date_sim= fuzzy_match(inv1["date"], inv2["date"])
    vendor_sim = fuzzy_match(inv1["vendor"], inv2["vendor"])

    overall_sim = 0.4* cosine_sim[inv1["index"]][inv2["index"]] + 0.2* number_sim/ 100 + 0.2 * amount_sim + 0.1* date_sim/ 100 + 0.1* vendor_sim/100
    return overall_sim

invoice_images= []
invoices=[]


for i, image_path in enumerate(invoice_images):
    text= extract_text_from_image(image_path)
    cleaned_text= preprocess_text(text)
    features= extract_features(cleaned_text)
    features["index"] =i
    features["text"]= cleaned_text
    invoices.append(features)


texts= [invoice["text"] for invoice in invoices]
vectorizer= TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
cosine_sim= cosine_similarity(tfidf_matrix, tfidf_matrix)

threshold=0.8
duplicates=[]

for i in range(len(invoices)):
    for j in range(i+1, len(invoices)):
        sim= compare_invoices(invoices[i], invoices[j], cosine_sim)
        if sim> threshold:
            duplicates.append((invoices[i]["index"], invoices[j]["index"], sim))


print("Duplicate Invoices:", duplicates)




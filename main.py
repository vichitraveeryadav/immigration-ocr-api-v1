from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import os
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Copy ALL your working classes exactly
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="Immigration OCR API", version="1.0")

# Add CORS for website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EXACT same configuration as your Streamlit
BASE_DIR = Path.cwd()
DATABASE_URL = "sqlite:///immigration_docs.db"

DOCUMENT_TYPES = {
    "passport": ["passport", "travel document"],
    "visa": ["visa", "entry permit"], 
    "permit": ["work permit", "study permit"],
    "certificate": ["birth certificate", "marriage certificate"],
    "identification": ["driver license", "id card"]
}

# EXACT same database classes
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    extracted_text = Column(Text)
    structured_data = Column(Text)
    confidence_score = Column(Float)
    processed_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine

def get_db_session():
    engine = init_database()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()

# EXACT same OCR processor from your code
class OCRProcessor:
    def __init__(self):
        self.languages = "eng+hin"

    def process_document(self, image_path):
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang=self.languages, config=config)
            
            return {
                "text": text.strip(), 
                "confidence": 0.8,
                "engine": "tesseract"
            }
        except Exception as e:
            return {
                "text": "", 
                "confidence": 0.0, 
                "error": str(e),
                "engine": "tesseract"
            }

# EXACT same document classifier
class DocumentClassifier:
    def __init__(self):
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('nb', MultinomialNB())
        ])
        self.is_trained = False

    def prepare_training_data(self):
        training_data = [
            ("passport number personal details republic india", "passport"),
            ("visa entry permit immigration canada", "visa"),
            ("work permit employment authorization", "permit"),
            ("birth certificate date of birth", "certificate"),
            ("driver license identification card", "identification"),
            ("travel document immigration status", "passport"),
            ("study permit student visa education", "permit"),
            ("marriage certificate spouse husband wife", "certificate"),
            ("temporary resident visa immigration", "visa"),
            ("permanent resident card citizenship", "identification"),
            ("passport issued government travel", "passport"),
            ("visa stamp entry country", "visa")
        ]
        texts, labels = zip(*training_data)
        return list(texts), list(labels)

    def train_classifier(self):
        texts, labels = self.prepare_training_data()
        self.classifier.fit(texts, labels)
        self.is_trained = True

    def classify_document(self, text):
        if not self.is_trained:
            self.train_classifier()
        
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        try:
            proba = self.classifier.predict_proba([cleaned_text])
            pred = self.classifier.classes_[proba.argmax()]
            confidence = float(proba.max())
        except:
            pred = "unknown"
            confidence = 0.5
        
        return {"document_type": pred, "confidence": confidence}

# EXACT same data extractor
class DataExtractor:
    def __init__(self):
        self.patterns = {
            "passport_number": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            "name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "visa_number": r'\b[A-Z0-9]{8,12}\b',
            "phone": r'\+?[0-9]{10,13}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }

    def extract_structured_data(self, text, document_type):
        base = {
            "extraction_date": datetime.now().isoformat(),
            "raw_text_length": len(text),
            "document_type": document_type
        }
        
        if document_type == "passport":
            m = re.search(self.patterns['passport_number'], text, re.IGNORECASE)
            if m:
                base['passport_number'] = m.group()
        
        if document_type == "visa":
            m = re.search(self.patterns['visa_number'], text, re.IGNORECASE)
            if m:
                base['visa_number'] = m.group()
        
        dates = re.findall(self.patterns['date'], text)
        if dates:
            base['dates_found'] = dates[:3]
        
        names = re.findall(self.patterns['name'], text)
        if names:
            base['names_found'] = names[:2]
        
        email = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email:
            base['email'] = email.group()
        
        return base

# EXACT same file organizer
class FileOrganizer:
    def __init__(self, base_path="processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    def organize_document(self, username, document_type, source_file, extracted_data):
        user_path = self.base_path / username
        user_path.mkdir(exist_ok=True)
        
        doc_folder = user_path / document_type
        doc_folder.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_path = Path(source_file)
        new_filename = f"{document_type}_{timestamp}{source_path.suffix}"
        
        destination = doc_folder / new_filename
        shutil.copy2(source_path, destination)
        
        return str(destination)

# Initialize services (same as Streamlit)
ocr_processor = OCRProcessor()
doc_classifier = DocumentClassifier()
data_extractor = DataExtractor()
file_organizer = FileOrganizer()

# Response model
class ProcessResponse(BaseModel):
    success: bool
    document_type: str
    confidence: float
    extracted_text: str
    structured_data: dict
    organized_path: str
    error_message: Optional[str] = None

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Immigration Document API is running!", "status": "healthy"}

@app.post("/process-document", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    username: str = "default_user"
):
    """Process immigration document - EXACT same logic as Streamlit"""
    temp_path = Path(f"temp_{file.filename}")
    
    try:
        # Initialize database
        init_database()
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Step 1: OCR (same as Streamlit)
        ocr_result = ocr_processor.process_document(temp_path)
        
        if not ocr_result.get('text'):
            return ProcessResponse(
                success=False,
                document_type="unknown",
                confidence=0.0,
                extracted_text="",
                structured_data={},
                organized_path="",
                error_message=f"Could not extract text: {ocr_result.get('error', 'Unknown error')}"
            )

        # Step 2: Classification (same as Streamlit)
        classification = doc_classifier.classify_document(ocr_result['text'])

        # Step 3: Data extraction (same as Streamlit)
        structured_data = data_extractor.extract_structured_data(
            ocr_result['text'], classification['document_type']
        )

        # Step 4: File organization (same as Streamlit)
        organized_path = file_organizer.organize_document(
            username, classification['document_type'], temp_path, structured_data
        )

        # Step 5: Database (same as Streamlit)
        try:
            db_session = get_db_session()
            
            user = db_session.query(User).filter_by(username=username).first()
            if not user:
                user = User(username=username)
                db_session.add(user)
                db_session.commit()

            new_doc = Document(
                user_id=user.id,
                filename=file.filename,
                document_type=classification['document_type'],
                file_path=organized_path,
                extracted_text=ocr_result['text'][:1000],
                structured_data=json.dumps(structured_data),
                confidence_score=float(ocr_result.get('confidence', 0.0))
            )
            db_session.add(new_doc)
            db_session.commit()
            db_session.close()
        except Exception as e:
            print(f"Database save failed: {str(e)}")

        return ProcessResponse(
            success=True,
            document_type=classification['document_type'],
            confidence=classification['confidence'],
            extracted_text=ocr_result['text'][:2000],
            structured_data=structured_data,
            organized_path=organized_path
        )
        
    except Exception as e:
        return ProcessResponse(
            success=False,
            document_type="unknown",
            confidence=0.0,
            extracted_text="",
            structured_data={},
            organized_path="",
            error_message=f"Processing error: {str(e)}"
        )
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/documents/{username}")
async def get_user_documents(username: str):
    """Get all documents for a user"""
    try:
        db_session = get_db_session()
        documents = db_session.query(Document).join(User).filter(User.username == username).all()
        
        result = []
        for doc in documents:
            result.append({
                "filename": doc.filename,
                "document_type": doc.document_type,
                "confidence_score": doc.confidence_score,
                "processed_at": doc.processed_at.isoformat(),
                "structured_data": json.loads(doc.structured_data) if doc.structured_data else {}
            })
        
        db_session.close()
        return {"documents": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

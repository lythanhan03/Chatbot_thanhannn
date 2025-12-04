import os
import shutil
import logging
from typing import Dict, Any, List, TypedDict, Literal
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from passlib.context import CryptContext

# Import SQLAlchemy và các thư viện liên quan
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Enum
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy import desc
import enum
import datetime
# Import RAG Core
from core.embeding.HuggingEmbed import HuggingEmbed
from core.llm.gemini_llm import LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
)

# ----------------------------------------------------------------------
# KHỞI TẠO VÀ CẤU HÌNH CƠ BẢN
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("Gemini_api_key")

DOCUMENT_DIR = "data"
VECTOR_DB_PATH = "vectordb"
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# Khởi tạo context băm mật khẩu
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# ----------------------------------------------------------------------
# CẤU HÌNH DATABASE MYSQL (SQLAlchemy)
# ----------------------------------------------------------------------

# THAY THẾ CHUỖI NÀY BẰNG THÔNG TIN KẾT NỐI CỦA BẠN:
# Format: mysql+pymysql://USER:PASSWORD@HOST:PORT/DATABASE_NAME
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:admin@127.0.0.1:3306/chatbot_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependency cho Database Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ----------------------------------------------------------------------
# ĐỊNH NGHĨA MODELS CHO MYSQL (Tên cột tiếng Việt không dấu)
# ----------------------------------------------------------------------
class UserRole(enum.Enum):
    """Enum cho cột vaitro"""
    student = "student"
    teacher = "teacher"
    admin = "admin"
class DocumentBase(BaseModel):
    tieude: str
    mota: str
    loai_vanban: Literal["Thông báo", "Quy định"]
    phong_ban: str
    ngay_ban_hanh: datetime.date # Sử dụng date cho việc nhập liệu


class DocumentCreate(DocumentBase):
    # Đường dẫn file sẽ được xử lý khi upload thực tế,
    # nhưng ở đây ta đơn giản hóa để mô tả API CRUD
    duong_dan_file: str


class DocumentUpdate(DocumentBase):
    duong_dan_file: str # Có thể cho phép thay đổi file

class DBUser(Base):
    __tablename__ = "nguoidung"  # Tên bảng đã thống nhất

    id = Column(Integer, primary_key=True, index=True)
    hoten = Column(String(100), index=True, nullable=False)  # fullname
    email = Column(String(100), unique=True, index=True, nullable=False)
    vaitro = Column(Enum(UserRole), default=UserRole.student, nullable=False)  # role
    matkhau = Column(String(255), nullable=False)  # hashed_password
    daduyet = Column(Boolean, default=False, nullable=False)  # is_approved

class SenderRole(enum.Enum):
    """Enum cho cột nguoigui"""
    user = "user"
    bot = "bot"


class DBChatConversation(Base):
    __tablename__ = "cuoctrochuyen"

    id = Column(Integer, primary_key=True, index=True)
    nguoidung_id = Column(Integer, ForeignKey("nguoidung.id"), nullable=False)
    tieude = Column(String(255), nullable=False)
    thoigian_taobang = Column(DateTime, default=datetime.datetime.utcnow)

    # Định nghĩa mối quan hệ ngược lại với lichsuchat
    messages = relationship("DBChatHistory", back_populates="conversation")


class DBChatHistory(Base):
    __tablename__ = "lichsuchat"

    id = Column(Integer, primary_key=True, index=True)

    # KHÓA NGOẠI MỚI: Liên kết tới bảng cuoctrochuyen
    conversation_id = Column(Integer, ForeignKey("cuoctrochuyen.id"), nullable=False)

    nguoidung_id = Column(Integer, ForeignKey("nguoidung.id"), nullable=False)
    nguoigui = Column(Enum(SenderRole), nullable=False)
    noidung = Column(String(5000), nullable=False)
    thoigian = Column(DateTime, default=datetime.datetime.utcnow)

    # Mối quan hệ để SQLAlchemy quản lý
    conversation = relationship("DBChatConversation", back_populates="messages")
    user = relationship("DBUser")

class DBDocument(Base):
    __tablename__ = "vanban" # Tên bảng mới

    id = Column(Integer, primary_key=True, index=True)
    tieude = Column(String(255), nullable=False) # Tiêu đề văn bản
    mota = Column(String(1000), nullable=False) # Mô tả ngắn
    loai_vanban = Column(String(50), nullable=False) # Loại: 'Thông báo', 'Quy định'
    phong_ban = Column(String(100), nullable=False) # Phòng ban ban hành
    ngay_ban_hanh = Column(DateTime, default=datetime.datetime.utcnow) # Ngày ban hành
    # Giả định đường dẫn tệp (file_path)
    duong_dan_file = Column(String(255), nullable=False)
# TẠO BẢNG TRONG DATABASE (GỌI 1 LẦN DUY NHẤT SAU KHI ĐỊNH NGHĨA TẤT CẢ MODELS)
Base.metadata.create_all(bind=engine)
# ----------------------------------------------------------------------
# LOGIC TẠO ADMIN MẶC ĐỊNH (Chỉ chạy một lần)
# ----------------------------------------------------------------------
def create_initial_admin():
    db = SessionLocal()
    admin_email = "admin@tnut.edu.vn"
    admin_password_plain = "admin123"

    try:
        if db.query(DBUser).filter(DBUser.email == admin_email).first() is None:
            hashed_password = pwd_context.hash(admin_password_plain)
            admin_user = DBUser(
                hoten="Super Admin",
                email=admin_email,
                vaitro=UserRole.admin,
                matkhau=hashed_password,
                daduyet=True
            )
            db.add(admin_user)
            db.commit()
            logger.info("Tài khoản Admin mặc định đã được tạo trong MySQL.")
    except Exception as e:
        logger.error(f"Lỗi khi tạo tài khoản admin: {e}")
    finally:
        db.close()


# Gọi hàm tạo Admin khi ứng dụng khởi động
create_initial_admin()


# ----------------------------------------------------------------------
# CÁC CLASS VÀ HÀM RAG (GIỮ NGUYÊN)
# ----------------------------------------------------------------------

# --- Class RAG ---
class ProcessData:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_text(self, documents: List[Document]) -> List[Document]:
        if not documents: return []
        chunks = self.text_splitter.split_documents(documents)
        return chunks


vector_Hugging = HuggingEmbed()
processor = ProcessData(chunk_size=500, chunk_overlap=100)


class QuestionRequest(BaseModel):
    question: str
    user_id: int
    conversation_id: int = 0
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def load_documents_from_dir(directory: str) -> List[Document]:
    all_documents = []
    if not os.path.exists(directory): return all_documents
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path): continue
        loader = None
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.lower().endswith((".docx", ".doc")):
            loader = Docx2txtLoader(file_path)
        if loader:
            try:
                all_documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    return all_documents


def retrain_vector_store_full():
    logger.info("Starting FULL vector store retraining process...")
    all_documents = load_documents_from_dir(DOCUMENT_DIR)
    if not all_documents:
        logger.warning("No documents found to train the vector store. Skipping retraining.")
        vector_Hugging.vector_db = None
        if os.path.exists(VECTOR_DB_PATH): shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
        return
    chunks = processor.split_text(all_documents)
    if not chunks:
        logger.warning("No chunks created from documents. Skipping vector store creation.")
        vector_Hugging.vector_db = None
        if os.path.exists(VECTOR_DB_PATH): shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
        return
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
        except OSError as e:
            logger.error(f"Error deleting old vector store {e.filename}: {e.strerror}.")
    new_vector_store = vector_Hugging.create_vector_store(chunks)
    vector_Hugging.save_vector_store(VECTOR_DB_PATH)
    logger.info("New vector store created and saved successfully")
    return new_vector_store


try:
    vector_Hugging.load_vector_store(VECTOR_DB_PATH)
    logger.info("Vector store loaded successfully on startup.")
except (FileNotFoundError, RuntimeError) as e:
    if "not found" in str(e) or "could not open" in str(e):
        logger.info("Vector store index.faiss not found on startup. Checking for documents...")
        retrain_vector_store_full()
    else:
        raise
except Exception as e:
    logger.error(f"An unexpected error occurred during vector store loading: {e}")
    retrain_vector_store_full()


def retrivel(state: State) -> State:
    if vector_Hugging.vector_db is None: return {**state, "context": []}
    similarity_search = vector_Hugging.vector_db.similarity_search(query=state['question'], k=7)
    return {**state, "context": similarity_search}


def classify_intent(question: str) -> Dict[str, bool]:
    lower_q = question.lower()
    is_student_query = (
            "điểm" in lower_q or "sinh viên" in lower_q or "lớp" in lower_q or "thời khóa biểu" in lower_q or "mã số sinh viên" in lower_q or "học kỳ" in lower_q)
    is_admission_query = (
            "tuyển sinh" in lower_q or "ngành" in lower_q or "điểm chuẩn" in lower_q or "xét tuyển" in lower_q or "khối" in lower_q or "học phí" in lower_q)
    return {"is_student_query": is_student_query, "is_admission_query": is_admission_query}


def generate(state: State):
    llm = LLM(api_key=api_key)
    context_text = "\n".join([doc.page_content for doc in state['context']])
    question = state['question']

    html_instruction = (
        "QUAN TRỌNG: Câu trả lời phải được định dạng bằng **HTML hợp lệ** (sử dụng các thẻ <h3>, <table>, <b>, <br>, <p>) để hiển thị chuyên nghiệp trên giao diện web. "
        "Sử dụng thẻ <table> cho dữ liệu có cấu trúc (như danh sách, bảng). Tuyệt đối không sử dụng định dạng Markdown (như ##, *, -). "
        "Hãy **TỔNG HỢP** thông tin từ tất cả các đoạn trích liên quan để đưa ra câu trả lời đầy đủ nhất."
    )

    if not context_text:
        return {**state,
                "answer": "<p>Tôi xin lỗi, tôi không tìm thấy bất kỳ thông tin liên quan nào trong cơ sở dữ liệu của Nhà trường. Vui lòng thử câu hỏi khác.</p>"}

    intents = classify_intent(question)
    is_student_query = intents["is_student_query"]
    is_admission_query = intents["is_admission_query"]

    if is_student_query and not is_admission_query:
        prompt = (
            f"Bạn là trợ lý thông tin sinh viên của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên. {html_instruction}"
            f"Ưu tiên trả lời câu hỏi dựa trên các thông tin từ bảng nếu có. Dựa trên các thông tin sau, hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn. Nếu thông tin không liên quan hoặc không đủ để trả lời, hãy trả lời bằng một thẻ <p> rằng 'Tôi không tìm thấy thông tin phù hợp về sinh viên này'."
            f"\nNội dung được cung cấp:\n{context_text}\nCâu hỏi của sinh viên: {state['question']}")

    elif is_admission_query and not is_student_query:
        prompt = (
            f"Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên. {html_instruction}"
            f"Dựa trên nội dung sau, hãy trả lời câu hỏi của người dùng một cách đầy đủ và chính xác. Nếu không có thông tin phù hợp, hãy trả lời bằng một thẻ <p> rằng 'Tôi không tìm thấy thông tin phù hợp về tuyển sinh'."
            f"Sử dụng thẻ <table> cho thông tin học phí, điểm chuẩn hoặc các danh sách liên quan. "
            f"\nNội dung được cung cấp:\n{context_text}\nCâu hỏi về tuyển sinh: {state['question']}")

    else:
        prompt = (
            f"Bạn là Trợ lý Thông tin chính thức của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên (TNUT). {html_instruction}"
            f"Dựa vào **DUY NHẤT** các thông tin được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng. "
            f"Sử dụng các tiêu đề <h3> và thẻ <table> cho dữ liệu cấu trúc phức tạp. "
            f"\n\n--- BẮT ĐẦU NỘI DUNG CUNG CẤP ---\n{context_text}\n--- KẾT THÚC NỘI DUNG CUNG CUNG CẤP ---\n\n"
            f"Câu hỏi của người dùng: {state['question']}\n\n"
            f"QUY TẮC PHẢN HỒI:\n1. **Chỉ sử dụng** thông tin trong phần 'Nội dung được cung cấp' để trả lời.\n2. Nếu thông tin được cung cấp **không đủ** hoặc **không liên quan** để trả lời câu hỏi, hãy trả lời bằng một thẻ <p> rằng: 'Tôi xin lỗi, tôi không tìm thấy thông tin chính thức phù hợp trong cơ sở dữ liệu của Nhà trường để trả lời câu hỏi này.' Tuyệt đối **không được tự ý bịa đặt hoặc suy đoán**.")

    answer = llm.post_request(prompt)
    return {**state, "answer": answer}


# ----------------------------------------------------------------------
# Cấu Hình FastAPI Endpoints
# ----------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def get_index():
    if not os.path.exists("static/index.html"):
        return FileResponse("static/login.html")
    return FileResponse("static/index.html")


@app.get("/api/conversations/{user_id}", tags=["Chatbot"])
def get_conversations(user_id: int, db: Session = Depends(get_db)):
    """API lấy danh sách các cuộc trò chuyện của người dùng."""
    conversations = db.query(DBChatConversation).filter(
        DBChatConversation.nguoidung_id == user_id
    ).order_by(desc(DBChatConversation.thoigian_taobang)).all()

    return [
        {
            "id": conv.id,
            "tieude": conv.tieude,
            "thoigian": conv.thoigian_taobang.strftime("%Y-%m-%d %H:%M")
        }
        for conv in conversations
    ]

@app.get("/login")
@app.get("/login.html")
async def get_login_page():
    if not os.path.exists("static/login.html"):
        raise HTTPException(status_code=404, detail="Login page (login.html) not found in static folder.")
    return FileResponse("static/login.html")


@app.get("/register")
@app.get("/register.html")
async def get_register_page():
    if not os.path.exists("static/register.html"):
        raise HTTPException(status_code=404, detail="Registration page (register.html) not found in static folder.")
    return FileResponse("static/register.html")


@app.get("/user")
@app.get("/user.html")
async def get_user_page():
    if not os.path.exists("static/user.html"):
        raise HTTPException(status_code=404, detail="User page (user.html) not found in static folder.")
    return FileResponse("static/user.html")


@app.get("/admin")
@app.get("/admin.html")
async def get_admin_page():
    if not os.path.exists("static/admin.html"):
        raise HTTPException(status_code=404, detail="Admin page (admin.html) not found in static folder.")
    return FileResponse("static/admin.html")


@app.get("/contact")
@app.get("/contact.html")
async def get_contact_page():
    if not os.path.exists("static/contact.html"):
        raise HTTPException(status_code=404, detail="Contact page (contact.html) not found in static folder.")
    return FileResponse("static/contact.html")


@app.get("/khach")
@app.get("/khach.html")
async def get_contact_page():
    if not os.path.exists("static/khach.html"):
        raise HTTPException(status_code=404, detail="Contact page (khach.html) not found in static folder.")
    return FileResponse("static/khach.html")


@app.get("/quycai")
@app.get("/quycai.html")
async def get_quycai_page():
    if not os.path.exists("static/quycai.html"):
        raise HTTPException(status_code=404, detail="Regulation page (quycai.html) not found in static folder.")
    return FileResponse("static/quycai.html")

@app.get("/api/chathistory/{conversation_id}", tags=["Chatbot"])
def get_chat_history(conversation_id: int, db: Session = Depends(get_db)):
    """API lấy lịch sử chat cho một cuộc trò chuyện cụ thể."""
    messages = db.query(DBChatHistory).filter(
        DBChatHistory.conversation_id == conversation_id
    ).order_by(DBChatHistory.thoigian).all()

    return [
        {
            "id": msg.id,
            "sender": msg.nguoigui.value,
            "content": msg.noidung,
            "timestamp": msg.thoigian.strftime("%H:%M")
        }
        for msg in messages
    ]

# ----------------------------------------------------------------------
# ENDPOINTS QUẢN LÝ NGƯỜI DÙNG (Sử dụng MySQL)
# ----------------------------------------------------------------------

class UserRegister(BaseModel):
    """Cấu trúc dữ liệu nhận vào khi người dùng đăng ký."""
    fullname: str
    email: str
    role: Literal["student", "teacher", "admin"]
    password: str


@app.post("/api/register", tags=["User Management"])
async def register_user(user: UserRegister, db: Session = Depends(get_db)):
    """API xử lý việc đăng ký tài khoản mới. Tài khoản mới luôn cần phê duyệt."""

    # 1. Kiểm tra Email đã tồn tại
    if db.query(DBUser).filter(DBUser.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email đã được đăng ký.")

    # 2. Băm mật khẩu và tạo user mới
    hashed_password = pwd_context.hash(user.password)

    new_user = DBUser(
        hoten=user.fullname,
        email=user.email,
        vaitro=UserRole(user.role),
        matkhau=hashed_password,
        daduyet=False,
    )

    # 3. Thêm vào DB
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Đăng ký thành công. Tài khoản đang chờ quản trị viên phê duyệt."}


@app.post("/api/login", tags=["User Management"])
async def login(credentials: Dict[str, str], db: Session = Depends(get_db)):
    """API xử lý đăng nhập và xác thực người dùng."""
    email = credentials.get("email")
    password = credentials.get("password")

    # 1. Truy vấn DB
    user = db.query(DBUser).filter(DBUser.email == email).first()

    if not user:
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")

    if not pwd_context.verify(password, user.matkhau):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")

    if not user.daduyet:
        raise HTTPException(status_code=403, detail="Tài khoản chưa được quản trị viên phê duyệt.")

    # 2. Trả về thông tin
    return {"message": "Đăng nhập thành công!",
            "id": user.id,
            "role": user.vaitro.value,
            "fullname": user.hoten}


@app.get("/api/pending-users", tags=["Admin"])
async def get_pending_users(db: Session = Depends(get_db)):
    """API lấy danh sách các tài khoản đang chờ duyệt (Dành cho Admin)."""
    pending_users_db = db.query(DBUser).filter(DBUser.daduyet == False).all()

    # Chuyển đổi từ DBUser objects sang dictionary
    pending_users = [
        {"id": u.id, "fullname": u.hoten, "email": u.email, "role": u.vaitro.value}
        for u in pending_users_db
    ]
    return pending_users


@app.post("/api/approve-user/{user_id}", tags=["Admin"])
async def approve_user(user_id: int, db: Session = Depends(get_db)):
    """API phê duyệt tài khoản (Dành cho Admin)."""
    user = db.query(DBUser).filter(DBUser.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")

    if user.daduyet:
        raise HTTPException(status_code=400, detail="Tài khoản này đã được phê duyệt.")

    # Cập nhật trạng thái và lưu vào DB
    user.daduyet = True
    db.commit()
    logger.info(f"User ID {user_id} approved.")
    return {"message": f"Tài khoản ID {user_id} đã được phê duyệt thành công."}


@app.get("/api/users", tags=["Admin"])
async def get_approved_users(db: Session = Depends(get_db)):
    """API lấy danh sách TẤT CẢ các tài khoản đã được phê duyệt (Sửa lỗi 404)."""
    approved_users_db = db.query(DBUser).filter(DBUser.daduyet == True).all()

    # Chuyển đổi từ DBUser objects sang dictionary
    approved_users = [
        {"id": u.id, "fullname": u.hoten, "email": u.email, "role": u.vaitro.value}
        for u in approved_users_db
    ]
    return approved_users


# ----------------------------------------------------------------------
# ENDPOINTS CHATBOT VÀ ADMIN RAG (GIỮ NGUYÊN)
# ----------------------------------------------------------------------

@app.post("/ask", tags=["Chatbot"])
async def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    # 1. Xử lý Logic Cuộc Trò Chuyện
    conversation_id = request.conversation_id

    if conversation_id == 0:
        # TẠO CUỘC TRÒ CHUYỆN MỚI
        new_conversation = DBChatConversation(
            nguoidung_id=request.user_id,
            tieude=request.question[:250]  # Lấy câu hỏi đầu tiên làm tiêu đề
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id

    # 2. LƯU LỊCH SỬ TIN NHẮN USER (Sử dụng conversation_id mới/hiện tại)
    user_message = DBChatHistory(
        conversation_id=conversation_id,  # <--- Dùng ID mới/hiện tại
        nguoidung_id=request.user_id,
        nguoigui=SenderRole.user,
        noidung=request.question,
        thoigian=datetime.datetime.utcnow()
    )
    db.add(user_message)

    # ... (Giữ nguyên phần gọi retrivel và generate)
    state = {"question": request.question, "context": [], "answer": ""}
    state = retrivel(state)
    final_state = generate(state)
    answer_text = final_state.get("answer", "Xin lỗi, tôi không tìm thấy thông tin.")

    # 3. LƯU LỊCH SỬ TIN NHẮN BOT
    bot_message = DBChatHistory(
        conversation_id=conversation_id,  # <--- Dùng ID mới/hiện tại
        nguoidung_id=request.user_id,
        nguoigui=SenderRole.bot,
        noidung=answer_text,
        thoigian=datetime.datetime.utcnow()
    )
    db.add(bot_message)
    db.commit()
    db.refresh(bot_message)

    # TRẢ VỀ CẢ ANSWER VÀ CONVERSATION ID (để frontend cập nhật nếu đây là cuộc trò chuyện mới)
    return {"answer": answer_text, "conversation_id": conversation_id}

@app.post("/retrain", tags=["Admin"])
async def retrain_model_full():
    """Endpoint để tạo lại toàn bộ Vector Store."""
    try:
        retrain_vector_store_full()
        vector_Hugging.load_vector_store(VECTOR_DB_PATH)
        return {"message": "Model retrained successfully. Vector store has been updated."}
    except Exception as e:
        logger.error(f"Failed to retrain model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")


@app.post("/uploadfile/", tags=["Admin"])
async def create_upload_file(file: UploadFile = File(...)):
    """Endpoint upload file và tự động retrain."""
    file_location = os.path.join(DOCUMENT_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File '{file.filename}' uploaded successfully.")
        retrain_vector_store_full()
        vector_Hugging.load_vector_store(VECTOR_DB_PATH)

        return {"message": f"File '{file.filename}' uploaded and system updated successfully."}
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")

#----QUAN LY TAI LIEU-----------

# API Tìm kiếm (Cho mọi người: Student, Teacher, Admin)
@app.get("/api/documents", tags=["Document Management"])
def get_documents(
        search: str = None,
        doc_type: str = None,
        department: str = None,
        db: Session = Depends(get_db)
):
    """API tìm kiếm và lọc tài liệu."""
    query = db.query(DBDocument)

    if search:
        query = query.filter(DBDocument.tieude.contains(search) | DBDocument.mota.contains(search))

    if doc_type and doc_type != "Tất cả":
        query = query.filter(DBDocument.loai_vanban == doc_type)

    if department and department != "Tất cả":
        query = query.filter(DBDocument.phong_ban == department)

    # Sắp xếp theo ngày ban hành mới nhất
    documents = query.order_by(desc(DBDocument.ngay_ban_hanh)).all()

    return [
        {
            "id": doc.id,
            "tieude": doc.tieude,
            "mota": doc.mota,
            "loai_vanban": doc.loai_vanban,
            "phong_ban": doc.phong_ban,
            "ngay_ban_hanh": doc.ngay_ban_hanh.strftime("%d/%m/%Y"),
            "duong_dan_file": doc.duong_dan_file
        }
        for doc in documents
    ]


# API Thêm mới (Chỉ Admin/Teacher)
@app.post("/api/documents", tags=["Document Management"])
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    # LƯU Ý: Cần thêm logic xác thực vai trò (role) ở đây (Admin/Teacher)

    new_doc = DBDocument(
        tieude=doc.tieude,
        mota=doc.mota,
        loai_vanban=doc.loai_vanban,
        phong_ban=doc.phong_ban,
        ngay_ban_hanh=doc.ngay_ban_hanh,
        duong_dan_file=doc.duong_dan_file  # Giả định file đã upload
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return {"message": "Tài liệu đã được thêm thành công.", "id": new_doc.id}


# API Cập nhật (Chỉ Admin/Teacher)
@app.put("/api/documents/{doc_id}", tags=["Document Management"])
def update_document(doc_id: int, doc: DocumentUpdate, db: Session = Depends(get_db)):
    # LƯU Ý: Cần thêm logic xác thực vai trò (role) ở đây (Admin/Teacher)

    db_doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not db_doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")

    # Cập nhật các trường
    db_doc.tieude = doc.tieude
    db_doc.mota = doc.mota
    db_doc.loai_vanban = doc.loai_vanban
    db_doc.phong_ban = doc.phong_ban
    db_doc.ngay_ban_hanh = doc.ngay_ban_hanh
    db_doc.duong_dan_file = doc.duong_dan_file

    db.commit()
    return {"message": f"Tài liệu ID {doc_id} đã được cập nhật."}


# API Xóa (Chỉ Admin/Teacher)
@app.delete("/api/documents/{doc_id}", tags=["Document Management"])
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    # LƯU Ý: Cần thêm logic xác thực vai trò (role) ở đây (Admin/Teacher)

    db_doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
    if not db_doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")

    db.delete(db_doc)
    db.commit()
    return {"message": f"Tài liệu ID {doc_id} đã được xóa thành công."}
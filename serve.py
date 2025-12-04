import os
import shutil
import logging
from typing import Dict, Any, List, TypedDict, Literal
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from passlib.context import CryptContext
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Đảm bảo 2 file này tồn tại trong thư mục 'core/'
# Cần có các file: core/embeding/HuggingEmbed.py và core/llm/gemini_llm.py
from core.embeding.HuggingEmbed import HuggingEmbed
from core.llm.gemini_llm import LLM

# Import Loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
)

# ----------------------------------------------------------------------
# KHỞI TẠO VÀ CẤU HÌNH CƠ BẢN
# ----------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
# Đảm bảo có file .env với biến Gemini_api_key
api_key = os.getenv("Gemini_api_key")

DOCUMENT_DIR = "data"
VECTOR_DB_PATH = "vectordb"
os.makedirs(DOCUMENT_DIR, exist_ok=True)

# Khởi tạo context băm mật khẩu (Bắt buộc cho bảo mật)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# --- Khởi tạo Admin CỐ ĐỊNH ---
ADMIN_EMAIL = "admin@tnut.edu.vn"
ADMIN_PASSWORD_PLAIN = "admin123"

ADMIN_HASHED_PASSWORD = pwd_context.hash(ADMIN_PASSWORD_PLAIN)

# Giả lập Database (Lưu trữ người dùng trong bộ nhớ)
fake_db: List[Dict[str, Any]] = [
    {
        "id": 1,
        "fullname": "Super Admin",
        "email": ADMIN_EMAIL,
        "role": "admin",
        "hashed_password": ADMIN_HASHED_PASSWORD,
        "is_approved": True,  # Luôn được phê duyệt
    }
]


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


# ==================== ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU (Pydantic Models) ====================

class UserRegister(BaseModel):
    """Cấu trúc dữ liệu nhận vào khi người dùng đăng ký."""
    fullname: str
    email: str
    role: Literal["student", "teacher", "admin"]
    password: str


class QuestionRequest(BaseModel):
    question: str


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# ==================== Logic RAG ====================

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
        # Bổ sung các loại file khác nếu cần:
        # elif filename.lower().endswith(".csv"):
        #     loader = CSVLoader(file_path)
        # elif filename.lower().endswith((".xls", ".xlsx")):
        #     loader = UnstructuredExcelLoader(file_path)

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


# Khởi tạo vector store khi ứng dụng khởi động
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
    """
    Hàm truy xuất dữ liệu.
    k=7 được giữ lại để đảm bảo lấy đủ Context cho LLM, vì các đoạn đã được tối ưu hóa.
    """
    if vector_Hugging.vector_db is None: return {**state, "context": []}
    # Tăng k lên 7-10 để lấy được nhiều context hơn theo yêu cầu
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

    # --- CHỈ THỊ HTML MỚI ĐÃ TỐI ƯU ---
    html_instruction = (
        "QUAN TRỌNG: Câu trả lời phải được định dạng bằng **HTML hợp lệ** (sử dụng các thẻ <h3>, <table>, <b>, <br>, <p>) để hiển thị chuyên nghiệp trên giao diện web. "
        "Sử dụng thẻ <table> cho dữ liệu có cấu trúc (như danh sách, bảng). Tuyệt đối không sử dụng định dạng Markdown (như ##, *, -). "
        "Hãy **TỔNG HỢP** thông tin từ tất cả các đoạn trích liên quan để đưa ra câu trả lời đầy đủ nhất."
    )
    # --- KẾT THÚC CHỈ THỊ HTML MỚI ---

    if not context_text:
        # Trả lời lỗi bằng thẻ HTML <p>
        return {**state,
                "answer": "<p>Tôi xin lỗi, tôi không tìm thấy bất kỳ thông tin liên quan nào trong cơ sở dữ liệu của Nhà trường. Vui lòng thử câu hỏi khác.</p>"}

    intents = classify_intent(question)
    is_student_query = intents["is_student_query"]
    is_admission_query = intents["is_admission_query"]

    # Prompt generation logic
    if is_student_query and not is_admission_query:
        # Sinh viên
        prompt = (
            f"Bạn là trợ lý thông tin sinh viên của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên. {html_instruction}"
            f"Ưu tiên trả lời câu hỏi dựa trên các thông tin từ bảng nếu có. Dựa trên các thông tin sau, hãy trả lời câu hỏi của người dùng một cách chính xác và ngắn gọn. Nếu thông tin không liên quan hoặc không đủ để trả lời, hãy trả lời bằng một thẻ <p> rằng 'Tôi không tìm thấy thông tin phù hợp về sinh viên này'."
            f"\nNội dung được cung cấp:\n{context_text}\nCâu hỏi của sinh viên: {state['question']}")

    elif is_admission_query and not is_student_query:
        # Tuyển sinh
        prompt = (
            f"Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên. {html_instruction}"
            f"Dựa trên nội dung sau, hãy trả lời câu hỏi của người dùng một cách đầy đủ và chính xác. Nếu không có thông tin phù hợp, hãy trả lời bằng một thẻ <p> rằng 'Tôi không tìm thấy thông tin phù hợp về tuyển sinh'."
            f"Sử dụng thẻ <table> cho thông tin học phí, điểm chuẩn hoặc các danh sách liên quan. "
            f"\nNội dung được cung cấp:\n{context_text}\nCâu hỏi về tuyển sinh: {state['question']}")

    else:
        # Prompt tổng quát/mặc định
        prompt = (
            f"Bạn là Trợ lý Thông tin chính thức của Trường Đại học Kỹ thuật Công nghiệp, Thái Nguyên (TNUT). {html_instruction}"
            f"Dựa vào **DUY NHẤT** các thông tin được cung cấp dưới đây, hãy trả lời câu hỏi của người dùng. "
            f"Sử dụng các tiêu đề <h3> và thẻ <table> cho dữ liệu cấu trúc phức tạp. "
            f"\n\n--- BẮT ĐẦU NỘI DUNG CUNG CẤP ---\n{context_text}\n--- KẾT THÚC NỘI DUNG CUNG CUNG CẤP ---\n\n"
            f"Câu hỏi của người dùng: {state['question']}\n\n"
            f"QUY TẮC PHẢN HỒI:\n1. **Chỉ sử dụng** thông tin trong phần 'Nội dung được cung cấp' để trả lời.\n2. Nếu thông tin được cung cấp **không đủ** hoặc **không liên quan** để trả lời câu hỏi, hãy trả lời bằng một thẻ <p> rằng: 'Tôi xin lỗi, tôi không tìm thấy thông tin chính thức phù hợp trong cơ sở dữ liệu của Nhà trường để trả lời câu hỏi này.' Tuyệt đối **không được tự ý bịa đặt hoặc suy đoán**.")

    answer = llm.post_request(prompt)
    return {**state, "answer": answer}


# ==================== Cấu Hình FastAPI Endpoints ====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Đảm bảo thư mục static/ tồn tại và chứa các file HTML
# Bạn cần tạo một thư mục tên là 'static' và đặt các file HTML vào đó
app.mount("/static", StaticFiles(directory="static"), name="static")


# ENDPOINT GỐC ĐÃ CẬP NHẬT: Ưu tiên index.html
@app.get("/", include_in_schema=False)
async def get_index():
    if not os.path.exists("static/index.html"):
        # Nếu trang chủ mới không có, chuyển hướng người dùng đến trang login
        return FileResponse("static/login.html")
    return FileResponse("static/index.html")


# ==================== CÁC ENDPOINT FRONTEND ĐÃ CẬP NHẬT ====================

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

# NEW ENDPOINT: contact.html
@app.get("/contact")
@app.get("/contact.html")
async def get_contact_page():
    if not os.path.exists("static/contact.html"):
        raise HTTPException(status_code=404, detail="Contact page (contact.html) not found in static folder.")
    return FileResponse("static/contact.html")

#khach
@app.get("/khach")
@app.get("/khach.html")
async def get_contact_page():
    if not os.path.exists("static/khach.html"):
        raise HTTPException(status_code=404, detail="Contact page (khach.html) not found in static folder.")
    return FileResponse("static/khach.html")

# NEW ENDPOINT: quycai.html
@app.get("/quycai")
@app.get("/quycai.html")
async def get_quycai_page():
    if not os.path.exists("static/quycai.html"):
        raise HTTPException(status_code=404, detail="Regulation page (quycai.html) not found in static folder.")
    return FileResponse("static/quycai.html")


# ==================== ENDPOINTS QUẢN LÝ NGƯỜI DÙNG (Đã sửa lỗi) ====================

@app.post("/api/register", tags=["User Management"])
async def register_user(user: UserRegister):
    """API xử lý việc đăng ký tài khoản mới. Tài khoản mới luôn cần phê duyệt."""

    if user.email.lower() == ADMIN_EMAIL:
        raise HTTPException(status_code=400,
                            detail="Email này đã được sử dụng cho tài khoản quản trị cố định. Vui lòng sử dụng email khác.")

    if any(u['email'] == user.email for u in fake_db):
        raise HTTPException(status_code=400, detail="Email đã được đăng ký.")

    hashed_password = pwd_context.hash(user.password)

    new_id = max([u['id'] for u in fake_db]) + 1 if fake_db else 1

    new_user = {
        "id": new_id,
        "fullname": user.fullname,
        "email": user.email,
        "role": user.role,
        "hashed_password": hashed_password,
        "is_approved": False,
    }
    fake_db.append(new_user)

    return {"message": "Đăng ký thành công. Tài khoản đang chờ quản trị viên phê duyệt."}


@app.post("/api/login", tags=["User Management"])
async def login(credentials: Dict[str, str]):
    """API xử lý đăng nhập và xác thực người dùng."""
    email = credentials.get("email")
    password = credentials.get("password")

    user = next((u for u in fake_db if u['email'] == email), None)

    if not user:
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")

    if not pwd_context.verify(password, user['hashed_password']):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")

    if not user['is_approved']:
        raise HTTPException(status_code=403, detail="Tài khoản chưa được quản trị viên phê duyệt.")

    # SỬA LỖI QUAN TRỌNG: Đã đổi 'user_id' thành 'id' để khớp với frontend
    return {"message": "Đăng nhập thành công!",
            **{"id": user['id'], "role": user['role'], "fullname": user['fullname']}}


@app.get("/api/pending-users", tags=["Admin"])
async def get_pending_users():
    """API lấy danh sách các tài khoản đang chờ duyệt (Dành cho Admin)."""
    pending_users = [
        {"id": u['id'], "fullname": u['fullname'], "email": u['email'], "role": u['role']}
        for u in fake_db if u['is_approved'] == False
    ]
    return pending_users


@app.post("/api/approve-user/{user_id}", tags=["Admin"])
async def approve_user(user_id: int):
    """API phê duyệt tài khoản (Dành cho Admin)."""
    for user in fake_db:
        if user['id'] == user_id:
            if user['is_approved']:
                raise HTTPException(status_code=400, detail="Tài khoản này đã được phê duyệt.")

            user['is_approved'] = True
            logger.info(f"User ID {user_id} approved.")
            return {"message": f"Tài khoản ID {user_id} đã được phê duyệt thành công."}

    raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")


# ==================== ENDPOINTS CHATBOT VÀ ADMIN RAG ====================

@app.post("/ask", tags=["Chatbot"])
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    state = {"question": request.question, "context": [], "answer": ""}
    state = retrivel(state)
    final_state = generate(state)
    return final_state


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


@app.get("/api/users", tags=["Admin"])
async def get_approved_users():
    """API lấy danh sách TẤT CẢ các tài khoản đã được phê duyệt (Sửa lỗi 404)."""
    approved_users = [
        {"id": u['id'], "fullname": u['fullname'], "email": u['email'], "role": u['role']}
        for u in fake_db if u['is_approved'] == True
    ]
    return approved_users
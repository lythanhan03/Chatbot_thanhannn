from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time

# Khởi tạo App
app = FastAPI()

# Cấu hình CORS (quan trọng để frontend kết nối)
# CHỈ NÊN DÙNG origins=["*"] CHO MÔI TRƯỜNG DEV
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model cho dữ liệu người dùng
class User(BaseModel):
    id: int
    email: str
    fullname: str
    role: str  # 'student', 'teacher', 'admin', 'co-admin'
    is_approved: bool = False
    password: Optional[str] = "hashed_mock_password"  # Chỉ để mock


# Dữ liệu mô phỏng: Danh sách người dùng
# - ID 1, 2, 5: Đã duyệt
# - ID 3, 4: Chờ duyệt (is_approved=False)
mock_users: List[User] = [
    User(id=1, email="a.nv@e.tlu.edu.vn", fullname="Nguyễn Văn A", role="student", is_approved=True),
    User(id=2, email="b.pv@e.tlu.edu.vn", fullname="Phạm Văn B", role="teacher", is_approved=True),
    User(id=3, email="c.ht@e.tlu.edu.vn", fullname="Hoàng Thị C", role="student", is_approved=False),
    User(id=4, email="d.pt@e.tlu.edu.vn", fullname="Phan Thanh D", role="co-admin", is_approved=False),
    User(id=5, email="admin@tnut.edu.vn", fullname="Quản Trị Viên", role="admin", is_approved=True),
]


# -----------------
# API USERS (QUẢN LÝ TÀI KHOẢN)
# -----------------

# Đảm bảo bạn đã import các thư viện cần thiết:
# from typing import List
# from fastapi import FastAPI
# ... (các cấu hình CORS và mock_users)

@app.get("/api/users", response_model=List[User])
def get_approved_users():
    """
    API để lấy danh sách TẤT CẢ các tài khoản đã được phê duyệt.
    """
    # Thay thế mock_users bằng logic truy vấn CSDL của bạn nếu đã có
    return [user for user in mock_users if user.is_approved]


@app.get("/api/pending-users", response_model=List[User])
def get_pending_users():
    """API 2: Lấy danh sách các tài khoản đang chờ phê duyệt."""
    return [user for user in mock_users if not user.is_approved]


@app.post("/api/approve-user/{user_id}")
def approve_user(user_id: int):
    """API 3: Phê duyệt một tài khoản và trả về user đã được cập nhật."""
    for user in mock_users:
        if user.id == user_id:
            if user.is_approved:
                raise HTTPException(status_code=400, detail="Tài khoản đã được phê duyệt.")

            # Giả lập quá trình xử lý mất 1 giây
            time.sleep(1)

            user.is_approved = True
            # **QUAN TRỌNG:** Trả về thông tin người dùng vừa được duyệt để frontend cập nhật
            return {"message": f"Đã phê duyệt tài khoản ID {user_id}", "user": user.dict()}
    raise HTTPException(status_code=404, detail="Không tìm thấy tài khoản.")


# -----------------
# API KHÁC (CẦN CHO ADMIN.HTML VÀ CÁC TRANG KHÁC)
# -----------------

@app.post("/api/login")
def login(login_data: dict):
    """API giả lập đăng nhập."""
    email = login_data.get("email")
    password = login_data.get("password")

    # Giả lập logic kiểm tra đăng nhập đơn giản
    user = next((u for u in mock_users if u.email == email and u.is_approved), None)

    if user:
        return {"message": "Đăng nhập thành công", "role": user.role, "user_id": user.id}

    raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng, hoặc tài khoản chưa được phê duyệt.")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """API giả lập xử lý upload file RAG."""
    # Giả lập việc xử lý file
    time.sleep(2)
    return {
        "message": f"Tệp '{file.filename}' đã được tải lên và xử lý thành công. {file.size} bytes đã được trích xuất vào cơ sở dữ liệu tri thức."}
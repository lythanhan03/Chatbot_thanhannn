#streamlit run app.py
import streamlit as st
import requests
import os
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_option_menu import option_menu
import bcrypt
import pandas as pd
import re

BASE_URL = "http://localhost:8000"

# --- TẠO DANH SÁCH NGƯỜI DÙNG VÀ CẤU HÌNH ĐĂNG NHẬP ---
# Mật khẩu đơn giản cho admin và sinh viên
plain_passwords = ['admin', 'sv001']
hashed_passwords = [
    bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    for password in plain_passwords
]

config = {
    'cookie': {
        'expiry_days': 30,
        'key': 'some_secret_key',
        'name': 'some_cookie_name'
    },
    'credentials': {
        'usernames': {
            'admin': {
                'email': 'admin@example.com',
                'name': 'Quản trị viên',
                'password': hashed_passwords[0],
                'role': 'admin'
            },
            'sinhvien': {
                'email': 'student@example.com',
                'name': 'Sinh viên',
                'password': hashed_passwords[1],
                'role': 'sinhvien'
            }
        }
    }
}

with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

# --- Tải cấu hình và khởi tạo Authenticator ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# --- Xử lý đăng nhập ---
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None
if 'username' not in st.session_state:
    st.session_state.username = None

if st.session_state.authentication_status is False or st.session_state.authentication_status is None:
    st.warning('Vui lòng nhập tên người dùng và mật khẩu của bạn')
    col1, col2 = st.columns([1, 1])
    with col1:
        username_input = st.text_input("Tên đăng nhập", key="username_input")
    with col2:
        password_input = st.text_input("Mật khẩu", type="password", key="password_input")
    if st.button('Đăng nhập'):
        if username_input == 'admin' and password_input == 'admin':
            st.session_state.authentication_status = True
            st.session_state.username = 'admin'
            st.session_state.name = 'Quản trị viên'
            st.session_state['role'] = 'admin'
            st.rerun()
        elif username_input == 'sinhvien' and password_input == 'sv001':
            st.session_state.authentication_status = True
            st.session_state.username = 'sinhvien'
            st.session_state.name = 'Sinh viên'
            st.session_state['role'] = 'sinhvien'
            st.rerun()
        else:
            st.session_state.authentication_status = False
            st.rerun()

if st.session_state.authentication_status:
    # --- Header của ứng dụng ---
    st.markdown(
        """
        <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            text-align: center;
        }
        .header-logo {
            width: 80px;
            height: 80px;
            margin-right: 20px;
        }
        .header-title-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header-title {
            color: #004d99;
            font-size: 1.5em;
            margin: 0;
            font-weight: bold;
        }
        .header-subtitle {
            color: #666;
            font-size: 1em;
            margin-top: 5px;
        }
        </style>
        <div class="header-container">
            <img src="logo.jpg"
                 alt="Logo Đại học Kỹ thuật Công nghiệp"
                 class="header-logo">
            <div class="header-title-container">
                <h1 class="header-title">TRƯỜNG ĐẠI HỌC KỸ THUẬT CÔNG NGHIỆP</h1>
                <h2 class="header-subtitle">Tư vấn tuyển sinh & Hỗ trợ sinh viên</h2>
            </div>
        </div>
        <hr style="border: 1px solid #ddd;">
        """,
        unsafe_allow_html=True
    )

    # --- Trang chính sau khi đăng nhập thành công ---
    st.sidebar.title(f'Chào mừng, {st.session_state.name}!')
    if st.sidebar.button("Đăng xuất"):
        st.session_state.authentication_status = False
        st.session_state.username = None
        st.rerun()

    if 'role' in st.session_state and st.session_state['role'] == 'admin':
        st.sidebar.subheader("Quản lý dữ liệu")
        uploaded_file = st.sidebar.file_uploader(
            "Tải file PDF, DOCX, TXT, CSV, XLSX",
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=False
        )

        if uploaded_file is not None:
            with st.spinner(f"Đang tải lên '{uploaded_file.name}' và cập nhật hệ thống..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BASE_URL}/uploadfile/", files=files)
                    if response.status_code == 200:
                        st.sidebar.success(f"File '{uploaded_file.name}' tải lên và hệ thống được cập nhật thành công!")
                    else:
                        st.sidebar.error(f"Lỗi khi tải lên file: {response.json().get('detail', 'Lỗi không xác định.')}")
                except Exception as e:
                    st.sidebar.error(f"Lỗi kết nối: {e}")

        st.sidebar.subheader("Cập nhật lại hệ thống")
        if st.sidebar.button("Train lại toàn bộ dữ liệu"):
            with st.spinner('Đang train lại toàn bộ dữ liệu...'):
                try:
                    response = requests.post(f"{BASE_URL}/retrain")
                    if response.status_code == 200:
                        st.sidebar.success("Train dữ liệu thành công! Hệ thống đã được cập nhật.")
                    else:
                        st.sidebar.error(f"Lỗi khi train: {response.json().get('detail', 'Lỗi không xác định.')}")
                except Exception as e:
                    st.sidebar.error(f"Lỗi kết nối: {e}")

    # --- Phần chính của ứng dụng: Giao diện Chatbot ---
    st.title("Hệ thống hỗ trợ thông tin")
    st.sidebar.markdown("---")
    st.sidebar.info("Hệ thống được phát triển bởi .")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message_id, (role, message) in enumerate(st.session_state.chat_history):
        with st.chat_message(role):
            st.markdown(message)

    if user_input := st.chat_input("Hỏi bất kỳ điều gì về Viện chúng tôi..."):
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        student_id_match = re.search(r'([A-Z]{2}\d{11})', user_input)
        if not student_id_match:
            student_id_match = re.search(r'([A-Z]{2}\d{10})', user_input)

        if 'role' in st.session_state and st.session_state['role'] == 'sinhvien' and student_id_match:
            student_id = student_id_match.group(1)
            try:
                df_diem = pd.read_csv('250701 Sổ tay Điện tử.xlsx - Dữ liệu điểm.csv')
                df_ctdt = pd.read_csv('250701 Sổ tay Điện tử.xlsx - Mô tả CTĐT.csv')

                student_courses = df_diem[df_diem['Mã sinh viên'].str.strip() == student_id.strip()]

                if student_courses.empty:
                    st.session_state.chat_history.append(("assistant", "Không tìm thấy thông tin của sinh viên này. Vui lòng kiểm tra lại mã sinh viên."))
                    st.rerun()

                merged_data = pd.merge(student_courses, df_ctdt, left_on='Mã MH', right_on='Mã HP', how='left')
                total_credits_earned = merged_data[merged_data['Điểm'] >= 4.0]['Số TC'].sum()
                total_credits_required = 141
                credits_remaining = total_credits_required - total_credits_earned

                all_courses = df_ctdt[['Mã HP', 'Tên HP', 'Số TC']]
                taken_courses = set(student_courses['Mã MH'])
                untaken_courses = all_courses[~all_courses['Mã HP'].isin(taken_courses)]
                suggestions = untaken_courses.sort_values(by='Số TC', ascending=False)

                result = f"""
                **Tổng số tín chỉ đã học:** {total_credits_earned} / {total_credits_required}
                **Số tín chỉ còn thiếu:** {credits_remaining}

                **Gợi ý cho kỳ học tiếp theo:**
                Dưới đây là một số môn bạn nên cân nhắc học:
                """
                for index, row in suggestions.head(5).iterrows():
                    result += f"\n- **{row['Tên HP']}** ({row['Mã HP']}): {row['Số TC']} tín chỉ"

                result += f"""
                \n**Lời khuyên:**
                - Hãy tập trung vào các môn học còn thiếu để hoàn thành chương trình đào tạo.
                - Ưu tiên các môn tiên quyết (prerequisite courses) để có thể học các môn chuyên ngành sau này.
                - Thường xuyên trao đổi với cố vấn học tập để có lộ trình phù hợp.
                - Đảm bảo điểm số các môn học đạt yêu cầu để được tính tín chỉ.
                """
                st.session_state.chat_history.append(("assistant", result))
                st.rerun()

            except FileNotFoundError:
                st.session_state.chat_history.append(("assistant", "Lỗi: Không tìm thấy file dữ liệu. Vui lòng đảm bảo các file CSV đã được đặt đúng chỗ."))
                st.rerun()
            except Exception as e:
                st.session_state.chat_history.append(("assistant", f"Đã xảy ra lỗi: {e}"))
                st.rerun()

        else:
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    chat_history_for_api = [
                        {"role": role, "content": message}
                        for role, message in st.session_state.chat_history
                    ]
                    response = requests.post(f"{BASE_URL}/ask", json={"question": user_input, "chat_history": chat_history_for_api})
                    if response.status_code == 200:
                        answer = response.json().get("answer", "Xin lỗi, tôi không thể tìm thấy câu trả lời cho câu hỏi này.")
                    else:
                        answer = f"Lỗi: Không thể lấy phản hồi từ máy chủ. (Mã lỗi: {response.status_code})"
                except requests.exceptions.ConnectionError:
                    answer = "Lỗi kết nối: Vui lòng đảm bảo máy chủ backend đang chạy."
                except Exception as e:
                    answer = f"Đã xảy ra lỗi không mong muốn: {e}"

            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.rerun()
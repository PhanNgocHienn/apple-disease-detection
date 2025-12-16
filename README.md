HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG - ỨNG DỤNG BÁC SĨ TÁO AI

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy ứng dụng trên máy tính cá nhân.

1. Yêu cầu Hệ thống

Hệ điều hành: Windows 10/11, macOS, hoặc Linux.

Python: Phiên bản 3.8 đến 3.10 (Khuyên dùng 3.9).

Phần cứng:

RAM: Tối thiểu 4GB (Khuyên dùng 8GB).

GPU: Không bắt buộc (Code sẽ tự động chạy bằng CPU nếu không có GPU NVIDIA).

2. Chuẩn bị Thư mục

Đảm bảo bạn đã nhận đủ các file và thư mục sau, đặt chúng cùng một nơi:

Apple_Disease_App/
├── app_mobile_compare.py              
├── requirements.txt      
└── models/               
    ├── resnet50_best.pth
    ├── mobilenet_v2_best.pth
    └── efficientnet_b0_best.pth


3. Thiết lập Môi trường (Chọn 1 trong 2 cách)

Cách 1: Sử dụng Anaconda (Khuyên dùng)

Tải và cài đặt Anaconda/Miniconda.

Mở Anaconda Prompt (Windows) hoặc Terminal.

Tạo môi trường ảo:

conda create -n leaf-disease python=3.9


Kích hoạt môi trường:

conda activate leaf-disease


Cách 2: Sử dụng Python venv (Mặc định)

Mở Command Prompt (CMD) hoặc Terminal tại thư mục dự án.

Tạo môi trường ảo:

python -m venv venv


Kích hoạt môi trường:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

4. Cài đặt Thư viện

Sau khi kích hoạt môi trường, chạy lệnh sau để cài đặt các gói cần thiết:

pip install -r requirements.txt


Lưu ý: Nếu máy có GPU NVIDIA và muốn chạy nhanh hơn, hãy cài thêm PyTorch bản hỗ trợ CUDA từ pytorch.org. Nếu không, lệnh trên sẽ cài bản CPU mặc định (vẫn chạy tốt).

5. Chạy Ứng dụng

Tại cửa sổ dòng lệnh (đang kích hoạt môi trường ảo), gõ lệnh sau:

streamlit run app_mobile_compare.py


Trình duyệt web sẽ tự động mở ra. Nếu không, hãy truy cập địa chỉ: http://localhost:8501.

Để dừng ứng dụng: Quay lại cửa sổ dòng lệnh và nhấn Ctrl + C.

6. Khắc phục lỗi thường gặp

Lỗi "File not found" hoặc "Models not found":

Đảm bảo bạn đang đứng đúng thư mục chứa file app.py khi gõ lệnh.

Kiểm tra xem thư mục models có nằm cùng cấp với file app.py không.

Lỗi "Command not found: streamlit":

Có thể bạn chưa kích hoạt môi trường ảo. Hãy chạy lại bước kích hoạt (activate).

Hoặc thử chạy: python -m streamlit run app.py.

Lỗi Camera không hoạt động trên điện thoại:

Nếu bạn truy cập ứng dụng qua mạng LAN (ví dụ: http://192.168.1.5:8501) trên điện thoại, trình duyệt có thể chặn Camera vì không có bảo mật HTTPS.

Giải pháp: Chạy trên máy tính (localhost) hoặc triển khai lên Streamlit Cloud/Ngrok để có HTTPS.

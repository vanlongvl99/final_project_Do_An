# Đồ Án Môn Học
## Đề tài: Face Recognition
### Thành viên
- Nguyễn Văn Long Mssv 1712024
- Nguyễn Thị Bích Lan Mssv 1711884
### Môi trường:
- Ubuntu 18.04
- Python 3.6.9
- Keras 2.3.1
- Tensorflow 2.0.0-alpha0
- Sklearn 0.22.2
- mtcnn 0.1.0

### Chạy chương trình:
- Lưu ý các bạn phải cài đúng các phiên bản của các thư viện trên. Bạn có thể cài đặt riêng từng thư viện hoặc có thể chạy file `requirements.txt`: `pip3 install -r requirements.txt`.
- Clone thư mục này về, `cd` vô forder `final_report_TTNT` tương tự như sau: `~/vanlong/ky6/TriTNT/finish_model/final_report_TTNT$`. Ta thực hiện tất cả các lệnh tại thư mục này.
- Vì file model nặng nên không thể upload lên github, các bạn tải file mode [tại đây](https://drive.google.com/drive/folders/1zG6k__Ndp0vGWWpyOfYIn0CTjc81GvYw?usp=sharing). Sau khi download các file model về, ta lưu các forder chứa file model tại thư mục `final_report_TTNT` . Lưu ý không chỉnh sửa tên các file model, nếu không sẽ bị 1 số lỗi gọi tên file khi chạy chương trình.
- Sau khi cài đặt xong các thư viện và tải file model về, ta chạy lệnh sau để demo kết quả:
`python3 final_source/gui_demo.py`.
### Phần giao diện của model chính sẽ có 4 chức năng:
#### **Take new face**
- Nếu bạn muốn chương trình có thể nhận diện thêm người mới (chưa có trong dataset). Ghi tên người đó vào ô `Name of new person` và nhấn nút `Take new face`.
#### **Training**
- Sau khi đã thêm người mới vào dataset thì phải train lại model
#### **Testing**
- Phần này để demo nhận diện khuôn mặt với file pre-train có sẵn gồm 28 người khác nhau.
#### **New Testing** 
- Sau khi đã thêm người mới và train lại, thì phần này sẽ demo chương trình, và dataset đã thêm vô.
### 2 phương pháp nhận diện khác: 
- Nhận diện bằng CNN không dùng model pre-train
- Nhận diện dùng transfer learning, với pre-train là resnet50
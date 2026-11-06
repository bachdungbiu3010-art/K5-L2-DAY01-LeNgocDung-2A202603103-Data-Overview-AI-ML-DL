# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \*\*Cách dùng mẫu này:\*\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \*\*upload\*\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\_lab\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \*\*link repo\*\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\_lab\_outputs/classification\_predictions.json` và `visuals/classification\_top5.png`.  
Chỉ dùng ảnh `sample\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\_id`=… ; `class\_name`=… ; `rank`=1 ; `score`=… ; `taxonomy\_name`=…
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
4. Vì sao cần ghi cả `class\_id`, `class\_name` và `taxonomy\_name`, không chỉ ghi mỗi tên lớp?
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
6. Vì sao không được lấy `score` của model làm ground truth?





1\. Phân loại ảnh



Câu 1. Record hạng 1 của ảnh traffic



class\_id: 468



class\_name: cab



rank: 1



score: 0.510915



taxonomy\_name: ImageNet-1K



Câu 2. Record này nói về cả ảnh hay một vật thể riêng?



Record này dự đoán nhãn cho toàn bộ bức ảnh, không xác định vị trí từng vật thể. Trong ảnh traffic có nhiều phương tiện, nhưng mô hình phân loại xếp nhãn cab ở vị trí cao nhất.



Câu 3. Danh sách lớp do đâu mà có?



Danh sách lớp được xác định bởi taxonomy ImageNet-1K đi kèm checkpoint đã huấn luyện, không phải do mô hình tự tạo ra khi dự đoán.



Câu 4. Vì sao cần ghi class\_id, class\_name và taxonomy\_name?



Để xác định rõ lớp dự đoán và hệ thống phân loại được sử dụng. Các taxonomy khác nhau có thể sử dụng mã lớp hoặc tên lớp khác nhau, nên chỉ ghi tên lớp sẽ không đủ để truy xuất và đối chiếu chính xác.



Câu 5. Guideline cần quy định điều gì khi ảnh có nhiều vật thể?



Cần quy định tiêu chí chọn một nhãn đại diện cho toàn bộ ảnh, chẳng hạn đối tượng chính hoặc chủ đề nổi bật, đồng thời hướng dẫn cách xử lý khi có nhiều đối tượng quan trọng hoặc không có lớp phù hợp.



Câu 6. Vì sao không lấy model score làm ground truth?



Model score là điểm dự đoán của mô hình, không phải nhãn đúng đã được con người kiểm tra. Mô hình có thể dự đoán sai dù điểm cao. Ground truth phải được xác lập theo quy tắc gán nhãn và quy trình kiểm tra chất lượng.





### **2. Phát hiện vật thể**

**Mở file:** `day1\_lab\_outputs/detection\_predictions.json` và `visuals/detection\_predictions.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\_name`=… ; `score`=… ; `bbox\_xyxy`=\[…, …, …, …] ; `bbox\_width`=… ; `bbox\_height`=…
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?



Câu 1. Chép một record

sample\_id: kitchen



class\_name: person



score: 0.912625



bbox\_xyxy: \[385.33, 69.24, 498.92, 348.92]



bbox\_width: 113.58 pixel



bbox\_height: 279.68 pixel



Câu 2. Giải thích Bounding Box



Khung phát hiện người đầu bếp có góc trên bên trái tại tọa độ (385.33, 69.24), góc dưới bên phải tại (498.92, 348.92). Chiều rộng khoảng 113.58 pixel và chiều cao 279.68 pixel.



Đây là vị trí và kích thước của người trên khung ảnh



Câu 3. So sánh hai ngưỡng



Threshold 0.20: 17 prediction.



Threshold 0.60: 6 prediction.





Câu 4. Khi hạ ngưỡng, reviewer cần làm gì?



Reviewer phải kiểm tra thêm các khung có điểm dự đoán thấp để phát hiện nhãn sai, khung sai vị trí, đối tượng bị đếm trùng và các trường hợp không đủ bằng chứng nhận diện.



Câu 5. Một quy tắc vẽ Bounding Box



Mỗi đối tượng được gán một khung chữ nhật riêng, bao sát phần đối tượng nhìn thấy trong ảnh.



Bước 1- Xác định khung: Góc trên trái và góc dưới phải phải bao hết phần vật thể nhìn thấy, hạn chế lấy phần thừa.



Bước 2 - Khi đối tượng bị che khuất hoặc cắt mép: Chỉ khoanh phần nhìn thấy, không tự suy đoán phần bị che hoặc nằm ngoài ảnh.



Bước 3 - Với đối tượng khó xác định: Nếu không đủ căn cứ nhận diện hoặc xác định ranh giới, chuyển cho quản lý tìm phương án xử lý và thống nhất làm theo, không tự gán nhãn.



Câu 6. Một trường hợp vật thể bị che khuất hoặc cắt mép



Trong ảnh kitchen, phía mép trái có một phần cánh tay người xuất hiện tại khu vực bàn bếp. Mô hình dự đoán nhãn person với score 0.610931 và khung \[0.12, 263.18, 61.55, 310.78].



Đây là trường hợp cần có guideline rõ ràng về vật thể chỉ xuất hiện một phần trong ảnh. Nếu guideline chưa xác định mức độ hiển thị tối thiểu để gán nhãn, cần đến Lab Coach thay vì tự quyết định.Bởi vì sẽ hạn chế sai sót về sau









## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\_lab\_outputs/segmentation\_predictions.json` và `visuals/segmentation\_prediction.png`.  
Chỉ dùng ảnh `sample\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\_id`=… ; `class\_name`=… ; `score`=… ; số điểm polygon=`polygon\_point\_count`=… ; 3–5 điểm đầu của `polygon\_xy`=…
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
3. `instance\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?



Câu 1. Chép một record 

instance\_id: kitchen-001



class\_name: person



score: 0.899318



polygon\_point\_count: 348



5 điểm đầu của polygon\_xy: \[\[446, 70], \[445, 71], \[444, 71], \[443, 72], \[442, 72]]



Câu 2. So với Bounding Box, Polygon cho thêm thông tin gì?



Bounding Box chỉ xác định vùng chữ nhật bao quanh vật thể. Polygon mô tả đường viền bao quanh vật thể bằng nhiều điểm tọa độ, nhờ đó thể hiện hình dạng chi tiết hơn và có thể xác định vùng điểm ảnh thuộc đối tượng.



Câu 3. Instance ID dùng để làm gì?



instance\_id dùng để phân biệt từng cá thể trong cùng một ảnh, kể cả khi chúng thuộc cùng một loại đối tượng



Ví dụ, hai người trong ảnh đều có class\_name = person nhưng phải có instance\_id khác nhau.



Ví dụ có 5 kỹ sư xây dựng đội mũ bảo hộ khác nhau, loại đối tượng là mũ bảo hộ nhưng instance là 5 cái mũ



Câu 4. Quy tắc vẽ biên Mask



Vẽ đường biên bám sát phần vật thể thực sự nhìn thấy, không lấy nền hoặc vùng thuộc vật thể khác. Khi hai vật thể chạm nhau, phải phân biệt và gán vùng cho từng cá thể riêng biệt.



Với vùng bị che khuất, chỉ gán phần nhìn thấy, không tự suy đoán phần nằm sau vật cản. Nếu ranh giới không xác định rõ, chuyển lên quản lý yêu cầu phương án xử lý và thực thi theo



Câu 5. Trường hợp khó xác định biên trong ảnh kitchen



Ở mép trái ảnh kitchen, mô hình tạo phân đoạn person với instance\_id = kitchen-009 và score 0.431437. Đối tượng chỉ xuất hiện một phần, gây khó khăn trong việc xác định đường biên và phạm vi gán nhãn.



Cần áp dụng guideline về vật thể bị cắt mép. Nếu chưa có quy định về mức độ hiển thị tối thiểu hoặc ranh giới không đủ rõ, phải chuyển Lab Coach xem xét thay vì tự suy đoán.





## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Một nhãn đại diện cho toàn bộ ảnh.|Ảnh traffic có nhiều xe buýt, ô tô nhưng AI xếp cab cao nhất (score 0.510915). Khó xác định nhãn đại diện.|Quan sát toàn ảnh, chọn một lớp theo guideline.|Kiểm tra lớp được chọn có phù hợp với nội dung ảnh và guideline không.|
|Phát hiện vật thể|Mỗi vật thể có một nhãn lớp và một Bounding Box.|Ảnh kitchen có người bị cắt ở mép trái (score 0.610931), khó xác định khung bao.|Gán đúng lớp, vẽ khung sát vật thể theo quy tắc.|Kiểm tra nhãn, vị trí khung, vật thể bị bỏ sót hoặc khoanh sai.|
|Instance segmentation|Mỗi cá thể có một nhãn lớp và một Mask/Polygon riêng|Ảnh kitchen có người và các vật thể bị che khuất; khó xác định chính xác đường biên.|Gán nhãn, vẽ vùng Mask theo phần vật thể nhìn thấy, phân biệt từng cá thể.|Kiểm tra đường biên, vùng bị lấy nhầm và các cá thể có được tách riêng đúng không.|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Chỉ sử dụng hình ảnh và dữ liệu được phép sử dụng, có nguồn gốc rõ ràng; không đưa thông tin về bài tập thực hành chia sẻ ra bên ngoài (cả link hướng dẫn và kết quả làm bài)

1. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Dừng xử lý và không đưa dữ liệu vào hệ thống; báo cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn xử lý.

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[ ] `REPORT.md` (file này đã điền)
* \[ ] `day1\_lab\_outputs/classification\_predictions.json`
* \[ ] `day1\_lab\_outputs/detection\_predictions.json`
* \[ ] `day1\_lab\_outputs/segmentation\_predictions.json`
* \[ ] `day1\_lab\_outputs/IMAGE\_ATTRIBUTION.md`
* \[ ] `day1\_lab\_outputs/visuals/classification\_top5.png`
* \[ ] `day1\_lab\_outputs/visuals/detection\_predictions.png`
* \[ ] `day1\_lab\_outputs/visuals/segmentation\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS


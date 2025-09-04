# Báo cáo: Ứng dụng LSTM trong dự báo giá và xu hướng biến động của Bitcoin

## Chương 1: Mở đầu và Đặt vấn đề

### 1.1. Bối cảnh và Tính cấp thiết của đề tài

Thị trường tiền điện tử, với Bitcoin (BTC) là đại diện hàng đầu, đã trở thành một lĩnh vực tài chính đầy biến động và thu hút sự quan tâm to lớn từ các nhà đầu tư cá nhân, tổ chức tài chính và cộng đồng nghiên cứu. Đặc điểm nổi bật của Bitcoin là tính biến động giá mạnh, chịu ảnh hưởng phức tạp từ nhiều yếu tố như cung cầu thị trường, các chính sách pháp lý, tin tức kinh tế vĩ mô và tâm lý đám đông. Sự phức tạp và tính phi tuyến tính của dữ liệu giá Bitcoin làm cho các phương pháp dự báo truyền thống gặp nhiều thách thức. Do đó, việc xây dựng các mô hình dự báo tiên tiến, có khả năng học được các mẫu hình ẩn sâu và mối quan hệ phụ thuộc thời gian trong dữ liệu là vô cùng quan trọng.  

Việc dự báo giá trị và xu hướng biến động của Bitcoin không chỉ đơn thuần là một bài toán học thuật, mà còn mang lại giá trị thực tiễn to lớn. Đối với nhà đầu tư, các mô hình dự báo chính xác có thể hỗ trợ đưa ra quyết định giao dịch kịp thời, quản lý rủi ro hiệu quả và xây dựng chiến lược đầu tư bền vững. Do đó, nghiên cứu này tập trung vào việc ứng dụng mô hình Long Short-Term Memory (LSTM), một loại mạng nơ-ron hồi quy có khả năng xử lý và ghi nhớ thông tin trong chuỗi thời gian, để giải quyết hai nhiệm vụ cốt lõi: dự báo giá (bài toán hồi quy) và dự báo xu hướng (bài toán phân loại) của Bitcoin dựa trên dữ liệu lịch sử.  

### 1.2. Mục tiêu nghiên cứu

Đề tài được xây dựng với các mục tiêu cụ thể như sau :  

- **Thu thập và Tiền xử lý Dữ liệu**: Sử dụng dữ liệu đa biến về Bitcoin từ sàn giao dịch Binance, bao gồm các đặc trưng giá, khối lượng giao dịch và các chỉ báo kỹ thuật, nhằm đảm bảo chất lượng dữ liệu đầu vào cho quá trình phân tích và dự báo.  
- **Phân tích và Trực quan hóa Dữ liệu**: Khám phá mối quan hệ giữa các đặc trưng và biến mục tiêu thông qua các kỹ thuật phân tích tương quan và Mutual Information, từ đó lựa chọn tập đặc trưng tối ưu cho từng bài toán.  
- **Xây dựng Mô hình Học sâu**: Phát triển và huấn luyện hai mô hình LSTM riêng biệt để giải quyết bài toán dự báo giá và bài toán dự báo xu hướng tăng/giảm.  
- **Đánh giá và Tối ưu hóa**: Sử dụng các chỉ số đánh giá chuyên biệt và biểu đồ trực quan để phân tích hiệu quả của từng mô hình, từ đó rút ra nhận xét, ưu điểm, hạn chế và đề xuất các hướng cải thiện trong tương lai.  

### 1.3. Tổng quan về mô hình Long Short-Term Memory (LSTM)

Mô hình Long Short-Term Memory (LSTM) là một biến thể của mạng nơ-ron hồi quy (RNN), được thiết kế đặc biệt để xử lý dữ liệu chuỗi thời gian và khắc phục các nhược điểm của RNN truyền thống như vấn đề tiêu biến hoặc bùng nổ gradient. Kiến trúc LSTM được Hochreiter và Schmidhuber giới thiệu vào năm 1997, mang lại khả năng ghi nhớ các phụ thuộc dài hạn trong chuỗi dữ liệu.  

Mỗi đơn vị LSTM bao gồm một ô nhớ (cell state) và ba "cổng" chính, hoạt động như các bộ lọc thông tin :  

- **Cổng quên (forget gate)**: Quyết định thông tin nào từ ô nhớ cũ cần được loại bỏ.  
- **Cổng vào (input gate)**: Quyết định thông tin mới nào cần được thêm vào ô nhớ.  
- **Cổng ra (output gate)**: Quyết định phần thông tin nào từ ô nhớ sẽ được sử dụng để tạo ra đầu ra tại bước thời gian hiện tại.  

Các cơ chế này cho phép LSTM duy trì thông tin quan trọng qua nhiều bước thời gian mà không bị mất đi, đồng thời loại bỏ các nhiễu không cần thiết. Nhờ khả năng này, LSTM trở thành một lựa chọn phù hợp để mô hình hóa các chuỗi thời gian tài chính phức tạp và phi tuyến tính như giá Bitcoin, giúp mô hình học được các quy luật và mẫu hình giá từ dữ liệu nến 15 phút một cách hiệu quả.  

### 1.4. Cấu trúc báo cáo

Báo cáo này được tổ chức thành năm chương. Chương 1 trình bày bối cảnh, lý do chọn đề tài và tổng quan về phương pháp. Chương 2 mô tả chi tiết quy trình xử lý dữ liệu, phân tích các đặc trưng và lựa chọn tập đặc trưng phù hợp cho từng bài toán. Chương 3 trình bày kiến trúc và tham số huấn luyện của hai mô hình LSTM. Chương 4 đi sâu vào phân tích và đánh giá kết quả thực nghiệm. Cuối cùng, Chương 5 đưa ra kết luận tổng quan, nêu bật ưu điểm, hạn chế và đề xuất các định hướng phát triển trong tương lai.

## Chương 2: Xử lý và Phân tích Dữ liệu

### 2.1. Quy trình Tiền xử lý Dữ liệu

Quá trình tiền xử lý dữ liệu là một bước nền tảng, đảm bảo chất lượng dữ liệu đầu vào cho mô hình học máy. Dữ liệu giá Bitcoin được thu thập từ sàn Binance với khung thời gian 15 phút, bao gồm 280.253 dòng dữ liệu từ năm 2017 đến tháng 8 năm 2025.  

Quy trình tiền xử lý được thực hiện theo các bước chính sau đây :  

- **Thêm tiêu đề cho các cột**: Dữ liệu thô từ Binance API ban đầu không có tiêu đề, do đó, các tiêu đề chuẩn (open_time, open, high, low, close, volume,...) đã được thêm vào để dễ dàng xử lý.  
- **Loại bỏ các cột không cần thiết**: Để giảm nhiễu và tối ưu hóa dữ liệu, các cột không có tác động trực tiếp đến việc dự báo giá ngắn hạn như quote_asset_volume hay number_of_trades đã được loại bỏ. Các đặc trưng quan trọng được giữ lại bao gồm open_time, các đặc trưng giá (open, high, low, close) và volume.  
- **Kiểm tra và xử lý dữ liệu**: Dữ liệu được kiểm tra để đảm bảo không có giá trị thiếu, không hợp lệ hoặc trùng lặp, tạo ra một tập dữ liệu sạch và đáng tin cậy.  
- **Chuyển đổi kiểu dữ liệu**: Cột open_time được chuyển từ kiểu số nguyên (timestamp) sang định dạng ngày giờ (datetime) và được đặt làm chỉ mục (index) của bộ dữ liệu, hỗ trợ việc phân tích chuỗi thời gian một cách thuận tiện hơn.  
- **Bổ sung chỉ số kỹ thuật**: Đây là một bước quan trọng nhằm làm giàu thông tin cho mô hình. Hai chỉ số kỹ thuật đã được thêm vào là Đường trung bình động 20 kỳ (MA20) và Chỉ số sức mạnh tương đối 14 kỳ (RSI14).  

MA20 làm mượt dữ liệu giá, giúp mô hình nhận diện xu hướng ngắn hạn và giảm nhiễu, trong khi RSI14 cung cấp tín hiệu về tâm lý thị trường, đo lường trạng thái quá mua (>70) hoặc quá bán (<30) của tài sản, rất hữu ích cho bài toán dự báo xu hướng.  

Quá trình tiền xử lý cũng bao gồm việc kiểm tra các giá trị ngoại lệ (outlier) bằng phương pháp IQR. Kết quả cho thấy các giá trị giá Bitcoin trên 100.000 USD không phải là nhiễu hay lỗi dữ liệu, mà là sự phản ánh của các đợt tăng trưởng mạnh trong giai đoạn 2024-2025. Do đó, các giá trị này không được loại bỏ vì chúng chứa thông tin quan trọng về các chu kỳ thị trường, là yếu tố then chốt giúp mô hình học được bản chất biến động của Bitcoin.  

### 2.2. Phân tích Tương quan giữa các Đặc trưng

Phân tích tương quan được sử dụng để đo lường mức độ liên hệ tuyến tính giữa các đặc trưng. Ma trận tương quan được trình bày dưới dạng biểu đồ nhiệt (heatmap), giúp trực quan hóa mối quan hệ này.  

Biểu đồ ma trận tương quan các đặc trưng (xem Hình 8) cho thấy một số điểm đáng chú ý:

- **Tương quan rất cao**: Các đặc trưng giá (Open, High, Low, Close) và chỉ số MA20 đều có mối tương quan tuyến tính rất mạnh với nhau, với hệ số tương quan gần bằng 1. Điều này chứng tỏ các biến này chứa thông tin rất tương đồng, đều phản ánh cùng một sự biến động giá.  
- **Tương quan thấp**: Các đặc trưng Volume và RSI14 có mối tương quan rất thấp với các đặc trưng giá (hệ số ~0.1 và ~0.03). Điều này cho thấy Volume và RSI14 là những đặc trưng độc lập, cung cấp thông tin mới không có trong dữ liệu giá, làm giàu thêm bộ dữ liệu đầu vào.  

Mặc dù mối tương quan cao giữa các đặc trưng giá có thể gây ra vấn đề đa cộng tuyến cho các mô hình hồi quy tuyến tính truyền thống, điều này không ảnh hưởng tiêu cực đến mô hình LSTM. Thay vào đó, nó giúp mô hình học sâu dễ dàng nắm bắt các xu hướng giá tổng thể và biến động nội nến một cách hiệu quả.  

### 2.3. Phân tích Mutual Information (MI)

Mutual Information (MI) là một phương pháp mạnh mẽ hơn tương quan tuyến tính, bởi nó có khả năng đo lường mối quan hệ phi tuyến giữa các biến và phản ánh lượng thông tin mà một đặc trưng cung cấp cho biến mục tiêu. MI càng cao thì đặc trưng càng quan trọng đối với bài toán. Phân tích MI được thực hiện cho cả hai bài toán hồi quy và phân loại.  

#### 2.3.1. Phân tích MI cho bài toán Hồi quy (dự báo giá)

Đối với bài toán dự báo giá, biến mục tiêu là Close tại thời điểm $t+1$. Phân tích MI (xem Hình 7) cho thấy sự phân cấp rõ rệt về mức độ quan trọng của các đặc trưng:

- Các đặc trưng giá (Close, High, Low, Open) và MA20 có điểm MI rất cao, dao động từ 2.89 đến 3.75. Điều này xác nhận rằng lịch sử giá và đường trung bình động là các yếu tố cung cấp thông tin quan trọng nhất để dự báo giá trị tuyệt đối của Bitcoin trong tương lai.  
- Volume có điểm MI thấp hơn đáng kể (0.13), nhưng vẫn đóng góp một lượng thông tin nhất định về sức mạnh của thị trường.  
- RSI14 có điểm MI rất thấp (0.05), cho thấy nó gần như không có giá trị thông tin đối với bài toán dự báo giá trị tuyệt đối.  

#### 2.3.2. Phân tích MI cho bài toán Phân loại (dự báo xu hướng)

Đối với bài toán dự báo xu hướng, biến mục tiêu là xu hướng tăng/giảm. Phân tích MI (xem Hình 6) mang lại một kết quả khác biệt đáng kể:

- RSI14 và MA20 là hai đặc trưng có điểm MI cao nhất, với RSI14 đạt 0.097 và MA20 đạt 0.034. Điều này chứng tỏ các chỉ báo kỹ thuật này là đặc trưng quan trọng nhất để dự đoán xu hướng biến động của Bitcoin.  
- Các đặc trưng giá (Open, High, Low, Close) có điểm MI rất thấp, dao động từ 0.0073 đến 0.011.  

Sự khác biệt rõ ràng giữa kết quả MI cho hai bài toán cho thấy bản chất của vấn đề quyết định việc lựa chọn đặc trưng. Dự báo giá trị (hồi quy) cần thông tin về chính giá trong quá khứ, trong khi dự báo xu hướng (phân loại) phụ thuộc nhiều hơn vào các chỉ báo về động lượng và sức mạnh của thị trường, được phản ánh qua RSI và MA20.

### 2.4. Tổng kết và Lựa chọn Đặc trưng

Dựa trên kết quả phân tích tương quan và Mutual Information, tập đặc trưng tối ưu đã được lựa chọn cho từng bài toán, nhằm tối đa hóa hiệu suất của mô hình.  

**Bảng 2.1: Lựa chọn Đặc trưng cho các Mô hình**

| **Bài toán** | **Đặc trưng được chọn** | **Lý do** |
|--------------|-------------------------|-----------|
| Dự báo giá (Hồi quy) | Open, High, Low, Close, MA20, Volume | Có tương quan và MI cao với biến mục tiêu giá, cung cấp đủ thông tin lịch sử giá để dự báo giá trị tuyệt đối. |
| Dự báo xu hướng (Phân loại) | RSI14, MA20, Close | RSI14 và MA20 có MI cao nhất, cung cấp tín hiệu mạnh mẽ về động lượng và xu hướng thị trường, là yếu tố then chốt cho bài toán phân loại. |

Sau khi lựa chọn đặc trưng, dữ liệu được chia thành tập huấn luyện và tập kiểm thử theo tỷ lệ 80/20. Cuối cùng, dữ liệu được chuẩn hóa bằng phương pháp MinMaxScaler để đưa về khoảng . Bước này là cần thiết vì các đặc trưng có thang đo khác nhau (giá BTC hàng chục nghìn USD, RSI từ 0-100, Volume dao động lớn), việc chuẩn hóa giúp mô hình học hiệu quả hơn và tránh hiện tượng gradient vanishing/exploding.  

## Chương 3: Xây dựng và Huấn luyện Mô hình LSTM

### 3.1. Kỹ thuật tạo dữ liệu dạng chuỗi thời gian (Sliding Window)

Mô hình LSTM được thiết kế để xử lý dữ liệu chuỗi thời gian. Để cung cấp đầu vào phù hợp, kỹ thuật cửa sổ trượt (sliding window) đã được áp dụng. Kỹ thuật này chia chuỗi dữ liệu lịch sử thành các chuỗi con (sequences), mỗi chuỗi con bao gồm một số bước thời gian (timesteps) và các đặc trưng (features) tương ứng. Dữ liệu được chuyển từ dạng 2 chiều ([samples, features]) sang dạng 3 chiều ([samples, timesteps, features]), là định dạng đầu vào chuẩn của mô hình LSTM.

Kích thước cửa sổ trượt được lựa chọn khác nhau cho mỗi bài toán:

- **Bài toán hồi quy**: Cửa sổ trượt 32 bước (tương đương 8 giờ dữ liệu).  
- **Bài toán phân loại**: Cửa sổ trượt 96 bước (tương đương 24 giờ dữ liệu).  

Sự khác biệt này xuất phát từ bản chất của từng bài toán. Dự báo giá trị tuyệt đối (hồi quy) có thể phụ thuộc nhiều vào các diễn biến gần nhất, trong khi dự báo xu hướng (phân loại) yêu cầu một bối cảnh rộng hơn để nhận diện các mẫu hình dao động và tâm lý thị trường.

### 3.2. Cấu hình và Tham số Huấn luyện cho Bài toán Hồi quy

Mô hình LSTM được xây dựng cho bài toán dự báo giá có kiến trúc gồm nhiều tầng, được thiết kế để trích xuất đặc trưng và dự đoán giá trị liên tục :  

- **Tầng 1**: LSTM(128) với return_sequences=True, nhằm trả về toàn bộ chuỗi ẩn để tầng tiếp theo có thể học các đặc trưng chi tiết hơn.  
- **Tầng Dropout**: Dropout(0.3), nhằm ngẫu nhiên bỏ đi 30% nơ-ron trong quá trình huấn luyện để giảm thiểu hiện tượng overfitting.  
- **Tầng 2**: LSTM(64) với return_sequences=False, tập trung học các đặc trưng sâu hơn và chỉ trả về trạng thái cuối cùng của chuỗi.  
- **Tầng Fully-connected**: Dense(32, activation='relu'), để trích xuất các đặc trưng phi tuyến tính.  
- **Tầng Đầu ra**: Dense(1), với một nơ-ron duy nhất để dự đoán giá trị liên tục của giá đóng cửa.  

Mô hình được biên dịch với hàm mất mát Mean Squared Error (MSE) và thuật toán tối ưu hóa Adam, với tốc độ học 0.0001. Quá trình huấn luyện sử dụng các callback EarlyStopping để dừng sớm khi validation loss không cải thiện sau 10 epoch, và ReduceLROnPlateau để tự động giảm tốc độ học khi mô hình chững lại sau 5 epoch. Các callback này giúp tối ưu hóa quá trình huấn luyện và đảm bảo mô hình đạt hiệu suất cao nhất. Mô hình được huấn luyện trong 200 epoch với kích thước batch là 64.  

### 3.3. Cấu hình và Tham số Huấn luyện cho Bài toán Phân loại

Tương tự như mô hình hồi quy, mô hình phân loại cũng có kiến trúc tương tự nhưng được điều chỉnh cho bài toán dự đoán xu hướng :  

- **Tầng LSTM 1**: LSTM(128) với return_sequences=True.  
- **Tầng Dropout**: Dropout(0.3).  
- **Tầng LSTM 2**: LSTM(64) với return_sequences=False.  
- **Tầng Fully-connected**: Dense(32, activation='relu').  
- **Tầng Đầu ra**: Dense(1) với hàm kích hoạt sigmoid, cho phép mô hình dự đoán xác suất của hai lớp (tăng hoặc giảm).  

Mô hình phân loại sử dụng hàm mất mát binary_crossentropy và thuật toán tối ưu Adam với tốc độ học 0.0001. Các callback EarlyStopping và ReduceLROnPlateau cũng được sử dụng để tối ưu quá trình huấn luyện. Mô hình được huấn luyện trong 150 epoch với kích thước batch là 64.  

## Chương 4: Kết quả và Đánh giá Mô hình

### 4.1. Kết quả Dự báo Giá (Bài toán Hồi quy)

#### 4.1.1. Phân tích Biểu đồ Loss qua các Epoch

Biểu đồ loss qua các epoch (xem Hình 5) cho thấy mô hình hồi quy đã hội tụ rất tốt và không bị overfitting. Cả training loss và validation loss đều giảm nhanh chóng trong những epoch đầu và tiếp tục giảm dần, song song với nhau, cho đến khi đạt mức rất thấp (gần 10−4). Sự hội tụ này chứng minh rằng mô hình đã học được các quy luật cơ bản từ dữ liệu huấn luyện và có khả năng tổng quát hóa tốt trên dữ liệu mới. Quá trình huấn luyện đã dừng sớm ở epoch thứ 98 nhờ cơ chế EarlyStopping, cho thấy mô hình đã đạt được hiệu suất tối ưu mà không cần chạy hết số epoch đã định.  

#### 4.1.2. Đánh giá Biểu đồ Dự đoán so với Giá Thực tế

Biểu đồ so sánh giá dự đoán và giá thực tế (xem Hình 4) minh họa rõ ràng hiệu suất vượt trội của mô hình. Đường giá dự đoán (màu cam) gần như trùng khớp hoàn hảo với đường giá thực tế (màu xanh), bám sát mọi biến động và xu hướng chính trong giai đoạn kiểm thử.  

Tuy nhiên, khi phân tích sâu hơn, có thể nhận thấy một hạn chế nhỏ: tại các đỉnh giá (local maxima) và đáy giá (local minima), mô hình có xu hướng "làm mượt" và dự đoán thấp hơn đỉnh hoặc cao hơn đáy. Hiện tượng này là đặc điểm thường thấy ở các mô hình chuỗi thời gian, do chúng phản ứng chậm hơn với các biến động đột ngột và cực đoan. Mặc dù vậy, khả năng nắm bắt xu hướng chung của mô hình vẫn rất mạnh, khiến nó trở thành một công cụ hỗ trợ đáng tin cậy.  

#### 4.1.3. Phân tích Phân bố Sai số

Biểu đồ phân bố sai số (True - Predicted) (xem Hình 3) cho thấy phần lớn sai số của mô hình tập trung chặt chẽ quanh giá trị 0. Phân bố này có hình dạng gần giống với phân bố chuẩn Gaussian, với sai số phổ biến nằm trong khoảng ±300−400 USD. Điều này cho thấy mô hình hoạt động ổn định và các dự đoán có sai số nhỏ, có tính ngẫu nhiên.  

Sự xuất hiện của một vài sai số lớn hơn (>1000 USD) ở hai đầu của biểu đồ có thể tương ứng với các thời điểm thị trường có biến động cực mạnh, khi mô hình gặp khó khăn trong việc dự đoán chính xác.  

Tổng thể, các chỉ số đánh giá của mô hình hồi quy cũng khẳng định độ chính xác cao :  

- **MSE (Mean Squared Error)**: 167,996.99  
- **RMSE (Root Mean Squared Error)**: 409.87 USD  
- **MAE (Mean Absolute Error)**: 294.18 USD  
- **R2 Score (Coefficient of Determination)**: 0.98  

Với R2 score đạt 0.98, mô hình có khả năng giải thích tới 98% sự biến động của giá Bitcoin trong tập kiểm thử. Sai số trung bình tuyệt đối (MAE) chỉ khoảng 294 USD, một con số rất nhỏ so với mức giá trung bình trên 100.000 USD, tương đương với sai số chỉ khoảng 0.3-0.4%.  

### 4.2. Kết quả Dự báo Xu hướng (Bài toán Phân loại)

#### 4.2.1. Phân tích Biểu đồ Loss qua các Epoch

Biểu đồ training loss và validation loss cho bài toán phân loại (xem Hình 2) cho thấy một hiệu suất huấn luyện kém hơn hẳn so với mô hình hồi quy. Mặc dù training loss giảm đều, validation loss lại không giảm sâu và có sự dao động mạnh, cho thấy mô hình đang gặp khó khăn trong việc học và tổng quát hóa trên dữ liệu chưa từng thấy. Điều này báo hiệu mô hình có thể bị underfitting, tức là chưa đủ mạnh để nắm bắt được các đặc trưng phức tạp của bài toán dự đoán xu hướng.  

#### 4.2.2. Đánh giá Ma trận Nhầm lẫn

Ma trận nhầm lẫn (Confusion Matrix) (xem Hình 1) cung cấp một cái nhìn chi tiết về hiệu suất của mô hình phân loại :  

- **TP (True Positive)**: 770 (Dự đoán tăng, thực tế tăng)  
- **FP (False Positive)**: 331 (Dự đoán tăng, thực tế giảm)  
- **TN (True Negative)**: 1749 (Dự đoán giảm, thực tế giảm)  
- **FN (False Negative)**: 1496 (Dự đoán giảm, thực tế tăng)  

Phân tích ma trận cho thấy mô hình có xu hướng thiên lệch về dự đoán "giảm" (lớp 0). Mặc dù có khả năng dự đoán đúng các trường hợp giảm (TN = 1749), mô hình lại bỏ sót một lượng lớn các trường hợp tăng thực tế (FN = 1496).  

Các chỉ số đánh giá cũng phản ánh hiệu suất kém này :  

- **Accuracy**: 0.58 (Chỉ cao hơn một chút so với dự đoán ngẫu nhiên 0.5)  
- **Precision (class 1 - tăng)**: 0.70 (Khi mô hình dự đoán tăng, độ chính xác là 70%)  
- **Recall (class 1 - tăng)**: 0.34 (Mô hình chỉ phát hiện được 34% số phiên tăng thực sự)  
- **F1-score (class 1 - tăng)**: 0.46 (Thấp, thể hiện sự mất cân bằng giữa precision và recall)  

Đặc biệt, recall thấp cho thấy mô hình đã bỏ sót tới gần 2/3 các cơ hội tăng giá thực tế, điều này làm giảm đáng kể tính hữu ích của mô hình trong môi trường giao dịch.  

## Chương 5: Kết luận và Định hướng Phát triển

### 5.1. Tóm tắt kết quả đạt được

Nghiên cứu đã chứng minh tính khả thi của việc ứng dụng mô hình LSTM trong việc dự báo chuỗi thời gian tài chính, cụ thể là giá và xu hướng biến động của Bitcoin.  

- **Mô hình hồi quy (dự báo giá)** đã cho kết quả xuất sắc, với sai số nhỏ, ổn định và khả năng giải thích biến động giá lên tới 98% (R2=0.98). Mô hình này có tiềm năng ứng dụng cao trong việc hỗ trợ các quyết định giao dịch ngắn hạn.  
- **Mô hình phân loại (dự báo xu hướng)** cho kết quả kém khả quan hơn, với accuracy chỉ 58% và recall thấp (0.34), cho thấy mô hình còn nhiều hạn chế trong việc phát hiện các cơ hội tăng giá thực sự.  

### 5.2. Đánh giá ưu điểm và hạn chế

**Ưu điểm**:

- **Sự phù hợp của LSTM**: Báo cáo khẳng định mô hình LSTM rất hiệu quả trong việc xử lý các chuỗi dữ liệu phức tạp và có phụ thuộc dài hạn như giá Bitcoin.  
- **Giá trị thực tiễn**: Mô hình hồi quy có độ chính xác cao, có thể được sử dụng như một công cụ hỗ trợ đáng tin cậy trong phân tích kỹ thuật và quản lý rủi ro.  
- **Khả năng xử lý dữ liệu đa biến**: Việc kết hợp các đặc trưng giá, khối lượng và chỉ báo kỹ thuật đã giúp mô hình khai thác được nhiều khía cạnh của thị trường.  

**Hạn chế**:

- **Khả năng dự đoán đỉnh/đáy**: Mô hình hồi quy vẫn còn hạn chế trong việc phản ứng tức thời với các biến động cực đoan, thường dự đoán thấp hơn đỉnh và cao hơn đáy.  
- **Hiệu suất phân loại**: Mô hình phân loại xu hướng chưa đủ tin cậy để đưa ra các tín hiệu giao dịch độc lập, do hiệu suất thấp và có xu hướng thiên lệch về một lớp.  
- **Giới hạn về dữ liệu**: Các mô hình hiện tại chỉ dựa trên dữ liệu lịch sử giá, chưa tích hợp các yếu tố bên ngoài như tin tức, tâm lý thị trường, và các sự kiện vĩ mô, vốn có ảnh hưởng lớn đến giá Bitcoin.  

### 5.3. Định hướng cải thiện và phát triển trong tương lai

Dựa trên những hạn chế đã được chỉ ra, các hướng phát triển trong tương lai có thể tập trung vào các điểm sau:

- **Mở rộng tập đặc trưng**: Tích hợp thêm các chỉ báo kỹ thuật khác như MACD, Bollinger Bands, Stochastic, cùng với dữ liệu phi cấu trúc như phân tích cảm xúc từ tin tức, bài đăng trên mạng xã hội, để mô hình có thể "hiểu" được bối cảnh thị trường và dự báo các biến động đột ngột một cách chính xác hơn.  
- **Tối ưu hóa mô hình phân loại**: Cần giải quyết vấn đề mất cân bằng trong việc dự đoán các lớp bằng cách sử dụng các kỹ thuật cân bằng dữ liệu (oversampling/undersampling) hoặc các hàm mất mát phù hợp hơn như focal loss.  
- **Thử nghiệm các kiến trúc mô hình khác**: So sánh hiệu suất của LSTM với các kiến trúc học sâu khác như GRU (Gated Recurrent Unit) hoặc Transformer, vốn đang rất thành công trong các bài toán xử lý chuỗi dữ liệu, để tìm ra mô hình tối ưu hơn cho bài toán dự báo chuỗi thời gian tài chính.

### 5.4. Tổng kết

Tóm lại, nghiên cứu đã đạt được những thành tựu quan trọng trong việc ứng dụng LSTM để dự báo giá Bitcoin, mở ra một hướng tiếp cận đầy tiềm năng cho việc phân tích và dự báo thị trường tiền điện tử. Mặc dù mô hình phân loại còn cần nhiều cải tiến, kết quả xuất sắc của mô hình hồi quy đã cung cấp một công cụ mạnh mẽ, có thể áp dụng ngay trong thực tiễn, đồng thời làm nền tảng vững chắc cho các nghiên cứu toàn diện hơn trong tương lai.
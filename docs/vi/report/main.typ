#import "@preview/ilm:2.0.0": *

#set text(lang: "vi")

#show: ilm.with(
  title: [Huấn luyện Mạng CNN bằng Phương pháp QuickProp Parabolic],
  authors: (
    "Phạm Kỳ Anh, MSSV: 25C15033",
    "Lớp: Toán cho trí tuệ nhân tạo nâng cao",
    "Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia TP.HCM",
    "Ngày: " + str(datetime.today().display()),
  ),
  abstract: [],
  preface: [
    #align(horizon)[
      GitHub: https://github.com/itskyf/MTH088-2026-ParaProp \
      HuggingFace: https://huggingface.co/spaces/itskyf/MTH088-2026-ParaProp
    ]
  ],
  footer: "page-number-right-with-chapter",
  chapter-pagebreak: false,
  bibliography: bibliography("refs.bib"),
  raw-text: "use-typst-default",
  figure-index: (enabled: true),
  table-index: (enabled: true),
  listing-index: (enabled: true),
)

= Giới thiệu
Quá trình học máy bản chất là bài toán xấp xỉ hàm và cực tiểu hóa hàm mất mát. Mạng nơ-ron tích chập (CNN) là lựa chọn ưu tiên cho xử lý ảnh nhờ khả năng trích xuất đặc trưng không gian. Tuy nhiên, tốc độ và sự ổn định của quá trình huấn luyện phụ thuộc lớn vào thuật toán tối ưu (optimizer). Báo cáo này trình bày hệ thống toán học của quá trình lan truyền thuận/ngược trong mạng CNN và thực nghiệm thay thế phương pháp tối ưu truyền thống bằng thuật toán QuickProp trên tập dữ liệu FashionMNIST. Thuật toán này sử dụng kỹ thuật xấp xỉ đạo hàm riêng phần bằng các parabol để hội tụ nhanh hơn.

= Ký hiệu và Phát biểu bài toán
- Tập dữ liệu: $Omega = { (x^{(i)}, y^{(i)}) }_(i=1)^N$, với đầu vào $x in bb(R)^(C times H times W)$.
- Nhãn mục tiêu: Được mã hóa dưới dạng vector nhị phân (one-hot) $y_"onehot" in {0, 1}^K$ tương ứng với $K$ phân lớp.
- Trọng số mô hình: $bold(w)$ (đại diện chung cho ma trận kernel và bias).
- Mô hình: Hàm số $f_bold(w)(x) -> z in bb(R)^K$ ánh xạ đầu vào thành vector điểm số (logits).
- Hàm mất mát: Trung bình Cross-entropy $cal(L)(bold(w)) = 1/N sum_i ell(z^{(i)}, y_"onehot"^{(i)})$.
- Mục tiêu: Tìm bộ trọng số tối ưu $bold(w)^* = op("arg min")_bold(w) cal(L)(bold(w))$.

= Tổng quan về mô hình ParaConv
Mô hình `ParaConv` là một kiến trúc All-convolutional (thuần tích chập) được thiết kế để duy trì luồng gradient liên tục, tương thích tốt với các thuật toán xấp xỉ đạo hàm như QuickProp. Thay vì sử dụng phép gộp (pooling) gây ra sự gián đoạn về mặt không gian toán học, mạng sử dụng các lớp tích chập có bước nhảy (stride $s=2$) để giảm độ phân giải. Phần đầu ra sử dụng lớp tích chập $1 times 1$ và Global Average Pooling (GAP) để trực tiếp trích xuất logits $z$.

= Quá trình Lan truyền thuận (Forward Pass)
== Lớp tích chập 2D
Cho đầu vào $X$, trọng số kernel $bold(w)$ và bias $b$. Tại vị trí $(h, w)$ của kênh đầu ra $c_"out"$, với bước nhảy $s$, quá trình lan truyền thuận là phép tương quan chéo (cross-correlation):
$
  Y_(c_"out", h, w) = b_(c_"out") + sum_(c_"in") sum_(m=0)^(K-1) sum_(n=0)^(K-1) w_(c_"out", c_"in", m, n) X_(c_"in", h dot s + m, w dot s + n)
$
Kích thước không gian đầu ra ($p$ là phần đệm padding):
$ H_"out" = floor((H_"in" + 2p - K) / s) + 1 $

== Hàm kích hoạt SiLU
Gọi hàm sigmoid logistic là $g(u) = 1 / (1 + e^(-u))$, hàm SiLU được định nghĩa:
$ "SiLU"(u) = u dot g(u) $
Đây là hàm trơn, liên tục và đạo hàm không bị triệt tiêu đột ngột, đảm bảo tính ổn định khi tính toán sai phân đạo hàm trong thuật toán QuickProp.

== Tích chập 1x1 và GAP
Lớp tích chập $1 times 1$ biến đổi thành $K$ kênh. GAP tính trung bình không gian:
$ z_k = 1 / (H W) sum_(h,w) a_(k,h,w) $

== Phân bố xác suất và Cross-entropy
Đầu ra được chuẩn hóa thành phân bố xác suất:
$ p_k = (e^(z_k)) / (sum_j e^(z_j)) $

= Quá trình Lan truyền ngược (Backward Pass)
Ký hiệu $Delta^{(t)} = (partial cal(L)) / (partial bold(w))$ là đạo hàm riêng phần tại bước lặp $t$.

== Gradient khởi nguồn và GAP
Gradient tại lớp logits:
$ delta_z = p - y_"onehot" $
Gradient truyền qua lớp GAP:
$ (partial ell) / (partial a_(k,h,w)) = 1 / (H W) delta_(z_k) $

== Lan truyền ngược qua lớp tích chập
Gọi $delta^"out" = (partial cal(L)) / (partial Y)$ là gradient nhận được từ lớp trên.
1. Gradient đối với bias:
  $
    (partial cal(L)) / (partial b_(c_"out")) = sum_(h, w) delta^"out"_(c_"out", h, w)
  $
2. Gradient đối với trọng số (Weights):
  $
    (partial cal(L)) / (partial w_(c_"out", c_"in", m, n)) = sum_(h, w) delta^"out"_(c_"out", h, w) X_(c_"in", h dot s + m, w dot s + n)
  $
3. Gradient đối với đầu vào (Input):
  Bản chất là phép tích chập chuyển vị (Transposed Convolution). Gọi $delta^"out"_"dilated"$ là gradient $delta^"out"$ đã được chèn $(s-1)$ số 0 vào giữa các phần tử không gian. Gradient truyền xuống lớp dưới là:
  $
    (partial cal(L)) / (partial X_(c_"in", h, w)) = sum_(c_"out") sum_(m=0)^(K-1) sum_(n=0)^(K-1) delta^"out"_"dilated"_(c_"out", h + m, w + n) w_(c_"out", c_"in", K-1-m, K-1-n)
  $

== Lan truyền ngược qua hàm SiLU
Đạo hàm của hàm SiLU:
$ d / (d u) "SiLU"(u) = g(u) + u dot g(u)(1-g(u)) $
Gradient lan truyền qua lớp SiLU (nhân từng phần tử):
$ delta_"in" = delta_"out" dot.o "SiLU"'(u) $

= Các phương pháp tối ưu hóa
== Baseline: SGD với momentum
Ký hiệu $C^{(t)}$ là biến thiên trọng thích nghi (weight step). Quy tắc cập nhật với hệ số học $alpha$ và momentum $beta$:
$ C^{(t)} = beta C^{(t-1)} - alpha Delta^{(t)} $
$ bold(w)^((t+1)) = bold(w)^((t)) + C^{(t)} $

== QuickProp (Scott Fahlman)
QuickProp sử dụng kỹ thuật xấp xỉ đạo hàm riêng phần bằng một parabol. Gọi $Delta^{(t)}$ và $Delta^{(t-1)}$ lần lượt là đạo hàm tại bước hiện tại và bước trước; $C^{(t-1)}$ là biến thiên trọng số bước trước. Mục tiêu là nhảy thẳng đến điểm cực tiểu của parabol, nơi đạo hàm $Delta^{(t+1)} = 0$.

Tỷ lệ giảm đạo hàm là:
$ (Delta^{(t)} - Delta^{(t+1)}) / (Delta^{(t-1)} - Delta^{(t)}) $
Tại điểm cực trị ($Delta^{(t+1)} = 0$), ta có công thức tính biến thiên trọng số cơ sở của QuickProp:
$ C_"QP"^{(t)} = (Delta^{(t)}) / (Delta^{(t-1)} - Delta^{(t)}) C^{(t-1)} $

Để đảm bảo thuật toán hội tụ ổn định trong thực tiễn, biến thiên trọng số thực tế $C^{(t)}$ được kiểm soát nghiêm ngặt qua hệ điều kiện sau:

$
  C^{(t)} = cases(
    - alpha Delta^{(t)} & "nếu" Delta^{(t-1)} = Delta^{(t)} "hoặc" C^{(t-1)} = 0 quad "(Bootstrapping)",
    mu C^{(t-1)} & "nếu" C_"QP"^{(t)} > mu C^{(t-1)} "và" Delta^{(t)} Delta^{(t-1)} > 0 quad "(Max Growth)",
    C_"QP"^{(t)} & "nếu" Delta^{(t)} Delta^{(t-1)} < 0 quad "(Overshoot)",
    C_"QP"^{(t)} - alpha Delta^{(t)} & "trường hợp còn lại"
  )
$

Trong đó:
- $alpha$: Hệ số học (learning rate), đóng vai trò mồi (bootstrap) hoặc bổ sung động năng.
- $mu$: Hệ số kẹp bước nhảy tối đa (thường chọn $mu = 1.75$) để tránh bùng nổ khi độ cong parabol quá nhỏ.
- *Weight Decay*: Để chống tràn số (overflow) khi trọng số tiến tới vô cực, một lượng suy hao được cộng trực tiếp vào đạo hàm trước khi đưa vào hệ phương trình trên:
  $ Delta^{(t)} arrow.l Delta^{(t)} + lambda bold(w)^((t)) $

= Thực nghiệm

== Tập dữ liệu và tiền xử lý
Sử dụng tập dữ liệu FashionMNIST ($N_"train" = 60000, N_"test" = 10000$, $10$ phân lớp, ảnh xám $28 times 28$).

Thực nghiệm không sử dụng các kỹ thuật tăng cường dữ liệu (data augmentation). Lý do là việc biến đổi ảnh ngẫu nhiên sẽ tạo ra nhiễu, làm sai lệch mẫu số $(Delta^{(t-1)} - Delta^{(t)})$ của QuickProp. Điều này khiến thuật toán xấp xỉ parabol hoạt động không ổn định. Ảnh đầu vào chỉ được chuẩn hóa đơn giản:
$ x_"norm" = (x - mu) / sigma $
Với $mu = 0.2860$ và $sigma = 0.3530$ tính từ tập huấn luyện.

== Mô hình và khởi tạo
Mô hình `ParaConv` ($c_"base" = 16$) dùng kiến trúc thuần tích chập (All-convolutional) kết hợp hàm kích hoạt SiLU. Hàm SiLU được chọn vì nó trơn và dễ tính đạo hàm, giúp quá trình truyền ngược ổn định hơn.

Trọng số của mạng được khởi tạo theo phương pháp Xavier (Glorot):
$ bold(w) tilde cal(N)(0, 2 / (n_"in" + n_"out")) $
Khởi tạo Xavier an toàn hơn Kaiming (He) vì nó giữ cho phương sai nhỏ. Nếu khởi tạo trọng số quá lớn, bước nhảy của QuickProp (vốn tỷ lệ thuận với đạo hàm) có thể bị bùng nổ (exploding steps) ngay từ những epoch đầu.

== Chế độ huấn luyện
Để kết quả đáng tin cậy, mô hình được huấn luyện trong 30 epoch với 3 seed ngẫu nhiên cố định ($42, 3407, 1337$). Các thuật toán được so sánh trong hai chế độ:

1. Full-batch (Cập nhật 1 lần/epoch): Đây là cách làm gốc của QuickProp. Đạo hàm được cộng dồn trên toàn bộ tập dữ liệu rồi mới cập nhật trọng số, giúp triệt tiêu hoàn toàn nhiễu:
  $
    Delta^{(t)} = 1 / N_"train" sum_(i=1)^(N_"train") (partial ell^{(i)}) / (partial bold(w))
  $
2. Mini-batch (Cập nhật liên tục theo batch): Đây là cách huấn luyện phổ biến hiện nay, dùng để kiểm tra độ thực dụng của thuật toán.

Các thông số được giữ cố định (không dùng lịch trình giảm learning-rate):
- SGD (Baseline): Dùng momentum $beta = 0.9$.
- QuickProp: Hệ số mồi $alpha = 0.01$, giới hạn bước nhảy $mu = 1.75$, hệ số suy hao (weight decay) $lambda$ nhỏ.
Cả hai thuật toán đều dùng chung kỹ thuật cắt xén đạo hàm (Gradient Clipping) chuẩn $L_2$ để đảm bảo công bằng:
$ Delta^{(t)} arrow.l Delta^{(t)} dot min(1, c_"max" / (||Delta^{(t)}||_2)) $

== Tiêu chí đánh giá
Thực nghiệm đánh giá dựa trên hai yếu tố:
- Tốc độ và độ ổn định: Dựa trên giá trị hàm mất mát $cal(L)$ và độ lớn đạo hàm $||Delta^{(t)}||_2$ lúc huấn luyện.
- Khả năng dự đoán thực tế: Dựa trên Accuracy và Macro F1-score trên tập kiểm tra.

= Kết quả và Thảo luận

== Quỹ đạo hội tụ và Diện tích dưới đường cong (Loss AUC)
Để đo xem mô hình học nhanh hay chậm trong cả quá trình, ta tính diện tích dưới đường cong hàm mất mát (AUC). AUC càng nhỏ chứng tỏ mô hình học càng nhanh:
$
  "AUC" = integral_0^T cal(L)(bold(w)^((t))) d t approx sum_(t=1)^T cal(L)(bold(w)^((t)))
$

#figure(
  image("figures/bar_loss_auc.png", width: 80%),
  caption: [So sánh Loss AUC giữa QuickProp và SGD trên hai chế độ huấn luyện.],
)

#table(
  columns: 3,
  align: (left, center, center),
  [Chế độ (Regime)], [AUC (QuickProp)], [AUC (SGD)],
  [Full-batch ($T=150$)],
  [*$132.15 plus.minus 3.53$*],
  [$312.65 plus.minus 31.58$],

  [Mini-batch ($T=30$)], [$17.84 plus.minus 0.96$], [*$10.72 plus.minus 0.48$*],
)

Ở chế độ Full-batch, QuickProp xấp xỉ parabol rất chuẩn xác. Nhờ đó, tổng sai số của QuickProp thấp hơn SGD tới $2.4$ lần.

== Tốc độ đạt ngưỡng mục tiêu
Gọi $T_(75%)$ là số epoch ít nhất cần thiết để mô hình đạt độ chính xác $75%$ trên tập kiểm tra:
$ T_(75%) = min { t mid "Acc"(bold(w)^((t))) >= 0.75 } $

- Full-batch: QuickProp chỉ mất $T_(75%)^"QP" = 62.3 plus.minus 6.0$ epoch. Trong khi đó, SGD không thể chạm mốc này trong suốt 150 epoch ($T_(75%)^"SGD" -> oo$).
- Mini-batch: Cả hai phương pháp đều qua mốc này rất nhanh, với $T_(75%)^"QP" = 2.3 plus.minus 0.6$ và $T_(75%)^"SGD" = 3.0 plus.minus 0.0$.

#figure(
  image("figures/bar_epochs_to_75pct_accuracy.png", width: 80%),
  caption: [Số lượng Epoch cần thiết để mô hình đạt ngưỡng Accuracy 75%.],
)

== Độ chính xác cuối cùng (Final Performance)
Bảng dưới đây thống kê độ chính xác tại epoch cuối cùng (trung bình $\pm$ độ lệch chuẩn trên 3 seeds):

#table(
  columns: 4,
  align: (left, left, center, center),
  [Chế độ], [Thuật toán], [Val Accuracy], [Val F1 (Macro)],
  [Full-batch],
  [*QuickProp*],
  [*$0.8500 plus.minus 0.0041$*],
  [*$0.8490 plus.minus 0.0042$*],

  [Full-batch],
  [SGD],
  [$0.5494 plus.minus 0.0739$],
  [$0.5068 plus.minus 0.0905$],

  [Mini-batch],
  [QuickProp],
  [$0.7897 plus.minus 0.0108$],
  [$0.7853 plus.minus 0.0124$],

  [Mini-batch],
  [*SGD*],
  [*$0.9050 plus.minus 0.0051$*],
  [*$0.9048 plus.minus 0.0046$*],
)

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  figure(
    image("figures/fullbatch_val_accuracy.png"),
    caption: [Validation Accuracy (Full-batch)],
  ),
  figure(
    image("figures/minibatch_val_accuracy.png"),
    caption: [Validation Accuracy (Mini-batch)],
  ),
)

== Thảo luận
Hiện tượng QuickProp thắng ở Full-batch nhưng thua ở Mini-batch có thể được giải thích như sau:
1. Ở chế độ Full-batch (Không có nhiễu): Đạo hàm $Delta^{(t)}$ được tính cực kỳ chuẩn xác trên toàn bộ tập dữ liệu $Omega$. Lúc này, mẫu số $(Delta^{(t-1)} - Delta^{(t)})$ phản ánh đúng độ cong của hàm số. Nhờ vậy, QuickProp có thể nhảy một bước chính xác đến ngay đáy parabol ($>>$ SGD).
2. Ở chế độ Mini-batch (Nhiễu nhiều): Đạo hàm tính trên từng batch nhỏ sẽ nhảy lung tung. Sự nhiễu loạn này phá vỡ cấu trúc trơn của parabol, khiến mẫu số dao động mạnh. Để không bị bùng nổ, QuickProp buộc phải liên tục kích hoạt cơ chế an toàn $C_"QP"^{(t)} > mu C^{(t-1)}$, ép bước nhảy bị hãm lại ở mức $C^{(t)} = mu C^{(t-1)}$. Việc "phanh" liên tục này làm thuật toán mất đà (chỉ đạt $78.97%$). Trong khi đó, SGD nhờ có quán tính (momentum $beta$) lại vượt qua được nhiễu và tiến sâu hơn ($90.50%$).


= Kết luận
Từ các kết quả trên, ta có thể rút ra:
- Thuật toán QuickProp của Scott Fahlman thực sự có cơ sở toán học vững chắc. Việc dùng parabol xấp xỉ đạo hàm giúp nó học cực kỳ nhanh và ổn định ở chế độ Full-batch.
- Tuy nhiên, điểm yếu chí mạng của QuickProp là quá nhạy cảm với nhiễu đạo hàm. Trong các bài toán học sâu hiện đại (vốn luôn dùng Mini-batch và chứa nhiều yếu tố ngẫu nhiên), SGD kèm momentum $beta$ vẫn là lựa chọn an toàn và hiệu quả hơn.

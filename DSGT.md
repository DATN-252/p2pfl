### SYSTEM PROMPT VÀ MÔ TẢ GIẢI THUẬT DSGTm-TV

**1. Bài toán mục tiêu (Problem Formulation)**
Hệ thống gồm $n$ tác tử (agents) kết nối qua mạng có hướng, thay đổi theo thời gian (time-varying directed networks). Mục tiêu là tối ưu hóa hàm mất mát toàn cục:


$$min_{x \in \mathbb{R}^p} f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)$$


Trong đó, $f_i(x) = \mathbb{E}_{\xi_i \sim \mathcal{D}_i}[\mathcal{L}(x; \xi_i)]$ là hàm chi phí cục bộ tại tác tử $i$, đánh giá trên tập dữ liệu riêng $\mathcal{D}_i$. Mỗi tác tử tính toán ước lượng đạo hàm ngẫu nhiên $g_i(x, \xi_i)$.

**2. Đồ thị giao tiếp và Ma trận trọng số (Communication Graphs & Mixing Matrices)**
Tại mỗi bước lặp $k$, đồ thị giao tiếp là $\mathbb{G}_k = ([n], \mathcal{E}_k)$. Quá trình trao đổi thông tin dựa trên cấu trúc **Push-Pull** với hai ma trận:

* **Ma trận $A_k$ (Row-stochastic / Kéo thông tin):** Dùng để cập nhật tham số mô hình. Tổng mỗi hàng bằng 1 ($A_k \mathbf{1} = \mathbf{1}$). $[A_k]_{ij} > 0$ nếu tác tử $i$ nhận thông tin từ $j$.
* **Ma trận $B_k$ (Column-stochastic / Đẩy thông tin):** Dùng để cập nhật biến theo dõi đạo hàm. Tổng mỗi cột bằng 1 ($\mathbf{1}^T B_k = \mathbf{1}^T$). $[B_k]_{ji} > 0$ nếu tác tử $i$ gửi thông tin đến $j$.

**3. Các biến số và Siêu tham số tại tác tử $i$**
Khác với các thuật toán truyền thống, DSGTm-TV cho phép các tác tử sử dụng **siêu tham số không đồng bộ (uncoordinated)**:

* $x_k^i \in \mathbb{R}^p$: Tham số mô hình cục bộ tại bước $k$.
* $y_k^i \in \mathbb{R}^p$: Biến theo dõi đạo hàm (Gradient tracker) tại bước $k$.
* $\alpha_i > 0$: Tốc độ học (stepsize) riêng biệt của tác tử $i$.
* $\beta_i \ge 0$: Hệ số động lượng Heavy-ball (momentum) riêng biệt của tác tử $i$.

**4. Cập nhật toán học cục bộ (Local Update Rules - Tương đương Phương trình 5a, 5b)**
Tại bước $k \ge 0$, mỗi tác tử $i \in [n]$ thực hiện hai cập nhật sau:

* **Cập nhật Trạng thái cục bộ (Consensus + Gradient Step + Momentum):**

$$x_{k+1}^i = \sum_{j=1}^n [A_k]_{ij} x_k^j - \alpha_i y_k^i + \beta_i (x_k^i - x_{k-1}^i)$$


* **Cập nhật Biến theo dõi đạo hàm (Gradient Tracking Update):**

$$y_{k+1}^i = \sum_{j=1}^n [B_k]_{ij} y_k^j + g_i(x_{k+1}^i, \xi_{k+1}^i) - g_i(x_k^i, \xi_k^i)$$



**5. Biểu diễn Ma trận tổng quát (Compact Form - Tương đương Phương trình 6a, 6b)**
Rất hữu ích khi vectorize code trong PyTorch/JAX.
Gọi $x_k = [x_k^1, \dots, x_k^n]^T \in \mathbb{R}^{n \times p}$ và $y_k = [y_k^1, \dots, y_k^n]^T \in \mathbb{R}^{n \times p}$.
Gọi $D_\alpha = \text{Diag}(\alpha_1, \dots, \alpha_n)$ và $D_\beta = \text{Diag}(\beta_1, \dots, \beta_n)$ là các ma trận đường chéo chứa siêu tham số.
Hệ thống được cập nhật theo dạng ma trận như sau:


$$x_{k+1} = A_k x_k - D_\alpha y_k + D_\beta (x_k - x_{k-1})$$

$$y_{k+1} = B_k y_k + g(x_{k+1}, \xi_{k+1}) - g(x_k, \xi_k)$$

---

**6. Pseudocode chi tiết để AI CLI khởi tạo bộ khung lập trình:**

> **Algorithm 1: The DSGTm-TV Algorithm**
> **Input:** Local cost function $f_i(x)$ for all agents $i \in [n]$.
> **Initialize:** > - Every agent $i \in [n]$ sets arbitrary initial vectors $x_{-1}^i, x_0^i \in \mathbb{R}^p$.
> * Evaluates local gradient estimate $y_0^i = g_i(x_0^i, \xi_0^i)$.
> * Sets uncoordinated stepsize $\alpha_i$ and heavy-ball momentum parameter $\beta_i$.
> 
> 
> **For iteration $k = 0, 1, 2, \dots$ do**
>     **Each agent $i \in [n]$ does:**
>     **1. Communication Step:**
>         - Choose the weights $[B_k]_{ji}$ for $j \in \mathcal{N}_{ik}^{out}$.
>         - Send $x_k^i$ and $[B_k]_{ji} y_k^i$ to out-neighbors $j \in \mathcal{N}_{ik}^{out}$.
>         - Receive $x_k^j$ and $[B_k]_{ij} y_k^j$ from in-neighbors $j \in \mathcal{N}_{ik}^{in}$.
>         - Choose the weights $[A_k]_{ij}$ for $j \in \mathcal{N}_{ik}^{in}$.
>     **2. Local State Update:**
>         - Update model: $x_{k+1}^i = \sum_{j=1}^n [A_k]_{ij} x_k^j - \alpha_i y_k^i + \beta_i (x_k^i - x_{k-1}^i)$.
>     **3. Obtain Gradient Estimate:**
>         - Query the stochastic oracle: $g_i(x_{k+1}^i, \xi_{k+1}^i)$.
>     **4. Gradient Tracking Update:**
>         - Update tracker: $y_{k+1}^i = \sum_{j=1}^n [B_k]_{ij} y_k^j + g_i(x_{k+1}^i, \xi_{k+1}^i) - g_i(x_k^i, \xi_k^i)$.
> **End For**
> **Output:** Optimal global decision $x^*$.
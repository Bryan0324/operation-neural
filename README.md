# 什麼是AI神經網路及AI的基本概念

---

### 🔸 什麼是人工智慧（AI）？

**人工智慧（Artificial Intelligence，簡稱 AI）** 是一種讓電腦或機器「模仿人類智慧」的技術。  
它可以讓電腦執行一些原本需要人腦才能完成的工作，例如：

- 認圖（像是人臉辨識）
- 聽懂語音（像 Siri、Google 助理）
- 玩遊戲（像 AlphaGo）
- 預測（像天氣預測、股市走勢）

AI 不只是程式碼，它是透過**學習資料**來改善自己的表現。

這次上課主要是認圖方面的專案

---

### 🔹 什麼是神經網路（Neural Network）？

神經網路是實現 AI 的一種**重要技術**。  
它是**模仿人腦運作方式**而設計出來的模型。

就像人腦由很多神經元（Neurons）連接起來，神經網路也是由**很多「節點」或「人工神經元」組成的網路**，這些節點會根據輸入資料來做出判斷。

---

### 🔸 神經網路的基本概念

一個神經網路大致分成三個部分：

1. **輸入層（Input Layer）**  
   - 負責接收資料（例如圖片的像素值、音訊的波形）
   - 每個節點對應一個特徵

2. **隱藏層（Hidden Layer）**  
   - 真正處理資料的地方
   - 可以有一層或多層
   - 每個節點會做數學運算，然後傳遞到下一層
   - 這些層會學會如何從輸入資料中找出**模式（pattern）**

3. **輸出層（Output Layer）**  
   - 輸出結果（例如：這是「狗」還是「貓」）

---

### 🔹 節點在做什麼？

每個節點會做這幾件事：

1. **接收前一層的輸入**
2. **加權相加（加上每個輸入對應的權重）**
3. **加上偏差值（bias）**
4. **丟進激活函數（activation function）產生輸出**
   - 激活函數可以讓神經網路變得「非線性」，有更強的表現力

動畫：https://www.youtube.com/embed/Aop4rGjMskI?si=F0vzRe9fSPPZOHyV

---

### 🔸 學習的方式：反向傳播

神經網路透過一種叫 **「反向傳播（Backpropagation）」** 的方法學習：

1. 比較預測值與正確答案的差距（也就是「誤差 (error)」）
2. 用誤差來調整每個權重（讓下一次表現更好）
3. 重複很多次，直到結果滿意為止

---

## ➕ 加法、✖️ 乘法（點乘、矩陣乘法）與 Hadamard 乘積說明

在人工智慧（AI）和神經網路中，**向量和矩陣的運算**是很基本也非常重要的工具。我們常會遇到以下幾種運算方式：

---

### 🔹 向量或矩陣的「加法」

矩陣或向量的加法非常直覺，就是「**同位置相加**」。

#### ✏️ 例子：
如果有兩個向量：

```
A = [1, 2, 3]  
B = [4, 5, 6]
```

它們相加就是：

```
A + B = [1+4, 2+5, 3+6] = [5, 7, 9]
```

✅ 前提：兩個向量或矩陣的**形狀（大小）要相同**。

---

### 🔸 向量的「點乘」（Dot Product）

點乘是**兩個向量**相乘後，得到一個「數字（純量）」的運算。這在神經網路中非常常見。

#### ✏️ 例子：
```
A = [1, 2, 3]  
B = [4, 5, 6]

A ⋅ B = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

這是一種「權重加總」的概念，就像神經元接收到的訊號強度。

---

### 🔹 矩陣的「標準乘法」（Matrix Multiplication）

矩陣乘法有點不直覺，但它是非常重要的運算。  
它不是單純的同位置相乘，而是**行乘列、加總後得到新數值**。

#### ✏️ 例子：
```
A = [[1, 2],     B = [[7, 8],
     [3, 4]]          [9, 10]]

A × B = [[1×7 + 2×9, 1×8 + 2×10],
         [3×7 + 4×9, 3×8 + 4×10]]

       = [[25, 28],
          [57, 64]]
```

✅ 前提是：A 的「列數」要等於 B 的「行數」。

---

### 🔸 Hadamard 乘積（逐元素乘法）

Hadamard 乘積是指**矩陣的同位置元素相乘**，結果還是一個矩陣。  
這跟矩陣的標準乘法不一樣！

#### ✏️ 例子：
```
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

A ⊙ B = [[1×5, 2×6],
         [3×7, 4×8]]

       = [[5, 12],
          [21, 32]]
```

✅ 前提：兩個矩陣大小要一樣。

---

### ✅ 總結比較表：

| 運算         | 需求大小相同 | 結果類型     | 特點                     |
|--------------|--------------|--------------|--------------------------|
| 加法         | ✅           | 矩陣或向量   | 同位置相加               |
| 點乘         | ✅（一維向量）| 數字（純量） | 元素相乘後再加總         |
| 矩陣乘法     | 不一定相同   | 矩陣         | 行乘列後加總（線性變換） |
| Hadamard 乘積 | ✅           | 矩陣         | 同位置元素相乘           |

---

## 🔍 神經網路的三大結構：輸入層、隱藏層、輸出層

一個神經網路就像是一個工廠，把「原料」變成「產品」，每一層就像是生產線的一段。以下是它的三個主要部分：

---

### 🔹 輸入層（Input Layer）

**功能：接收原始資料**

- 輸入層就像是「接收器」，是神經網路的起點。
- 每個節點（也叫神經元）對應一個輸入特徵。

#### ✅ 範例：
假如你有一張 28×28 像素的灰階圖片要辨識，那就是：
```
28 × 28 = 784 個像素
→ 輸入層會有 784 個神經元

```

這些數字會變成神經網路的「第一道訊號」。

---

### 🔸 隱藏層（Hidden Layer）

**功能：分析、處理、學習資料內的「模式」**

- 隱藏層是神經網路的「大腦」，它會對輸入資料進行數學運算。
- 神經網路的學習能力，幾乎都來自這些隱藏層。
- 你可以有 1 層、2 層，甚至上百層（深度學習就是指很多層的網路）。

#### ✅ 每個神經元做的事情：

1. 收到輸入層傳來的數值  
2. 每個輸入都乘上一個「權重」  
3. 把結果加總後，加上「偏差值（bias）」  
4. 套用一個「激活函數（activation function）」  
   → 這樣可以增加網路的非線性能力，像 ReLU、Sigmoid、tanh 等

#### 🔧 結果：
處理完的資料會傳到下一層（可以是下一個隱藏層，也可以是輸出層）

---

### 🔹 輸出層（Output Layer）

**功能：給出最終預測結果**

- 輸出層會根據隱藏層處理完的資訊，做出「決策」。
- 輸出層的結構通常根據任務而設計：

#### ✅ 常見例子：

- **二元分類（是 / 否）** → 1 個輸出（值在 0～1 之間）
- **多分類（貓 / 狗 / 馬）** → 3 個輸出（每個對應一個類別）
- **回歸問題（例如預測房價）** → 輸出是一個連續數值

輸出層也會使用激活函數，例如：
- **Sigmoid**：讓輸出在 0～1 之間（用於機率）
- **Softmax**：讓所有輸出加起來為 1，常用在多分類

---

### 🔸 總結流程圖（簡化版）：

```
輸入層（特徵）→ 隱藏層（處理）→ 輸出層（預測結果）
```

---

### 🔹 容易理解的比喻：

- 輸入層 = 眼睛看到東西
- 隱藏層 = 大腦思考處理
- 輸出層 = 嘴巴說出答案

---

## 把兩份程式碼（`matrix.h` 與 `neural_network_v1.cpp`）加上**註解**，重點放在：

1. 這段程式在做什麼？
2. 它為什麼這樣做？
3. 用生活或圖像的方式說明數學原理。

---

## **【matrix.h】— 負責「矩陣」的運算工具箱**

```cpp
// 引入標準工具
#include <bits/stdc++.h> // 包含幾乎所有常用 C++ 標頭檔
#include <cassert>
#include <random>
#include <chrono>
using namespace std;

// 我們定義一個叫 matrix 的類別（像是一個可以加減乘的表格）
template <typename T> class matrix {
public:
  // 建構函式：建立一個 m x n 的矩陣，內容預設都填上 init（通常是 0）
  matrix(unsigned int row = 2, unsigned int col = 2, T init = 0) {
    m = row;
    n = col;
    a = vector<vector<T>>(row, vector<T>(col, init));
  }

  // 這邊定義的是加法運算：這個矩陣 + 另一個矩陣
  matrix operator+(const matrix &b) {
    if (m ^ b.m || n ^ b.n) {
      cout << "Can't do addition\n";
      exit(0);
    }
    matrix res(m, n);
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < n; j++)
        res.a[i][j] = a[i][j] + b.a[i][j];
    return res;
  }

  // += 是「加上某個矩陣後更新自己」
  matrix& operator+=(const matrix &b) {
    if (m ^ b.m || n ^ b.n) {
      cout << "Can't do addition inplacement\n";
      exit(0);
    }
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < n; j++)
        a[i][j] += b.a[i][j];
    return *this;
  }

  // 減法也一樣的邏輯
  matrix operator-(const matrix &b) { ... }

  // 乘法：這邊是「每個數自己乘自己對應位置」→ 叫 Hadamard 乘積
  matrix operator*(T &b) { ... }

  // 標準矩陣乘法（這在神經網路裡非常重要！）
  matrix operator*(const matrix &b) {
    if (n ^ b.m) {
      cout << "Can't do multiplication\n";
      exit(0);
    }
    matrix res(m, b.n);
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < b.n; j++)
        for (unsigned int k = 0; k < n; k++)
          res.a[i][j] += a[i][k] * b.a[k][j];
    return res;
  }

  // hadamard 乘積：兩個矩陣大小一樣，對應位置相乘
  matrix hadamard(const matrix &b) { ... }

  // 隨機填入值：用來初始化神經網路的「權重」
  matrix& random(T min = 0, T max = 1) {
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<> dis(min, max);
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < n; j++)
        a[i][j] = dis(rng);
    return *this;
  }

  // 轉置矩陣：行變成列 → 用在反向傳播時計算權重更新
  matrix<T> transpose() {
    matrix<T> res(n, m);
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < n; j++)
        res[j][i] = a[i][j];
    return res;
  }

  // 其他輔助功能：列數、欄數、列印、平方、加總…
};
```

---

## **【neural_network_v1.cpp】— 整個神經網路的大腦**

```cpp
// 引入所有必要的程式庫
#include <bits/stdc++.h>
#include "matrix.h"  // 引入剛剛我們自己寫的「矩陣工具」

// 建立神經網路類別
template <typename T = float> class neural_network {
public:
  float learn_rate = 0.01;  // 學習速率，控制每次調整的「幅度」

  // 建構函式：給定輸入大小，建立網路結構
  neural_network(unsigned int input_size, float rate = 0.01) {
    insize = input_size;
    learn_rate = rate;
    output.push_back(matrix<T>(1, input_size, 0));  // 初始輸入是全 0
  }

  // 前向傳播：資料從輸入層流經隱藏層，最後輸出
  matrix<T> forward(matrix<T> mat) {
    insert(mat);  // 將輸入放進網路第一層
    for (unsigned int i = 0; i < size(); i++) {
      output[i + 1] = output[i] * weight[i]; // 計算加權輸出
      output[i + 1] += bias[i];              // 加上偏差（bias）
      sigmoid(output[i + 1]);                // 套用激活函數
    }
    return output.back();  // 回傳最終輸出
  }

  // 反向傳播：根據錯誤修正權重與偏差
  void backprop(const matrix<T>& expected) {
    matrix<T> delta = output.back() - expected;  // 找出錯誤
    for (int i = size() - 1; i >= 0; --i) {
      bias[i] -= delta * learn_rate;
      weight[i] -= output[i].transpose() * delta * learn_rate;
      matrix<T> d_sigmoid = output[i].hadamard(matrix<T>(output[i].row(), output[i].col(), 1) - output[i]);
      delta = (delta * weight[i].transpose()).hadamard_inplace(d_sigmoid);
    }
  }

  // 訓練：給多筆資料、反覆訓練多次
  void train(vector<matrix<T>> input, vector<matrix<T>> expectation, unsigned int epoch = 10) {
    for (unsigned int cnt = epoch; cnt--; ) {
      for (unsigned int i = 0; i < input.size(); ++i) {
        forward(input[i]);
        backprop(expectation[i]);
      }
      cout << "Epoch: " << epoch - cnt << '/' << epoch << "\tError: " << error(output.back(), expectation.back()) << '\n';
    }
  }

  // 其他功能：新增層、儲存權重、讀取模型檔案等
};
```

---

## **main() 主程式邏輯簡介**

```cpp
int main() {
  // 建立 1000 筆訓練資料 x 與標籤 y（手寫數字圖像）
  vector<matrix<float>> x(tests, matrix<float> (1, 28*28));
  vector<matrix<float>> y(tests, matrix<float> (1,10,0));

  // 建立神經網路，輸入是 28*28 個像素（784 個值）
  NN<float> nn(28*28);
  nn.pb(48);  // 隱藏層1：48 個神經元
  nn.pb(32);  // 隱藏層2：32 個神經元
  nn.pb(10);  // 輸出層：對應數字 0~9

  // 讀取圖像檔案與答案
  ifstream file_x("train_x.txt"), file_y("train_y.txt");
  for (unsigned int i = 0; i < tests; ++i) {
    for (unsigned int j = 0; j < 28*28; ++j) {
      file_x >> x[i][0][j];  // 讀像素
      x[i][0][j] /= 255;     // 標準化（0~1之間）
    }
    file_y >> temp;     // 讀答案
    y[i][0][temp] = 1;  // one-hot 編碼
  }

  // 開始訓練！
  nn.train(x, y, 10);

  // 測試模型的準確率
  ...
}
```

---


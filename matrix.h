// initialize
#include <bits/stdc++.h>
#include <cassert>
#include <random>
#include <chrono>
using namespace std;

// custom containers
template <typename T> class matrix{
public:
  // n-->
  //m[   ]
  //|[   ]
  //v[   ]

  matrix(unsigned int row = 2, unsigned int col = 2, T init = 0){
    m = row;
    n = col;
    a = vector<vector<T>>(row,vector<T>(col,init));
  }
  ~matrix(){}

  matrix operator+(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do addition\n";exit(0); }
    matrix res(m,n);
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; j++){
        res.a[i][j] = a[i][j] + b.a[i][j];
      }
    }
    return res;
  }

  matrix& operator+=(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do addition inplacement\n";exit(0); }
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; ++j){
        a[i][j] += b.a[i][j];
      }
    }
    return *this;
  }

  matrix operator-(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do subtraction\n";exit(0); }
    matrix res(m,n);
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; ++j){
        res.a[i][j] = a[i][j] - b.a[i][j];
      }
    }
    return res;
  }

  matrix& operator-=(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do subtraction inplacement\n";exit(0); }
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; ++j){
        a[i][j] -= b.a[i][j];
      }
    }
    return *this;
  }

  matrix operator*(T &b){  // O(n^2)
    matrix res(m,n,b);
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; ++j){
        res.a[i][j] *= a[i][j];
      }
    }
    return res;
  }

  matrix operator*(const matrix &b){  // O(n^3)
    if(n ^ b.m){ cout << "Can't do multiplication\n";exit(0); }
    matrix res(m,b.n);
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < b.n; j++){
        for(unsigned int k = 0; k < n; ++k){
          res.a[i][j] += a[i][k] * b.a[k][j];
        }
      }
    }
    return res;
  }

  matrix& operator*=(T &b){  // O(n^2)
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; ++j){
        a[i][j] *= b;
      }
    }
    return *this;
  }

  matrix operator*=(const matrix &b){  // O(n^3)
    if(n ^ b.m){ cout << "Can't do multiplication inplacement\n";exit(0); }
    return *this = *this * b;
  }

  vector<T>& operator[](unsigned int k){ return a[k]; } // O(1)

  inline unsigned int col(){ return n; }  // O(1)

  matrix hadamard(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do hadamard product\n";exit(0); }
    matrix res(m,n);
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; res.a[i][j] = a[i][j] * b.a[i][j], ++j);
    }
    return res;
  }

  matrix& hadamard_inplace(const matrix &b){  // O(n^2)
    if(m ^ b.m | n ^ b.n){ cout << "Can't do hadamard inplacement\n";exit(0); }
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; ++j){
        a[i][j] *= b.a[i][j];
      }
    }
    return *this;
  }

  void fill(T k){
    for(unsigned int i = 0; i < m; ++i){
      fill(a[i].begin(), a[i].end(), k);
    }
  }

  void fill(const matrix &b){
    for(unsigned int i = 0; i < m; ++i){
      for(unsigned int j = 0; j < n; a[i][j] = b.a[i][j], ++j);
    }
  }

  void print(){  // O(n^2)
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n-1; j++){
        cout << a[i][j] << ' ';
      }
      cout << a[i][n-1] << '\n';
    }
  }

  matrix& random(T min = 0, T max = 1){  // O(n^2)
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution<> dis(min, max);
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; j++){
        a[i][j] = dis(rng);
      }
    }
    return *this;
  }

  inline unsigned int row(){ return m; }  // O(1)

  matrix<T> submatrix(unsigned int x, unsigned int y, unsigned int k, unsigned int h){
    matrix<T> res(k,h);
    for(unsigned int i = 0; i < k; ++i){
      for(unsigned int j = 0; j < h; ++j){
        res[i][j] = a[x+i][y+j];
      }
    }
    return res;
  }

  inline pair<unsigned int, unsigned int> size(){ return make_pair(m, n); }  // O(1)

  matrix& square(){  // O(n^2)
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; j++){
        a[i][j] *= a[i][j];
      }
    }
    return *this;
  }

  T sum(){  // O(n^2)
    T res = 0;
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; j++){
        res += a[i][j];
      }
    }
    return res;
  }

  matrix<T> transpose(){  // O(n^2)
    matrix<T> res(n,m);
    for(unsigned int i = 0; i < m; i++){
      for(unsigned int j = 0; j < n; j++){
        res[j][i] += a[i][j];
      }
    }
    return res;
  }

private:
  unsigned int m, n;
  vector<vector<T>> a;
};
template <typename T> ostream& operator<<(ostream& os, matrix<T>& m){  // O(n^2)
  for(unsigned int i = 0; i < m.row(); os << '\n', ++i){
    os << m[i].front();
    for(unsigned int j = 1; j < m.col(); ++j){
    	os << ' ' << m[i][j];
	}
  }
  return os;
}

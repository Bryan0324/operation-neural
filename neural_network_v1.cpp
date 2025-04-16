// smarter compiler
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx,avx2,fma,bmi,bmi2,lzcnt,popcnt")
// initialize
#include <bits/stdc++.h>
#include <cassert>
#include <random>
#include <chrono>
using namespace std;
#define fastio() std::cin.tie(0) -> std::ios_base::sync_with_stdio(0);
// easier classes names
#define ui unsigned int
#define ll long long
#define ull unsigned long long
#define ld long double
#define ff first
#define ss second
#define mp make_pair
#define pf push_front
#define pb push_back
#define eb emplace_back
#define popcount(n) __builtin_popcount(n)
// operator overloading
template <typename T> istream& operator>>(istream& is, vector<T>& v){  // O(n)
  for (unsigned int i = 0; i < v.size(); ++i){
    is >> v[i];
  }
  return is;
}
template <typename T> ostream& operator<<(ostream& os, const vector<T>& v){  // O(n)
  os << v.front();
  for (unsigned int i = 1; i < v.size(); os << ' ' << v[i++]);
  return os;
}

// custom containers
#include "matrix.h"

#define NN neural_network  // Warning: requires class matrix & should be used as an expansion
template <typename T = float> class neural_network{
public:
  // sigmoid( current output * current weight + current bias ) = next output
  // weight  * [0][1] ... [n-2][n-1]
  // bias    + [0][1] ... [n-2][n-1]
  //       -------------------------
  // output [0][1][2] ... [n-1][ n ]--> final output

  float learn_rate = 0.01;

  neural_network(unsigned int input_size, float rate = 0.01){
    if(typeid(T) != typeid(float) && typeid(T) != typeid(double)){ cout << "class neural_network only have float & double support rn\n";exit(0); }
    insize = input_size;
    learn_rate = rate;
    output.push_back(matrix<T> (1,input_size,0));
  }

  neural_network(string load_file){
    if(typeid(T) != typeid(float) && typeid(T) != typeid(double)){ cout << "class neural_network only have float & double support rn\n";exit(0); }
    load(load_file);
  }

  void backprop(const matrix<T>& expected){
    matrix<T> delta = output.back() - expected;
    for(int i = size()-1; i >= 0; --i){
      bias[i] -= delta * learn_rate;  // bias += σ'(z)
      weight[i] -= output[i].transpose() * delta * learn_rate;  // weights += prev.Y.T * σ'(z)
      matrix<T> d_sigmoid = output[i].hadamard(matrix<T>(output[i].row(), output[i].col(), 1) - output[i]);  // σ'(z) = σ(z) * (1 - σ(z))
      delta = (delta * weight[i].transpose()).hadamard_inplace(d_sigmoid);  // delta = (delta * prev.W.T) x σ'(z);
    }
  }

  void change_input(unsigned int input_size){  // O(1)
    insize = input_size;
    clear();
  }

  void clear(){  // O(1)
    bias.clear(), output.clear(), weight.clear(), output.push_back(matrix<T> (1,insize,0));
  }

  matrix<T> forward(matrix<T> mat){  // O(n^4)
    insert(mat);
    for(unsigned int i = 0; i < size(); output[i+1] = output[i] * weight[i], output[i+1] += bias[i], sigmoid(output[i+1]), ++i);
    return output.back();
  }

  T error(matrix<T> &exp) {
    return error(output.back(), exp);
  }

  inline unsigned int input_size(){ return insize; }  // O(1)

  inline void insert(matrix<T> mat, unsigned int layer = 0){ output[layer].fill(mat); }  // O(n^2)

  void load(string fname){  // O(n^4)
    unsigned int i, tempui;
    ifstream ifile(fname);
    ifile >> tempui >> learn_rate >> i;
    change_input(tempui);
    for(; i--;){
      ifile >> tempui;
      push_back(tempui);
      for(unsigned int j = 0; j < weight.back().row(); ++j){
        for(unsigned int k = 0; k < weight.back().col(); ifile >> weight.back()[j][k++]);
      }
      for(unsigned int j = 0; j < bias.back().col(); ifile >> bias.back()[0][j++]);
    }
  }

  void pop_back(){ bias.pop_back(), output.pop_back(), weight.pop_back(); }  // O(1)

  void push_back(unsigned int nodes){  // O(1)
    bias.push_back(matrix<T> (1,nodes,0));
    weight.push_back(matrix<T> (output.back().col(),nodes,0));
    weight.back().random(-0.5,0.5);
    output.push_back(matrix<T> (1,nodes,0));
  }

  inline matrix<T> recall(unsigned int layer = 0){ return output[layer]; }  // O(1)

  void save(string fname){  // O(n^4)
    ofstream ofile(fname);
    ofile << insize << ' ' << learn_rate << ' ' << size() << '\n';
    for(unsigned int i = 0; i < weight.size(); ofile << bias[i++][0].back() << '\n'){
      ofile << weight[i].col() << '\n';
      for(unsigned int j = 0; j < weight[i].row(); ofile << weight[i][j].back() << '\n', j++){
        for(unsigned int k = 0; k < weight[i].col()-1; ofile << weight[i][j][k++] << ' ');
      }
      for(unsigned int j = 0; j < bias[i].col()-1; ofile << bias[i][0][j++] << ' ');
    }
    ofile.close();
  }

  inline unsigned int size(){ return weight.size(); }  // O(1)

  void train(matrix<T> input, matrix<T> expectation, unsigned int epoch = 1){  // O(n^4)
    for(unsigned int cnt = epoch; cnt--; ){
      forward(input);
      cout << "Epoch: " << epoch - cnt << '/' << epoch  << "\tError: " << error(output.back(), expectation) << '\n';
      backprop(expectation);
    }
  }

  void train(vector<matrix<T>> input, vector<matrix<T>> expectation, unsigned int epoch = 10){  // O(n^5)
    for(unsigned int cnt = epoch; cnt--; ){
      for(unsigned int i = 0; i < input.size(); ++i){
        forward(input[i]);
        backprop(expectation[i]);
      }
      cout << "Epoch: " << epoch - cnt << '/' << epoch << "\tError: " << error(output.back(), expectation.back()) << '\n';
    }
  }

private:
  unsigned int insize = 0;
  vector<matrix<T>> bias;
  vector<matrix<T>> output;
  vector<matrix<T>> weight;

  T error(matrix<T>& out, matrix<T>& exp) {
    return (out - exp).square().sum() / out.col();
  }

  template <typename S> inline T sigmoid(S n){ return (T)1 / (exp(-n) + 1); }  // O(1)

  matrix<T>& sigmoid(matrix<T>& m){  // O(n^2)
    for(unsigned int i = 0; i < m.row(); i++){
      for(unsigned int j = 0; j < m.col(); j++){
        m[i][j] = (T)1 / (exp(-m[i][j]) + 1);
      }
    }
    return m;
  }
};

// template ends here

int main(){
  //fastio();
  const static unsigned int tests = 1000;
  long long temp;
  cout << "Building test tensors...\n";
  vector<matrix<float>> x(tests, matrix<float> (1, 28*28)), y(tests, matrix<float> (1,10,0));
  cout << "Test tensors built\nConsructing neural network...\n";
  NN<float> nn(28*28); nn.pb(48), nn.pb(32), nn.pb(10);
  ifstream file_x("train_x.txt"), file_y("train_y.txt");
  cout << "Neural network completed\nInserting files...\n";
  for(unsigned int i = 0; i < tests; ++i){
    for(unsigned int j = 0; j < 28*28; ++j){
      file_x >> x[i][0][j];
      x[i][0][j] /= 255;
    }
    file_y >> temp;
    y[i][0][temp] = 1;
  }
  cout << "Files successfully inserted\nTraining...\n";
  nn.train(x,y,10);
  nn.save("mnist_bot");


  int tries = 1000;
  long long ans, opt = 0;
  double accuracy = 0;
  matrix<float> tx(1,28*28), ty(1,10);
  // NN<float> nn("mnist_bot");
  ifstream test_x("test_x.txt"), test_y("test_y.txt");
  cout << "Testing...\n";
  for(int i = tries; i--; ){
    for(unsigned int j = 0; j < 28*28; test_x >> tx[0][j], tx[0][j] /= 255, ++j);
    ty = nn.forward(tx);
    for(unsigned int j = 0; j < 10; ty[0][opt] <= ty[0][j] ? opt = j : opt, ++j);
    test_y >> ans;
    // cout << "\nExp: " << ans << "\tGet: " << opt << '\n';
    if(ans == opt){
      accuracy += 1;
    }
  }
  cout << "Accuracy: " << accuracy / tries << '\n';
  return 0;

}

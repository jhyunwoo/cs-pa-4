#include "gpt_mini.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using namespace std;

namespace {

const string kWeightRoot = "/tmp/cs_pa4/weights/";
const string kAnswerLayerPath = "/tmp/cs_pa4/answer/layer_final.out";

string weight_path(const string &name) { return kWeightRoot + name + ".txt"; }

void ensure_directory(const string &path) {
  struct stat st{};
  if (stat(path.c_str(), &st) == 0) {
    if (!S_ISDIR(st.st_mode)) {
      throw runtime_error("Path exists but is not a directory: " + path);
    }
    return;
  }
  if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    throw runtime_error("Failed to create directory " + path + ": " +
                        std::strerror(errno));
  }
}

string shape_string(int rows, int cols) {
  ostringstream oss;
  oss << "[" << rows << "x" << cols << "]";
  return oss.str();
}

float *load_weights(const string &name, int expected_rows, int expected_cols,
                    double &total_bytes) {
  const string path = weight_path(name);
  ifstream in(path);
  if (!in) {
    throw runtime_error("Failed to open weight file: " + path);
  }

  int rows = 0;
  int cols = 0;
  if (!(in >> rows >> cols)) {
    throw runtime_error("Failed to read header for " + path);
  }
  if (rows != expected_rows || cols != expected_cols) {
    ostringstream err;
    err << "Shape mismatch for " << name << ": expected "
        << shape_string(expected_rows, expected_cols) << ", found "
        << shape_string(rows, cols);
    throw runtime_error(err.str());
  }

  size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  float *buffer = new float[count];
  for (size_t i = 0; i < count; ++i) {
    if (!(in >> buffer[i])) {
      delete[] buffer;
      throw runtime_error("Unexpected EOF while reading " + path);
    }
  }

  size_t bytes = count * sizeof(float);
  total_bytes += static_cast<double>(bytes);

  double mb = bytes / (1024.0 * 1024.0);
  double gb = mb / 1024.0;

  auto old_flags = cout.flags();
  auto old_prec = cout.precision();
  cout.setf(ios::fixed, ios::floatfield);
  cout << setprecision(3) << "Load " << name << " " << shape_string(rows, cols)
       << " <- " << mb << " MB (" << gb << " GB)" << endl;
  cout.flags(old_flags);
  cout.precision(old_prec);

  return buffer;
}

pair<double, size_t> accumulate_layer_diff(const string &reference_path,
                                           const string &output_path) {
  ifstream ref(reference_path);
  ifstream out(output_path);
  if (!ref) {
    throw runtime_error("Failed to open reference layer file: " +
                        reference_path);
  }
  if (!out) {
    throw runtime_error("Failed to open output layer file: " + output_path);
  }

  int ref_rows = 0, ref_cols = 0;
  int out_rows = 0, out_cols = 0;
  if (!(ref >> ref_rows >> ref_cols)) {
    throw runtime_error("Failed to read header from reference file: " +
                        reference_path);
  }
  if (!(out >> out_rows >> out_cols)) {
    throw runtime_error("Failed to read header from output file: " +
                        output_path);
  }
  if (ref_rows != out_rows || ref_cols != out_cols) {
    ostringstream err;
    err << "Layer output shape mismatch: reference "
        << shape_string(ref_rows, ref_cols) << " vs output "
        << shape_string(out_rows, out_cols);
    throw runtime_error(err.str());
  }

  size_t count = static_cast<size_t>(ref_rows) * static_cast<size_t>(ref_cols);
  double total_diff = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double ref_val = 0.0;
    double out_val = 0.0;
    if (!(ref >> ref_val) || !(out >> out_val)) {
      throw runtime_error("Failed to read layer value at index " +
                          to_string(i));
    }
    total_diff += std::fabs(ref_val - out_val);
  }

  return {total_diff, count};
}

} // namespace

int main(int argc, char **argv) {
  bool verbose = false;
  const string output_dir = "output";
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "-verbose" || arg == "--verbose") {
      verbose = true;
    } else {
      cerr << "Unknown argument: " << arg << endl;
      cerr << "Usage: " << argv[0] << " [-verbose]" << endl;
      return 1;
    }
  }
  if (verbose)
    ensure_directory(output_dir);

  const unsigned int num_steps = 40;
  const int vocab = 4096;
  const int d_model = 2048;
  const int n_head = 1;
  const int d_ff = 2560;
  const int n_layer = 1;

  double total_weight_bytes = 0.0;

  float *embed_weights =
      load_weights("embed", vocab, d_model, total_weight_bytes);
  float *lm_head_weights =
      load_weights("lm_head", d_model, vocab, total_weight_bytes);

  vector<GPTMini::BlockWeights> block_weights;
  block_weights.reserve(n_layer);
  for (int i = 0; i < n_layer; ++i) {
    GPTMini::BlockWeights bw;
    string prefix = "block" + to_string(i) + ".";
    bw.Wq = load_weights(prefix + "Wq", d_model, d_model, total_weight_bytes);
    bw.Wk = load_weights(prefix + "Wk", d_model, d_model, total_weight_bytes);
    bw.Wv = load_weights(prefix + "Wv", d_model, d_model, total_weight_bytes);
    bw.Wo = load_weights(prefix + "Wo", d_model, d_model, total_weight_bytes);
    bw.fc1 = load_weights(prefix + "fc1", d_model, d_ff, total_weight_bytes);
    bw.fc2 = load_weights(prefix + "fc2", d_ff, d_model, total_weight_bytes);
    block_weights.push_back(bw);
  }

  {
    auto old_flags = cout.flags();
    auto old_prec = cout.precision();
    cout.setf(ios::fixed, ios::floatfield);
    double total_mb = total_weight_bytes / (1024.0 * 1024.0);
    double total_gb = total_mb / 1024.0;
    cout << setprecision(3) << "Total weight memory -> " << total_mb << " MB ("
         << total_gb << " GB)" << endl;
    cout.flags(old_flags);
    cout.precision(old_prec);
  }

  GPTMini model(vocab, d_model, n_head, d_ff, n_layer, embed_weights,
                lm_head_weights, block_weights);
  if (verbose) {
    model.enable_layer_dumping(output_dir);
  }
  vector<int> context = {1, 5, 10};

  cout << "Initial context: ";
  for (auto t : context)
    cout << t << " ";
  cout << endl;

  auto start = chrono::high_resolution_clock::now();

  for (unsigned int step = 0; step < num_steps; step++) {
    int next = model.generate_next(context);
    context.push_back(next);
    cout << "Step " << step << " -> next token: " << next << endl;
  }

  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;
  double tokens_per_sec = num_steps / elapsed.count();

  cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
  cout << "Tokens per second: " << tokens_per_sec << " tokens/sec" << endl;

  if (verbose) {
    const string reference_file = kAnswerLayerPath;
    const string output_file = output_dir + "/layer_final.out";
    auto [layer_diff, element_count] =
        accumulate_layer_diff(reference_file, output_file);
    double mean_diff = element_count > 0
                           ? layer_diff / static_cast<double>(element_count)
                           : 0.0;
    auto old_flags = cout.flags();
    auto old_prec = cout.precision();
    cout.setf(ios::fixed, ios::floatfield);
    cout << setprecision(6)
         << "Sum of |answer - output| over layer_final.out: " << layer_diff
         << endl;
    cout << setprecision(9) << "Mean absolute error per element: " << mean_diff
         << endl;
    if (layer_diff < 1.0) {
      cout << "=== VALID ===" << endl;
    } else {
      cout << "=== INVALID ===" << endl;
    }
    cout.flags(old_flags);
    cout.precision(old_prec);
  }

  return 0;
}
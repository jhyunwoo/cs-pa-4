#pragma once

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace utils {

inline void enable_layer_dumping(bool &dump_enabled, std::string &dump_dir,
                                 const std::string &directory) {
  dump_enabled = true;
  dump_dir = directory;
}

inline void dump_layer_output(bool dump_enabled, const std::string &dump_dir,
                              const float *data, int rows, int cols) {
  if (!dump_enabled)
    return;

  std::string filename = dump_dir + "/layer_final.out";
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open layer dump file: " + filename);
  }

  out.setf(std::ios::fixed, std::ios::floatfield);
  out << rows << " " << cols << '\n';
  out << std::setprecision(8);

  const int total = rows * cols;
  for (int i = 0; i < total; ++i) {
    out << data[i];
    if ((i + 1) % cols == 0) {
      out << '\n';
    } else {
      out << ' ';
    }
  }
}

} // namespace utils
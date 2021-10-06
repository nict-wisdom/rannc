//
// Created by Masahiro Tanaka on 2018-12-10.
//
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include "Common.h"
#include "graph/ir.h"

namespace {
static int REF_LENGTH = 16;
static const char* REF_PREF = "ref_";

void gen_random(char* s, const int len) {
  static const char alphanum[] =
      "0123456789"
      //                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  s[len] = 0;
}
} // namespace

namespace rannc {
const std::string RANNC_CONF_DIR = ".pyrannc";

std::string generateRef() {
  char id[REF_LENGTH + 1];
  gen_random(id, REF_LENGTH);
  return std::string(REF_PREF) + std::string(id);
}

std::string generateName(const std::string& prefix) {
  char id[REF_LENGTH + 1];
  gen_random(id, REF_LENGTH);
  return prefix + std::string(id);
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.push_back(item);
    }
  }
  return elems;
}

std::vector<std::string> split(const std::string& s, const std::string& delim) {
  std::vector<std::string> ret;
  for (size_t i = 0, n; i <= s.length(); i = n + 1) {
    n = s.find_first_of(delim, i);
    if (n == std::string::npos)
      n = s.length();
    std::string tmp = s.substr(i, n - i);
    ret.push_back(tmp);
  }
  return ret;
}

bool begins_with(const std::string& str, const std::string& pattern) {
  if (str.size() >= pattern.size()) {
    return std::equal(std::begin(pattern), std::end(pattern), std::begin(str));
  }
  return false;
}

bool ends_with(const std::string& str, const std::string& pattern) {
  return str.size() >= pattern.size() &&
      str.find(pattern, str.size() - pattern.size()) != std::string::npos;
}

bool passedForBackward(const IRScalarType scalar_type) {
  switch (scalar_type) {
    case IRScalarType::FLOAT:
      return true;
    case IRScalarType::NONE:
    case IRScalarType::NUMBER:
    case IRScalarType::INT:
    case IRScalarType::BOOL:
    case IRScalarType::DEVICE:
      return false;
  }
}

bool passedForBackward(const IRTensorElemType tensor_elem_type) {
  switch (tensor_elem_type) {
    case IRTensorElemType::FLOAT:
    case IRTensorElemType::HALF:
    case IRTensorElemType::BFLOAT16:
    case IRTensorElemType::DOUBLE:
    case IRTensorElemType::UNDEF:
      return true;
    case IRTensorElemType::INT:
    case IRTensorElemType::LONG:
    case IRTensorElemType::BOOL:
      return false;
  }
}

bool passedForBackward(const IRType& type) {
  auto base_type = type.getBaseType();
  switch (base_type) {
    case IRBaseType::SCALAR:
      return passedForBackward(type.getScalarType());
    case IRBaseType::TENSOR:
      return passedForBackward(type.getTensorElemType());
    case IRBaseType::LIST: {
      const auto& list_type = type.getListType();
      if (list_type == IRListType::TENSOR || list_type == IRListType::GENERIC) {
        for (const auto& et : type.getCompoundTypes()) {
          if (!passedForBackward(et)) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
    case IRBaseType::TUPLE: {
      const auto& elem_types = type.getCompoundTypes();
      bool ret = true;
      for (const auto& et : elem_types) {
        ret &= passedForBackward(et);
        if (!ret)
          return false;
      }
      return true;
    }
    case IRBaseType::STRING:
      return false;
    case IRBaseType::OPTIONAL:
      return false;
    case IRBaseType::NONE:
      return false;
    case IRBaseType::FUNCTION:
      return false;
  }
}

bool isTensorOrTensorList(const IRType& type) {
  auto base_type = type.getBaseType();
  switch (base_type) {
    case IRBaseType::SCALAR:
      return false;
    case IRBaseType::TENSOR:
      return true;
    case IRBaseType::LIST: {
      const auto& list_type = type.getListType();
      return list_type == IRListType::TENSOR;
    }
    case IRBaseType::TUPLE: {
      const auto& elem_types = type.getCompoundTypes();
      bool ret = true;
      for (const auto& et : elem_types) {
        ret &= isTensorOrTensorList(et);
        if (!ret)
          return false;
      }
      return true;
    }
    case IRBaseType::STRING:
      return false;
    case IRBaseType::OPTIONAL:
      return false;
    case IRBaseType::NONE:
      return false;
  }
}

fs::path getHomeDir() {
  const char* home_dir;
  if ((home_dir = getenv("HOME")) == nullptr) {
    home_dir = getpwuid(getuid())->pw_dir;
  }
  return home_dir;
}

void calcCombination(
    std::unordered_set<int> comb, const std::vector<int>& group, int size,
    int offset, std::vector<std::unordered_set<int>>& results) {
  if (size == 0) {
    results.push_back(comb);
    return;
  }

  for (size_t i = offset; i <= group.size() - size; i++) {
    comb.insert(group.at(i));
    calcCombination(comb, group, size - 1, i + 1, results);
    comb.erase(group.at(i));
  }
}

std::vector<std::unordered_set<int>> combination(
    const std::vector<int>& group, int size) {
  std::vector<std::unordered_set<int>> results;
  calcCombination(std::unordered_set<int>(), group, size, 0, results);
  return results;
}
} // namespace rannc

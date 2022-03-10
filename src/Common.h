//
// Created by Masahiro Tanaka on 2018-12-10.
//

#ifndef PT_RANNC_COMMON_H
#define PT_RANNC_COMMON_H

#include <boost/filesystem.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>

#include <msgpack.hpp>

#include "Logging.h"

namespace fs = boost::filesystem;

namespace rannc {
class IRType;
class IRValue;

std::string generateName(const std::string& prefix);

bool begins_with(const std::string& str, const std::string& pattern);
bool ends_with(const std::string& str, const std::string& pattern);

template <typename T, typename H>
bool contains(const std::unordered_set<T, H>& v, const T& elem) {
  auto result = v.find(elem);
  return result != v.end();
}

template <typename T>
bool contains(const std::vector<T>& v, const T& elem) {
  auto result = std::find(v.begin(), v.end(), elem);
  return result != v.end();
}

template <typename C, typename T>
bool contains(const C& set, const T& elem) {
  return set.find(elem) != set.end();
}

template <typename K, typename V, typename H = std::hash<K>>
std::vector<K> keys(
    const std::unordered_map<K, V, H>& map, bool sorted = true) {
  std::vector<K> keys;
  for (const auto& e : map) {
    keys.push_back(e.first);
  }

  if (sorted) {
    std::sort(keys.begin(), keys.end());
  }
  return keys;
}

template <typename K, typename V, typename H>
std::unordered_set<K> key_set(const std::unordered_map<K, V, H>& map) {
  std::unordered_set<K> keys;
  keys.reserve(map.size());
  for (const auto& e : map) {
    keys.insert(e.first);
  }
  return keys;
}

template <typename K, typename V>
std::unordered_set<V> value_set(const std::unordered_map<K, V>& map) {
  std::unordered_set<V> keys;
  keys.reserve(map.size());
  for (const auto& e : map) {
    keys.insert(e.second);
  }
  return keys;
}

template <typename K, typename V, typename H>
std::vector<V> values(const std::unordered_map<K, V, H>& map) {
  std::vector<V> values;
  for (const auto& e : map) {
    values.push_back(e.second);
  }
  return values;
}

template <typename T>
inline std::string to_string(const T& v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

template <typename T>
std::string join_as_str(
    const T& v, const char* delim = ",", const size_t maxlen = 0) {
  std::stringstream ss;

  if (!v.empty()) {
    auto it = v.begin();
    ss << to_string(*it);
    it++;
    for (; it != v.end(); ++it) {
      if (delim)
        ss << delim;
      ss << to_string(*it);
    }
  }

  std::string s = ss.str();
  if (maxlen > 0 && s.length() > maxlen) {
    s = s.substr(0, maxlen) + " ...";
  }

  return "[" + s + "]";
}

template <typename K, typename V>
std::string join_map_as_str(
    const std::unordered_map<K, V>& map, const char* delim = ",",
    const int maxlen = 0) {
  std::vector<std::string> elems;
  elems.reserve(map.size());
  for (const auto& it : map) {
    std::stringstream ss;
    ss << it.first << "=" << it.second;
    elems.push_back(ss.str());
  }
  return join_as_str(elems, delim, maxlen);
}

template <typename L>
size_t productDim(const L& dim) {
  size_t prod = 1;
  for (auto d : dim) {
    prod *= d;
  }
  return prod;
}

template <typename T>
std::string tensorPtrToString(T* ptr, size_t size, size_t str_len = 100) {
  std::vector<T> vals;
  for (size_t i = 0; i < size; i++) {
    vals.push_back(*ptr);
    ptr++;
  }
  return join_as_str(vals, ",", str_len);
}

template <typename T>
std::string toString(T d) {
  std::stringstream ss;
  ss << d;
  return ss.str();
}

std::vector<std::string> split(const std::string& s, char delim);
std::vector<std::string> split(const std::string& s, const std::string& delim);

bool passedForBackward(const IRType& type);
bool isTensorOrTensorList(const IRType& type);
} // namespace rannc

namespace rannc {
using unique_void_ptr = std::unique_ptr<void, std::function<void(void const*)>>;

template <typename T>
auto unique_void(T* ptr) -> unique_void_ptr {
  return unique_void_ptr(ptr, [](void const* data) {
    if (data != nullptr) {
      T const* p = static_cast<T const*>(data);
      delete p;
    }
  });
}

template <typename T>
auto unique_void_with_key(T* ptr, const std::string key) -> unique_void_ptr {
  return unique_void_ptr(ptr, [key](void const* data) {
    if (data != nullptr) {
      std::cout << "Releasing " << key << std::endl;
      T const* p = static_cast<T const*>(data);
      delete p;
    }
  });
}

template <typename T>
auto unique_void_array(T* ptr) -> unique_void_ptr {
  return unique_void_ptr(ptr, [](void const* data) {
    T const* p = static_cast<T const*>(data);
    delete[] p;
  });
}

template <typename I, typename O>
std::vector<O> flatMap(
    const std::vector<I>& vec, std::function<std::vector<O>(I)> f) {
  std::vector<O> results;

  for (const auto& v : vec) {
    for (const auto& o : f(v)) {
      results.push_back(o);
    }
  }
  return results;
}

template <typename T>
std::vector<T> setToVector(const std::unordered_set<T>& set) {
  std::vector<T> vec;
  vec.reserve(set.size());
  for (const auto& e : set) {
    vec.push_back(e);
  }
  return vec;
}

template <typename T>
std::unordered_set<T> vectorToSet(const std::vector<T>& vector) {
  std::unordered_set<T> set;
  set.reserve(vector.size());
  for (const auto& e : vector) {
    set.insert(e);
  }
  return set;
}

template <typename T>
std::vector<T> addAll(const std::vector<T>& v1, const std::vector<T>& v2) {
  std::vector<T> ret;
  ret.reserve(v1.size() + v2.size());
  for (const auto& e : v1) {
    ret.push_back(e);
  }
  for (const auto& e : v2) {
    ret.push_back(e);
  }
  return ret;
}

template <typename K, typename V>
std::unordered_map<K, V> addAll(
    const std::unordered_map<K, V>& v1, const std::unordered_map<K, V>& v2) {
  std::unordered_map<K, V> ret;
  ret.reserve(v1.size() + v2.size());
  for (const auto& it : v1) {
    ret[it.first] = it.second;
  }
  for (const auto& it : v2) {
    ret[it.first] = it.second;
  }
  return ret;
}

template <typename T>
std::vector<T> reverse(const std::vector<T>& vec) {
  std::vector<T> rev = vec;
  std::reverse(rev.begin(), rev.end());
  return rev;
}

fs::path getHomeDir();

std::vector<std::unordered_set<int>> combination(
    const std::vector<int>& group, int size);

std::vector<int> createDummyRanks(int num);

struct IntSetHash {
  std::size_t operator()(const std::unordered_set<int>& iset) const {
    auto ivec = setToVector(iset);
    std::sort(ivec.begin(), ivec.end());
    return std::hash<std::string>()(join_as_str(ivec));
  };
};

struct SizeVectorHash {
  std::size_t operator()(const std::vector<size_t>& sizes) const {
    return std::hash<std::string>()(join_as_str(sizes));
  };
};

struct StringPairHash {
  std::size_t operator()(
      const std::pair<std::string, std::string>& pair) const {
    return std::hash<std::string>()(pair.first + "_" + pair.second);
  };
};

struct IntPairHash {
  std::size_t operator()(const std::pair<int, int>& pair) const {
    return pair.first ^ pair.second;
  };
};

template <typename T1, typename T2>
struct PairKey {
  T1 first;
  T2 second;

  bool operator==(const PairKey& rhs) const {
    return first == rhs.first && second == rhs.second;
  }

  bool operator!=(const PairKey& rhs) const {
    return !(rhs == *this);
  }
};

template <typename T1, typename T2>
struct PairHash {
  std::size_t operator()(const PairKey<T1, T2>& pair) const {
    std::stringstream ss;
    ss << pair.first << "_" << pair.second;
    return std::hash<std::string>()(ss.str());
  };
};

template <class T>
struct EnumHash {
  static_assert(
      std::is_enum<T>::value, "This hash only works for enumeration types");
  size_t operator()(T x) const noexcept {
    using type = typename std::underlying_type<T>::type;
    return std::hash<type>{}(static_cast<type>(x));
  }
};

template <typename T>
T sum(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 0);
}

template <typename T>
T average(const std::vector<T>& v) {
  return sum(v) / v.size();
}

template <typename T>
T max(const std::vector<T>& v) {
  return *std::max_element(v.begin(), v.end());
}

template <typename T>
std::vector<T> shuffle(const std::vector<T>& vec) {
  std::vector<T> ret = vec;

  std::random_device seed_gen;
  static std::mt19937 engine(seed_gen());
  std::shuffle(ret.begin(), ret.end(), engine);

  return vec;
}

template <typename T>
T gcd(T a, T b) {
  if (a < 0)
    a = -a;
  if (b < 0)
    b = -b;
  while (b != 0) {
    a %= b;
    if (a == 0)
      return b;
    b %= a;
  }
  return a;
}

template <typename T>
T gcd(const std::vector<T>& v) {
  if (v.size() == 1) {
    return v.front();
  }
  T a = v.front();
  for (size_t i = 1; i < v.size(); ++i) {
    a = gcd(a, v[i]);
  }
  return a;
}

template <typename T>
std::vector<char> serialize(const T data) {
  std::stringstream buffer;
  msgpack::pack(buffer, data);

  buffer.seekg(0);
  std::string str_buf(buffer.str());

  std::vector<char> vec_buf(str_buf.size());
  memcpy(&vec_buf[0], str_buf.c_str(), str_buf.size());
  return vec_buf;
}

template <typename T>
T deserialize(const std::vector<char>& data) {
  msgpack::object_handle oh = msgpack::unpack(&data[0], data.size());
  msgpack::object deserialized = oh.get();

  T obj;
  deserialized.convert(obj);
  return obj;
}

template <typename T>
void saveToFile(const std::string& path, const T& obj) {
  const auto cache_data = serialize(obj);

  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::invalid_argument("Failed to open file: " + path);
  }
  out.write(reinterpret_cast<const char*>(&cache_data[0]), cache_data.size());
  out.close();
}

template <typename T>
T loadFromFile(const std::string& path) {
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Failed to open file: " + path);
  }

  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return deserialize<T>(buffer);
}

class CommErrorException : public std::runtime_error {
 public:
  CommErrorException(const char* msg, int code)
      : std::runtime_error(msg), code_(code) {}

  int getCode() const {
    return code_;
  }

 private:
  int code_;
};

} // namespace rannc

#endif // PT_RANNC_COMMON_H

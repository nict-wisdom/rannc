//
// Created by Masahiro Tanaka on 2022/03/11.
//

#ifndef PYRANNC_GRAPHVALUECACHE_H
#define PYRANNC_GRAPHVALUECACHE_H

#include <Common.h>
#include <torch/torch.h>

namespace rannc {

// Modified https://github.com/lamerman/cpp-lru-cache
template <typename K, typename V>
class LRUCache {
 public:
  typedef typename std::pair<K, V> key_value_pair_t;
  typedef typename std::list<key_value_pair_t>::iterator list_iterator_t;

  LRUCache(size_t max_size) : max_size_(max_size), current_size_(0) {}

  void put(const K& key, const V& value) {
    auto it = item_map_.find(key);
    items_.push_front(key_value_pair_t(key, value));
    current_size_ += getValueSize(value);

    if (it != item_map_.end()) {
      items_.erase(it->second);
      item_map_.erase(it);
      current_size_ -= getValueSize(value);
    }
    item_map_[key] = items_.begin();

    while (max_size_ != 0 && max_size_ < current_size_) {
      auto last = items_.end();
      last--;
      item_map_.erase(last->first);

      current_size_ -= getValueSize(last->second);
      items_.pop_back();
    }
  }

  const V& get(const K& key) {
    auto it = item_map_.find(key);
    if (it == item_map_.end()) {
      throw std::range_error("No key found.");
    } else {
      items_.splice(items_.begin(), items_, it->second);
      return it->second->second;
    }
  }

  bool exists(const K& key) const {
    return item_map_.find(key) != item_map_.end();
  }

  size_t elemCount() const {
    return item_map_.size();
  }

  void clear() {
    items_.clear();
    item_map_.clear();
    current_size_ = 0;
  }

 protected:
  virtual size_t getValueSize(const V& v) const = 0;

 private:
  std::list<std::pair<K, V>> items_;
  std::unordered_map<K, list_iterator_t> item_map_;
  size_t max_size_;
  size_t current_size_;
};

class ParamCache : public LRUCache<long, at::Tensor> {
 public:
  ParamCache(size_t max_size) : LRUCache(max_size){};

 protected:
  size_t getValueSize(const at::Tensor& v) const override;
};

} // namespace rannc
#endif // PYRANNC_GRAPHVALUECACHE_H

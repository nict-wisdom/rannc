//
// Created by Masahiro Tanaka on 2019-07-12.
//

#include <sstream>

#include "Common.h"
#include "IValueLocation.h"

namespace rannc {

std::string toString(const IValueLocation& loc) {
  std::stringstream ss;
  ss << loc.value_name << ":" << toString(loc.path);
  return ss.str();
}

std::string toString(const std::vector<IValueLocation>& locs) {
  std::vector<std::string> str_locs;
  str_locs.reserve(locs.size());
  for (const auto& l : locs) {
    str_locs.push_back(toString(l));
  }
  return join_as_str(str_locs);
}

std::string toString(const PathInIValue& path) {
  std::vector<std::string> steps;
  for (const auto& st : path) {
    std::stringstream ss;
    switch (st.type) {
      case StepTypeInIValue::LIST:
        ss << "LIST:" << st.index;
        break;
      case StepTypeInIValue::TUPLE:
        ss << "TUPLE:" << st.index;
        break;
    }
    steps.push_back(ss.str());
  }
  return join_as_str(steps, "->");
}

IValueLocation createListElem(const IValueLocation& loc, int index) {
  PathInIValue path = loc.path;
  path.emplace_back(StepTypeInIValue::LIST, index);
  return IValueLocation(loc.value_name, path);
}

IValueLocation createTupleElem(const IValueLocation& loc, int index) {
  PathInIValue path = loc.path;
  path.emplace_back(StepTypeInIValue::TUPLE, index);
  return IValueLocation(loc.value_name, path);
}
} // namespace rannc
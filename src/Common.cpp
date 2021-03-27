//
// Created by Masahiro Tanaka on 2018-12-10.
//
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>


#include "graph/ir.h"
#include "Common.h"

namespace {
    static int REF_LENGTH = 16;
    static const char *REF_PREF = "ref_";

    void gen_random(char *s, const int len) {
        static const char alphanum[] =
                "0123456789"
                //                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz";

        for (int i = 0; i < len; ++i) {
            s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
        }

        s[len] = 0;
    }
}

namespace rannc {
    const std::string RANNC_CONF_DIR = ".pyrannc";

    std::string generateRef() {
        char id[REF_LENGTH + 1];
        gen_random(id, REF_LENGTH);
        return std::string(REF_PREF) + std::string(id);
    }

    std::string generateName(const std::string &prefix) {
        char id[REF_LENGTH + 1];
        gen_random(id, REF_LENGTH);
        return prefix + std::string(id);
    }

    std::vector<std::string> split(const std::string &s, char delim) {
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

    std::vector<std::string> split(const std::string &s, const std::string &delim) {
        std::vector<std::string> ret;
        for (size_t i = 0, n; i <= s.length(); i = n + 1) {

            n = s.find_first_of(delim, i);
            if (n == std::string::npos) n = s.length();
            std::string tmp = s.substr(i, n - i);
            ret.push_back(tmp);
        }
        return ret;
    }

    bool begins_with(const std::string &str, const std::string &pattern) {
        if (str.size() >= pattern.size()) {
            return std::equal(std::begin(pattern), std::end(pattern), std::begin(str));
        }
        return false;
    }

    bool ends_with(const std::string &str, const std::string &pattern) {
        return str.size() >= pattern.size()
               && str.find(pattern, str.size() - pattern.size()) != std::string::npos;
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
            case IRTensorElemType::DOUBLE:
            case IRTensorElemType::UNDEF:
                return true;
            case IRTensorElemType::INT:
            case IRTensorElemType::LONG:
            case IRTensorElemType::BOOL:
                return false;
        }
    }

    bool passedForBackward(const IRType &type) {
        auto base_type = type.getBaseType();
        switch (base_type) {
            case IRBaseType::SCALAR:
                return passedForBackward(type.getScalarType());
            case IRBaseType::TENSOR:
                return passedForBackward(type.getTensorElemType());
            case IRBaseType::LIST: {
                const auto& list_type = type.getListType();
                if (list_type == IRListType::TENSOR || list_type == IRListType::GENERIC) {
                    for (const auto &et: type.getCompoundTypes()) {
                        if (!passedForBackward(et)) {
                            return false;
                        }
                    }
                    return true;
                }
                return false;
            }
            case IRBaseType::TUPLE: {
                const auto &elem_types = type.getCompoundTypes();
                bool ret = true;
                for (const auto &et: elem_types) {
                    ret &= passedForBackward(et);
                    if (!ret) return false;
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

    bool isTensorOrTensorList(const IRType &type) {
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
                const auto &elem_types = type.getCompoundTypes();
                bool ret = true;
                for (const auto &et: elem_types) {
                    ret &= isTensorOrTensorList(et);
                    if (!ret) return false;
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
        const char *home_dir;
        if ((home_dir = getenv("HOME")) == nullptr) {
            home_dir = getpwuid(getuid())->pw_dir;
        }
        return home_dir;
    }


    void calcCombination(std::unordered_set<int> comb, const std::vector<int>& group, int size, int offset,
                         std::vector<std::unordered_set<int>>& results) {
        if (size == 0) {
            results.push_back(comb);
            return;
        }

        for (size_t i=offset; i <= group.size()-size; i++) {
            comb.insert(group.at(i));
            calcCombination(comb, group, size-1,  i+1, results);
            comb.erase(group.at(i));
        }
    }

    std::vector<std::unordered_set<int>> combination(const std::vector<int>& group, int size) {
        std::vector<std::unordered_set<int>> results;
        calcCombination(std::unordered_set<int>(), group, size, 0, results);
        return results;
    }

    std::unordered_map<int, int64_t> getSplitBatchSizes(int64_t batch_size, const std::unordered_set<int>& ranks,
                                            int split_index) {
        std::vector<int> vec_ranks = rannc::setToVector(ranks);
        std::sort(vec_ranks.begin(), vec_ranks.end());

        size_t rank_num = ranks.size();
        int64_t base_size = batch_size / rank_num;
        size_t mod = batch_size % rank_num;

        std::unordered_map<int, int64_t> batch_sizes;
        for (size_t i = 0; i < rank_num; i++) {
            int64_t ex = i < mod ? 1 : 0;
            int64_t split_size = base_size + ex;

            int rank = vec_ranks.at(i);
            batch_sizes[rank] = split_size;
        }

        if (split_index == 0) {
            return batch_sizes;
        }

        int offset = 0;
        for (int i=0; i<split_index; i++) {
            offset = (offset + batch_size) % ranks.size();
        }
        std::unordered_map<int, int64_t> shifted_batch_sizes;
        shifted_batch_sizes.reserve(batch_sizes.size());

        for (int i=0; i<vec_ranks.size(); i++) {
            int shift_i = (i + offset)%vec_ranks.size();
            shifted_batch_sizes[vec_ranks.at(shift_i)] = batch_sizes.at(vec_ranks.at(i));
        }

        return shifted_batch_sizes;
    }

    std::vector<int64_t> getSplitBatchSizes(int64_t batch_size, int split_num) {

        std::vector<int> dummy_ranks;
        for (int i=0; i<split_num; i++) {
            dummy_ranks.push_back(i);
        }

        const auto batch_sizes = getSplitBatchSizes(batch_size, vectorToSet(dummy_ranks));
        std::vector<int64_t> result;
        for (int r: dummy_ranks) {
            result.push_back(batch_sizes.at(r));
        }
        return result;
    }

    std::vector<int64_t> getLocalSplitBatchSizes(const std::vector<int64_t>& split_batch_sizes,
            int world_size, int rank) {
        std::vector<int64_t> ret_split;
        int64_t offset = rank;

        for (const auto& bs: split_batch_sizes) {
            int64_t ret_bs = (int64_t) std::ceil((bs-offset) / (double) world_size);
            ret_split.push_back(ret_bs);
            offset = (offset + ret_bs*world_size) - bs;
        }
        return ret_split;
    }

    std::vector<int64_t> getGlobalDim(int64_t batch_size, const std::vector<int64_t> &local_dim,
                                      const std::unordered_set<int> &ranks, int my_rank) {

        const std::unordered_map<int, int64_t> batch_sizes = getSplitBatchSizes(batch_size, ranks);
        std::vector<int64_t> global_dim = local_dim;
        global_dim[0] = local_dim[0] * batch_size / batch_sizes.at(my_rank);
        return global_dim;
    }

    std::unordered_map<int, std::vector<int64_t>> calcDistBatchDims(int64_t batch_size,
                                                                    const std::vector<int64_t> &global_dim,
                                                                    const std::unordered_set<int> &ranks,
                                                                    int split_index) {
        std::vector<int> vec_ranks = rannc::setToVector(ranks);
        std::sort(vec_ranks.begin(), vec_ranks.end());

        assert(!global_dim.empty());
        std::unordered_map<int, int64_t> batch_sizes = getSplitBatchSizes(batch_size, ranks, split_index);

        std::unordered_map<int, std::vector<int64_t>> results;
        for (const auto& it: batch_sizes) {
            int rank = it.first;
            auto rank_bs = it.second;

            std::vector<int64_t> rank_dim = global_dim;
            rank_dim[0] = global_dim[0] * rank_bs / batch_size;
            results[rank] = rank_dim;
        }

        return results;
    }
 }
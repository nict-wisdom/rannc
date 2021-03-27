
#include "FunctionStorage.h"

#include <ATen/core/function.h>

namespace rannc {

//
//    Construct function table from graph.
//
void FunctionStorage::deploy(const std::shared_ptr<torch::jit::Graph> & graph)
{
    for (const auto & node : graph->nodes() ) {
        if (node->kind() != c10::prim::Constant) {
            continue;
        }

        c10::ConstTypePtr tp = node->output()->type();
        if ( tp->kind() != c10::TypeKind::FunctionType ) {
            continue;
        }

        std::shared_ptr<const c10::FunctionType> ftype
            =  std::static_pointer_cast<const c10::FunctionType>(tp->shared_from_this());
        torch::jit::Function *  const  func = ftype->function();
        const std::string  name = node->output()->debugName() + '@' + func->qualname().name();
        this->functions_[name]  = func;

        //  Save the attr name of the function.
        this->func_attr_[name]  = node->s(c10::attr::name);
    }
}

//
//    Get attr::name of the function.
//

const std::string &
FunctionStorage::getAttrName(const std::string name) const
{
    return  func_attr_.at(name);
}


}   //  End of namespace rannc.

#include "autodiff/autodiff.h"
#include <algorithm>

namespace autodiff {

    std::shared_ptr<op> mult(std::shared_ptr<op> t1, std::shared_ptr<op> t2)
    {
        std::shared_ptr<op> result { new op };
    
        t1->parent = result.get();
    
        result->children.emplace_back(t1);
        result->children.emplace_back(t2);
    
        result->name = "mult";
    
        return result;
    }
    
    std::shared_ptr<op> logistic(std::shared_ptr<op> input)
    {
        std::shared_ptr<op> result { new op };

        input->parent = result.get();

        result->children.emplace_back(input);
    
        result->name = "logistic";
    
        return result;
    }

    void eval(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        std::vector<std::shared_ptr<op>> stack { root };
        std::vector<std::shared_ptr<op>> path;

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            while (path.size() > 0 && path.back().get() != t->parent) {
                funcs.at(path.back()->name)(path.back());
                path.pop_back();
            }
            path.push_back(t);

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }
    }

    void grad(std::shared_ptr<op> root,
        std::unordered_map<std::string, std::function<void(std::shared_ptr<op>)>> funcs)
    {
        std::vector<std::shared_ptr<op>> stack { root };

        while (stack.size() != 0) {
            std::shared_ptr<op> t = stack.back();
            stack.pop_back();

            funcs.at(t->name)(t);

            for (auto c: t->children) {
                stack.push_back(c);
            }
        }
    }

}

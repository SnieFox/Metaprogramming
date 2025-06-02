#ifndef GRADIENT_ASCENT_OPTIMIZER_HPP
#define GRADIENT_ASCENT_OPTIMIZER_HPP

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <tuple>
#include "tuple_utils.hpp"
#include "gradient_calculator.hpp"

template<int TOTAL_STEPS>
class GradientAscentOptimizer {
private:
    template<int STEPS_REMAINING, typename Func, typename... Args>
    static std::tuple<Args...> perform_steps_recursive(
        Func&& func_to_eval,
        std::tuple<Args...> current_point,
        const double fixed_learning_rate,
        int current_step_number) {

        if constexpr (STEPS_REMAINING == 0) {
            return current_point;
        }
        else {
            std::tuple<Args...> grad = TupleGradientCalculator::calculate(func_to_eval, current_point);
            std::tuple<Args...> scaled_grad = TupleUtils::multiply_tuple_scalar(fixed_learning_rate, grad);
            std::tuple<Args...> next_point = TupleUtils::add_tuples(current_point, scaled_grad);
            return perform_steps_recursive<STEPS_REMAINING - 1>(
                std::forward<Func>(func_to_eval),
                next_point,
                fixed_learning_rate,
                current_step_number + 1
            );
        }
    }

public:
    template<typename Func, typename... Args>
    std::tuple<Args...> optimize(
        Func&& func_to_eval,
        const std::tuple<Args...>& initial_point,
        const double fixed_learning_rate) {

        if (fixed_learning_rate <= 0.0) {
            throw std::invalid_argument("Learning rate must be positive.");
        }
        if constexpr (sizeof...(Args) == 0 && TOTAL_STEPS > 0) {
            throw std::invalid_argument("Initial point cannot be empty if steps > 0 for 0-arity function.");
        }
        if constexpr (sizeof...(Args) == 0 && TOTAL_STEPS == 0) {
            return {};
        }

        return perform_steps_recursive<TOTAL_STEPS>(
            std::forward<Func>(func_to_eval),
            initial_point,
            fixed_learning_rate,
            0
        );
    }
};

#endif
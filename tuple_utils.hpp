#ifndef TUPLE_UTILS_HPP
#define TUPLE_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <tuple>
#include <utility>
#include <string>

namespace TupleUtils {
    template<typename Tuple, std::size_t... Is>
    void print_tuple_impl(std::ostream& os, const Tuple& t, std::index_sequence<Is...>) {
        os << "[";
        ((os << (Is == 0 ? "" : ", ") << std::fixed << std::setprecision(6) << std::get<Is>(t)), ...);
        os << "]";
    }

    template<typename... Args>
    void print_tuple(std::ostream& os, const std::string& label, const std::tuple<Args...>& t, int step_num = -1) {
        if (step_num >= 0) {
            os << "Step " << std::setw(2) << step_num << ": ";
        }
        os << label;
        print_tuple_impl(os, t, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<typename Tuple1, typename Tuple2, std::size_t... Is>
    auto add_tuples_impl(const Tuple1& t1, const Tuple2& t2, std::index_sequence<Is...>) {
        return std::make_tuple((std::get<Is>(t1) + std::get<Is>(t2))...);
    }

    template<typename... Args1, typename... Args2>
    auto add_tuples(const std::tuple<Args1...>& t1, const std::tuple<Args2...>& t2) {
        static_assert(sizeof...(Args1) == sizeof...(Args2), "Tuples must have the same size for addition.");
        return add_tuples_impl(t1, t2, std::make_index_sequence<sizeof...(Args1)>{});
    }

    template<typename Scalar, typename Tuple, std::size_t... Is>
    auto multiply_tuple_scalar_impl(const Scalar& s, const Tuple& t, std::index_sequence<Is...>) {
        return std::make_tuple((s * std::get<Is>(t))...);
    }

    template<typename Scalar, typename... Args>
    auto multiply_tuple_scalar(const Scalar& s, const std::tuple<Args...>& t) {
        return multiply_tuple_scalar_impl(s, t, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<std::size_t I, typename... Args>
    std::tuple<Args...> perturb_tuple_element(const std::tuple<Args...>& t,
        typename std::tuple_element<I, std::tuple<Args...>>::type delta) {
        std::tuple<Args...> perturbed_t = t;
        std::get<I>(perturbed_t) += delta;
        return perturbed_t;
    }
}

#endif
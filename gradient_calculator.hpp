#ifndef GRADIENT_CALCULATOR_HPP
#define GRADIENT_CALCULATOR_HPP

#include <tuple>
#include <stdexcept>
#include <string>
#include "tuple_utils.hpp"

#ifndef M_PI
#define M_PI (std::acos(-1.0))
#endif

template <typename T = double>
constexpr T default_h_for_gradient_v = 1e-5;

class TupleGradientCalculator {
public:
    template <typename Func, typename... Args, std::size_t... Is>
    static std::tuple<Args...> calculate_impl(
        Func&& func_to_eval,
        const std::tuple<Args...>& point,
        double h,
        std::index_sequence<Is...>) {

        auto partial_derivative_k = [&](auto K_val) {
            constexpr std::size_t K = decltype(K_val)::value;
            std::tuple<Args...> point_plus_h = TupleUtils::perturb_tuple_element<K>(point, static_cast<typename std::tuple_element<K, std::tuple<Args...>>::type>(h));
            std::tuple<Args...> point_minus_h = TupleUtils::perturb_tuple_element<K>(point, static_cast<typename std::tuple_element<K, std::tuple<Args...>>::type>(-h));
            double f_plus_h = std::apply(func_to_eval, point_plus_h);
            double f_minus_h = std::apply(func_to_eval, point_minus_h);
            return (f_plus_h - f_minus_h) / (2.0 * h);
            };

        return std::make_tuple(partial_derivative_k(std::integral_constant<std::size_t, Is>{})...);
    }

    template <typename Func, typename... Args>
    static std::tuple<Args...> calculate(
        Func&& func_to_eval,
        const std::tuple<Args...>& point,
        double h = default_h_for_gradient_v<>) {

        if (h <= 0.0) {
            throw std::invalid_argument("Step size h for finite difference must be positive.");
        }
        if constexpr (sizeof...(Args) == 0) {
            return {};
        }
        else {
            return calculate_impl(std::forward<Func>(func_to_eval), point, h, std::make_index_sequence<sizeof...(Args)>{});
        }
    }
};

#endif
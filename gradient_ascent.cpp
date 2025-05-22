#include "gradient_ascent.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>

void print_vector_template_detail(const std::string& label, const std::vector<double>& vec, int step_num) {
    if (step_num >= 0) {
        std::cout << "Step " << std::setw(2) << step_num << ": ";
    }
    std::cout << label << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}

std::vector<double> calculate_gradient(MultiVarFunc func, const std::vector<double>& point, double h) {
    if (point.empty()) {
        throw std::invalid_argument("Input point vector cannot be empty.");
    }
    if (h <= 0.0) {
        throw std::invalid_argument("Step size h for finite difference must be positive.");
    }
    size_t num_variables = point.size();
    std::vector<double> gradient(num_variables);
    std::vector<double> perturbed_point = point;

    for (size_t i = 0; i < num_variables; ++i) {
        double original_value_at_i = perturbed_point[i];
        perturbed_point[i] = original_value_at_i + h;
        double f_plus_h = func(perturbed_point);
        perturbed_point[i] = original_value_at_i - h;
        double f_minus_h = func(perturbed_point);
        gradient[i] = (f_plus_h - f_minus_h) / (2.0 * h);
        perturbed_point[i] = original_value_at_i;
    }
    return gradient;
}

template<int STEPS_REMAINING>
std::vector<double> perform_gradient_steps_static_templated(
    MultiVarFunc func,
    std::vector<double> current_point,
    const double fixed_learning_rate,
    int current_step_number,
    typename std::enable_if_t<STEPS_REMAINING == 0>*
) {
    double current_func_value = func(current_point);
    print_vector_template_detail("Point (Base Case): ", current_point, current_step_number);
    std::cout << ", f(p): " << std::fixed << std::setprecision(6) << current_func_value
        << ", LR (fixed): " << fixed_learning_rate << std::endl;
    std::cout << "Reached maximum steps (compile-time defined via template). Final point." << std::endl;
    return current_point;
}

template<int STEPS_REMAINING>
std::vector<double> perform_gradient_steps_static_templated(
    MultiVarFunc func,
    std::vector<double> current_point,
    const double fixed_learning_rate,
    int current_step_number,
    typename std::enable_if_t<(STEPS_REMAINING > 0)>*
) {
    double current_func_value = func(current_point);
    print_vector_template_detail("Point: ", current_point, current_step_number);
    std::cout << ", f(p): " << std::fixed << std::setprecision(6) << current_func_value
        << ", LR (fixed): " << fixed_learning_rate
        << ", Steps Left (Template): " << STEPS_REMAINING << std::endl;

    std::vector<double> grad = calculate_gradient(func, current_point, 1e-5);
    std::vector<double> next_point = current_point;
    for (size_t i = 0; i < current_point.size(); ++i) {
        next_point[i] += fixed_learning_rate * grad[i];
    }

    return perform_gradient_steps_static_templated<STEPS_REMAINING - 1>(
        func,
        next_point,
        fixed_learning_rate,
        current_step_number + 1
    );
}

// Explicit template instantiations
template std::vector<double> perform_gradient_steps_static_templated<0>(
    MultiVarFunc, std::vector<double>, const double, int, typename std::enable_if_t<0 == 0>*);
template std::vector<double> perform_gradient_steps_static_templated<100>(
    MultiVarFunc, std::vector<double>, const double, int, typename std::enable_if_t<(100 > 0)>*);
template std::vector<double> perform_gradient_steps_static_templated<5>(
    MultiVarFunc, std::vector<double>, const double, int, typename std::enable_if_t<(5 > 0)>*);
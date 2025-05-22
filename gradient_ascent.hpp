#ifndef GRADIENT_ASCENT_HPP
#define GRADIENT_ASCENT_HPP

#include <vector>
#include <functional>
#include <string>
#include <type_traits>

// Define PI for trigonometric functions if M_PI is not available
#ifndef M_PI
#define M_PI (std::acos(-1.0))
#endif

// Type alias for a multivariable scalar function
using MultiVarFunc = std::function<double(const std::vector<double>&)>;

// Forward declarations
std::vector<double> calculate_gradient(MultiVarFunc func, const std::vector<double>& point, double h);
void print_vector_template_detail(const std::string& label, const std::vector<double>& vec, int step_num = -1);

// Templated Gradient Ascent Function - Base case (STEPS_REMAINING = 0)
template<int STEPS_REMAINING>
std::vector<double> perform_gradient_steps_static_templated(
    MultiVarFunc func,
    std::vector<double> current_point,
    const double fixed_learning_rate,
    int current_step_number = 0,
    typename std::enable_if_t<STEPS_REMAINING == 0>* = nullptr
);

// Recursive step for the templated function (STEPS_REMAINING > 0)
template<int STEPS_REMAINING>
std::vector<double> perform_gradient_steps_static_templated(
    MultiVarFunc func,
    std::vector<double> current_point,
    const double fixed_learning_rate,
    int current_step_number = 0,
    typename std::enable_if_t<(STEPS_REMAINING > 0)>* = nullptr
);

#endif
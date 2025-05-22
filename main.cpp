#include "gradient_ascent.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <chrono>

int main() {
    MultiVarFunc func_to_optimize = [](const std::vector<double>& p) {
        if (p.size() != 2) {
            throw std::invalid_argument("Function expects 2 variables (x, y).");
        }
        double x = p[0], y = p[1];
        return -((x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0)); // Max at (2,3)
        };

    const double fixed_learning_rate = 0.1;
    const int TOTAL_COMPILE_TIME_STEPS = 100;
    const int FEWER_STEPS = 5;
    const int NUM_TEST_POINTS = 50;

    // Random number generator for test points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    std::cout << "--- Recursive Gradient Ascent with Compile-Time Fixed Steps (Templates) ---" << std::endl;
    std::cout << "Optimizing f(x,y) = -( (x-2)^2 + (y-3)^2 ), max at (2,3)" << std::endl;
    std::cout << "Testing with " << NUM_TEST_POINTS << " random starting points" << std::endl;
    std::cout << "Fixed LR: " << fixed_learning_rate << ", Total Compile-Time Steps: " << TOTAL_COMPILE_TIME_STEPS << std::endl;
    std::cout << "==================================================================================" << std::endl;

    try {
        // Measure time for the main test loop
        auto start = std::chrono::high_resolution_clock::now();
        long long total_point_time = 0; // Accumulate time for individual points (in microseconds)

        for (int i = 0; i < NUM_TEST_POINTS; ++i) {
            std::vector<double> starting_point = { dis(gen), dis(gen) };
            std::cout << "\nTest Point " << i + 1 << ":" << std::endl;
            print_vector_template_detail("Initial Point: ", starting_point);
            std::cout << std::endl;

            // Measure time for this test point
            auto point_start = std::chrono::high_resolution_clock::now();

            std::vector<double> final_point = perform_gradient_steps_static_templated<TOTAL_COMPILE_TIME_STEPS>(
                func_to_optimize,
                starting_point,
                fixed_learning_rate
            );

            auto point_end = std::chrono::high_resolution_clock::now();
            auto point_duration = std::chrono::duration_cast<std::chrono::microseconds>(point_end - point_start);
            total_point_time += point_duration.count();

            std::cout << "==================================================================================" << std::endl;
            print_vector_template_detail("Final Point after " + std::to_string(TOTAL_COMPILE_TIME_STEPS) + " steps: ", final_point);
            std::cout << ", f(p): " << std::fixed << std::setprecision(6) << func_to_optimize(final_point) << std::endl;
            std::cout << "Analytical maximum is at [2.000000, 3.000000] with f(p): 0.000000" << std::endl;
        }

        // Calculate and print execution time for test points
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1'000'000.0;
        std::cout << "\nExecution time for " << NUM_TEST_POINTS << " test points (" << TOTAL_COMPILE_TIME_STEPS
            << " steps each): " << std::fixed << std::setprecision(3) << seconds << " seconds" << std::endl;

        // Calculate and print average time per test point
        double avg_point_time = (total_point_time / NUM_TEST_POINTS) / 1'000'000.0;
        std::cout << "Average execution time per test point: " << std::fixed << std::setprecision(3)
            << avg_point_time << " seconds" << std::endl;

        // Example with fewer steps
        std::vector<double> starting_point = { 0.0, 0.0 };
        std::cout << "\n--- Example with " << FEWER_STEPS << " compile-time steps ---" << std::endl;
        print_vector_template_detail("Initial Point: ", starting_point);
        std::cout << std::endl;

        // Measure time for the fewer steps example
        start = std::chrono::high_resolution_clock::now();

        std::vector<double> final_point_fewer = perform_gradient_steps_static_templated<FEWER_STEPS>(
            func_to_optimize,
            starting_point,
            fixed_learning_rate
        );

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        seconds = duration.count() / 1'000'000.0;

        std::cout << "==================================================================================" << std::endl;
        print_vector_template_detail("Final Point after " + std::to_string(FEWER_STEPS) + " steps: ", final_point_fewer);
        std::cout << ", f(p): " << std::fixed << std::setprecision(6) << func_to_optimize(final_point_fewer) << std::endl;
        std::cout << "Execution time for " << FEWER_STEPS << " steps: " << std::fixed << std::setprecision(3)
            << seconds << " seconds" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
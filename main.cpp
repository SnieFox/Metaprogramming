#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <string>
#include "gradient_ascent_optimizer.hpp"

int main() {
    auto func_to_optimize = [](double x, double y) {
        return -((x - 2.0) * (x - 2.0) + (y - 3.0) * (y - 3.0));
        };

    const double fixed_learning_rate = 0.1;
    const int NUM_TEST_POINTS = 10;

    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    std::vector<std::tuple<double, double>> test_starting_points;
    test_starting_points.reserve(NUM_TEST_POINTS);
    for (int i = 0; i < NUM_TEST_POINTS; ++i) {
        test_starting_points.push_back(std::make_tuple(dis(gen), dis(gen)));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "--- Gradient Ascent with Tuples and Compile-Time Fixed Steps ---" << std::endl;
    std::cout << "Optimizing f(x,y) = -( (x-2)^2 + (y-3)^2 ), max at (2,3)" << std::endl;
    std::cout << "Testing with " << NUM_TEST_POINTS << " pre-generated starting points" << std::endl;

    const int TOTAL_STEPS_MAIN_LOOP = 100;
    GradientAscentOptimizer<TOTAL_STEPS_MAIN_LOOP> optimizer_main;
    std::cout << "Fixed LR: " << fixed_learning_rate
        << ", Total Compile-Time Steps: " << TOTAL_STEPS_MAIN_LOOP << std::endl;
    std::cout << "Default h for gradient: " << default_h_for_gradient_v<> << std::endl;
    std::cout << "==================================================================================" << std::endl;

    try {
        auto overall_start_time = std::chrono::high_resolution_clock::now();
        long long total_optimization_time_us = 0;

        for (int i = 0; i < NUM_TEST_POINTS; ++i) {
            const auto& starting_point = test_starting_points[i];

            auto point_start_time = std::chrono::high_resolution_clock::now();
            std::tuple<double, double> final_point = optimizer_main.optimize(
                func_to_optimize,
                starting_point,
                fixed_learning_rate
            );
            auto point_end_time = std::chrono::high_resolution_clock::now();
            auto point_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(point_end_time - point_start_time);
            total_optimization_time_us += point_duration_us.count();
        }

        auto overall_end_time = std::chrono::high_resolution_clock::now();
        auto overall_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(overall_end_time - overall_start_time);
        double overall_seconds = static_cast<double>(overall_duration_us.count()) / 1'000.0;

        std::cout << "\nTotal execution time for " << NUM_TEST_POINTS << " test points ("
            << TOTAL_STEPS_MAIN_LOOP << " steps each): "
            << std::fixed << std::setprecision(3) << overall_seconds << " ms" << std::endl;

        if (NUM_TEST_POINTS > 0) {
            double avg_point_time_s = (static_cast<double>(total_optimization_time_us) / NUM_TEST_POINTS) / 1'000.0;
            std::cout << "Average optimization time per test point: "
                << std::fixed << std::setprecision(3) << avg_point_time_s << " ms" << std::endl;
        }

        const int FEWER_STEPS = 5;
        GradientAscentOptimizer<FEWER_STEPS> optimizer_fewer_steps;
        std::tuple<double, double> start_point_fewer = (NUM_TEST_POINTS > 0) ? test_starting_points[0] : std::make_tuple(0.0, 0.0);

        std::cout << "\n--- Example with " << FEWER_STEPS << " compile-time steps ---" << std::endl;
        TupleUtils::print_tuple(std::cout, "Initial Point: ", start_point_fewer);
        std::cout << std::endl;

        auto fewer_steps_start_time = std::chrono::high_resolution_clock::now();
        std::tuple<double, double> final_point_fewer = optimizer_fewer_steps.optimize(
            func_to_optimize,
            start_point_fewer,
            fixed_learning_rate
        );
        auto fewer_steps_end_time = std::chrono::high_resolution_clock::now();
        auto fewer_steps_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(fewer_steps_end_time - fewer_steps_start_time);
        double fewer_steps_seconds = static_cast<double>(fewer_steps_duration_us.count()) / 1'000.0;

        std::cout << "----------------------------------------------------------------------------------" << std::endl;
        TupleUtils::print_tuple(std::cout, "Final Point after " + std::to_string(FEWER_STEPS) + " steps: ", final_point_fewer);
        std::cout << ", f(p): " << std::apply(func_to_optimize, final_point_fewer) << std::endl;
        std::cout << "Execution time for " << FEWER_STEPS << " steps: "
            << std::fixed << std::setprecision(3) << fewer_steps_seconds << " ms" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
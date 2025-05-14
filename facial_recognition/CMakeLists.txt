cmake_minimum_required(VERSION 3.8)
project(facial_recognition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Nájdeme potrebné balíky
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Kompilácia C++ uzla
add_executable(prediction_node src/prediction.cpp)
ament_target_dependencies(prediction_node rclcpp std_msgs OpenCV)

# Inštalácia C++ uzla
install(TARGETS prediction_node
  DESTINATION lib/${PROJECT_NAME})

# Povolenie spúšťania Python uzla
find_package(ament_cmake_python REQUIRED)

# Nastavenie exekutovateľného Python skriptu
install(PROGRAMS
  scripts/recognition_publisher.py
  DESTINATION lib/${PROJECT_NAME})
# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/
)
ament_package()

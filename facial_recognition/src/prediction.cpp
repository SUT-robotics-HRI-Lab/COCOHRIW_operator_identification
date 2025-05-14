#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

class PredictionSubscriberRecognition : public rclcpp::Node {
public:
    PredictionSubscriberRecognition() : Node("prediction_subscriber_recognition") {
        subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "recognition_status", 10,
            std::bind(&PredictionSubscriberRecognition::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::Int32::SharedPtr msg) const {
        RCLCPP_INFO(this->get_logger(), "Received prediction: '%d'", msg->data);
    }

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    std::cout << "Prediction subscriber started" << std::endl;
    rclcpp::spin(std::make_shared<PredictionSubscriberRecognition>());
    rclcpp::shutdown();
    return 0;
}

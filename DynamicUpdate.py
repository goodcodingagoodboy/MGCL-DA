import torch
import torch.nn as nn


class DynamicUpdate:
    def __init__(self, initial_omega=0.5, min_omega=0.3, max_omega=0.95):
        """
        Initialize the dynamic update class.
        Args:
            initial_omega: Initial weight factor.
            min_omega: Minimum weight factor.
            max_omega: Maximum weight factor.
        """
        self.omega = initial_omega
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.prev_SA = None
        self.prev_CA = None

    def update_omega(self, accuracy, threshold_acc=0.67):
        """
        Dynamically adjust omega based on accuracy with a smoother adjustment strategy.
        Args:
            accuracy: Current training accuracy.
            threshold_acc: Accuracy threshold.
        """
        # Compute the gap between accuracy and the threshold
        acc_gap = abs(threshold_acc - accuracy)

        # Sigmoid function to smooth adjustment rate changes
        def sigmoid_scale(x, scale=5.0):
            return 1 / (1 + torch.exp(-scale * x))

        # Normalize acc_gap to the range [0, 0.2] for smoother adjustment
        normalized_gap = min(0.2, acc_gap)

        if accuracy > threshold_acc:  # Accuracy exceeds the threshold
            # Smoother increase: range from 1.02 to 1.2
            adjust_rate = 1.02 + normalized_gap
            new_omega = self.omega * adjust_rate
            # Apply buffer to prevent abrupt changes
            self.omega = min(new_omega, min(self.max_omega, self.omega * 1.2))
        else:  # Accuracy below the threshold
            # Smoother decay: range from 0.98 to 0.8
            adjust_rate = 0.98 - normalized_gap
            new_omega = self.omega * adjust_rate
            # Apply buffer to prevent abrupt changes
            self.omega = max(new_omega, max(self.min_omega, self.omega * 0.8))

    def update_views(self, current_SA, current_CA):
        """
        Update enhanced views.
        Args:
            current_SA: Current self-aware enhanced view.
            current_CA: Current cross-sample enhanced view.
        Returns:
            Updated SA and CA views.
        """
        # Initialize previous views on the first run
        if self.prev_SA is None:
            self.prev_SA = current_SA.clone()
            self.prev_CA = current_CA.clone()
            return current_SA, current_CA

        # Use exponential moving average for smooth updates
        updated_SA = (1 - self.omega) * self.prev_SA + self.omega * current_SA
        updated_CA = (1 - self.omega) * self.prev_CA + self.omega * current_CA

        # Update previous views
        self.prev_SA = updated_SA.clone()
        self.prev_CA = updated_CA.clone()

        return updated_SA, updated_CA

    def reset(self):
        """
        Reset stored states.
        """
        self.prev_SA = None
        self.prev_CA = None
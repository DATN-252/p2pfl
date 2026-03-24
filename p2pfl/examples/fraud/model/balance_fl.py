import torch

class BalanceFL:
    """
    Class implementation for balancing data strategies in Federated Learning.
    Designed to be integrated with fraud detection models to handle extreme class imbalance.
    """

    @staticmethod
    def calculate_class_covariance(features: torch.Tensor, labels: torch.Tensor):
        """
        Calculate the covariance matrix for each class in the provided batch/dataset.

        Args:
            features: Input features tensor of shape (N, feature_dim).
            labels: Labels tensor of shape (N,) or (N, 1).

        Returns:
            dict: A dictionary where keys are class labels (0, 1, ...) and 
                  values are their respective covariance matrices of shape (feature_dim, feature_dim).
        """
        # Ensure labels are 1D for indexing
        if labels.dim() > 1:
            labels = labels.squeeze()

        unique_classes = torch.unique(labels)
        class_covariances = {}

        for cls in unique_classes:
            # Filter features belonging to the current class
            cls_mask = (labels == cls)
            cls_features = features[cls_mask]

            # We need at least 2 samples to calculate covariance
            if cls_features.size(0) > 1:
                # Step 1: Center the data (subtract mean)
                mean = torch.mean(cls_features, dim=0)
                centered_features = cls_features - mean

                # Step 2: Calculate covariance: (X^T * X) / (N - 1)
                # Resulting matrix size: (feature_dim, feature_dim)
                n_samples = cls_features.size(0)
                cov = torch.mm(centered_features.t(), centered_features) / (n_samples - 1)
                
                class_covariances[int(cls.item())] = cov
            else:
                # If only 1 or 0 samples, we can't determine the distribution variance
                class_covariances[int(cls.item())] = None

        return class_covariances

    @staticmethod
    def calculate_selection_probabilities(labels: torch.Tensor, lambda_param: float = 1.0):
        """
        Calculate selection probabilities for each class based on sample counts.
        Formula: pi = mi^lambda / sum(mj^lambda)

        Args:
            labels: Labels tensor of shape (N,) or (N, 1).
            lambda_param: Smoothing parameter (lambda). Default is 1.0.

        Returns:
            dict: A dictionary mapping class labels to their selection probabilities.
        """
        if labels.dim() > 1:
            labels = labels.squeeze()

        unique_labels, counts = torch.unique(labels, return_counts=True)
        counts = counts.float()

        # Calculate mi^lambda
        powered_counts = torch.pow(counts, lambda_param)

        # Calculate sum(mj^lambda)
        sum_powered_counts = torch.sum(powered_counts)

        # Calculate probabilities: pi = mi^lambda / sum(mj^lambda)
        probabilities = powered_counts / sum_powered_counts

        return {int(label.item()): prob.item() for label, prob in zip(unique_labels, probabilities)}

    @staticmethod
    def calculate_max_imbalance_probabilities(labels: torch.Tensor):
        """
        Calculate selection probabilities for each class based on the maximum imbalance.
        Formula: pi = (m_max - mi) / m_max

        Args:
            labels: Labels tensor of shape (N,) or (N, 1).

        Returns:
            dict: A dictionary mapping class labels to their selection probabilities.
        """
        if labels.dim() > 1:
            labels = labels.squeeze()

        unique_labels, counts = torch.unique(labels, return_counts=True)
        counts = counts.float()
        m_max = torch.max(counts)

        if m_max > 0:
            probabilities = (m_max - counts) / m_max
        else:
            probabilities = torch.zeros_like(counts)

        return {int(label.item()): prob.item() for label, prob in zip(unique_labels, probabilities)}

    @staticmethod
    def calculate_global_covariance(features: torch.Tensor, labels: torch.Tensor):
        """
        Calculate the global (pooled) covariance matrix for all labels.
        Formula: covariance = sum(mj * covariance_j) / sum(mj)

        Args:
            features: Input features tensor of shape (N, feature_dim).
            labels: Labels tensor of shape (N,) or (N, 1).

        Returns:
            torch.Tensor: The global covariance matrix of shape (feature_dim, feature_dim).
        """
        if labels.dim() > 1:
            labels = labels.squeeze()

        unique_labels, counts = torch.unique(labels, return_counts=True)
        # Sử dụng hàm calculate_class_covariance để lấy cov của từng class
        class_covs = BalanceFL.calculate_class_covariance(features, labels)

        feature_dim = features.size(1)
        global_cov = torch.zeros((feature_dim, feature_dim), device=features.device)
        total_valid_samples = 0

        for label, count in zip(unique_labels, counts):
            cls_idx = int(label.item())
            cov = class_covs.get(cls_idx)
            
            if cov is not None:
                m_j = count.item()
                global_cov += m_j * cov
                total_valid_samples += m_j

        if total_valid_samples > 0:
            global_cov /= total_valid_samples
            return global_cov
        return None

    @staticmethod
    def compute_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7):
        """
        Compute the Tversky Loss for binary classification.
        Tversky Index = TP / (TP + alpha * FN + beta * FP)
        Loss = 1 - Tversky Index

        Args:
            logits: Predicted logits of shape (N, 1) or (N,).
            targets: Ground truth labels of shape (N, 1) or (N,).
            alpha: Penalty for False Positives (FP). Default is 0.3.
            beta: Penalty for False Negatives (FN). Default is 0.7.
            eps: Small epsilon for numerical stability.

        Returns:
            torch.Tensor: The Tversky loss value.
        """
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.float().view(-1)

        true_pos = (probs * targets).sum()
        false_neg = ((1 - probs) * targets).sum()
        false_pos = (probs * (1 - targets)).sum()

        tversky_index = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)

        return 1 - tversky_index

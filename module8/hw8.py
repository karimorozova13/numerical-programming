import numpy as np

def g(x, X_mean, X_cov, X_inv_cov, X_prob):
    g_i = -1/2 * np.dot(np.dot((x - X_mean), X_inv_cov), (x - X_mean)) - 1/2 * np.log(np.linalg.det(X_cov)) + np.log(X_prob)
    return g_i
# setosa_g = g(X_test[0], np.mean(setosa_data), setosa_cov_matrix, inv_setosa_cov_matrix, setosa_prior_prob)
# print(setosa_g)
# versicolor_g = g(X_test[1], np.mean(versicolor_data), versicolor_cov_matrix, inv_versicolor_cov_matrix, versicolor_prior_prob)
# print(versicolor_g)
# virginica_g = g(X_test[2], np.mean(virginica_data), virginica_cov_matrix, inv_virginica_cov_matrix, virginica_prior_prob)
# print(virginica_g)

def calculate_discriminant_values(X_test, class_means, class_covariances, class_inv_covariances, class_priors):
    num_classes = len(class_means)
    num_samples = X_test.shape[0]
    num_features = X_test.shape[1]

    discriminant_values = np.zeros((num_samples, num_classes))
    probabilities = np.zeros((num_samples, num_classes))

    for i in range(num_classes):
        class_mean = class_means[i]
        class_cov = class_covariances[i]
        class_inv_cov = class_inv_covariances[i]
        class_prior = class_priors[i]

        for j in range(num_samples):
          x = X_test[j]
          discriminant_values[j, i] = g(x, class_mean, class_cov, class_inv_cov, class_prior )

    likelihood = np.exp(discriminant_values - np.max(discriminant_values, axis=1)[:, np.newaxis])
    probabilities = likelihood / np.sum(likelihood, axis=1)[:, np.newaxis]

    return discriminant_values, probabilities
# class_means = [np.mean(setosa_data, axis=0), np.mean(versicolor_data, axis=0), np.mean(virginica_data, axis=0)]
# class_covariances = [setosa_cov_matrix, versicolor_cov_matrix, virginica_cov_matrix]
# class_inv_covariances = [inv_setosa_cov_matrix, inv_versicolor_cov_matrix, inv_virginica_cov_matrix]
# class_priors = [setosa_prior_prob, versicolor_prior_prob, virginica_prior_prob]

# values, likelihood = calculate_discriminant_values(X_test, class_means, class_covariances, class_inv_covariances, class_priors)
# print("Імовірності належності до класів за власною функцією:\n", likelihood)

# print("Значення дискримінантних функцій:\n", values)
# print("\nІмовірності належності до класів:\n", likelihood)
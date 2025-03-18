import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, expon

# Set random seed for reproducibility
np.random.seed(0)

# ============================================
# Problem 1: Simulate from f(x) = (2/a^2)x
# ============================================
a = 2
n_samples = 1000
U = np.random.uniform(0, 1, n_samples)
X = a * np.sqrt(U)  # Inverse transform sampling

# Plot histogram for Problem 1
plt.figure(figsize=(8, 6))
plt.hist(X, bins=30, density=True, alpha=0.6, color='blue', label='Simulated Data')
x = np.linspace(0, a, 1000)
true_pdf = (2 / a**2) * x  # True PDF
plt.plot(x, true_pdf, 'r-', label='True PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Histogram of Simulated Data vs True PDF (Problem 1)')
plt.legend()
plt.savefig('histogram_problem1.png')  # Save the histogram
plt.show()

# ============================================
# Problem 2: Simulate from f(x) = (2/3)e^{-2x} + 2e^{-3x}
# ============================================
n_samples = 1000
U = np.random.uniform(0, 1, n_samples)
X = np.where(U <= 2/3, expon(scale=1/2).rvs(size=n_samples), expon(scale=1/3).rvs(size=n_samples))

# Plot histogram for Problem 2
plt.figure(figsize=(8, 6))
plt.hist(X, bins=30, density=True, alpha=0.6, color='blue', label='Simulated Data')
x = np.linspace(0, 5, 1000)
true_pdf = (2/3) * 2 * np.exp(-2 * x) + (1/3) * 3 * np.exp(-3 * x)  # True PDF
plt.plot(x, true_pdf, 'r-', label='True PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Histogram of Simulated Data vs True PDF (Problem 2)')
plt.legend()
plt.savefig('histogram_problem2.png')  # Save the histogram
plt.show()

# ============================================
# Problem 3: Simulate from Beta(3,3) using accept-reject
# ============================================
n_samples = 500
samples = []
while len(samples) < n_samples:
    Y = np.random.uniform(0, 1)
    U = np.random.uniform(0, 1)
    if U <= (beta.pdf(Y, 3, 3) / 1.875):
        samples.append(Y)

# Plot histogram for Problem 3
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue', label='Simulated Data')
x = np.linspace(0, 1, 1000)
plt.plot(x, beta.pdf(x, 3, 3), 'r-', label='True Beta(3,3) Density')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Histogram of Simulated Data vs True Beta(3,3) Density (Problem 3)')
plt.legend()
plt.savefig('histogram_problem3.png')  # Save the histogram
plt.show()
# Reproducing Figure 2 (Middle) — Influence Functions

This project reproduces the **middle plot of Figure 2** from the paper:

**Understanding Black-box Predictions via Influence Functions**  
*Koh and Liang, ICML 2017*

The goal is to verify that **influence function estimates closely match actual
leave-one-out (LOO) retraining effects**, without explicitly retraining the model
for every training point.

---

## 1. Method Overview

This implementation follows the influence function formulation under the
empirical risk minimization (ERM) framework.

Given a trained model θ̂, the influence of a training point z on the loss
at a test point z_test is approximated as:

I_up,loss(z, z_test)
= - ∇_θ L(z_test, θ̂)^T H^{-1} ∇_θ L(z, θ̂)


Instead of explicitly computing or inverting the Hessian, this project
approximates the **inverse Hessian–vector product (IHVP)** using a **stochastic
Neumann-series-based method (LiSSA)**, as described in **Section 3** of the paper.

---

## 2. Implementation Details

- **Model**: Multiclass logistic regression on MNIST  
- **Optimization**: Full-batch LBFGS for ERM training  
- **Inverse Hessian approximation**: LiSSA-style stochastic estimation  
- **Validation**: Actual leave-one-out retraining using LBFGS warm start  
- **Visualization**: Scatter plot comparing predicted and actual loss change  

---

## 3. Reproduced Figure

Each point in the reproduced figure corresponds to **one training sample**
(only the **top-k samples with the largest absolute influence** are shown).

- **x-axis**: Actual change in test loss from leave-one-out retraining  
- **y-axis**: Influence function linear approximation  
- **Reference line**: The diagonal \(y = x\) indicates perfect agreement  

---

## Conclusion

The **Pearson correlation is 0.9694**, showing that influence function predictions
closely match the actual LOO retraining results.

The reproduced plot is more scattered than the original figure in the paper.
This is expected for several reasons:

- The inverse Hessian–vector product is approximated using a **stochastic method**,
  which introduces noise  
- Only **top-k high-influence training points** are visualized  
- Leave-one-out retraining itself is a **numerical approximation** with limited
  optimization steps  

---

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib
python reproduce_fig2_mid.py



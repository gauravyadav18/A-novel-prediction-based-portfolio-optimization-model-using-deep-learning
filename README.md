This repository implements the AE+LSTM+OMEGA model, a novel prediction-based portfolio optimization framework that 
combines deep learning with robust optimization techniques to enhance portfolio returns.
* The AE+LSTM+OMEGA framework consists of:
* Autoencoder (AE): For dimensionality reduction and feature extraction from historical returns.
* LSTM Network: For predicting stock returns based on AE-extracted features.
* Worst-case Omega Model: A robust optimization approach using predicted returns and errors to construct an efficient portfolio.

Key Features:

* Combines unsupervised learning (AE) and time series forecasting (LSTM).
* Integrates a robust Omega ratio-based optimization for portfolio construction.
* Outperforms equally-weighted and existing state-of-the-art prediction-based portfolio strategies.
 Download Paper PDF:  [(PDF)]([https://doi.org/10.1016/j.cie.2023.109023](https://www.sciencedirect.com/science/article/pii/S0360835223000475?casa_token=CGwPkcOTMvcAAAAA:dt-1bn_Pq4Od0iNFlVrlYxgDV0EbNXvrGyA46-SsI8m_LPsJkG2UrxXN0aahwbEf7hKxI1wuog))

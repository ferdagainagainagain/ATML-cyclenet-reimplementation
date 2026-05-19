1.⁠ ⁠Introduction and Method Background
   1.1 Motivation: long-horizon time series forecasting
   1.2 Required time series background
   1.3 CycleNet paper summary
   1.4 Residual Cycle Forecasting
   1.5 Assumptions and expected limitations

2.⁠ ⁠Reimplementation and Experiments
   2.1 Implementation details
       - PyTorch reimplementation
       - Linear baseline
       - CycleNet-Linear
       - MLP / CycleNet-MLP
       - CycleiTransformer ablation
   2.2 Experimental setup
       - Synthetic dataset
       - ETTh1 dataset
       - Train/validation/test split
       - Preprocessing and normalization
       - Metrics: MSE and MAE
       - Hyperparameters, seeds, hardware
   2.3 Synthetic experiment
       - Full-cycle visible case
       - Partial-cycle visible case
       - Learned cycle visualization
   2.4 ETTh1 experiments
       - Linear vs CycleNet-Linear
       - Horizon study
       - Cycle length sensitivity
       - MLP backbone
       - CycleiTransformer ablation

3.⁠ ⁠Discussion and Conclusion
   3.1 Summary of reproduced findings
   3.2 What worked and what did not
   3.3 Limitations of our reproduction
   3.4 Relation to other papers / future work
   3.5 Final conclusion
   References
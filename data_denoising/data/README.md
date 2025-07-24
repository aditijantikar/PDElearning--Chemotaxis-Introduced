We investigate methods for discovering the governing parameters of a partial differential equa-
tion (PDE) model for 1D chemotaxis from spatiotemporal data. Recent progress in this area
includes sparse regression via SINDy and Physics-Informed Neural Networks (PINNs), both of
which present unique challenges in the presence of noise. We analyze the performance and key
vulnerabilities of both a Finite-Difference SINDy (FD-SINDy) framework and a PINN-based ap-
proach for this parameter discovery task. We then develop a novel, decoupled methodology that
uses an artificial neural network (ANN) to denoise data and accurately approximate partial deriva-
tives. We test these frameworks on the 1D chemotaxis PDE, a canonical model for biological
transport. Our results highlight the critical challenges associated with derivative estimation and
optimization in existing frameworks and show that our proposed decoupled approach provides a
more robust path toward accurate model discovery from imperfect data.


You can view the detailed report here: [BENG_227_Project (1).pdf](https://github.com/user-attachments/files/21415620/BENG_227_Project.1.pdf)

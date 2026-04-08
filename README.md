# PINO-for-wave-data-assimilation
The repository contains code related to the paper "Bridging ocean wave physics and deep learning: Physics-informed neural operators for nonlinear wavefield reconstruction in real-time" by Ehlers et al. (2025), which explores using PINOs to rapidly reconstruct wave fields from sparse measurement data without providing ground truth data for training.


**DOI:** 

[10.48550/arXiv.2508.03315](https://doi.org/10.48550/arXiv.2508.03315) preprint v1

[https://doi.org/10.1063/5.0294655](https://doi.org/10.1063/5.0294655) published version



**Key Features:**

*   **PINO-based wave data assimilation framework:**  Demonstrates the ability to reconstruct spatio-temporal phase-resolved wave fields from sparse measurements by incorporating physical constraints.
*   **No fully-resolved ground truth data required for training:**: Works from sparse measurements only, e.g. buoy time series or radar snapshots
*   **Embeds free-surface wave physics directly into the loss function:** The surface velocity potential is constructed from the surface elevation via a HOSM-related formulation, and the free surface boundary conditions (deduced from potential flow) constrain the loss function during training to account for the missing ground truth data and fill the measurement gaps in physiccally consistent way.
*   **PINO allows for rapid wave field reconstruction:** Learns a generalizable inverse operator instead of solving only a single case, implying that inference is fast for new inputs, once the PINO is trained.
*   **Code Example for case A:** This repository provides a code example specifically for the setup and results presented in Section 3A of the paper, showcasing the PINO implementation for the case where wave fields are reconstructed from point-sensor data (buoys).


**Why this is different from a PINN?**

Unlike a classical Physics-Informed Neural Network (PINN, [my implementation](https://github.com/svenjahlrs/PINN-for-ocean-waves)), which is typically trained for a single realization or single inverse problem ("neural solver"), this repository implements a Physics-Informed Neural Operator (PINO).
This means the model learns a mapping operator from sparse measurements to full nonlinear wavefields across many wave conditions, rather than solving only one case at a time.
As a result, the PINO offers better generalization, requires no retraining for each new wave instance, allows for much faster inference, and thus greater practical relevance for real-time ocean applications.


**Paper Abstract:**

Accurate real-time reconstruction of phase-resolved ocean wave fields remains a critical yet largely unsolved problem, primarily due to the absence of practical data assimilation methods for obtaining initial conditions from sparse or indirect wave measurements. While recent advances in supervised deep learning have shown potential for this purpose, they require large labeled datasets of ground truth wave data, which are infeasible to obtain in real-world scenarios. To overcome this limitation, we propose a physics-informed neural operator (PINO) framework for reconstructing spatially and temporally phase-resolved, nonlinear ocean wave fields from sparse measurements, without the need for ground truth data during training. This is achieved by embedding residuals of the free surface boundary conditions of ocean gravity waves into the loss function, constraining the solution space in a soft manner. In the current implementation, the framework is demonstrated for long-crested, unidirectional wave surfaces, where the wave propagation direction is aligned with the radar scanning direction. Within this setting, we validate our approach using highly realistic synthetic wave measurements by demonstrating the accurate reconstruction of nonlinear wave fields from both buoy time series and radar snapshots. Our results indicate that PINOs enable accurate, real-time reconstruction and generalize robustly across a wide range of wave conditions, thereby paving the way for future extensions of this framework toward multidirectional sea states and thus operational wave reconstruction in realistic marine environments.




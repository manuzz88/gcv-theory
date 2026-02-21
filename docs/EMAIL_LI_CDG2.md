# Email to Li et al. — GCV prediction for CDG-2

**To**: Dayi Li (first author), Roberto Abraham, Pieter van Dokkum
**Subject**: A priori velocity dispersion prediction for CDG-2 from modified gravity (GCV)

---

Dear Dr. Li and collaborators,

I read with great interest your recent paper on CDG-2 (arXiv:2506.15644), the candidate dark galaxy in the Perseus cluster. The detection of a galaxy purely through its globular cluster population is a remarkable achievement.

I am writing because CDG-2 provides an ideal test case for a modified gravity framework I have been developing, called Vacuum Coherence Gravity (GCV). In GCV, the quantum vacuum responds to local matter density through a susceptibility function, reproducing the MOND phenomenology at galactic scales while extending to cosmological scales via a modified Boltzmann solver (CLASS).

**The key point**: CDG-2 sits deep in the low-acceleration regime (g_N/a_0 ≈ 0.006 at R_eff), where GCV predicts a large gravitational enhancement from vacuum susceptibility — naturally explaining the extreme apparent dark matter fraction without invoking dark matter particles.

**A priori prediction (no free parameters beyond a_0 = 1.2 × 10⁻¹⁰ m/s²)**:

| Baryonic mass | Predicted σ (km/s) |
|---|---|
| M_b = 6 × 10⁶ M☉ (low) | 17.6 |
| M_b = 1.2 × 10⁷ M☉ (best) | **20.9** |
| M_b = 1.8 × 10⁷ M☉ (high) | 23.1 |

For comparison, the Newtonian prediction (no dark matter) gives σ ≈ 6 km/s.

This prediction is derived from σ ~ (G · M_b · a_0)^{1/4}, which is the standard deep-MOND estimator for pressure-supported systems. It is published in our preprint (DOI: 10.5281/zenodo.18723731, Section 5) and the analysis code is publicly available at:

- Paper: https://doi.org/10.5281/zenodo.17505641
- Code (Script 139): https://github.com/manuzz88/gcv-theory

The prediction was committed to the public repository on February 21, 2026 — before any spectroscopic measurement of CDG-2's velocity dispersion, ensuring it cannot be adjusted post-hoc.

I understand you noted in your paper that "it is essential to conduct further observations with high precision kinematic and spectroscopy data to constrain and confirm the dark matter content of CDG-2." If such observations are planned, I would be very interested to know whether the measured velocity dispersion is consistent with ~21 km/s.

I want to be transparent: GCV is a preprint by an independent researcher, not yet peer-reviewed. The framework has shown promising preliminary results (Δχ² = -17.70 vs ΛCDM on a simplified CMB+LSS likelihood, S8 tension reduced from 3.9σ to 2.6σ), but a full MCMC with official Planck likelihoods is still pending. I make no strong claims — only a falsifiable prediction.

Thank you for your excellent work, and I would welcome any feedback.

Best regards,
Manuel Lazzaro
Independent Researcher, Italy
manuel.lazzaro@me.com
https://github.com/manuzz88/gcv-theory

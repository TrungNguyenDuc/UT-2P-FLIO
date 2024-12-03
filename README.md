# Retinal Fluorescence Lifetime Imaging Software

## Overview

This software provides advanced tools for monitoring the visual cycle, analyzing fluorescence lifetime data, and unmixing fluorescence signals from retinal structures. It is designed to assist researchers in ophthalmology, retinal biology, and fluorescence lifetime imaging, offering powerful and efficient methods for studying the retina and its complex processes. This software was used with our Two-photon fluorescence lifetime ophthalmoscopy (2P-FLIO) https://opg.optica.org/boe/fulltext.cfm?uri=boe-15-5-3094&id=549059.

## Key Features

### 1. Understanding the Visual Cycle
This software focuses on the visual cycle, specifically the interaction between light and dark adaptation, retinoid isomerization, and the functioning of photoreceptors and the retinal pigment epithelium (RPE). By tracking the biochemical processes in the retina during light exposure and recovery in the dark, it provides in-depth insights into retinal health. The software is designed to measure fluorescence signals associated with these processes, giving researchers a unique way to observe how the visual cycle affects retinoid metabolism.

- **Light/Dark Adaptation:** Study the adaptive processes of the retina in response to changes in lighting conditions.
- **Retinoid Isomerization:** Track the conversion of retinoids in photoreceptors and RPE cells, essential for vision.
- **Photoreceptor and RPE Functioning:** Evaluate how light exposure and dark recovery impact cellular activity and metabolism.

### 2. Rapid Analysis of Fluorescence Lifetime Data
The software enables efficient analysis of multi-exponential fluorescence decay data using **Discrete Fourier Transform (DFT)** to generate phasor plots. Traditional fitting methods often require trial-and-error to estimate the parameters of complex exponential decays, which can be time-consuming and computationally intensive. This software overcomes these limitations by directly transforming decay curves into phasor plots without the need for fit-based approaches.

- **Phasor Plot Generation:** The DFT method produces phasor plots that represent the decay of fluorescence lifetimes, offering a more intuitive and faster way to analyze data.
- **Fit-Free Analysis:** By eliminating the need for parameter fitting, this method significantly speeds up the analysis of large datasets, particularly those with low photon counts.
- **Efficient Data Processing:** The software is optimized for high-speed data processing, enabling real-time analysis of fluorescence lifetime images.

### 3. Unmixing Fluorescence Signals
One of the key challenges in fluorescence lifetime imaging is the separation of signals from multiple fluorophores, especially when they share overlapping emission spectra or when the signal-to-noise ratio is low. To address this, the software uses **Gaussian Mixture Models (GMM)** to unmix fluorescence signals from different retinal structures, such as photoreceptors and the RPE.

- **Gaussian Mixture Models (GMM):** GMM is applied to decompose complex fluorescence lifetime data into distinct components corresponding to different structures in the retina.
- **Signal Separation:** The software effectively isolates signals from multiple fluorophores by using lifetime characteristics to differentiate them.
- **Overcoming Low Signal-to-Noise Ratios:** The GMM approach works effectively even with challenging data, including low signal-to-noise ratios, by modeling the fluorescence decay profiles of each fluorophore individually.

## How It Works

### Data Input and Preprocessing
1. **Fluorescence Lifetime Imaging:** The software accepts raw fluorescence lifetime imaging data, which contains the decay signals from different regions of the retina. These signals are typically collected using a time-correlated single photon counting (TCSPC) system or similar equipment.
   
2. **Data Preprocessing:** The input data is first preprocessed to filter out noise and artifacts. This involves the application of basic denoising techniques and alignment of the data from multiple time points.

### Phasor Plot Generation (DFT Analysis)
1. **Discrete Fourier Transform (DFT):** The multi-exponential decay curves from each pixel in the image are processed using DFT. This transform converts the decay data into phasor plots, where each point represents the phase and frequency of the fluorescence decay.

2. **Visualizing Fluorescence Lifetimes:** The phasor plot provides an intuitive visual representation of fluorescence lifetimes, where the color and position of each point on the plot correspond to the characteristics of the fluorophores in the image. This method simplifies the analysis of complex datasets and allows researchers to quickly identify distinct fluorescence lifetimes.

### Signal Unmixing (Gaussian Mixture Models)
1. **Gaussian Mixture Models (GMM):** The software applies GMM to separate overlapping fluorescence signals. It assumes that the fluorescence decay curves from different structures (e.g., photoreceptors and RPE) are composed of Gaussian distributions, and the model fits each signal component individually.

2. **Signal Decomposition:** Each GMM component corresponds to a distinct fluorophore or structure in the retina. The software calculates the expected lifetime and amplitude of each signal, effectively separating the contributions of each fluorophore.

3. **Iterative Optimization:** The GMM algorithm iterates over the data, adjusting the parameters of the Gaussian distributions to minimize the difference between the observed and predicted signals. This results in a high-fidelity separation of the fluorescence signals.

### Results Visualization and Export
- **Visual Outputs:** After the data is processed, the software generates both phasor plots and unmixed signal maps for each retinal structure. These outputs are visually presented for easy interpretation.
- **Export Data:** Processed data, including fluorescence lifetimes and unmixed signal maps, can be exported in various formats (e.g., CSV, TIFF) for further analysis or publication.

## Target Audience

This software is intended for researchers working in the following areas:

- **Ophthalmology**  
- **Retinal Biology**  
- **Fluorescence Lifetime Imaging**

Potential users include academic labs, vision research centers, and biomedical companies focused on retinal diagnostics and therapeutics.

## Installation



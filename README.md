# FCBFormer

Official codebase for: [FCN-Transformer Feature Fusion for Polyp Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-12053-4_65) (MIUA 2022 paper)

Authors: [Edward Sanderson](https://scholar.google.com/citations?user=ea4c7r0AAAAJ&hl=en&oi=ao) and [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

## 1. Overview

### 1.1 Abstract

Colonoscopy is widely recognised as the gold standard procedure for the early detection of colorectal cancer (CRC). Segmentation is valuable for two significant clinical applications, namely lesion detection and classification, providing means to improve accuracy and robustness. The manual segmentation of polyps in colonoscopy images is timeconsuming. As a result, the use of deep learning (DL) for automation of polyp segmentation has become important. However, DL-based solutions can be vulnerable to overfitting and the resulting inability to generalise to images captured by different colonoscopes. Recent transformer-based architectures for semantic segmentation both achieve higher performance and generalise better than alternatives, however typically predict a segmentation map of $\frac{h}{4} × \frac{w}{4}$ spatial dimensions for a $h \times w$ input image. To
this end, we propose a new architecture for full-size segmentation which leverages the strengths of a transformer in extracting the most important features for segmentation in a primary branch, while compensating for its limitations in full-size prediction with a secondary fully convolutional branch. The resulting features from both branches are then fused for final prediction of a $h × w$ segmentation map. We demonstrate our method’s state-of-the-art performance with respect to the mDice, mIoU, mPrecision, and mRecall metrics, on both the Kvasir-SEG and CVC-ClinicDB dataset benchmarks. Additionally, we train the model on each of these datasets and evaluate on the other to demonstrate its superior generalisation performance.

### 1.2 Architecture

<p align="center">
	<img width=900, src="Images/FCBformer.jpg"> <br />
	<em>
		Figure 1: Illustration of the proposed FCBFormer architecture
	</em>
</p>

### 1.3 Qualitative results

<p align="center">
	<img width=900, src="Images/Comparison.png"> <br />
	<em>
		Figure 2: Comparison of predictions of FCBFormer against baselines. FF is FCBFormer, PN is PraNet, MN is MSRF-Net, R++ is ResUNet++, UN is U-Net
	</em>
</p>

<p align="center">
	<img width=900, src="Images/FCB_benefit.png"> <br />
	<em>
		Figure 3: Visualisation of the benefit of the fully convolutional branch (FCB)
	</em>
</p>

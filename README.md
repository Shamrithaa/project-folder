# Classification of Bone Fracture

This project focuses on detecting bone fractures in X-ray images using a deep learning model based on ResNet18. The application is built with PyTorch and deployed using Streamlit to provide an intuitive interface for users to upload and analyze X-ray images.

## Project Overview

Accurate and timely detection of bone fractures is critical in medical diagnostics. This project provides an automated solution that classifies X-ray images into two categories:

- Fractured
- Not Fractured

The model is trained using a dataset of labeled X-ray images and is capable of making predictions along with confidence scores. The application also computes accuracy on a provided test dataset, if available.

## Features

- Deep learning model built with ResNet18
- Image preprocessing using TorchVision transforms
- Streamlit-based web interface for interactive use
- Softmax-based confidence scoring for predictions
- Optional Docker support for deployment

## Project Structure


## Getting Started

### Prerequisites

- Python 3.8 or later
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/classification-of-bone-fracture.git
cd classification-of-bone-fracture

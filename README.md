# Movie Prediction Project

## Overview

This project uses machine learning techniques to make movie-related predictions based on the user's summary.

## Project Structure

**Dataset**

The `./datasets/` folder includes the dataset preparation steps inside the related Jupyter notebook.

**Model**

The `./model/` folder includes the Python files related to the model training and testing processes.

**Backend**

The `./backend/` folder includes the Python files related to deployment and making predictions, using Flask for integration.

**Frontend**

The `./frontend/` folder includes the web application that interacts with the backend. It is built with Vue.js.

## Setup

1. Clone the repository
```bash
git clone https://github.com/nethbotheju/movie-prediction.git
cd movie-prediction
```

2. Backend
```bash
cd backend

# Create the environment using Conda
conda env create -f environment.yml

# Run the backend server
python serve_model_k2.py
```

3. Frontend
```bash
cd frontend

# Install the Node.js modules
npm install

# Run the frontend development server
npm run dev
```
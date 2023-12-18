# Document Question and Answering System using Langchain, Qdrant, Streamlit 

## Project Overview

This project aims to create a Squad application utilizing Langchain, Streamlit, and Qdrant. The application allows users to interact with a squad-related dataset using a user-friendly interface powered by Streamlit. Qdrant is used for efficient vector similarity search, and Langchain facilitates language-related functionalities.

## Project Structure

- **app.py:** The main script containing the Streamlit application code. It provides the user interface and interacts with Langchain and Qdrant to display squad-related information.

- **htmlTemplate.py:** This Python script includes the HTML template for custom styling using CSS.

- **requirements.txt:** This file lists all the dependencies required to run the project. Install them using `pip install -r requirements.txt`.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/minhtien2405/Document-Question-and-Answering-System.git
   cd Document-Question-and-Answering-System

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

3. **Run the Docker container:**

   ```bash
   docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
4. **Run the Streamlit application:**

   ```bash
   streamlit run app.py

Visit http://localhost:8501 in your web browser to interact with the Squad application.

## Customization
HTML/CSS Styling:
Customize the appearance of the application by modifying the HTML template in *htmlTemplate.py* according to your design preferences.

Integration with Langchain and Qdrant:
Adjust the code in *app.py* to integrate with Langchain and Qdrant based on your specific squad-related dataset and requirements.


   

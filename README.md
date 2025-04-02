# nexus_gstdost_

Project Overview

This repository contains a Streamlit-based web application (app.py), along with multiple HTML and JavaScript files for different arcade-style games, a FAISS-based document retrieval system, and various supporting assets.

Table of Contents

Installation

Usage

Files and Directories



*Installation*

To set up the project, follow these steps:

Clone this repository:

git clone https://github.com/anubhav-nekko/nexus_gstdost.git

Navigate to the project directory:

cd nexus_gstdost

Install dependencies:

pip install -r requirements.txt

*Usage*

Run the Streamlit app:

streamlit run app.py

You can also open individual HTML files in a browser to play the arcade-style games.

*Files and Directories*

app.py - The main Streamlit application.

HTML Files (atari.html, snake.html, space.html, surfer.html) - Different web-based games.

script.js - JavaScript logic for interactive elements.

style.css - Stylesheet for the games and web UI.

faiss_index.bin - FAISS-based document indexing file.

metadata_store.pkl - Stores metadata for document retrieval.

chat_history.json - JSON file to store chat history.

logo.png - Logo image for the application.

requirements.txt - List of dependencies for the Streamlit app.
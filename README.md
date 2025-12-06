        Instructions for Running the Code:\
1. Environment Setup: Python installed.\
2. Dependencies: Install all required libraries:\
` pip install scikit-learn numpy librosa `\
3. Data Placement: Create a folder named data in the same directory as the Python\
script. Inside data, create subfolders for each genre (e.g., chief keef, kanye) and place\
the corresponding .wav or .mp3 files inside.\
4. Execution: Run the script from the command line. The first run will perform the slow\
feature extraction and save the results to audio_features_cache.npz. Subsequent runs\
will load the cache instantly.\
typebeat-ai.py}